import torch
import torch.distributed
import torchvision.datasets 
import torchvision.transforms
import os
from hq_anomaly import common
from hq_anomaly import models
import sys
from tqdm import tqdm
import numpy as np
from .datasets import ImageSingleFolder


def create_model(config: common.ModelConfig) -> torch.nn.Module:
    # model = models.AutoEncoderViT()
    model = models.DistillViT2()
    return model

def train(config: common.TrainConfig):
    image_size = config.modelConfig.image_size
    data_path = config.data_path

    if "LOCAL_RANK" not in os.environ:
        print("Not using distributed training")
        device = torch.device(f"cuda:{config.devices[0]}" if torch.cuda.is_available() else "cpu")
        rank = 0
    else:
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
        acc = torch.accelerator.current_accelerator()
        backend = torch.distributed.get_default_backend_for_device(acc)
        
        torch.distributed.init_process_group(backend)
        rank = torch.distributed.get_rank()
        print(f"Start running basic DDP example on rank {rank}.")
        # create model and move it to GPU with id rank
        device_id = rank % torch.accelerator.device_count()
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        pass

    # data loader
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])
    train_path = os.path.join(data_path, 'train', 'good')
    valid_path = os.path.join(data_path, 'valid')
    
    train_dataset = ImageSingleFolder(
        folder=train_path, transform=transforms)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_data_workers,
        sampler=torch.utils.data.DistributedSampler(train_dataset) if torch.distributed.is_initialized() else torch.utils.data.RandomSampler(train_dataset),
    )

    model = create_model(config.model_config)
    model.to(device)
    if torch.distributed.is_initialized():
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
    else:
        ddp_model = model
        pass

    optimizer = torch.optim.AdamW(model.get_param_dict(1e-4))
    scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, total_iters=config.num_epochs,
            end_factor=config.lr_min / config.lr0
    )
    os.makedirs(config.output_path, exist_ok=True)
    for i_epoch in range(config.num_epochs):
        if rank == 0:
            bar = tqdm(train_loader)
        else:
            bar = train_loader
            pass
        train_losses = []
        infos = dict()
        # model.set_require_grad_(i_epoch)
        for i_batch, images in enumerate(bar):
            images = images.to(device)
            forward_result = ddp_model(images)
            loss, info = model.compute_loss(forward_result)
            train_losses.append(loss.item())
            for key, value in info.items():
                if key in infos:
                    infos[key].append(value)
                else:
                    infos[key] = [value]
                    pass
                pass
            if rank == 0:
                bar.set_postfix({k: np.mean(v) for k, v in infos.items()} | {"epoch": i_epoch})
                pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            pass

        scheduler.step()

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                model.save(os.path.join(config.output_path, "ckpt.pth"))
        else:
            model.save(os.path.join(config.output_path, "ckpt.pth"))
            pass
        pass
    pass


if __name__ == "__main__":
    train_config = common.TrainConfig(
        data_path=sys.argv[1],
        batch_size=16,
        num_epochs=200,
        num_data_workers=16,
        modelConfig=common.ModelConfig(
            image_size=512,
        ),
    )
    train(train_config)
    pass