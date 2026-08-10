

import timm
import torch.nn
from .. import common
import torch
from ..memory import MemoryBank
import torchvision
from typing import List
import cv2
import math
import numpy as np


class ViTWithMemoryBank(torch.nn.Module):
    def __init__(
            self, model_config: common.ModelConfig = None,
            # backbone_name: str = "vit_small_patch16_dinov3.lvd1689m",
            # layer_indices = [1],
            ):
        super().__init__()
        if model_config is not None:
            image_size = model_config.image_size
            layer_indices = model_config.layer_indices
            memory_size = model_config.memory_size
            backbone_name = model_config.backbone_name
    
            if len(model_config.checkpoint_path) > 0:
                data = torch.load(model_config.checkpoint_path, weights_only=False)
                self.image_size = data.get("image_size", image_size)
                self.layer_indices = data.get("layer_indices", layer_indices)
                self.memory_size = data.get("memory_size", memory_size)
                self.backbone_name = data.get("backbone_name", backbone_name)
            else:
                print(backbone_name)
                # print(self.state_dict().keys())
                # exit(-1)
                self.image_size = image_size
                self.layer_indices = layer_indices
                self.memory_size = memory_size
                self.backbone_name = backbone_name
                pass
            pass
        else:
            raise RuntimeError("model_config is None")

        timm_args = {
            "model_name": self.backbone_name,
            "pretrained": True,
            "num_classes": 0,
        }
        self.temperature = 0.1

        if self.backbone_name.startswith("vit_"):
            timm_args["dynamic_img_size"] = True
        elif self.backbone_name.startswith("wide_resnet"):
            pass
        else:
            raise RuntimeError(f"backbone_name {self.backbone_name} is not supported")
        
        self.backbone = timm.create_model(**timm_args)

        dim = self.backbone.num_features
        self.memories = torch.nn.ModuleList(
            [MemoryBank(size=self.memory_size, dim=dim, max_size=3000000) for _ in self.layer_indices]
        )
        self.register_buffer("middle_distance", torch.tensor(0.5))
        self.register_buffer("scale_distance", torch.tensor(1.0))
        self.train_backbone = False
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        if len(model_config.checkpoint_path) > 0:
            self.load(model_config.checkpoint_path)
            pass

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def to(self, device):
        super().to(device)
        self.device = device
        pass

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        # forward pass
        intermediates = self.backbone.forward_intermediates(x)[1]

        embs = []
        for i, emb in enumerate(intermediates):
            if emb.shape[-1] * emb.shape[-2] > 1024:
                # downsample
                emb = torch.nn.functional.interpolate(emb, size=(32, 32), mode="bilinear", align_corners=False)
                pass
            emb = emb.permute(0, 2, 3, 1).reshape(emb.shape[0], -1, emb.shape[1])
            embs.append(emb)
            pass

        return embs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediates  = self.forward_backbone(x)

        return intermediates
        pass
    
    def add_memory(self, forward_result, memory_index):
        intermediates = forward_result
        layer_index = self.layer_indices[memory_index]
        embeddings = intermediates[layer_index]
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        self.memories[memory_index].update(embeddings)
        pass

    def compute_loss(self, forward_result):
        dists, _ = self.compute_distance_every_layer(forward_result)

        dists = torch.stack(dists, dim=-1)
        # dists: [B, num_patches, num_layers]
        dists = dists.reshape(-1)
        weights = torch.softmax(dists / self.temperature, dim=-1).detach() * dists.shape[0]

        loss = dists * weights

        return loss.mean()


    def shrink_memory(self, memory_index):
        self.memories[memory_index].shrink()
        pass

    def compute_distance_every_layer(self, forward_result):
        intermediates = forward_result
        dists = []
        indices = []
        for i, ilayer in enumerate(self.layer_indices):
            embeddings = intermediates[ilayer]
            bsize = embeddings.shape[0]

            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
            # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            dist, idx = self.memories[i].compute_min_distance(embeddings)
            # dist = (dist - self.memories[i].dist_mean) / (self.memories[i].dist_std + 1e-8)

            dist = dist.reshape((bsize, -1))
            idx = idx.reshape((bsize, -1))
            dists.append(dist)
            indices.append(idx)
            pass
        
        return dists, indices
    
    def compute_distance(self, forward_result):
        dists, indices = self.compute_distance_every_layer(forward_result)

        max_dist, idx_max_dist = torch.max(torch.stack(dists, dim=0), dim=0)
        # max_dist: [batch, num_patches], idx_max_dist: [batch, num_patches]
        max_indices = torch.stack(indices, dim=0)
        max_indices = max_indices[idx_max_dist, torch.arange(idx_max_dist.shape[0]).unsqueeze(-1), torch.arange(idx_max_dist.shape[1]).unsqueeze(0)]
        
        return max_dist, max_indices

    def postprocess(self, forward_result, num_neighbours=1):
        max_dist, max_idx = self.compute_distance(forward_result)
        # score = self.compute_anomaly_score(forward_result, max_dist, max_idx, num_neighbours=9)
        score = torch.sigmoid((score - self.middle_distance) * self.scale_distance)
        # use sigmoid
        # proba = torch.sigmoid((max_dist - self.middle_distance) * self.scale_distance)
        min_max_dist = max_dist.min(dim=-1, keepdim=True)[0]
        max_max_dist = max_dist.max(dim=-1, keepdim=True)[0]
        proba = (max_dist - min_max_dist) / (max_max_dist - min_max_dist + 1e-8)

        return proba, score


    def get_default_transforms(self):
        transforms = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.ToPILImage(),
            torchvision.transforms.v2.Resize((self.image_size, self.image_size)),
            # NormalizeContrast(),
            torchvision.transforms.v2.GaussianBlur(kernel_size=5, sigma=1.0),
            torchvision.transforms.v2.ToTensor(),
            torchvision.transforms.v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        return transforms
    
    def imgs2batch(self, imgs: List[np.ndarray]) -> torch.Tensor:
        transforms = self.get_default_transforms()

        imgs = [transforms(img) for img in imgs]
        batch = torch.stack(imgs, dim=0)
        return batch
        pass

    def set_distance_stats(self, stats):
        middle_dist, max_dist = stats
        # max_dist should have the probability of 0.95
        # compute the scale of sigmoid((d-middle_dist) * scale) = 0.95
        scale = - torch.log(torch.tensor(0.05)) / (max_dist - middle_dist)
        self.middle_distance.copy_(torch.tensor(middle_dist))
        self.scale_distance.copy_(torch.tensor(scale))
        pass

    def distance2proba(self, stats, dists):
        middle_dist, max_dist = stats
        scale = - np.log(0.05) / (max_dist - middle_dist)
        proba = 1 / (1 + np.exp(-(dists - middle_dist) * scale))
        return proba

    def set_middle_probability(self, proba):
        dist = torch.sqrt(self.middle_distance * ((1 - proba) / proba))
        self.set_middle_distance(dist)
        pass

    def generate_heatmap(self, proba: np.array, img: np.array):
        # convert map of probability to visualizable heatmap
        heatmap = (proba * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
        
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

        return heatmap


    def predict(
            self, imgs: List[np.ndarray], is_bgr=True,
            return_heatmap=False, num_neighbours=1) -> List[common.PredictionResult]:
        if is_bgr:
            for i in range(len(imgs)):
                imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
                pass
            pass
        
        original_sizes = [img.shape[:2] for img in imgs]

        batch = self.imgs2batch(imgs)
        batch = batch.to(self.device)
        with torch.no_grad():
            forward_result = self.forward(batch)
            preds, scores = self.postprocess(forward_result, num_neighbours=num_neighbours)
            pass
        # pred: [B, E, E, 1]
        preds = preds.squeeze(-1)
        results = []
        embedding_h = int(math.sqrt(preds.shape[1]))
        assert embedding_h * embedding_h == preds.shape[1], "Embedding size is not a perfect square"
        for original_size, pred, score in zip(original_sizes, preds, scores):
            pred = pred.cpu().numpy()
            score = score.cpu().numpy()
            pred = pred.reshape(embedding_h, embedding_h)

            pred = cv2.resize(
                pred, (original_size[1], original_size[0]))
            prediction = common.PredictionResult(score=score)
            results.append(prediction)
            pass
        
        if return_heatmap:
            for i in range(len(results)):
                pred = preds[i].cpu().numpy()
                pred = pred.reshape(embedding_h, embedding_h)
                pred = cv2.resize(
                    pred, (original_sizes[i][1], original_sizes[i][0]))
                results[i].heat_map = self.generate_heatmap(
                    pred, imgs[i])
                pass
            
        return results


    def compute_stats(self, i):
        self.memories[i].compute_stats()
        pass

    def update_stats(self, embeddings, i):
        self.memories[i].update_stats(embeddings)
        pass

    def save(self, checkpoint_path):
        state_dict = self.state_dict()
        image_size = self.image_size
        torch.save({
            "state_dict": state_dict,
            "image_size": image_size,
            "layer_indices": self.layer_indices,
            "memory_size": self.memory_size,
            "backbone_name": self.backbone_name,
        }, checkpoint_path)
        pass

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            self.load_state_dict(state_dict, strict=True)
            pass
        else:
            # according to old version, directly load state dict
            self.load_state_dict(checkpoint)
            pass
        pass

    def set_train_backbone(self, train_backbone: bool):
        self.train_backbone = train_backbone
        self.backbone.requires_grad_(train_backbone)
        pass

    def train(self, mode=True):
        if mode:
            if self.train_backbone:
                self.backbone.train()
            else:
                self.backbone.eval()
                pass
            for memory in self.memories:
                memory.train()
                pass
            pass
        else:
            self.backbone.eval()
            for memory in self.memories:
                memory.eval()
                pass
            pass
        pass

    def get_param_dict(self, lr):
        param_dict = {}

        params = [p for p in self.parameters() if p.requires_grad]

        param_dict["params"] = params
        param_dict["lr"] = lr

        return [param_dict]
    pass
