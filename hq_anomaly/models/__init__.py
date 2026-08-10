

import torch.nn
import timm
import copy
# from vector_quantize_pytorch import VectorQuantize
import torch.distributed
from functools import partial
from timm.models.layers import LayerNorm
from hq_anomaly.memory import MemoryBank
import numpy as np
from typing import List
import cv2
import torch
from hq_anomaly import common
import torchvision.transforms.v2
import math
from PIL import Image
from .vit_with_memory_bank import ViTWithMemoryBank

class Quantize(torch.nn.Module):
    def __init__(self, num_bins=4):
        super().__init__()
        self.bin_value = 255 // num_bins

    
    def forward(self, img, bboxes=None, labels=None):
        # pil to array
        img = np.array(img)

        # make img's value be fixed values
        # img = img // self.bin_value * self.bin_value
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img[img>=200] = 255
        img[img<50] = 0

        # array to pil
        img = Image.fromarray(img)

        return img

class NormalizeContrast(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    def forward(self, img, bboxes=None, labels=None):
        # pil to array
        img = np.array(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # make img's value be fixed values
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # array to pil
        img = Image.fromarray(img)

        return img

class Zscore(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    def forward(self, img, bboxes=None, labels=None):
        # pil to array
        img = np.array(img)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(np.float32) / 255
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8) * 255
        img = img.astype(np.uint8)
        # make img's value be fixed values
        
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # array to pil
        img = Image.fromarray(img)

        return img

class AdaptiveNormalize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, img):
        img = np.array(img)
        
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

        # array to pil
        img = Image.fromarray(img)

        return img       
    
class AutoEncoderViT(torch.nn.Module):
    def __init__(
            self, backbone_name: str = "vit_small_patch16_dinov3.lvd1689m",
            num_reconstruct_layers: int = 8, temperature=0.5):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0)
        self.encoder = timm.create_model(
            backbone_name, pretrained=True, num_classes=0)
        
        if num_reconstruct_layers == -1:
            num_reconstruct_layers = len(self.backbone.blocks)
            pass

        hidden_dim = self.backbone.num_features
        self.vq_list = torch.nn.ModuleList(
            [VectorQuantize(
                dim=hidden_dim,
                codebook_size=8192,
                decay=0.99,
                commitment_weight=0.1,
                kmeans_init=True,
            ) for _ in range(num_reconstruct_layers)])
        # self.vq = VectorQuantize(
        #     dim=hidden_dim,
        #     codebook_size=8192,
        #     decay=0.99,
        #     commitment_weight=0.1,
        #     kmeans_init=True,
        # )

        self.num_reconstruct_layers = num_reconstruct_layers
        self.backbone.requires_grad_(False)
        self.encoder.requires_grad_(False)
        for i in range(num_reconstruct_layers):
            self.encoder.blocks[-(i+1)].requires_grad_(True)
            pass

        self.temperature = temperature
        pass

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        # forward pass
        x = self.backbone.patch_embed(x)
        x, rot_pos_embed = self.backbone._pos_embed(x)
        x = self.backbone.norm_pre(x)

        intermediates = []
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x, rope=rot_pos_embed)
            intermediates.append(x)
            pass

        # x = self.backbone.norm(x)
        # intermediates.append(x)

        return intermediates
        pass


    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.patch_embed(x)
        x, rot_pos_embed = self.encoder._pos_embed(x)
        x = self.encoder.norm_pre(x)

        intermediates = []
        quantize_layer_inex = len(self.encoder.blocks) - self.num_reconstruct_layers
        total_commit_loss = 0.0
        for i, blk in enumerate(self.encoder.blocks):
            if i < quantize_layer_inex:
                with torch.no_grad():
                    x = blk(x, rope=rot_pos_embed)
                    intermediates.append(x)
                pass
            elif i >= quantize_layer_inex:
                xq, _, commit_loss = self.vq_list[i - quantize_layer_inex](x)
                total_commit_loss = total_commit_loss + commit_loss
                x = blk(xq, rope=rot_pos_embed)
                intermediates.append(x)
                pass
            else:
                raise RuntimeError("Unexpected block index")
            pass

        # x = self.encoder.norm(x)
        # intermediates.append(x)
        commit_loss = total_commit_loss

        return intermediates, commit_loss
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            backbone_intermediates = self.forward_backbone(x)
            pass
        
        encoder_intermediates, commit_loss = self.forward_encoder(x)
        return backbone_intermediates, encoder_intermediates, commit_loss
    

    def extract_feat(self, intermediates):
        selected = intermediates[-self.num_reconstruct_layers: ]
        feat = sum(selected) / len(selected)
        feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat[:, 5:, :]

    def set_require_grad_(self, epoch):
        if epoch % 2 == 0:
            self.encoder.requires_grad_(False)
            self.vq_list.requires_grad_(True)
        else:
            for i in range(self.num_reconstruct_layers):
                self.encoder.blocks[-(i+1)].requires_grad_(True)
                self.vq_list[i].requires_grad_(False)
                pass
            pass
        pass
    
    def compute_loss(self, forward_result):
        backbone_intermediates, encoder_intermediates, commit_loss = forward_result
        backbone_intermediates = backbone_intermediates[-self.num_reconstruct_layers: ]
        encoder_intermediates = encoder_intermediates[-self.num_reconstruct_layers: ]
        origin_feat = self.extract_feat(backbone_intermediates)
        reconstructed_feat = self.extract_feat(encoder_intermediates)
        similarity = (origin_feat * reconstructed_feat).sum(-1)
        # similarity = self.compute_similarity(backbone_intermediates, encoder_intermediates)
        reconstruct_loss = 1 - similarity
        # proba = torch.sigmoid((similarity - 0.5) / self.temperature)
        # reconstruct_loss = torch.log(proba + 1e-8).mean() * -1
        # reconstruct_loss = torch.nn.functional.mse_loss(origin_feat, reconstructed_feat, reduction="none").sum(-1)
        max_loss = torch.max(reconstruct_loss)
        lower_bound = max_loss - 0.05
        mask_effective = reconstruct_loss >= lower_bound
        count_effective = mask_effective.sum()
        reconstruct_loss[torch.logical_not(mask_effective)] = 0.0
        reconstruct_loss = reconstruct_loss.sum() / count_effective
        # reconstruct_loss = reconstruct_loss.mean()

        info = {
            "recon_loss": reconstruct_loss.item(),
            "commit_loss": commit_loss.item(),
        }
        return reconstruct_loss + commit_loss, info
        pass

    def compute_similarity(self, backbone_intermediates, encoder_intermediates):
        similarity = 0
        for b, e in zip(backbone_intermediates, encoder_intermediates):
            b = torch.nn.functional.normalize(b[:, 5:, :], p=2, dim=-1)
            e = torch.nn.functional.normalize(e[:, 5:, :], p=2, dim=-1)
            similarity += (b * e).sum(-1)
            pass
        return similarity / len(backbone_intermediates)

    def predict(self, forward_result):
        backbone_intermediates, encoder_intermediates, commit_loss = forward_result
        backbone_intermediates = backbone_intermediates[-self.num_reconstruct_layers: ]
        encoder_intermediates = encoder_intermediates[-self.num_reconstruct_layers: ]
        origin_feat = self.extract_feat(backbone_intermediates)
        reconstructed_feat = self.extract_feat(encoder_intermediates)
        similarity = (origin_feat * reconstructed_feat).sum(-1)
        # similarity = self.compute_similarity(backbone_intermediates, encoder_intermediates)
        return similarity
    
    def get_param_dict(self, lr0):
        # Get the parameters of the model
        params_default = []
        params_backbone = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                params_backbone.append(param)
            else:
                params_default.append(param)

        return [
            {
                'params': params_default,
                'lr': lr0
            },
            {
                'params': params_backbone,
                'lr': lr0 * 1.0,
            }
        ]
    pass


class DistillViT(torch.nn.Module):
    def __init__(
            self, backbone_name: str = "vit_base_patch16_dinov3.lvd1689m",
            num_reconstruct_layers: int = 8, temperature=0.5):
        super().__init__()
        hidden_dim = 192
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0)
        self.encoder = self.create_model(hidden_dim)

        hidden_dim_backbone = self.backbone.num_features
        if num_reconstruct_layers == -1:
            num_reconstruct_layers = len(self.backbone.blocks)
            pass
        self.num_reconstruct_layers = num_reconstruct_layers
        self.linear_maps = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim_backbone) for _ in range(len(self.backbone.blocks))])
        self.backbone.requires_grad_(False)
        pass


    def create_model(self, hidden_dim):
        model_args = dict(
            patch_size=16,
            dynamic_img_size=True,
            embed_dim=hidden_dim,
            depth=12,
            num_heads=6,
            qkv_bias=False,
            init_values=1.0e-05, # layer-scale
            rope_type='dinov3',
            rope_temperature=100,
            #rope_rescale_coords=2,  # haven't added to interface
            rope_rotate_half=True,
            use_rot_pos_emb=True,
            use_abs_pos_emb=False,
            num_reg_tokens=4,
            use_fc_norm=False,
            norm_layer=partial(LayerNorm, eps=1e-5),
        )
        model = timm.models.eva.Eva(**model_args)
        return model

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        # forward pass
        x = self.backbone.patch_embed(x)
        x, rot_pos_embed = self.backbone._pos_embed(x)
        x = self.backbone.norm_pre(x)

        intermediates = []
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x, rope=rot_pos_embed)
            intermediates.append(x)
            pass

        # x = self.backbone.norm(x)
        # intermediates.append(x)

        return intermediates
        pass


    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.patch_embed(x)
        x, rot_pos_embed = self.encoder._pos_embed(x)
        x = self.encoder.norm_pre(x)

        intermediates = []
        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x, rope=rot_pos_embed)
            x_ = self.linear_maps[i](x)
            intermediates.append(x_)
            pass

        return intermediates
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            backbone_intermediates = self.forward_backbone(x)
            pass
        
        encoder_intermediates = self.forward_encoder(x)
        return backbone_intermediates, encoder_intermediates
    

    def extract_feat(self, intermediates):
        selected = intermediates[: self.num_reconstruct_layers]
        feat = sum(selected) / len(selected)
        feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat[:, 5:, :]

    def set_require_grad_(self, epoch):
        if epoch % 2 == 0:
            self.encoder.requires_grad_(False)
            self.vq_list.requires_grad_(True)
        else:
            for i in range(self.num_reconstruct_layers):
                self.encoder.blocks[-(i+1)].requires_grad_(True)
                self.vq_list[i].requires_grad_(False)
                pass
            pass
        pass
    
    def compute_similarity(self, backbone_intermediates, encoder_intermediates):
        similarity = 0
        for b, e in zip(backbone_intermediates, encoder_intermediates):
            b = torch.nn.functional.normalize(b[:, 5:, :], p=2, dim=-1)
            e = torch.nn.functional.normalize(e[:, 5:, :], p=2, dim=-1)
            similarity += (b * e).sum(-1)
            pass
        return similarity / len(backbone_intermediates)
    
    def compute_loss(self, forward_result):
        backbone_intermediates, encoder_intermediates = forward_result
        backbone_intermediates = backbone_intermediates[-self.num_reconstruct_layers: ]
        encoder_intermediates = encoder_intermediates[-self.num_reconstruct_layers: ]
        origin_feat = self.extract_feat(backbone_intermediates)
        reconstructed_feat = self.extract_feat(encoder_intermediates)
        similarity = (origin_feat * reconstructed_feat).sum(-1)
        # similarity = self.compute_similarity(backbone_intermediates, encoder_intermediates)
        reconstruct_loss = 1 - similarity
        # proba = torch.sigmoid((similarity - 0.5) / self.temperature)
        # reconstruct_loss = torch.log(proba + 1e-8).mean() * -1
        # reconstruct_loss = torch.nn.functional.mse_loss(origin_feat, reconstructed_feat, reduction="none").sum(-1)
        max_loss = torch.max(reconstruct_loss)
        lower_bound = max_loss - 0.02
        mask_effective = reconstruct_loss >= lower_bound
        count_effective = mask_effective.sum()
        effective_loss = reconstruct_loss.clone()
        effective_loss[torch.logical_not(mask_effective)] = 0.0
        effective_loss = effective_loss.sum() / count_effective
        # effective_loss = effective_loss.mean()

        reconstruct_loss = reconstruct_loss.mean()

        info = {
            "recon_loss": reconstruct_loss.item(),
            "effect_loss": effective_loss.item(),
        }
        return effective_loss, info
        pass

    def predict(self, forward_result):
        backbone_intermediates, encoder_intermediates = forward_result
        backbone_intermediates = backbone_intermediates[-self.num_reconstruct_layers: ]
        encoder_intermediates = encoder_intermediates[-self.num_reconstruct_layers: ]
        origin_feat = self.extract_feat(backbone_intermediates)
        reconstructed_feat = self.extract_feat(encoder_intermediates)
        similarity = (origin_feat * reconstructed_feat).sum(-1)
        # similarity = self.compute_similarity(backbone_intermediates, encoder_intermediates)
        return similarity
    
    def get_param_dict(self, lr0):
        # Get the parameters of the model
        params_default = []
        params_backbone = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                params_backbone.append(param)
            else:
                params_default.append(param)

        return [
            {
                'params': params_default,
                'lr': lr0
            },
            {
                'params': params_backbone,
                'lr': lr0 * 1.0,
            }
        ]
    pass


class DistillViT2(torch.nn.Module):
    def __init__(
            self, backbone_name: str = "vit_base_patch16_dinov3.lvd1689m",
            layer_indices=[-1, -3, -4, -6], temperature=0.5):
        super().__init__()
        hidden_dim = 192
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0)
        self.encoder = self.create_model(hidden_dim)

        hidden_dim_backbone = self.backbone.num_features
        self.layer_indices = layer_indices
        self.linear_maps = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim_backbone) for _ in range(len(self.backbone.blocks))])
        self.linear_input = torch.nn.Linear(hidden_dim_backbone, hidden_dim)
        self.backbone.requires_grad_(False)
        pass


    def create_model(self, hidden_dim):
        model_args = dict(
            patch_size=16,
            dynamic_img_size=True,
            embed_dim=hidden_dim,
            depth=12,
            num_heads=6,
            qkv_bias=False,
            init_values=1.0e-05, # layer-scale
            rope_type='dinov3',
            rope_temperature=100,
            #rope_rescale_coords=2,  # haven't added to interface
            rope_rotate_half=True,
            use_rot_pos_emb=True,
            use_abs_pos_emb=False,
            num_reg_tokens=4,
            use_fc_norm=False,
            norm_layer=partial(LayerNorm, eps=1e-5),
        )
        model = timm.models.eva.Eva(**model_args)
        return model

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        # forward pass
        x = self.backbone.patch_embed(x)
        x, rot_pos_embed = self.backbone._pos_embed(x)
        x = self.backbone.norm_pre(x)

        intermediates = []
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x, rope=rot_pos_embed)
            intermediates.append(x)
            pass

        # x = self.backbone.norm(x)
        # intermediates.append(x)

        return intermediates
        pass


    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_input(x)
        x_temp = torch.zeros((x.shape[0], 32, 32, x.shape[-1]), device=x.device)
        x, rot_pos_embed = self.encoder._pos_embed(x_temp)
        x = self.encoder.norm_pre(x)

        intermediates = []
        for i, blk in enumerate(self.encoder.blocks[::-1]):
            x = blk(x, rope=rot_pos_embed)
            x_ = self.linear_maps[i](x)
            intermediates.append(x_)
            pass

        return intermediates[::-1]
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            backbone_intermediates = self.forward_backbone(x)
            pass
        
        encoder_intermediates = self.forward_encoder(backbone_intermediates[-1])
        return backbone_intermediates, encoder_intermediates

    
    def compute_similarity(self, backbone_intermediates, encoder_intermediates):
        dists = []
        for ilayer in self.layer_indices:
            # compute distance
            dist = (backbone_intermediates[ilayer][:, 5:, :] - encoder_intermediates[ilayer][:, 5:, :]) ** 2
            dist = torch.sqrt(dist.sum(-1))
            dists.append(dist)
            pass
        dists = torch.stack(dists, dim=0)
        return -dists

    
    def compute_loss(self, forward_result):
        backbone_intermediates, encoder_intermediates = forward_result
        similarity = self.compute_similarity(backbone_intermediates, encoder_intermediates)
        reconstruct_loss = -similarity
        # proba = torch.sigmoid((similarity - 0.5) / self.temperature)
        # reconstruct_loss = torch.log(proba + 1e-8).mean() * -1
        # reconstruct_loss = torch.nn.functional.mse_loss(origin_feat, reconstructed_feat, reduction="none").sum(-1)
        max_loss = torch.max(reconstruct_loss)
        lower_bound = max_loss - 100.0
        mask_effective = reconstruct_loss >= lower_bound
        count_effective = mask_effective.sum()
        effective_loss = reconstruct_loss.clone()
        effective_loss[torch.logical_not(mask_effective)] = 0.0
        effective_loss = effective_loss.sum() / count_effective
        # effective_loss = effective_loss.mean()

        reconstruct_loss = reconstruct_loss.mean()

        info = {
            "recon_loss": reconstruct_loss.item(),
            "effect_loss": effective_loss.item(),
        }
        return effective_loss, info
        pass

    def predict(self, forward_result):
        backbone_intermediates, encoder_intermediates = forward_result
        similarity = self.compute_similarity(backbone_intermediates, encoder_intermediates)
        similarity = similarity.min(dim=0)[0]
        return similarity
    
    def get_param_dict(self, lr0):
        # Get the parameters of the model
        params_default = []
        params_backbone = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                params_backbone.append(param)
            else:
                params_default.append(param)

        return [
            {
                'params': params_default,
                'lr': lr0
            },
            {
                'params': params_backbone,
                'lr': lr0 * 1.0,
            }
        ]
    pass


class ViTPatchcore(torch.nn.Module):
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
        with torch.no_grad():
            intermediates  = self.forward_backbone(x)
            pass

        # blur
        # for i, emb in enumerate(intermediates):
        #     b, n, c = emb.shape
        #     h = w = int(math.sqrt(n))
        #     assert h*w == n, f"n={n} is not a perfect square"
        #     emb = emb.reshape(b, h, w, c).permute(0, 3, 1, 2)
        #     # gaussian blur
        #     emb = torch.nn.functional.gaussian_blur(emb, kernel_size=(5, 5), sigma=(5.0, 5.0))
        #     emb = emb.permute(0, 2, 3, 1).reshape(b, n, c)
        #     intermediates[i] = emb
        #     pass
        return intermediates
        pass
    
    def compute_loss(self, forward_result, memory_index):
        intermediates = forward_result
        layer_index = self.layer_indices[memory_index]
        embeddings = intermediates[layer_index]
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        self.memories[memory_index].update(embeddings)
        pass

    def shrink_memory(self, memory_index):
        self.memories[memory_index].shrink()
        pass

    def compute_distance(self, forward_result, num_neighbours=1):
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

    def compute_anomaly_score(self, intermediates, dists, indices, num_neighbours=9):
        # compute the anomaly score for each image in the batch
        batch_size = dists.shape[0]
        # find max idx
        max_idx = torch.argmax(dists, dim=1)
        # max_idx: [batch_size]
        max_indices = indices[torch.arange(batch_size), max_idx]
        # max_indices: [batch_size]
        max_memory = self.memories[0].memory_bank[max_indices]
        support_samples = self.memories[0].get_topk_sample(max_memory, k=num_neighbours)
        # support_samples: [B, K, D]

        # compute the distance between the max embedding and the support samples
        max_embedding = intermediates[self.layer_indices[0]][torch.arange(batch_size), max_idx]
        # max_embedding: [B, D]
        
        # compute distance between embedding and support samples
        support_dists = torch.cdist(max_embedding.unsqueeze(1), support_samples, p=2)
        # support_dists: [B, 1, K]
        support_dists = support_dists.squeeze(1)

        # do softmax
        support_weights = (1 - torch.softmax(support_dists, dim=1))[..., 0]
        # support_weights: [B, 1]

        score = dists[torch.arange(batch_size), max_idx] * support_weights

        return score

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

    def train(self, mode=True):
        if mode:
            self.backbone.eval()
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
    pass


class ViTPatchcore2(torch.nn.Module):
    def __init__(
            self, model_config: common.ModelConfig = None,
            backbone_name: str = "vit_base_patch16_dinov3.lvd1689m",
            # backbone_name: str = "vit_small_patch16_dinov3.lvd1689m",
            # layer_indices = [1],
            ):
        super().__init__()
        if model_config is not None:
            image_size = model_config.image_size
            layer_indices = model_config.layer_indices
            memory_size = model_config.memory_size
            if len(model_config.checkpoint_path) > 0:
                self.backbone = timm.create_model(
                    backbone_name, pretrained=False, num_classes=0)
                data = torch.load(model_config.checkpoint_path, weights_only=False)
                self.image_size = data.get("image_size", image_size)
                self.layer_indices = data.get("layer_indices", layer_indices)
                self.memory_size = data.get("memory_size", memory_size) 
            else:
                self.backbone = timm.create_model(
                    backbone_name, pretrained=True, num_classes=0)
                self.image_size = image_size
                self.layer_indices = layer_indices
                self.memory_size = memory_size
                pass
            pass

        dim = self.backbone.num_features
        self.memories = torch.nn.ModuleList(
            [MemoryBank(size=self.memory_size, dim=dim, max_size=3000000) for _ in self.layer_indices]
        )
        self.register_buffer("middle_distance", torch.tensor(0.5))
        self.register_buffer("scale_distance", torch.tensor(1.0))
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
        
        intermediates = [
            emb.permute(0, 2, 3, 1).reshape(emb.shape[0], -1, emb.shape[1]) for emb in intermediates
        ]

        return intermediates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            intermediates  = self.forward_backbone(x)
            pass

        return intermediates
        pass
    
    def compute_loss(self, forward_result, memory_index):
        intermediates = forward_result
        layer_index = self.layer_indices[memory_index]
        embeddings = intermediates[layer_index]
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        self.memories[memory_index].update(embeddings)
        pass

    def shrink_memory(self, memory_index):
        self.memories[memory_index].shrink()
        pass

    def compute_distance(self, forward_result, num_neighbours=1):
        idxs = None
        intermediates = forward_result
        dists = []
        for i, ilayer in enumerate(self.layer_indices):
            embeddings = intermediates[ilayer]
            bsize = embeddings.shape[0]

            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
            # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            dist, _ = self.memories[i].compute_min_distance(embeddings)
            # dist = (dist - self.memories[i].dist_mean) / (self.memories[i].dist_std + 1e-8)

            dist = dist.reshape((bsize, -1))
            dists.append(dist)
            pass

        max_dist = torch.max(torch.stack(dists, dim=0), dim=0)[0]
        return max_dist

    def postprocess(self, forward_result, num_neighbours=1):
        max_dist = self.compute_distance(forward_result, num_neighbours=num_neighbours)
        # use sigmoid
        # proba = torch.sigmoid((max_dist - self.middle_distance) * self.scale_distance)

        max_max_dist = max_dist.max(dim=0, keepdim=True)[0]
        min_max_dist = max_dist.min(dim=0, keepdim=True)[0]
        proba = (max_dist - min_max_dist) / (max_max_dist - min_max_dist)
        
        return proba

    def get_default_transforms(self):
        transforms = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.ToPILImage(),
            torchvision.transforms.v2.Resize((self.image_size, self.image_size)),
            torchvision.transforms.v2.GaussianBlur(kernel_size=3, sigma=1.0),
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
            preds = self.postprocess(forward_result, num_neighbours=num_neighbours)
            pass
        # pred: [B, E, E, 1]
        preds = preds.squeeze(-1)
        results = []
        for original_size, pred in zip(original_sizes, preds):
            pred = pred.cpu().numpy()
            pred = pred.reshape(32, 32)

            pred = cv2.resize(
                pred, (original_size[1], original_size[0]))
            prediction = common.PredictionResult(score=pred)
            results.append(prediction)
            pass
        
        if return_heatmap:
            for i in range(len(results)):
                results[i].heat_map = self.generate_heatmap(
                    results[i].score, imgs[i])
                pass
            
        return results


    def compute_stats(self, ):
        for memory in self.memories:
            memory.compute_stats()
            pass
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

    def train(self, mode=True):
        if mode:
            self.backbone.eval()
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
    pass
