

import torch.nn
import timm
import copy


class AutoEncoderNeck(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout=0.1):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x
    pass


class AutoEncoderViT(torch.nn.Module):
    def __init__(
            self, backbone_name: str = "vit_small_patch16_dinov3.lvd1689m",
            num_reconstruct_layers: int = 1):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0)
        hidden_dim = self.backbone.num_features
        self.neck = AutoEncoderNeck(hidden_dim, hidden_dim * 2, dropout=0.1)

        self.decoder_blocks = []
        for i in range(num_reconstruct_layers):
            new_module = copy.deepcopy(self.backbone.blocks[-(i+1)])
            # re inialize the module
            self.decoder_blocks.append(new_module)
            pass
        self.decoder = torch.nn.ModuleList(self.decoder_blocks)
        self.backbone.requires_grad_(False)
        pass

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        # forward pass
        B, _, height, width = x.shape
        x = self.backbone.patch_embed(x)
        x, rot_pos_embed = self.backbone._pos_embed(x)
        x = self.backbone.norm_pre(x)

        intermediates = []
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x, rope=rot_pos_embed)
            intermediates.append(x)
            pass

        x = self.backbone.norm(x)
        intermediates.append(x)

        return x, intermediates, rot_pos_embed
        pass


    def forward_decoder(self, x: torch.Tensor, rot_pos_embed: torch.Tensor) -> torch.Tensor:
        intermediates = [x]
        for i, blk in enumerate(self.decoder):
            x = blk(x, rope=rot_pos_embed)
            intermediates.append(x)
            pass
        return x, intermediates
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoded_x, encoder_intermediates, rot_pos_embed = self.forward_encoder(x)
            pass
        
        neck_x = self.neck(encoded_x)
        decoded_x, decoder_intermediates = self.forward_decoder(neck_x, rot_pos_embed)
        return encoder_intermediates, decoder_intermediates
    

    def compute_loss(self, forward_result):
        encoder_intermediates, decoder_intermediates = forward_result
        encoder_intermediates = encoder_intermediates[::-1]

        encoder_intermediates = encoder_intermediates[: len(decoder_intermediates)]
        origin_feat = torch.cat(encoder_intermediates, dim=-1)
        reconstructed_feat = torch.cat(decoder_intermediates, dim=-1)

        loss = torch.nn.functional.mse_loss(origin_feat, reconstructed_feat)
        return loss
        pass
    pass
