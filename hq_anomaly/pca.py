import torch

class PCA(torch.nn.Module):
    def __init__(self, original_dim: int, target_dim: int):
        super().__init__()
        self.original_dim = original_dim
        self.target_dim = target_dim
        self.register_buffer("v", torch.zeros((original_dim, target_dim)))
        self.register_buffer("mean", torch.zeros((original_dim,)))

        pass

    def fit(self, features: torch.Tensor):
        self.mean = features.mean(dim=0)
        features = features - self.mean
        u, s, v = torch.pca_lowrank(features.float(), q=self.target_dim, center=False, niter=10)
        self.v = v[:, :self.target_dim].to(dtype=torch.float16)
        return self

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        features = features - self.mean
        return features @ self.v