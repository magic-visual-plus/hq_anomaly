
from hq_anomaly.k_center_greedy import KCenterGreedy
import torch


class MemoryBank(torch.nn.Module):
    def __init__(self, size, dim=768, max_size=1000000, device="cuda:0"):
        super().__init__()
        self.max_size = max_size
        self.size = size
        self.register_buffer("memory_bank", torch.zeros((size, dim), dtype=torch.float32))
        self.register_buffer("dist_mean", torch.tensor(0.0))
        self.register_buffer("dist_std", torch.tensor(1.0))
        self.min_dist = 0
        self.memories = []
        self.device = torch.device(device)
        pass

    def update(self, embeddings):
        embeddings = embeddings.to(device="cpu", dtype=torch.float16)
        self.memories.append(embeddings)
        if sum([m.shape[0] for m in self.memories]) >= self.max_size:
            self.shrink()
            pass
        pass

    def shrink(self):
        if self.memory_bank.abs().sum() == 0:
            self.memories = torch.cat(self.memories, dim=0)
        else:
            self.memory_bank = self.memory_bank.to("cpu", dtype=torch.float16)
            self.memories = torch.cat([self.memory_bank] + self.memories, dim=0)
            pass
        self.memory_bank = self.memories
        self.memories = []
        self.memory_bank = self.memory_bank.to(device=self.device)
        r = self.size / self.memory_bank.shape[0]
        if r < 1:
            sampler = KCenterGreedy(embedding=self.memory_bank, size=self.size)
            with torch.no_grad():
                self.memory_bank = sampler.sample_coreset().to(dtype=torch.float32)
                pass
            pass
        else:
            while self.memory_bank.shape[0] < self.size:
                self.memory_bank = torch.cat([self.memory_bank, self.memory_bank], dim=0)
                pass
            self.memory_bank = self.memory_bank[:self.size]
            pass
        pass

    def compute_stats(self,):
        batch_size = 8
        dist_sum = 0.0
        dist2_sum = 0.0
        for i in range(0, self.memory_bank.shape[0], batch_size):
            batch_embeddings = self.memory_bank[i:i+batch_size]
            dist = torch.cdist(batch_embeddings, self.memory_bank)
            dist_sum += dist.sum().item()
            dist2_sum += (dist ** 2).sum().item()
            pass

        dist_mean = dist_sum / (self.memory_bank.shape[0] ** 2)
        dist_std = ((dist2_sum / (self.memory_bank.shape[0] ** 2)) - dist_mean ** 2) ** 0.5
        self.dist_mean = torch.tensor(dist_mean)
        self.dist_std = torch.tensor(dist_std)
        pass

    def compute_min_distance(self, embeddings):
        batch_size = 8
        dists = []
        indices = []
        embeddings = embeddings.to(dtype=self.memory_bank.dtype)
        for i in range(0, embeddings.shape[0], batch_size):
            batch_embeddings = embeddings[i:i+batch_size]
            # dist = torch.norm(batch_embeddings[:, None, :] - self.memory_bank[None, :, :], dim=-1)
            dist = torch.cdist(batch_embeddings, self.memory_bank, compute_mode="donot_use_mm_for_euclid_dist")
            dist, idx = torch.min(dist, dim=1)

            dists.append(dist)
            indices.append(idx)
            pass

        return torch.cat(dists, dim=0), torch.cat(indices, dim=0)
        pass


    def get_topk_sample(self, embeddings, k=10):
        batch_size = 8
        indices = []
        embeddings = embeddings.to(dtype=self.memory_bank.dtype)
        for i in range(0, embeddings.shape[0], batch_size):
            batch_embeddings = embeddings[i:i+batch_size]
            dist = torch.cdist(batch_embeddings, self.memory_bank, compute_mode="donot_use_mm_for_euclid_dist")
            _, idx = torch.topk(dist, k=k, dim=1, largest=False)
            indices.append(idx)
            pass

        indices = torch.cat(indices, dim=0)
        return self.memory_bank[indices]


    def compute_self_min_dinstance(self, ):
        batch_size = 8
        dists = []
        for i in range(0, self.memory_bank.shape[0], batch_size):
            batch_embeddings = self.memory_bank[i:i+batch_size]
            dist = torch.cdist(batch_embeddings, self.memory_bank)
            # set self to inf
            column_start = i
            dist[torch.arange(batch_embeddings.shape[0]), torch.arange(column_start, column_start + batch_embeddings.shape[0])] = 1e-6
            dist = torch.min(dist, dim=1)[0]
            dists.append(dist)
            pass

        return torch.cat(dists, dim=0)
        pass
    pass
