from hq_anomaly import models, common
import sys
import os
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TVF
from tqdm import tqdm


if __name__ == "__main__":
    input_path = sys.argv[1]
    # model = models.AutoEncoderViT()
    # model = models.DistillViT2()
    model = models.ViTPatchcore(common.ModelConfig())
    model.load("output/ckpt.pth")
    device = "cuda:0"
    model.to(device)
    model.eval()
    filenames = os.listdir(input_path)
    filenames = [os.path.join(input_path, f) for f in filenames if f.endswith('.jpg')]

    # model.compute_stats()
    cnt = 0
    for filename in tqdm(filenames):
        # if "ok_2_中壳_中壳反面机加面_ca6026bc17e945ef9ca84de833ed18fd_20260514_131814_5439940.jpg" not in filename:
        #     continue
        img = cv2.imread(filename)
        r = model.predict([img], return_heatmap=False)[0]
        print(f"{filename}:{r.score.max()}")
        pass
    pass