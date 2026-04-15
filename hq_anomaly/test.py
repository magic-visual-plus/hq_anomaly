from hq_anomaly import models
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
    model = models.ViTPatchcore()
    sd = torch.load('output/ckpt2.pth', map_location='cpu')
    model.load_state_dict(sd)
    device = "cuda:0"
    model.to(device)
    model.eval()
    filenames = os.listdir(input_path)
    filenames = [os.path.join(input_path, f) for f in filenames if f.endswith('.jpg')]

    # model.compute_stats()
    cnt = 0
    for filename in tqdm(filenames):
        img = cv2.imread(filename)
        img = cv2.resize(img, (512, 512))
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=1.0, sigmaY=1.0)  # Apply Gaussian blur
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = TVF.to_tensor(img).unsqueeze(0)
        img_tensor = TVF.normalize(
            img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            forward_result = model(img_tensor)
            score = model.predict(forward_result)
            pass
        print(score.min())
        if score.min() < -538.3447:
            cnt += 1
            print(cnt)
            pass
        pass
    pass