
import torchvision.datasets
import torch
from . import common
from .models import ViTPatchcore
from tqdm import tqdm
import sklearn.metrics
import numpy as np


def valid(model: ViTPatchcore, folder: str):
    model.eval()
    valid_dataset = torchvision.datasets.ImageFolder(
        root=folder,
        transform=model.get_default_transforms())

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=4,
        num_workers=8,
        shuffle=False
    )

    ground_truths = []
    dists = []
    for i, (images, labels) in enumerate(tqdm(valid_loader)):
        label_names = [valid_dataset.classes[label] for label in labels]
        with torch.no_grad():
            images = images.to(model.device)
            preds = model.forward(images)
            dist = model.compute_distance(preds)
            pass
        ground_truths.extend(label_names)
        dists.extend([d.cpu().numpy().max() for d in dist])
        pass
    
    ground_truths = [1 if gt != "good" else 0 for gt in ground_truths]
    # find minimal probability for ng images
    ng_dist = [dists[i] for i in range(len(dists)) if ground_truths[i] == 1]
    min_ng_dist = np.min(ng_dist)
    # find dist that smaller min_ng_dist
    max_ok_dist = np.max(
        [dists[i] for i in range(len(dists)) if ground_truths[i] == 0 and dists[i] < min_ng_dist])
    middle_dist = 0.5 * (min_ng_dist + max_ok_dist)
    
    predict_labels = np.asarray([1 if p > middle_dist else 0 for p in dists])
    predict_scores = np.asarray(dists)

    # calculate accuracy, f1_score, precision, recall
    accuracy = sklearn.metrics.accuracy_score(ground_truths, predict_labels)
    f1_score = sklearn.metrics.f1_score(ground_truths, predict_labels)
    precision = sklearn.metrics.precision_score(ground_truths, predict_labels)
    recall = sklearn.metrics.recall_score(ground_truths, predict_labels)

    precision_curve, recall_curve, _ = sklearn.metrics.precision_recall_curve(ground_truths, predict_scores)

    return middle_dist, accuracy, f1_score, precision, recall, (precision_curve, recall_curve)

if __name__ == "__main__":
    pass