
import torchvision.datasets
import torch
from hq_anomaly import common
from hq_anomaly.models import ViTPatchcore
from tqdm import tqdm
import sklearn.metrics
import numpy as np
import sys


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
            dist, idx = model.compute_distance(preds)
            scores = model.compute_anomaly_score(preds, dist, idx, num_neighbours=9)
            pass
        ground_truths.extend(label_names)
        dists.extend([d.cpu().numpy() for d in scores])
        pass
    
    ground_truths = [1 if gt != "good" else 0 for gt in ground_truths]
    # find minimal probability for ng images
    ng_dist = [dists[i] for i in range(len(dists)) if ground_truths[i] == 1]
    min_ng_dist = np.min(ng_dist)
    max_ng_dist = np.max(ng_dist)
    # find dist that smaller min_ng_dist
    max_ok_dist = np.max(
        [dists[i] for i in range(len(dists)) if ground_truths[i] == 0 and dists[i] < min_ng_dist])
    middle_dist = 0.5 * (min_ng_dist + max_ok_dist)
    
    predict_scores = model.distance2proba((middle_dist, max_ng_dist), np.asarray(dists))

    # calculate accuracy, f1_score, precision, recall
    precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(ground_truths, predict_scores)
    f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    max_f1_idx = np.argmax(f1)
    confidence = thresholds[max_f1_idx]

    predict_labels = [1 if score >= confidence else 0 for score in predict_scores]

    accuracy = sklearn.metrics.accuracy_score(ground_truths, predict_labels)
    f1_score = sklearn.metrics.f1_score(ground_truths, predict_labels)
    precision = sklearn.metrics.precision_score(ground_truths, predict_labels)
    recall = sklearn.metrics.recall_score(ground_truths, predict_labels)

    precision_curve, recall_curve, _ = sklearn.metrics.precision_recall_curve(ground_truths, predict_scores)

    returns = (middle_dist, max_ng_dist), confidence, accuracy, f1_score, precision, recall, (precision_curve, recall_curve)
    print(returns)
    return returns

if __name__ == "__main__":
    model = ViTPatchcore(
        model_config=common.ModelConfig(
            checkpoint_path=sys.argv[1]
        )
    )
    valid(model, sys.argv[2])
    pass