import torch
import numpy as np
from tqdm import tqdm

def extract_features(loader, model, device):
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            feats = model(images)
            features.append(feats.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)
