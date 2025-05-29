import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch

def denormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return image * std + mean

def show_finch_samples(cluster_labels, true_labels, dataset, samples_per_cluster=5):
    unique_clusters = np.unique(cluster_labels)
    for cluster_id in unique_clusters:
        st.markdown(f"### Cluster {cluster_id}")
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        true_in_cluster = true_labels[cluster_indices]
        unique, counts = np.unique(true_in_cluster, return_counts=True)

        st.write("Distribusi label asli:")
        for u, c in zip(unique, counts):
            st.write(f"- {dataset.classes[u]}: {c} samples")

        images_row = []
        for i in np.random.choice(cluster_indices, min(samples_per_cluster, len(cluster_indices)), replace=False):
            img, _ = dataset[i]
            img = denormalize(img).permute(1,2,0).numpy()
            images_row.append(img)

        st.image(images_row, width=100)
