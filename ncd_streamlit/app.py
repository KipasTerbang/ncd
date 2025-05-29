# app.py
import streamlit as st
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms
from sklearn.metrics import normalized_mutual_info_score
from clustering_utils import clustering_accuracy
from vit_utils import extract_features
from finch import FINCH
from visualization import show_finch_samples
import torch
import numpy as np

st.title("🌿 Novel Class Discovery dengan ViT + FINCH")

folder_path = st.text_input("Masukkan path ke folder unlabeled dataset:", "")

if folder_path and os.path.exists(folder_path):
    st.success("Folder ditemukan!")

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(root=folder_path, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    st.write(f"Jumlah gambar: {len(dataset)} | Jumlah class folder: {len(dataset.classes)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads = torch.nn.Identity()
    model.to(device)

    if st.button("Ekstrak Fitur dan Clustering"):
        with st.spinner("Ekstraksi fitur..."):
            features, labels = extract_features(loader, model, device)

        st.success("Fitur diekstrak!")

        with st.spinner("Clustering dengan FINCH..."):
            c, _, _ = FINCH(features, use_ann_above_samples=1000, verbose=True)
            clusters = c[:, 2] 

        acc = clustering_accuracy(labels, clusters)
        nmi = normalized_mutual_info_score(labels, clusters)

        st.subheader("📊 Evaluasi Clustering")
        st.metric("ACC", f"{acc:.4f}")
        st.metric("NMI", f"{nmi:.4f}")
        st.write(f"Jumlah cluster ditemukan: {len(np.unique(clusters))}")

        st.subheader("📸 Visualisasi Per Cluster")
        show_finch_samples(clusters, labels, dataset)

else:
    st.warning("Masukkan path dataset terlebih dahulu!")