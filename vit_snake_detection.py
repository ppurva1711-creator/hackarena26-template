#!/usr/bin/env python3
"""
vit_snake_detection.py

Usage:
    # Build embeddings + index from CSV
    python vit_snake_detection.py --csv "/mnt/data/Untitled spreadsheet - Sheet1 (2).csv" --build-index

    # Predict for a single image (uses existing embeddings/index)
    python vit_snake_detection.py --predict --image "./test/bite_photo.jpg" --k 3

Notes:
- The CSV is expected to have at least these columns:
    image_path, species_name, venom_type
- If paths in CSV are Windows-style, the script will try to normalize them.
"""

import os
import argparse
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import cast
import cv2
import uuid
import torch
from transformers import AutoImageProcessor, ViTModel

# Optional dependencies
USE_FAISS = False
try:
    import faiss  # type: ignore
    USE_FAISS = True
except Exception:
    USE_FAISS = False
    from sklearn.neighbors import NearestNeighbors


DEFAULT_CSV_PATH =  r"D:\snakebite_project2\snakebite_project\data\annotations\image_metadata.csv"
EMBEDDING_FILE = "snake_embeddings.npz"    
FAISS_INDEX_FILE = "snake_faiss.index"
SKLEARN_INDEX_FILE = "snake_sklearn_index.npy"  
EMBED_DIM = 768  
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_path(p: str) -> str:
    """Attempt to normalize path strings (handles Windows backslashes from CSV)."""
    if pd.isna(p):
        return ""
    p = str(p).strip()
    # If it's a Windows absolute path like C:\..., convert to unix path if running on linux
    # but prefer leaving as-is — PIL will fail if path doesn't exist.
    p = p.replace("\\", "/")
    return p

def load_csv(csv_path: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(csv_path)
    for col in ["image_path", "species_name", "venom_type"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    df["image_path"] = df["image_path"].apply(normalize_path)
    df = df[df["image_path"].astype(bool)].reset_index(drop=True)
    return df  # type: ignore[return-value]

def load_vit_model(model_name="google/vit-base-patch16-224-in21k"):
    
    # Use AutoImageProcessor for broader compatibility across transformers versions
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    model.eval()
    model = model.to(torch.device(DEVICE))  # type: ignore[arg-type]
    return processor, model

def image_to_tensor(image: Image.Image, processor):
    # processor will convert and normalize
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(torch.device(DEVICE))
    return pixel_values

def extract_embeddings_from_paths(image_paths, processor, model, batch_size=BATCH_SIZE):
    embeddings = []
    valid_paths = []
    failed = []
    n = len(image_paths)
    for i in tqdm(range(0, n, batch_size), desc="Extracting embeddings"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                failed.append((p, str(e)))
                images.append(None)
        # Filter None
        valid_images = []
        valid_indices = []
        for idx, img in enumerate(images):
            if img is not None:
                valid_images.append(img)
                valid_indices.append(idx)
        if not valid_images:
            # add placeholders for these as zeros
            for idx in range(len(batch_paths)):
                embeddings.append(None)
                valid_paths.append(batch_paths[idx])
            continue
        inputs = processor(images=valid_images, return_tensors="pt").to(torch.device(DEVICE))
        with torch.no_grad():
            outputs = model(**inputs)
            # outputs.last_hidden_state shape: (batch, seq_len, hidden_dim)
            # mean pool across seq dimension (except cls token specially isn't needed)
            emb = outputs.last_hidden_state.mean(dim=1)  # (batch, hidden_dim)
            # move to cpu and store numpy
            emb_cpu = emb.cpu().numpy()
        # map back to original positions in batch
        emb_iter = iter(emb_cpu)
        for idx in range(len(batch_paths)):
            if idx in valid_indices:
                e = next(emb_iter)
                embeddings.append(e)
            else:
                embeddings.append(None)
            valid_paths.append(batch_paths[idx])
    # convert to array, skipping None rows (we'll note indices)
    embedding_array = []
    valid_idx_map = []  # indices into original list that are valid
    for i, e in enumerate(embeddings):
        if e is None:
            continue
        embedding_array.append(e)
        valid_idx_map.append(i)
    if len(embedding_array) == 0:
        raise RuntimeError("No embeddings could be extracted from provided image paths.")
    embedding_array = np.vstack(embedding_array).astype("float32")
    return embedding_array, valid_idx_map, failed

# ---------------------------
# Index helpers (FAISS or sklearn)
# ---------------------------
def build_faiss_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d) 
    faiss.normalize_L2(embeddings)  # type: ignore[call-arg]
    index.add(embeddings)  # type: ignore[call-arg]
    return index

def build_sklearn_index(embeddings: np.ndarray):
    # Will use cosine similarity via NearestNeighbors with metric='cosine'
    nbrs = NearestNeighbors(n_neighbors=10, metric="cosine").fit(embeddings)
    return nbrs

def save_embeddings_npz(path, embeddings, metadata):
    # metadata can include lists for image_path, species_name, venom_type, original_row_idx
    np.savez_compressed(path, embeddings=embeddings, **metadata)
    print(f"Saved embeddings+metadata to {path}")

def load_embeddings_npz(path):
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    meta = {k: data[k].tolist() for k in data.files if k != "embeddings"}
    return embeddings, meta

# ---------------------------
# Camera Capture Helper
# ---------------------------
def capture_from_camera(save_dir="captured"):
    """
    Capture image from camera.
    Returns: image_path or None if capture failed
    """
    os.makedirs(save_dir, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Camera not found")
        return None

    print("📷 Press SPACE to capture image | ESC to exit")

    while True:
        ret, frame = cam.read()
        cv2.imshow("Snakebite Camera", frame)

        key = cv2.waitKey(1)

        if key % 256 == 27:   
            print("❌ Closing camera…")
            cam.release()
            cv2.destroyAllWindows()
            return None

        elif key % 256 == 32:  # SPACE
            filename = f"{save_dir}/capture_{uuid.uuid4().hex}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✔️ Image saved: {filename}")

            cam.release()
            cv2.destroyAllWindows()
            return filename

# ---------------------------
# Load index and metadata
# ---------------------------
def load_index_and_meta(embedding_file=EMBEDDING_FILE):
    
    loaded = load_embeddings_npz(embedding_file)
    if loaded is None:
        raise FileNotFoundError(f"No embeddings file found at {embedding_file}. Please run --build-index first.")
    embeddings, meta = loaded
    # If FAISS, load index or build on-the-fly:
    if USE_FAISS:
        if os.path.exists(FAISS_INDEX_FILE):
            index = faiss.read_index(FAISS_INDEX_FILE)
        else:
            print("FAISS index file not found. Building index in memory from saved embeddings...")
            emb_copy = embeddings.copy()
            faiss.normalize_L2(emb_copy)  # type: ignore[call-arg]
            index = faiss.IndexFlatIP(emb_copy.shape[1])
            index.add(emb_copy)  # type: ignore[call-arg]
            faiss.write_index(index, FAISS_INDEX_FILE)  # type: ignore[call-arg]
        return index, embeddings, meta
    else:
        # sklearn path: build NearestNeighbors on embeddings
        nbrs = NearestNeighbors(n_neighbors=10, metric="cosine").fit(embeddings)
        return nbrs, embeddings, meta

def predict_image(image_path: str, processor, model, index, embeddings_meta, k=3):
    """
    Predict snake species from image.
    Returns list of top-k matches: [(image_path, species_name, venom_type, score), ...]
    """
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(torch.device(DEVICE)) 
    with torch.no_grad():
        out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy().astype("float32")
    if USE_FAISS:
        # normalize emb to unit
        faiss.normalize_L2(emb)  # type: ignore[call-arg]
        D, I = index.search(emb, k)  # D: inner product similarities
        D = D.flatten().tolist()
        I = I.flatten().tolist()
        results = []
        for idx, score in zip(I, D):
            # metadata arrays correspond to embeddings order
            res = (
                embeddings_meta["image_path"][idx],
                embeddings_meta["species_name"][idx],
                embeddings_meta["venom_type"][idx],
                float(score),
            )
            results.append(res)
        return results
    else:
        # sklearn nearest neighbors: metric='cosine' returns distances in [0,2], lower is more similar
        nbrs = index
        distances, indices = nbrs.kneighbors(emb, n_neighbors=k)
        distances = distances.flatten().tolist()
        indices = indices.flatten().tolist()
        results = []
        for idx, dist in zip(indices, distances):
            # cosine similarity ~ 1 - dist (approx)
            score = 1.0 - dist
            res = (
                embeddings_meta["image_path"][idx],
                embeddings_meta["species_name"][idx],
                embeddings_meta["venom_type"][idx],
                float(score),
            )
            results.append(res)
        return results

def predict_one_image(image_path, processor, model, index, meta, k=1, csv_path=DEFAULT_CSV_PATH):
    """
    Main prediction function that returns comprehensive results.
    Returns: list of dictionaries with prediction results
    """
    df = pd.read_csv(csv_path)

    # Load input image
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(torch.device(DEVICE))

    with torch.no_grad():
        out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy().astype("float32")

    # Retrieve embeddings from FAISS/sklearn
    if USE_FAISS:
        faiss.normalize_L2(emb)  # type: ignore[call-arg]
        D, I = index.search(emb, k)
        similarities = D[0].tolist()
        indices = I[0].tolist()
    else:
        distances, indices = index.kneighbors(emb, n_neighbors=k)
        similarities = [1 - d for d in distances[0]]  # convert cosine distance → similarity

    results = []
    for idx, sim in zip(indices, similarities):
        idx = int(idx)
        original_idx = meta["original_idx"][idx]   # index of row in CSV
        row = df.loc[original_idx]  # index of row in CSV

        # Create a comprehensive result dictionary
        result_dict = {
            'image_path': row["image_path"],
            'species_name': row["species_name"],
            'venom_type': row["venom_type"],
            'hospital_importance': row.get("hospital_importance", "Not specified"),
            'first_aid': row.get("first_aid", "Not available"),
            'similarity_score': float(sim),
            'severity_level': row.get("severity_level", "Unknown"),
            'symptoms': row.get("symptoms", "Not available"),
            'reaction_stage': row.get("reaction_stage", "Not available"),
        }
        results.append(result_dict)

    return results

# ---------------------------
# Build index function (for CLI)
# ---------------------------
def build_index_from_csv(csv_path: str, embedding_output=EMBEDDING_FILE, rebuild_index=True):
    print(f"Loading CSV from: {csv_path}")
    df = load_csv(csv_path)
    image_paths = df["image_path"].tolist()
    print(f"Found {len(image_paths)} image paths in CSV (non-empty).")
    processor, model = load_vit_model()
    print("Extracting embeddings with ViT (this may take a while depending on number of images)...")
    emb_array, valid_idx_map, failed = extract_embeddings_from_paths(image_paths, processor, model)
    print(f"Extracted embeddings for {emb_array.shape[0]} images. Failures: {len(failed)}")
    # Build metadata arrays aligned with embeddings
    image_paths_valid = [image_paths[i] for i in valid_idx_map]
    species_valid = [df.loc[i, "species_name"] for i in valid_idx_map]
    venom_valid = [df.loc[i, "venom_type"] for i in valid_idx_map]
    original_idx_valid = valid_idx_map  # store original row idx if needed
    metadata = {
        "image_path": np.array(image_paths_valid),
        "species_name": np.array(species_valid),
        "venom_type": np.array(venom_valid),
        "original_idx": np.array(original_idx_valid),
    }
    # Save embeddings + metadata
    save_embeddings_npz(embedding_output, emb_array, metadata)
    # Build & save index
    if USE_FAISS:
        print("Building FAISS index (inner product on normalized vectors)...")
        emb_copy = emb_array.copy()
        faiss.normalize_L2(emb_copy)  # type: ignore[call-arg]
        index = faiss.IndexFlatIP(emb_copy.shape[1])
        index.add(emb_copy)  # type: ignore[call-arg]
        faiss.write_index(index, FAISS_INDEX_FILE)  # type: ignore[call-arg]
        print(f"Saved FAISS index to {FAISS_INDEX_FILE}")
    else:
        print("FAISS not available — building sklearn NearestNeighbors index with cosine metric.")
        nbrs = build_sklearn_index(emb_array)
        # sklearn NearestNeighbors doesn't have a direct save format here; we will keep embeddings file.
        # Optionally save a small pointer file
        np.save(SKLEARN_INDEX_FILE, np.array([0]))  # placeholder to indicate built
        print(f"Saved placeholder for sklearn index: {SKLEARN_INDEX_FILE}")
    print("Build complete.")
    if failed:
        print("Failed image opens (sample):")
        for p, e in failed[:10]:
            print(" -", p, "->", e)

# ---------------------------
# CLI functions (kept for backward compatibility)
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ViT-based snake detection (feature-extraction + similarity search).")
    p.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to CSV with image_path,species_name,venom_type")
    p.add_argument("--build-index", action="store_true", help="Build embeddings and index from CSV")
    p.add_argument("--predict", action="store_true", help="Run prediction for a single input image")
    p.add_argument("--camera", action="store_true", help="Capture image from webcam")
    p.add_argument("--image", type=str, help="Image path to predict")
    p.add_argument("--k", type=int, default=3, help="Top K matches to show")
    p.add_argument("--emb-file", type=str, default=EMBEDDING_FILE, help="Where to save/load embeddings npz")
    return p.parse_args()

def main():
    args = parse_args()
    if args.build_index:
        build_index_from_csv(args.csv, embedding_output=args.emb_file)
        return

    if args.predict:
        if not args.image:
            print("Please provide --image PATH to predict.")
            return
        # load index and metadata
        index, embeddings, meta = load_index_and_meta(args.emb_file)
        processor, model = load_vit_model()
        results = predict_image(args.image, processor, model, index, meta, k=args.k)
        print("Top matches (image_path, species_name, venom_type, similarity_score):")
        for r in results:
            print(f" - {r[0]}   | species: {r[1]}   | venom: {r[2]}   | score: {r[3]:.4f}")
        return

    # Camera mode
    if args.camera:
        print("📸 Starting camera...")
        image_path = capture_from_camera()
        
        if image_path is None:
            print("No image captured.")
            return

        processor, model = load_vit_model()
        index, embeddings, meta = load_index_and_meta(args.emb_file)

        results = predict_one_image(image_path, processor, model, index, meta, k=1)

        print("\n===== CAMERA PREDICTION RESULT =====")
        r = results[0]
        print(f"Image: {image_path}")
        print(f"Species: {r['species_name']}")
        print(f"Venom Type: {r['venom_type']}")
        print(f"Emergency: {r['hospital_importance']}")
        print(f"First Aid: {r['first_aid']}")
        print(f"Similarity: {r['similarity_score']:.4f}")
        return

    print("No action specified. Use --build-index or --predict or --camera. Run with -h for help.")

if __name__ == "__main__":
    main()