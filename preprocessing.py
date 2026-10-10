"""
Preprocessing module for Structural Damage Image Classification.
Extracts continuous normalized raw pixel vectors (48x48 = 2,304 dimensions),
handles stratified dataset pooling, and performs PCA dimensionality reduction.
"""

import os
import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from config import (
    IMG_SIZE, INPUT_DIM, NORM_RANGE, DEFAULT_DATA_DIRS,
    SAMPLE_SIZE, SPLIT_RATIOS, RANDOM_SEED, SAMPLE_DIR
)

# ImageNet BGR mean offset used in PEER Hub dataset
IMAGENET_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def preprocess_image(image_input, target_size=IMG_SIZE):
    """
    Standard preprocessing pipeline:
    1. Loads image from file path or array
    2. Restores color if stored with ImageNet mean subtraction
    3. Converts to single-channel Grayscale
    4. Resizes to canonical 48x48 resolution
    5. Normalizes continuous pixel intensities to [-1.0, 1.0]
    6. Flattens into a 2,304-dimensional 1D vector

    Returns:
        flat_vector: np.ndarray of shape (2304,) in range [-1.0, 1.0]
        gray_image: np.ndarray of shape (48, 48) uint8 for visualization
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Could not read image from {image_input}")
    else:
        img = image_input.copy()

    # Handle float images with potential ImageNet mean subtraction
    if img.dtype != np.uint8 and np.issubdtype(img.dtype, np.floating):
        if img.min() < 0:
            # Add back ImageNet BGR mean
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = img + IMAGENET_MEAN_BGR
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    # Convert to grayscale
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 1:
            gray = img[:, :, 0]
        else:
            gray = img[:, :, 0]
    else:
        gray = img

    # Resize to canonical 48x48
    if gray.shape != target_size:
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # Continuous normalization: [0, 255] -> [-1.0, 1.0]
    normalized = (gray.astype(np.float32) / 127.5) - 1.0
    flat_vector = normalized.flatten()

    return flat_vector, gray


def find_peer_dataset():
    """Locates the PEER Hub dataset directory across standard paths."""
    for d in DEFAULT_DATA_DIRS:
        if os.path.exists(d):
            test_x = os.path.join(d, "task2_X_test.npy")
            test_y = os.path.join(d, "task2_y_test.npy")
            if os.path.exists(test_x) and os.path.exists(test_y):
                return d
    return None


def load_and_preprocess_data(n_samples=SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    Loads dataset, balances Damaged and Undamaged classes 50/50,
    extracts continuous raw pixel features, and creates stratified splits.

    Returns:
        Dictionary containing X_train, X_val, X_test, y_train, y_val, y_test,
        pca, X_pca_train, X_pca_test.
    """
    np.random.seed(seed)
    data_dir = find_peer_dataset()

    if data_dir is not None:
        print(f"Loading official PEER Hub dataset from: {data_dir}")
        y_test_raw = np.load(os.path.join(data_dir, "task2_y_test.npy"))
        y_test_all = (y_test_raw[:, 0] == 1.0).astype(int)
        X_test_mmap = np.load(os.path.join(data_dir, "task2_X_test.npy"), mmap_mode='r')

        train_x_path = os.path.join(data_dir, "task2_X_train.npy")
        train_y_path = os.path.join(data_dir, "task2_y_train.npy")
        has_train = os.path.exists(train_x_path) and os.path.exists(train_y_path)

        if has_train:
            y_train_raw = np.load(train_y_path)
            y_train_all = (y_train_raw[:, 0] == 1.0).astype(int)
            X_train_mmap = np.load(train_x_path, mmap_mode='r')
            total_available = len(y_train_all) + len(y_test_all)
            print(f"  Found full dataset: {len(y_train_all)} Train + {len(y_test_all)} Test = {total_available} images")
        else:
            y_train_all = np.array([])
            X_train_mmap = None
            total_available = len(y_test_all)
            print(f"  Found {total_available} test images.")

        damaged_pool = [('test', i) for i, lbl in enumerate(y_test_all) if lbl == 1]
        undamaged_pool = [('test', i) for i, lbl in enumerate(y_test_all) if lbl == 0]

        if has_train:
            damaged_pool.extend([('train', i) for i, lbl in enumerate(y_train_all) if lbl == 1])
            undamaged_pool.extend([('train', i) for i, lbl in enumerate(y_train_all) if lbl == 0])

        n_per_class = min(n_samples // 2, len(damaged_pool), len(undamaged_pool))
        chosen_dmg = np.random.choice(len(damaged_pool), n_per_class, replace=False)
        chosen_undmg = np.random.choice(len(undamaged_pool), n_per_class, replace=False)

        selected_items = [(damaged_pool[idx], 1) for idx in chosen_dmg] + \
                         [(undamaged_pool[idx], 0) for idx in chosen_undmg]
        np.random.shuffle(selected_items)

        X_all = np.zeros((len(selected_items), INPUT_DIM), dtype=np.float32)
        y_all = np.zeros(len(selected_items), dtype=int)

        print(f"  Extracting raw pixel features for {len(selected_items)} balanced images...")
        for idx, (source, label) in enumerate(selected_items):
            split_type, img_idx = source
            img = X_train_mmap[img_idx] if split_type == 'train' else X_test_mmap[img_idx]
            vec, _ = preprocess_image(img, target_size=IMG_SIZE)
            X_all[idx] = vec
            y_all[idx] = label

    else:
        print("PEER Hub .npy files not found locally. Loading from sample_images/ directory...")
        dmg_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "Damaged", "*.*")))
        undmg_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "Undamaged", "*.*")))

        raw_dmg = [preprocess_image(f)[0] for f in dmg_files]
        raw_undmg = [preprocess_image(f)[0] for f in undmg_files]

        # Augment sample vectors to create balanced training pool
        aug_dmg = []
        aug_undmg = []
        for _ in range(150):
            base_d = raw_dmg[np.random.choice(len(raw_dmg))]
            noise = np.random.normal(0, 0.05, size=base_d.shape)
            aug_dmg.append(np.clip(base_d + noise, -1.0, 1.0))

            base_u = raw_undmg[np.random.choice(len(raw_undmg))]
            noise = np.random.normal(0, 0.05, size=base_u.shape)
            aug_undmg.append(np.clip(base_u + noise, -1.0, 1.0))

        X_all = np.vstack([aug_dmg, aug_undmg])
        y_all = np.array([1] * len(aug_dmg) + [0] * len(aug_undmg))
        shuffled = np.random.permutation(len(y_all))
        X_all = X_all[shuffled]
        y_all = y_all[shuffled]

    # Stratified Train / Val / Test Split
    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all, y_all, test_size=test_ratio, random_state=seed, stratify=y_all
    )
    val_adj = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_adj, random_state=seed, stratify=y_train_val
    )

    # 2D PCA Dimensionality Reduction for Visual Exploration
    pca = PCA(n_components=2, random_state=seed)
    X_pca_train = pca.fit_transform(X_train)
    X_pca_test = pca.transform(X_test)

    print(f"Dataset Preprocessed Successfully:")
    print(f"  Feature Dimensions: {INPUT_DIM} raw pixels ({IMG_SIZE[0]}x{IMG_SIZE[1]} Grayscale)")
    print(f"  Train set: {len(X_train)} samples (Damaged: {int(y_train.sum())}, Undamaged: {int(len(y_train) - y_train.sum())})")
    print(f"  Val set:   {len(X_val)} samples (Damaged: {int(y_val.sum())}, Undamaged: {int(len(y_val) - y_val.sum())})")
    print(f"  Test set:  {len(X_test)} samples (Damaged: {int(y_test.sum())}, Undamaged: {int(len(y_test) - y_test.sum())})")
    print(f"  2D PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'pca': pca,
        'X_pca_train': X_pca_train,
        'X_pca_test': X_pca_test
    }


if __name__ == "__main__":
    load_and_preprocess_data()
