# src/utils.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

EMOTIONS = {
    0: "angry", 1: "disgust", 2: "fear",
    3: "happy", 4: "sad",    5: "surprise", 6: "neutral"
}

def load_image_data(data_dir="data/fer2013", img_size=(48,48)):
    emotion_to_label = {v:k for k,v in EMOTIONS.items()}
    images, labels = [], []
    for split in ("train", "test"):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for emotion_name, label in emotion_to_label.items():
            class_dir = os.path.join(split_dir, emotion_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                img_path = os.path.join(class_dir, fname)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)
    X = np.array(images, dtype=np.uint8)
    y = np.array(labels)
    return X, y

def split_data(X, y, test_size=0.1, val_size=0.1, random_state=42):
    # First split off test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Then split train and validation
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_relative,
        stratify=y_trainval, random_state=random_state
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_numpy_splits(output_dir="data/processed"):
    X, y = load_image_data()
    # Split into train, validation, and test sets
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Save datasets as .npy files
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    print(f"Saved train ({X_train.shape}), validation ({X_val.shape}), test ({X_test.shape}) splits to '{output_dir}'")
