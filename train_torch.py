import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from PIL import Image
import os

class NumpyEmotionDataset(Dataset):
    """
    Custom Dataset for loading emotion recognition data from .npy files.
    """
    def __init__(self, X_path, y_path, transform=None):
        self.X = np.load(X_path)
        self.y = np.load(y_path)
        self.transform = transform
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        img = self.X[idx]
        # Ensure 3 channels for MobileNetV2
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        img = Image.fromarray((img*255).astype(np.uint8)) if img.max() <= 1.0 else Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        label = int(self.y[idx])
        return img, label

def get_data_loaders(batch_size=64):
    """
    Load training and validation data and return DataLoaders.
    """
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_ds = NumpyEmotionDataset('data/processed/X_train.npy', 'data/processed/y_train.npy', transform=train_transform)
    val_ds = NumpyEmotionDataset('data/processed/X_val.npy', 'data/processed/y_val.npy', transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, len(np.unique(train_ds.y))

def train_model(num_epochs=30, batch_size=64, learning_rate=1e-3, model_path='models/emotion_model_torch.pth'):
    """
    Train the MobileNetV2 model on the emotion dataset.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, num_classes = get_data_loaders(batch_size)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_acc = 0.0
    os.makedirs('models', exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved to {model_path}")
    print("Training complete.")

def main():
    """
    Entry point for training script.
    """
    train_model()

if __name__ == '__main__':
    main()
