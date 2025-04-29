import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

class NumpyEmotionDataset(Dataset):
    """
    Custom Dataset for loading emotion recognition data from .npy files.
    """
    def __init__(self, X_path, y_path, transform=None):
        """
        Initialize the dataset with data and labels from .npy files.

        Args:
            X_path (str): Path to the .npy file containing the data.
            y_path (str): Path to the .npy file containing the labels.
            transform (callable, optional): Optional transform to apply to the data. Defaults to None.
        """
        self.X = np.load(X_path)
        self.y = np.load(y_path)
        self.transform = transform

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing the sample data and label.
        """
        img = self.X[idx]
        # Ensure 3 channels for MobileNetV2
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        img = Image.fromarray((img*255).astype(np.uint8)) if img.max() <= 1.0 else Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        label = int(self.y[idx])
        return img, label

# Evaluation transforms (same as used in training)
eval_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main():
    """
    Evaluate the trained PyTorch model on the test set and plot the confusion matrix.
    """
    # Load test data
    test_ds = NumpyEmotionDataset('data/processed/X_test.npy', 'data/processed/y_test.npy', transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    num_classes = len(np.unique(test_ds.y))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load('models/emotion_model_torch.pth', map_location=device))
    model = model.to(device)
    model.eval()

    # Evaluation
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print classification report and plot confusion matrix
    print(classification_report(all_labels, all_preds, digits=4))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(num_classes)), yticklabels=list(range(num_classes)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    os.makedirs('evaluation', exist_ok=True)
    plt.savefig('evaluation/confusion_matrix_torch.png')
    print('Confusion matrix saved to evaluation/confusion_matrix_torch.png')

if __name__ == '__main__':
    """
    Entry point for evaluation script.
    """
    main()
