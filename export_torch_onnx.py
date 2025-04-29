import torch
import torch.nn as nn
from torchvision import models
import numpy as np

def export_model():
    """
    Export the trained PyTorch model to ONNX format for deployment.
    """
    num_classes = 7  # Change if you have a different number of classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load('models/emotion_model_torch.pth', map_location=device))
    model = model.to(device)
    model.eval()
    # Example input (batch_size=1, 3x96x96)
    dummy_input = torch.randn(1, 3, 96, 96, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        'models/emotion_model_torch.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=12
    )
    print('Model exported to models/emotion_model_torch.onnx')

if __name__ == '__main__':
    """
    Entry point for ONNX export script.
    """
    export_model()
