# face-emotion-recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Face emotion detection and recognition project using:

- **CNN (PyTorch, MobileNetV2 backbone)** trained on the FER-2013 dataset
- **OpenCV** for real-time face detection
- **ONNX export** for scalable deployment

[![CI](https://github.com/kosiorkosa47/face-emotion-recognition-main/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kosiorkosa47/face-emotion-recognition-main/actions/workflows/ci.yml)

## Features

- Train a convolutional neural network to classify facial expressions into 7 categories:
  - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Real-time face detection and emotion recognition via webcam
- Save and load trained model weights

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kosiorkosa47/face-emotion-recognition.git
   cd face-emotion-recognition
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

- Download and extract the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) so that images are organized in subfolders by emotion, for example:
  ```
  data/fer2013/train/happy/xxx.png
  data/fer2013/train/sad/yyy.png
  ...
  ```
- Preprocess the dataset into numpy arrays:
  ```bash
  python utils.py --data_dir data/fer2013 --output_dir data/processed
  ```
This generates `.npy` files in `data/processed` for training and evaluation.

> **Note:** If you only have the CSV version of FER-2013, you must first convert it to images organized by class folders. You can use, for example, [this Python script](https://github.com/Teamten47/Facial-Emotion-Image-Recognition/blob/main/convert_fer2013_to_images_and_landmarks.py) to perform the conversion.


## Usage

### 1. Train the model

To train the PyTorch model:
```bash
python train_torch.py
```

To test/evaluate the model:
```bash
python test_torch.py
```

To export the trained model to ONNX:
```bash
python export_torch_onnx.py
```

To generate confusion matrix and evaluation plots:
```bash
eval_torch.py
```

To run inference on a single image (PyTorch):
```python
from PIL import Image
import torch
from torchvision import transforms, models
img = Image.open('your_image.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, 7)
model.load_state_dict(torch.load('models/emotion_model_torch.pth', map_location='cpu'))
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    pred = output.argmax(1).item()
print(f'Predicted class: {pred}')
```

> **Note:** All training, evaluation, and inference scripts are now PyTorch-based and located in the main project folder. Scripts in the `src/` directory and any TensorFlow/Keras code are deprecated and not used in the current workflow.


### 2. Evaluate the model

```bash
python src/evaluate.py \
  --processed_dir data/processed \
  --model_path models/emotion_model.h5 \
  --output_dir evaluation
```

### 3. Run real-time detection

```bash
python detect.py \
  --model_path models/emotion_model.h5
```

The webcam window will open and display detected faces with predicted emotions.

## Project Structure

```
face-emotion-recognition/
├── train_torch.py           # PyTorch training script
├── test_torch.py            # PyTorch evaluation script
├── export_torch_onnx.py     # Export to ONNX
├── eval_torch.py            # Evaluation with confusion matrix plot
├── demo_inference.ipynb     # Jupyter notebook demo for single image inference
├── data/
│   ├── fer2013.csv
│   └── processed/           # Preprocessed .npy files
├── models/                  # Saved model weights
├── evaluation/              # Evaluation plots/results
├── requirements.txt
├── LICENSE
├── README.md
└── .github/
    └── workflows/
        └── ci.yml           # GitHub Actions workflow
```

## Contributing

Contributions welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
