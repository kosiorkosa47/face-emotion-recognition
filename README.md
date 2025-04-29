# face-emotion-recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Face emotion detection and recognition project using:

- **CNN (TensorFlow/Keras)** trained on the FER-2013 dataset
- **OpenCV** for real-time face detection

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

- Download the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) and place `fer2013.csv` in the `data/` directory.
- Preprocess the dataset:
    ```bash
    python src/utils.py \
      --input_csv data/fer2013.csv \
      --output_dir data/processed
    ```
This generates `.npy` files in `data/processed` for training and evaluation.

## Usage

### 1. Train the model

Use `train.py` with configurable backbones and advanced options:
```bash
python src/train.py \
  --processed_dir data/processed \
  --backbone mobilenetv2 \
  --resolution 96 \
  --use_focal_loss \
  --label_smoothing 0.1 \
  --use_cosine_lr \
  --fine_tune \
  --epochs 30 \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --fine_tune_epochs 10 \
  --fine_tune_lr 1e-5 \
  --model_path models/emotion_model.h5 \
  --log_dir logs
```

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
├── data/
│   ├── fer2013.csv
│   └── processed/              # Preprocessed .npy files
├── src/
│   ├── train.py                # Training script with advanced options
│   ├── evaluate.py             # Evaluation script for test set
│   ├── utils.py                # Data loading and preprocessing utilities
│   └── detect.py               # Real-time detection script
├── models/                     # Saved model weights
├── evaluation/                 # Generated evaluation reports and plots
├── requirements.txt
├── LICENSE
└── README.md
```

## Contributing

Contributions welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
