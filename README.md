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

This project uses the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013). Place the downloaded `fer2013.csv` file in the `data/` directory.

## Usage

### 1. Train the model

```bash
python train.py \
  --data_path data/fer2013.csv \
  --epochs 50 \
  --batch_size 64 \
  --model_path models/emotion_model.h5
```

### 2. Run real-time detection

```bash
python detect.py \
  --model_path models/emotion_model.h5
```

The webcam window will open and display detected faces with predicted emotions.

## Project Structure

```
face-emotion-recognition/
├── data/                   # Dataset directory
│   └── fer2013.csv         # FER-2013 CSV file
├── models/                 # Saved model weights
├── train.py                # Script to train CNN model
├── detect.py               # Real-time detection script
├── requirements.txt        # Python dependencies
├── LICENSE
└── README.md
```

## Contributing

Contributions welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
