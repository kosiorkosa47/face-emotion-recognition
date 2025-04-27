import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.utils.class_weight import compute_class_weight


def load_data(processed_dir):
    """Load processed numpy arrays for training and validation."""
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))

    # Normalize and add channel dimension if needed
    X_train = X_train.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0
    if X_train.ndim == 3:
        X_train = np.expand_dims(X_train, -1)
        X_val = np.expand_dims(X_val, -1)

    return (X_train, y_train), (X_val, y_val)


def build_model(input_shape, num_classes=7):
    """Build a simple CNN model for emotion recognition with data augmentation."""
    # Data augmentation pipeline
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomContrast(0.05),
    ])
    model = models.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,
        layers.Conv2D(32, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train emotion recognition CNN")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                        help="Directory with processed .npy data files")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--model_path", type=str, default="models/emotion_model.h5",
                        help="Path to save the best model")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="TensorBoard log directory")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    (X_train, y_train), (X_val, y_val) = load_data(args.processed_dir)
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))

    # Calculate class weights to address class imbalance
    classes = np.unique(y_train)
    class_weights_values = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, class_weights_values))

    # Build and compile model
    model = build_model(input_shape, num_classes)
    optimizer = optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Prepare directories
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Callbacks for checkpoints and TensorBoard
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=args.model_path,
        save_best_only=True,
        verbose=1
    )
    tensorboard_cb = callbacks.TensorBoard(log_dir=args.log_dir)

    # Learning rate scheduler and early stopping
    reduce_lr_cb = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # Train
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[checkpoint_cb, tensorboard_cb, reduce_lr_cb, early_stop_cb],
        class_weight=class_weights
    )

    # Save final model
    model.save(args.model_path)
    print(f"Training completed. Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
