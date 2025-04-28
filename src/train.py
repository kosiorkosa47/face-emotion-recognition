import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
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
    # Stronger data augmentation for better generalization
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
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


def build_transfer_model(input_shape, num_classes=7, resolution=96):
    """Build a transfer learning model using MobileNetV2 as feature extractor."""
    # Create MobileNetV2 feature extractor
    base_model = MobileNetV2(input_shape=(resolution,resolution,3), include_top=False, weights='imagenet', name='mobilenetv2_base')
    base_model.trainable = False
    inputs = layers.Input(shape=input_shape)
    # Resize to MobileNet input and convert grayscale to RGB
    x = layers.Resizing(resolution,resolution)(inputs)
    # Convert grayscale to RGB by duplicating channels
    x = layers.Concatenate()([x, x, x])
    # Feature extraction
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)


def build_efficientnet_model(input_shape, num_classes=7, resolution=96):
    """Build a transfer learning model using EfficientNetB0 as feature extractor."""
    base_model = EfficientNetB0(input_shape=(resolution,resolution,3), include_top=False, weights='imagenet', name='efficientnetb0_base')
    base_model.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = layers.Resizing(resolution,resolution)(inputs)
    x = layers.Concatenate()([x, x, x])
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)


def sparse_categorical_focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        ce = -tf.math.log(y_pred) * y_true_one_hot
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * ce
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return loss_fn


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
    parser.add_argument("--backbone", choices=["cnn","mobilenetv2","efficientnet"], default="cnn",
                        help="Model backbone: cnn, mobilenetv2, or efficientnetb0")
    parser.add_argument("--resolution", type=int, default=96,
                        help="Input resolution for backbone models")
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="Use focal loss instead of cross-entropy")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing factor for cross-entropy loss")
    parser.add_argument("--fine_tune", action="store_true",
                        help="Fine-tune the backbone after initial training")
    parser.add_argument("--fine_tune_epochs", type=int, default=10,
                        help="Number of epochs for fine-tuning")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--use_cosine_lr", action="store_true",
                        help="Use cosine decay learning rate schedule")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    (X_train, y_train), (X_val, y_val) = load_data(args.processed_dir)

    # Oversample minority classes to balance training set
    classes, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()
    X_resampled, y_resampled = [], []
    for cls, count in zip(classes, counts):
        idx = np.where(y_train == cls)[0]
        X_resampled.append(X_train[idx])
        y_resampled.append(y_train[idx])
        num_extra = max_count - count
        if num_extra > 0:
            extra_idx = np.random.choice(idx, size=num_extra, replace=True)
            X_resampled.append(X_train[extra_idx])
            y_resampled.append(y_train[extra_idx])
    # Concatenate and shuffle
    X_train = np.concatenate(X_resampled, axis=0)
    y_train = np.concatenate(y_resampled, axis=0)
    perm = np.random.permutation(len(y_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))

    # Calculate class weights to address class imbalance
    classes = np.unique(y_train)
    class_weights_values = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, class_weights_values))

    # Build model based on backbone choice
    if args.backbone == "cnn":
        model = build_model(input_shape, num_classes)
    elif args.backbone == "mobilenetv2":
        model = build_transfer_model(input_shape, num_classes, resolution=args.resolution)
    elif args.backbone == "efficientnet":
        model = build_efficientnet_model(input_shape, num_classes, resolution=args.resolution)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    # Optimizer with optional cosine decay schedule
    if args.use_cosine_lr:
        total_steps = args.epochs
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=total_steps
        )
        optimizer = optimizers.Adam(learning_rate=lr_schedule)
    else:
        optimizer = optimizers.Adam(learning_rate=args.learning_rate)

    # Configure loss
    if args.use_focal_loss:
        loss_fn = sparse_categorical_focal_loss()
    elif args.label_smoothing > 0:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=args.label_smoothing)
    else:
        loss_fn = "sparse_categorical_crossentropy"
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

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

    # Fine-tuning: unfreeze backbone if requested
    if args.backbone != "cnn" and args.fine_tune:
        # Unfreeze backbone for fine-tuning
        layer_name = 'mobilenetv2_base' if args.backbone == 'mobilenetv2' else 'efficientnetb0_base'
        base_model = model.get_layer(layer_name)
        base_model.trainable = True
        # Recompile with lower learning rate
        model.compile(
            optimizer=optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss=loss_fn,
            metrics=["accuracy"]
        )
        print("Starting fine-tuning of backbone...")
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            callbacks=[checkpoint_cb, tensorboard_cb, reduce_lr_cb, early_stop_cb],
            class_weight=class_weights
        )
        model.save(args.model_path)
        print(f"Fine-tuning completed. Model saved to {args.model_path}")

    # Save final model
    model.save(args.model_path)
    print(f"Training completed. Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
