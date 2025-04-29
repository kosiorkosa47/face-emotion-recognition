import os
import argparse
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(processed_dir):
    """Load test data from processed .npy files."""
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    X_test = X_test.astype("float32") / 255.0
    if X_test.ndim == 3:
        X_test = np.expand_dims(X_test, -1)
    return X_test, y_test


def plot_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained emotion model on test set.")
    parser.add_argument("--processed_dir", default="data/processed", help="Directory with processed data .npy files.")
    parser.add_argument("--model_path", default="models/emotion_model.h5", help="Path to saved model.")
    parser.add_argument("--output_dir", default="evaluation", help="Directory to save reports and plots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data and model
    X_test, y_test = load_data(args.processed_dir)
    # Load model without compile to bypass custom loss deserialization issues
    model = tf.keras.models.load_model(args.model_path, compile=False)
    # Compile model for evaluation
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

    # Predict
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Classification report
    labels = ["angry","disgust","fear","happy","sad","surprise","neutral"]
    report = classification_report(y_test, y_pred, target_names=labels, digits=4)
    print(report)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, labels, cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # Save summary JSON
    summary = {"test_loss": float(loss), "test_accuracy": float(acc)}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved evaluation summary to {args.output_dir}/summary.json")

if __name__ == "__main__":
    main()
