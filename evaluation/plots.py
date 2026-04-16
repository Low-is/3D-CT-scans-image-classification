import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


# -----------------------------
# 1. Training curves
# -----------------------------
def plot_training_history(history):
    """
    Plot accuracy and loss curves from model.fit()
    """

    plt.figure()

    # Accuracy
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Loss
    plt.figure()

    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# -----------------------------
# 2. Confusion matrix
# -----------------------------
def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    y_true: integer labels
    y_pred: integer labels
    """

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = cm.shape[0]
    tick_marks = np.arange(classes)

    plt.xticks(tick_marks, class_names if class_names else tick_marks)
    plt.yticks(tick_marks, class_names if class_names else tick_marks)

    # annotate cells
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(classes), range(classes)):
        plt.text(
            j, i,
            cm[i, j],
            ha="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 3. Visualize CT slices
# -----------------------------
def plot_volume_slices(volume, num_slices=6):
    """
    volume shape: (H, W, D, 1) or (H, W, D)
    """

    if volume.ndim == 4:
        volume = volume[..., 0]

    depth = volume.shape[2]
    indices = np.linspace(0, depth - 1, num_slices, dtype=int)

    plt.figure(figsize=(12, 3))

    for i, idx in enumerate(indices):
        plt.subplot(1, num_slices, i + 1)
        plt.imshow(volume[:, :, idx], cmap="gray")
        plt.title(f"Slice {idx}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
