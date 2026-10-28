"""
Mini Project: Handwritten Digit Recognition using a Deep Neural Network
-----------------------------------------------------------------------
Dataset : scikit-learn "digits" (1,797 images, 8x8 pixels, 10 classes)
Model   : Fully connected Deep Neural Network (Keras / TensorFlow)
Outputs : plots + metrics saved into the ./outputs folder

Run:  python digit_dnn.py
"""

import os
import json
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")           # save figures without needing a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------
# 0. Reproducibility and output folder
# ---------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load and explore the dataset
# ---------------------------------------------------------------
digits = load_digits()
X, y = digits.data, digits.target
print("Images:", X.shape[0], "| Features per image:", X.shape[1], "| Classes:", len(np.unique(y)))
print("Pixel value range:", X.min(), "to", X.max())

# Figure 1: sample images
fig, axes = plt.subplots(2, 10, figsize=(12, 3))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(digits.images[i], cmap="gray_r")
    ax.set_title(str(y[i]), fontsize=10)
    ax.axis("off")
plt.suptitle("Sample images from the digits dataset")
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_samples.png", dpi=150)
plt.close()

# Figure 2: class distribution
plt.figure(figsize=(6, 3.5))
counts = np.bincount(y)
plt.bar(range(10), counts, color="#3b6ea5")
plt.xticks(range(10))
plt.xlabel("Digit class")
plt.ylabel("Number of images")
plt.title("Class distribution")
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_class_distribution.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 2. Pre-processing: normalise pixels to [0, 1] and split the data
# ---------------------------------------------------------------
X = X / 16.0                      # pixel values originally 0..16

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

print("Train / Validation / Test:", len(X_train), len(X_val), len(X_test))

# ---------------------------------------------------------------
# 3. Model builder
# ---------------------------------------------------------------
def build_model(hidden_units, dropout=0.2, lr=1e-3):
    """hidden_units: list such as [128, 64]. Empty list = plain softmax classifier."""
    model = keras.Sequential()
    model.add(layers.Input(shape=(64,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(model, epochs=150):
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True)
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=32,
        callbacks=[early], verbose=0)

# ---------------------------------------------------------------
# 4. Baseline: Logistic Regression (a model with no hidden layers)
# ---------------------------------------------------------------
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, y_train)
base_acc = accuracy_score(y_test, logreg.predict(X_test))
print(f"Baseline (Logistic Regression) test accuracy: {base_acc:.4f}")

# ---------------------------------------------------------------
# 5. Main experiment: effect of network depth
# ---------------------------------------------------------------
configs = {
    "0 hidden layers (softmax)": [],
    "1 hidden layer  [128]": [128],
    "2 hidden layers [128, 64]": [128, 64],
    "3 hidden layers [128, 64, 32]": [128, 64, 32],
}

results = {}
histories = {}
models = {}
for name, units in configs.items():
    keras.backend.clear_session()
    tf.random.set_seed(SEED)
    m = build_model(units)
    h = train(m)
    test_loss, test_acc = m.evaluate(X_test, y_test, verbose=0)
    results[name] = {
        "params": int(m.count_params()),
        "epochs_run": len(h.history["loss"]),
        "val_acc": float(max(h.history["val_accuracy"])),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
    }
    histories[name] = h.history
    models[name] = m
    print(f"{name:32s} params={m.count_params():6d}  test_acc={test_acc:.4f}")

# Figure 3: depth comparison bar chart
names = list(results.keys())
accs = [results[n]["test_acc"] * 100 for n in names]
plt.figure(figsize=(8, 4))
bars = plt.bar(range(len(names)), accs, color=["#9aa5b1", "#5b8db8", "#2f5d8a", "#1b3a5c"])
plt.xticks(range(len(names)), ["0 hidden\n(softmax)", "1 hidden\nlayer", "2 hidden\nlayers", "3 hidden\nlayers"], fontsize=9)
plt.ylim(min(accs) - 3, 100)
plt.ylabel("Test accuracy (%)")
plt.title("Test accuracy vs network depth")
for b, a in zip(bars, accs):
    plt.text(b.get_x() + b.get_width() / 2, a + 0.1, f"{a:.2f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_depth_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Final model = 2 hidden layers [128, 64]
# ---------------------------------------------------------------
FINAL = "2 hidden layers [128, 64]"
final_model = models[FINAL]
final_hist = histories[FINAL]
final_model.save(f"{OUT}/digit_dnn_model.keras")

with open(f"{OUT}/model_summary.txt", "w") as f:
    final_model.summary(print_fn=lambda s: f.write(s + "\n"))

# Figure 4: training curves
ep = range(1, len(final_hist["loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ep, final_hist["accuracy"], label="Training")
ax[0].plot(ep, final_hist["val_accuracy"], label="Validation")
ax[0].set_title("Model accuracy")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Accuracy")
ax[0].legend()
ax[1].plot(ep, final_hist["loss"], label="Training")
ax[1].plot(ep, final_hist["val_loss"], label="Validation")
ax[1].set_title("Model loss")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_training_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. Evaluation of the final model
# ---------------------------------------------------------------
probs = final_model.predict(X_test, verbose=0)
y_pred = probs.argmax(axis=1)
final_acc = accuracy_score(y_test, y_pred)
print(f"\nFINAL MODEL test accuracy: {final_acc:.4f}")
print(classification_report(y_test, y_pred, digits=3))

report = classification_report(y_test, y_pred, digits=4, output_dict=True)

# Figure 5: confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6.5, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted digit")
plt.ylabel("True digit")
plt.title("Confusion matrix (test set)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_confusion_matrix.png", dpi=150)
plt.close()

# Figure 6: correct predictions
test_images = (X_test * 16).reshape(-1, 8, 8)
fig, axes = plt.subplots(2, 8, figsize=(12, 4.2))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(test_images[i], cmap="gray_r")
    colour = "green" if y_pred[i] == y_test[i] else "red"
    ax.set_title(f"T:{y_test[i]} P:{y_pred[i]}", color=colour, fontsize=9)
    ax.axis("off")
plt.suptitle("Sample predictions (T = true, P = predicted)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_predictions.png", dpi=150)
plt.close()

# Figure 7: misclassified images
wrong = np.where(y_pred != y_test)[0]
print("Misclassified test images:", len(wrong))
if len(wrong) > 0:
    n = min(len(wrong), 8)
    fig, axes = plt.subplots(1, n, figsize=(1.6 * n, 2.2))
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, wrong[:n]):
        ax.imshow(test_images[idx], cmap="gray_r")
        ax.set_title(f"T:{y_test[idx]} P:{y_pred[idx]}", color="red", fontsize=9)
        ax.axis("off")
    plt.suptitle("Misclassified test images")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig7_misclassified.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------
# 8. Save all numbers (used when writing the report)
# ---------------------------------------------------------------
summary = {
    "dataset": {"images": int(len(y)), "train": len(X_train), "val": len(X_val), "test": len(X_test)},
    "baseline_logreg_test_acc": float(base_acc),
    "depth_experiment": results,
    "final_model": FINAL,
    "final_test_acc": float(final_acc),
    "final_epochs_run": len(final_hist["loss"]),
    "misclassified": int(len(wrong)),
    "per_class_f1": {str(k): float(report[str(k)]["f1-score"]) for k in range(10)},
    "per_class_precision": {str(k): float(report[str(k)]["precision"]) for k in range(10)},
    "per_class_recall": {str(k): float(report[str(k)]["recall"]) for k in range(10)},
    "macro_f1": float(report["macro avg"]["f1-score"]),
    "confusion_matrix": cm.tolist(),
    "misclassified_pairs": [[int(y_test[i]), int(y_pred[i])] for i in wrong],
    "final_train_acc": float(final_hist["accuracy"][-1]),
    "final_val_acc": float(final_hist["val_accuracy"][-1]),
}
with open(f"{OUT}/results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nAll outputs saved in the 'outputs' folder.")
