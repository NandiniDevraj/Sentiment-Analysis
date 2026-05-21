"""
Deep Learning with Keras — Sentiment Analysis & Image Classification
=====================================================================
Two end-to-end deep learning experiments using Keras / TensorFlow:

  1. IMDB Sentiment Analysis — dense MLP on bag-of-words vectors
  2. MNIST Digit Recognition — CNN vs. fully-connected baseline

Concepts demonstrated:
  - Multi-hot / bag-of-words text vectorization
  - Dense feedforward networks for NLP (binary classification)
  - Convolutional neural networks for image classification
  - Overfitting detection via train vs. validation loss curves
  - Hyperparameter search: layer width, depth, dropout rate
  - CNN vs. fully-connected accuracy comparison on MNIST
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import imdb, mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")
np.random.seed(0)
tf.random.set_seed(0)


# ===========================================================================
# Part 1 — IMDB Sentiment Analysis
# ===========================================================================

NUM_WORDS = 10_000    # vocabulary size
VAL_SIZE  = 5_000     # validation split size


def vectorize_sequences(sequences, dimension=NUM_WORDS):
    """
    Convert integer-encoded reviews to multi-hot vectors.
    Each row has a 1 at every word index that appears in the review.
    """
    matrix = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        matrix[i, seq] = 1.0
    return matrix


def build_dense_model(hidden_units=(64, 64), activation="relu", input_dim=NUM_WORDS):
    """
    Configurable dense network for binary sentiment classification.

    Parameters
    ----------
    hidden_units : tuple of ints — widths of each hidden layer
    activation   : str — activation function for hidden layers
    input_dim    : int — input vector dimensionality
    """
    model = Sequential()
    model.add(Dense(hidden_units[0], activation=activation, input_shape=(input_dim,)))
    for units in hidden_units[1:]:
        model.add(Dense(units, activation=activation))
    model.add(Dense(1, activation="sigmoid"))   # binary output
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_training_history(history, title_prefix=""):
    """Plot loss and accuracy curves for train vs. validation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Val loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"{title_prefix} Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="Train acc")
    axes[1].plot(history.history["val_accuracy"], label="Val acc")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title=f"{title_prefix} Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def run_imdb_experiment():
    """
    Train and evaluate dense MLPs on IMDB sentiment.

    Steps:
      1. Load & vectorize data
      2. Train baseline model [64, 64] hidden units
      3. Plot train/val curves to inspect overfitting
      4. Compare wider/deeper architecture [128, 64, 32]
    """
    print("\n" + "=" * 60)
    print("PART 1 — IMDB Sentiment Analysis")
    print("=" * 60)

    # --- Load data ---
    (X_train_raw, y_train_raw), (X_test_raw, y_test) = imdb.load_data(num_words=NUM_WORDS)

    # Hold out a validation set from training data
    X_val_raw,   y_val   = X_train_raw[:VAL_SIZE],  y_train_raw[:VAL_SIZE]
    X_train_raw, y_train = X_train_raw[VAL_SIZE:],  y_train_raw[VAL_SIZE:]

    # --- Vectorize ---
    X_train = vectorize_sequences(X_train_raw)
    X_val   = vectorize_sequences(X_val_raw)
    X_test  = vectorize_sequences(X_test_raw)

    # --- Baseline model ---
    print("\n[Baseline] 2 hidden layers, 64 units each")
    model = build_dense_model(hidden_units=(64, 64))
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=512,
        validation_data=(X_val, y_val),
        verbose=2,
    )
    plot_training_history(history, title_prefix="IMDB Baseline")

    # --- Deeper / wider model ---
    print("\n[Tuned] 3 hidden layers: 128 → 64 → 32")
    model_tuned = build_dense_model(hidden_units=(128, 64, 32))
    history_tuned = model_tuned.fit(
        X_train, y_train,
        epochs=20,
        batch_size=512,
        validation_data=(X_val, y_val),
        verbose=2,
    )
    val_loss, val_acc = model_tuned.evaluate(X_val, y_val, verbose=0)
    print(f"\nTuned model — Val accuracy: {val_acc:.3f}, Val loss: {val_loss:.3f}")
    plot_training_history(history_tuned, title_prefix="IMDB Tuned")


# ===========================================================================
# Part 2 — MNIST Digit Recognition (CNN vs. Dense)
# ===========================================================================

def build_cnn(filters=(32, 64), kernel_sizes=(3, 3), activation="relu", dropout_rate=0.5):
    """
    Two-block convolutional network for 28×28 grayscale images.

    Architecture: Conv→Pool → Conv→Pool → Flatten → Dense(128) → Dropout → Softmax(10)
    """
    model = Sequential([
        Conv2D(filters[0], (kernel_sizes[0], kernel_sizes[0]),
               activation=activation, input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters[1], (kernel_sizes[1], kernel_sizes[1]), activation=activation),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation=activation),
        Dropout(dropout_rate),
        Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_dense_baseline(input_dim=784, n_classes=10):
    """Fully-connected baseline to compare against CNN."""
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        Dense(64,  activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def run_mnist_experiment():
    """
    Train CNN and fully-connected baseline on MNIST.
    Demonstrates the architectural advantage of convolutions for image data.
    """
    print("\n" + "=" * 60)
    print("PART 2 — MNIST Digit Recognition")
    print("=" * 60)

    # --- Load & preprocess ---
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

    X_train_raw = X_train_raw.astype("float32") / 255.0
    X_test_raw  = X_test_raw.astype("float32")  / 255.0

    # Reshape for Conv2D: (N, 28, 28, 1)
    X_train_4d = X_train_raw.reshape(-1, 28, 28, 1)
    X_test_4d  = X_test_raw.reshape(-1, 28, 28, 1)

    # Validation split
    X_val_4d,   y_val   = X_train_4d[:VAL_SIZE],  y_train_raw[:VAL_SIZE]
    X_train_4d, y_train = X_train_4d[VAL_SIZE:],  y_train_raw[VAL_SIZE:]
    X_test_4d,  y_test  = X_test_4d,              y_test_raw

    # One-hot encode
    y_train_oh = to_categorical(y_train, 10)
    y_val_oh   = to_categorical(y_val,   10)
    y_test_oh  = to_categorical(y_test,  10)

    # --- CNN ---
    print("\n[CNN] Filters: (32, 64), Kernel: 3×3, Dropout: 0.5")
    cnn = build_cnn()
    cnn.summary()

    history_cnn = cnn.fit(
        X_train_4d, y_train_oh,
        epochs=10,
        batch_size=128,
        validation_data=(X_val_4d, y_val_oh),
        verbose=2,
    )
    plot_training_history(history_cnn, title_prefix="MNIST CNN")

    # --- Hyperparameter search ---
    print("\n--- Hyperparameter search ---")
    configs = [
        dict(filters=(16, 32), kernel_sizes=(3, 3), dropout_rate=0.3),
        dict(filters=(32, 64), kernel_sizes=(5, 5), dropout_rate=0.5),
    ]
    for cfg in configs:
        print(f"\n  {cfg}")
        m = build_cnn(**cfg)
        h = m.fit(X_train_4d, y_train_oh, epochs=5, batch_size=128,
                  validation_data=(X_val_4d, y_val_oh), verbose=0)
        print(f"  Val accuracy: {h.history['val_accuracy'][-1]:.3f}")

    # --- Final CNN evaluation ---
    cnn_loss, cnn_acc = cnn.evaluate(X_test_4d, y_test_oh, verbose=0)
    y_pred_labels = np.argmax(cnn.predict(X_test_4d), axis=1)

    print(f"\n{'='*50}")
    print(f"CNN — Test accuracy: {cnn_acc:.4f} | Test loss: {cnn_loss:.4f}")
    print("\nClassification report (CNN):")
    print(classification_report(y_test, y_pred_labels))
    print("Confusion matrix (CNN):")
    print(confusion_matrix(y_test, y_pred_labels))

    # --- Dense baseline ---
    print(f"\n{'='*50}")
    print("Fully-connected baseline")
    X_train_fc = X_train_4d.reshape(-1, 784)
    X_val_fc   = X_val_4d.reshape(-1, 784)
    X_test_fc  = X_test_4d.reshape(-1, 784)

    fc_model = build_dense_baseline()
    fc_model.fit(X_train_fc, y_train_oh, epochs=10, batch_size=128,
                 validation_data=(X_val_fc, y_val_oh), verbose=0)
    fc_loss, fc_acc = fc_model.evaluate(X_test_fc, y_test_oh, verbose=0)

    print(f"Dense — Test accuracy: {fc_acc:.4f} | Test loss: {fc_loss:.4f}")
    print(f"\nCNN advantage: {(cnn_acc - fc_acc) * 100:.2f} percentage points")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    run_imdb_experiment()
    run_mnist_experiment()
