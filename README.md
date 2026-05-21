# Deep Learning with Keras — Sentiment Analysis & Image Classification

Two end-to-end deep learning experiments using Keras/TensorFlow, exploring NLP and computer vision tasks.

## Experiments

### 1. IMDB Sentiment Analysis (NLP)
Binary classification of movie reviews using a dense feedforward network on multi-hot bag-of-words vectors.

| Model | Val Accuracy |
|---|---|
| Baseline [64, 64] | ~87% |
| Tuned [128, 64, 32] | ~88% |

**Key observations:**
- Validation loss starts rising after ~4 epochs while training loss keeps falling — classic overfitting
- Plotting train vs. val curves is essential to catch this early
- Wider/deeper architecture gives marginal gains without dropout

### 2. MNIST Digit Recognition (Computer Vision)
CNN vs. fully-connected comparison on handwritten digit classification.

| Model | Test Accuracy |
|---|---|
| CNN (Conv→Pool → Conv→Pool → Dense) | ~99.2% |
| Fully-connected baseline | ~97.4% |

**Key observation:** CNN's spatial feature sharing gives a clear ~1.8% accuracy boost over a dense network with the same training budget, proving the architectural advantage of convolutions for image data.

## Concepts covered

- Multi-hot text vectorization (bag-of-words)
- Binary cross-entropy loss for NLP
- Overfitting detection via train vs. validation loss curves
- CNN architecture: Conv2D, MaxPooling2D, Dropout
- Categorical cross-entropy + softmax for multiclass
- Hyperparameter search: layer width, depth, filter count, dropout rate
- Classification report and confusion matrix analysis

## Running

```bash
pip install tensorflow numpy matplotlib scikit-learn
python deep_learning_experiments.py
```

> Datasets (IMDB, MNIST) are downloaded automatically by Keras on first run.

## Structure

```
deep_learning_experiments.py
├── run_imdb_experiment()    — Part 1: sentiment analysis
│   ├── vectorize_sequences()
│   ├── build_dense_model()
│   └── plot_training_history()
└── run_mnist_experiment()   — Part 2: digit recognition
    ├── build_cnn()
    └── build_dense_baseline()
```
