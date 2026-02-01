# ML Fundamentals

> **A from-scratch Python machine learning library built to understand core ML algorithms at a foundational level.**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-required-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

This library implements fundamental machine learning algorithms from scratch using only NumPy. No scikit-learn, TensorFlow, or PyTorch—just pure mathematical implementations. I created this to understand how ML really works under the hood.

---

## ✨ Features

### **Regression**
- **Linear Regression** — Gradient descent with MSE cost function
- **Logistic Regression** — Binary classification with sigmoid activation and cross-entropy loss
- **Regularization** — L2 regularization support for preventing overfitting

### **Clustering**
- **K-Means** — Unsupervised clustering with automatic centroid initialization
- **Cost Optimization** — Multiple random initializations to find global minima

### **Decision Trees**
- **Entropy-based Splits** — Information gain calculation for optimal feature selection
- **One-Hot Encoding** — Built-in support for categorical features
- **Bagging** — Bootstrap aggregating for ensemble methods

### **Neural Networks**
- **Feedforward Networks** — Fully-connected dense layers with sigmoid activation
- **Backpropagation** — Automatic gradient computation and weight updates
- **Sequential API** — Keras-style model building

### **Anomaly Detection**
- **Gaussian Distribution** — Mean and variance estimation for outlier detection
- **Threshold Selection** — F1-score optimization on validation data

### **Preprocessing**
- **Feature Scaling** — Min-max normalization
- **Mean Normalization** — Zero-centered features
- **Z-Score Normalization** — Standardization with unit variance

### **Utilities**
- **Data Generation** — Synthetic datasets for testing algorithms
- **Visualization** — Matplotlib-based plotting for classification, regression, and decision boundaries

---

## 🚀 Quick Start

### Basic Usage

#### **Linear Regression Example**

```python
import numpy as np
from linear_regression import gradient_descent, mean_squared_error, compute_mse_gradients

# Training data
X_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([300.0, 500.0, 700.0, 900.0])

# Initialize parameters
w_init = 0.0
b_init = 0.0

# Hyperparameters
learning_rate = 1e-2
iterations = 10_000

# Train the model
w, b, cost_history, _ = gradient_descent(
    X_train, y_train, 
    w_init, b_init, 
    learning_rate, iterations,
    mean_squared_error, 
    compute_mse_gradients
)

print(f"Trained parameters: w={w:.4f}, b={b:.4f}")
```

#### **Logistic Regression Example**

```python
import numpy as np
from logistic_regression import sigmoid, compute_sig_gradient, binary_cross_entropy, predict
from generate_data import generate_separable

# Generate synthetic binary classification data
X, y = generate_separable(num_examples=200)

# Initialize parameters
w = np.zeros(X.shape[1])
b = 0.0
learning_rate = 1e-3

# Train for 10,000 iterations
for _ in range(10_000):
    dj_dw, dj_db = compute_sig_gradient(X, y, w, b)
    w -= learning_rate * dj_dw
    b -= learning_rate * dj_db

# Evaluate
final_cost = binary_cross_entropy(X, y, w, b)
predictions = predict(X, w, b)
accuracy = (predictions == y).mean()

print(f"Final cost: {final_cost:.4f}")
print(f"Accuracy: {accuracy:.2%}")
```

#### **K-Means Clustering Example**

```python
import numpy as np
from clusters import run_kMeans

# Generate some random 2D data
np.random.seed(42)
X = np.random.randn(100, 2)

# Find 3 clusters, run algorithm 50 times to avoid local minima
best_centroids, cost = run_kMeans(X, cluster_count=3, iters=50)

print(f"Cluster centroids:\n{best_centroids}")
print(f"Final cost: {cost:.4f}")
```

#### **Neural Network Example**

```python
from deep_learning import SigmoidNN, DenseLayer
from generate_data import generate_separable

# Generate binary classification data
X, y = generate_separable(num_examples=200)

# Build a 2-layer neural network
model = SigmoidNN()
model.sequential([
    DenseLayer(units=3, input_features=X.shape[1]),  # Hidden layer with 3 neurons
    DenseLayer(units=1, input_features=3)             # Output layer
])

# Compile and train
model.compile(alpha=0.01)
model.fit(X, y, epochs=1000)

# Make predictions
predictions = model.predict(X)
```

---

## 📚 Documentation

For detailed API documentation including all functions and classes, see **[DOCS.md](DOCS.md)**.

### Module Overview

| Module | Description |
|--------|-------------|
| `linear_regression.py` | Linear regression with gradient descent |
| `logistic_regression.py` | Binary classification with regularization |
| `decision_tree.py` | Entropy-based decision trees |
| `deep_learning.py` | Feedforward neural networks |
| `clusters.py` | K-means clustering algorithm |
| `anomaly.py` | Gaussian-based anomaly detection |
| `preprocessing.py` | Feature scaling and normalization |
| `generate_data.py` | Synthetic dataset generation |
| `plot.py` | Visualization utilities |

---

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

---

## 🛠️ Project Structure

```
ml_fundamentals/
├── README.md              # You are here
├── DOCS.md               # Auto-generated API documentation
├── linear_regression.py  # Linear regression implementation
├── logistic_regression.py # Logistic regression implementation
├── decision_tree.py      # Decision tree classifier
├── deep_learning.py      # Neural network library
├── clusters.py           # K-means clustering
├── anomaly.py            # Anomaly detection
├── preprocessing.py      # Data preprocessing utilities
├── generate_data.py      # Synthetic data generators
├── plot.py               # Visualization tools
├── tests/                # Test suite
│   └── test_tree.py
└── create_readme.py      # Script to generate DOCS.md
```

---

## 🤝 Contributing

This is a learning project, but contributions are welcome! If you find bugs or want to add new algorithms:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Commit your changes (`git commit -m 'Add SVM implementation'`)
4. Push to the branch (`git push origin feature/new-algorithm`)
5. Open a Pull Request

---

## 📖 Learning Resources

These implementations are heavily inspired by concepts from:
- **Andrew Ng's Machine Learning Course** (Coursera)

---

<div align="center">

**Built with 🧠 to understand how machines learn**

[⭐ Star this repo](https://github.com/yourusername/ml_fundamentals) if you find it helpful!

</div>
