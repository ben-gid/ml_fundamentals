# API Documentation

> Auto-generated documentation for all modules, classes, and functions.

---

## File: `anomaly.py`

### Description
> Anomaly Detection

### Top-Level Functions

#### `def estimate_gaussian(X)`
Calculates mean and variance of all features 
in the dataset

Args:
    X (ndarray): (m, n) Data matrix

Returns:
    mu (ndarray): (n,) Mean of all features
    var (ndarray): (n,) Variance of all features

#### `def select_threshold(y_val, p_val)`
Finds the best threshold to use for selecting outliers 
based on the results from a validation set (p_val) 
and the ground truth (y_val)

Args:
    y_val (ndarray): Ground truth on validation set
    p_val (ndarray): Results on validation set
    
Returns:
    epsilon (float): Threshold chosen 
    F1 (float):      F1 score by choosing epsilon as threshold

---

## File: `clusters.py`

### Description
> K-means clustering

### Top-Level Functions

#### `def main()`
No docstring provided.

#### `def cluster_cost(X, centroids)`
computes cluster_cost

Args:
    X (np.ndarray): data
    centroids (np.ndarray): centroids from running Kmeans

Returns:
    float: cost

#### `def cluster(X, cluster_count, seed)`
clusters dataset into specified clusters count

Args:
    X (np.ndarray): data
    cluster_count (int): count to cluster data into
    seed (Optional[int], optional): seed to make rng ordered(not random). Defaults to None.

Returns:
    np.ndarray: centroids of clusters

#### `def find_closest_centroids(X, centroids)`
Computes the centroid memberships for every example

Args:
    X (ndarray): (m, n) Input values      
    centroids (ndarray): (K, n) centroids

Returns:
    idx (array_like): (m,) closest centroids

#### `def run_kMeans(X, cluster_count, iters)`
runs the Kmeans algorithm iters times to get the centroid 
with the lowest cost/diferentiaton

Args:
    X (np.ndarray): data with shape (m, n)
    cluster_count (int): desired number of clusters
    iters (int): iterations to perform Kmeans

Returns:
    tuple[np.ndarray, np.ndarray, float]: 
    best_centroids: centroids with the lowest cost with shape (cluster_count, n),
    cost: cost of best_centroids

---

## File: `generate_data.py`

### Description
> Generate data for machine learning

### Top-Level Functions

#### `def generate_separable(num_examples)`
generates separable data for binary classification.

Args:
    num_examples (int, optional): number of examples to generate. Defaults to 200.

Returns:
    tuple[np.ndarray, np.ndarray]: 
    X: 2d array of examples size=(num_examples, 2)
    y: targets, 1d array of 0 or 1

#### `def generate_separable_1d(num_examples)`
No docstring provided.

---

## File: `linear_regression.py`

### Description
> Linear Regression

### Top-Level Functions

#### `def main()`
No docstring provided.

#### `def main2()`
No docstring provided.

#### `def mean_squared_error(y_hats, ys)`
_summary_

Args:
    y_hats (np.ndarray): array of predictions
    ys (np.ndarray): array of targets

Returns:
    np.float64: cost

#### `def predict(xs, w, b)`
_summary_

Args:
    xs (np.ndarray): array of training data
    w (float | np.ndarray ): weight (float for single dimension training data, 
        array for > 1)
    b (float): base

Returns:
    np.ndarray: prediction (y_hat)

#### `def compute_mse_gradients(xs, ys, w, b)`
_summary_

Args:
    xs (np.ndarray): array of training data
    ys (np.ndarray): array of targets
    w (float | np.ndarray ): weight (float for single dimension training data, 
        array for > 1)
    b (float): base

Returns:
    tuple[float, float]: gradients (weight gradient, base gradient)

#### `def gradient_descent(xs, ys, w_in, b_in, alpha, iterations, cost_function, gradient_function)`
applies gradient descent to find cost minimum. 

Args:
    xs (np.ndarray): array of training data
    ys (np.ndarray): array of targets
    w_in w (float | np.ndarray ): starting weight (float for single dimension 
        training data, array for > 1)
    b_in (float): starting base
    alpha (float): learning rate
    iterations (int): number of iterations to calculate gradient and update parameters
    cost_function (Callable[[np.ndarray, np.ndarray], np.float64]): function to calculate cost 
    gradient_function (Callable[[np.ndarray, np.ndarray, float, float], tuple[float, float]]): 
        function to calculate cost function gradient

Returns:
    tuple[float, float, list[np.float64], list[list[float]]]: [final weight, final base, 
    cost_history, param_history]

---

## File: `plot.py`

### Description
> Plot various functions and data

### Top-Level Functions

#### `def quadratic(x, a, b, c)`
Computes a quadratic to a np.ndarray

Args:
    x (np.ndarray): array of values to apply quadratic to 
    a (float): first scalar in quadratic
    b (float): second scalar in quadratic
    c (float): third scalar in quadratic

Returns:
    np.ndarray: result of apply quadratic to x

#### `def vertex_2(x)`
No docstring provided.

#### `def compute_vertex(a, b, c)`
Computes vertex (x, y) for y = ax^2 + bx + c

#### `def plot_function(x, func)`
plots a function with matplotlip and saves it as the functions name .png

Args:
    x (np.ndarray): array to apply function to
    func (Callable[[np.ndarray], float]): function to plot

#### `def plot_binary_classification(X, y, x_names)`
creates a matplotlib plot that plots x and y. you can view the plot after calling this fuction
in jupyter with plt.show() and save it with plt.savefig.

Args:
    X (np.ndarray): data training set
    y (np.ndarray): data target
    x_names (list[str] | None, optional): names for each x feature. Defaults to None.

Raises:
    ValueError: _description_
    ValueError: _description_

#### `def plot_one_variable_classification(X, Y)`
No docstring provided.

#### `def plot_actual_vs_prediction(data, Y, y_hats, feature_names)`
No docstring provided.

#### `def plot_decision_boundary(X, y, w, b)`
No docstring provided.

---

## File: `logistic_regression.py`

### Description
> Logistic Regression

### Top-Level Functions

#### `def main()`
No docstring provided.

#### `def sigmoid(X, w, b)`
No docstring provided.

#### `def unsimplified_classification_cost(xs, ys, w, b)`
No docstring provided.

#### `def binary_cross_entropy(xs, ys, w, b, eps)`
No docstring provided.

#### `def predict(X, w, b)`
No docstring provided.

#### `def compute_sig_gradient(xs, ys, w, b)`
No docstring provided.

#### `def regularized_classification_cost(xs, ys, w, b, lamb, eps)`
No docstring provided.

#### `def compute_regularized_sig_gradient(xs, ys, w, b, lamb)`
No docstring provided.

---

## File: `decision_tree.py`

### Description
> Decision Tree

### Top-Level Functions

#### `def main()`
No docstring provided.

#### `def entropy(p1)`
calculates entropy/impurity of classified training data for
decision tree. 

Args:
    p1 (float | np.ndarray): mean of training data set examples as class 1.
    (# samples with class 1 in node) / (total samples in node)

Returns:
    float | np.ndarray: entropy or impurity of p1

#### `def split_at_feature(X, feature_idx)`
Splits X by chosen feature.

Args:
    X (np.ndarray): training data as 2d array
    feature_idx (int): feature (column) index of X to split on

Returns:
    tuple[np.ndarray, np.ndarray]: left_indices, right_indices

#### `def weighted_entropy(X, y, left_idcs, right_idcs)`
calculates entropy based on examples in new branch.

Args:
    X (np.ndarray): data as 2d array
    y (np.ndarray): binary values as 1d array
    left_idcs (np.ndarray): indices of examples in X to go to the left branch/node
    right_idcs (np.ndarray): indices of examples in X to go to the right branch/node

Returns:
    (float): weighted entropy as float

#### `def information_gain(X, y, left_idcs, right_idcs)`
calculates information gained if X is split using left and right indices.

Args:
    X (np.ndarray): data as 2d array
    y (np.ndarray): binary values as 1d array
    left_idcs (np.ndarray): indices of examples in X to go to the left branch/node
    right_idcs (np.ndarray): indices of examples in X to go to the right branch/node

Returns:
    (float): information gain as float

#### `def best_separation(X, y)`
returns the feature of X (X[:,j]) with the highest information gain to separate on.

Args:
    X (np.ndarray): data as 2d array
    y (np.ndarray): binary values as 1d array

Raises:
    ValueError: if X isn't a 2d array
    RuntimeError: if best_idx is never assigned a truthful value

Returns:
    int: index of highest information gain

#### `def one_hot_encode(X)`
distributes non binary features to new on hot encoded binary features.

Args:
    X (np.ndarray): data as 2d array

Returns:
    tuple[list[int], list[np.ndarray]]: list of non_binary features 
    and list of new binary one hot encoded features 

#### `def generate_bagged_sample(X, y, B)`
generates a new X and y to train a decision tree on using sampling with replacement.

Args:
    X (np.ndarray): data as 2d array
    y (np.ndarray): binary values as 1d array
    B (int): length of desired resulting X and y

Returns:
    tuple[np.ndarray, np.ndarray]: sampled_X and sampled_y

---

## File: `preprocessing.py`

### Description
> Functions to feature scale data to improve accuracy and speed convergence of ml models

### Top-Level Functions

#### `def scale(features)`
Scales each feature by dividing by its maximum value (feature-wise)

#### `def mean_normalization(features)`
Applies mean normalization feature-wise.

#### `def z_score_normalization(features)`
Applies Z-score normalization feature-wise.

---

## File: `__init__.py`

### Description
> No module docstring provided.

---

## File: `deep_learning.py`

### Description
> Classification Neural Network library with forward and backprop using sigmoid activation and Gradient descent

### Classes

#### `class DenseLayer`
> No class docstring.

**Methods:**

- **`__init__(self, units, input_features)`**
  - No docstring.
- **`W(self)`**
  - No docstring.
- **`W(self, W)`**
  - No docstring.
- **`b(self)`**
  - No docstring.
- **`b(self, b)`**
  - No docstring.
- **`forward(self, A_in)`**
  - No docstring.
- **`backward(self, delta, A_in, batch_size, alpha)`**
  - No docstring.

#### `class SigmoidNN`
> No class docstring.

**Methods:**

- **`__init__(self)`**
  - No docstring.
- **`sequential(self, layers)`**
  - No docstring.
- **`compile(self, alpha)`**
  - No docstring.
- **`fit(self, X, y, epochs)`**
  - No docstring.
- **`predict(self, X)`**
  - No docstring.

### Top-Level Functions

#### `def main()`
No docstring provided.

#### `def vec_sigmoid(X, W, b)`
vectorized sigmoid

Args:
    X (np.ndarray): 2d array of data
    W (np.ndarray): 2d array of weights
    b (np.ndarray): 1d array of bias's

Returns:
    np.ndarray: sigmoid

#### `def binary_cross_entropy(a_out, y)`
sigmoid cost function

Args:
    a_out (np.ndarray): 1d array of output from node
    y (np.ndarray): 1d target values

Returns:
    float: cost

---

