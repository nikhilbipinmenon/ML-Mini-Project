"""
Models module for Structural Damage Image Classification.
Strictly implements the 3 classical syllabus models:
1. Logistic Regression (L2 Regularization)
2. Decision Tree (ID3 Algorithm: Shannon Entropy & Information Gain)
3. Support Vector Machine (Soft-Margin with RBF Kernel)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from config import LOGISTIC_PARAMS, DECISION_TREE_PARAMS, SVM_PARAMS


def get_logistic_regression():
    """
    Model 1: Logistic Regression with L2 Regularization (Ridge penalty).
    Optimizes the convex cross-entropy log-loss with L2 weight shrinkage:
        J(w, b) = -1/m sum [ y_i log(sigma(z_i)) + (1 - y_i) log(1 - sigma(z_i)) ] + (1 / 2C) ||w||_2^2
    Inductive Bias:
        Assumes linear class separability in feature space. Predictions rely on a
        global weighted linear combination of all 2,304 input pixels.
    """
    return LogisticRegression(**LOGISTIC_PARAMS)


def get_decision_tree():
    """
    Model 2: Decision Tree Classifier (ID3 Algorithm with Shannon Entropy).
    Splits nodes by maximizing Information Gain:
        Entropy: H(S) = - sum p_i log_2(p_i)
        Information Gain: IG(S, A) = H(S) - sum (|S_v| / |S|) * H(S_v)
    Continuous thresholding evaluates candidate cutpoints along pixel dimensions.
    Regularized via pre-pruning: max_depth=10, min_samples_split=5.
    Inductive Bias:
        Assumes decision boundaries are orthogonal, axis-aligned hyperplanes.
        Focuses on localized pixel thresholds rather than global patterns.
    """
    return DecisionTreeClassifier(**DECISION_TREE_PARAMS)


def get_svm():
    """
    Model 3: Support Vector Machine with Radial Basis Function (RBF) Kernel.
    Solves the soft-margin dual quadratic program:
        max sum alpha_i - 0.5 sum sum alpha_i alpha_j y_i y_j K(x_i, x_j)
        subject to 0 <= alpha_i <= C, sum alpha_i y_i = 0
        where K(x_i, x_j) = exp(-gamma ||x_i - x_j||^2)
    Inductive Bias:
        Smoothness in RKHS (Reproducing Kernel Hilbert Space). Decision depends
        solely on Euclidean distance to critical boundary boundary samples (support vectors).
    """
    return SVC(**SVM_PARAMS)


def initialize_all_models():
    """
    Initializes and returns a dictionary of all 3 syllabus models.
    """
    return {
        "Logistic Regression (L2)": get_logistic_regression(),
        "Decision Tree (ID3 - Entropy)": get_decision_tree(),
        "SVM (RBF Kernel)": get_svm()
    }
