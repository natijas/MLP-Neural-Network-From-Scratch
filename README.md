# Multi-layer Perceptron (MLP) Implementation & Exploratory Data Analysis from Scratch

This repository contains the implementation of a Multi-layer Perceptron (MLP) and Exploratory Data Analysis (EDA) on a wine dataset.
# Project Description

The project involves two primary tasks:

- Exploratory Data Analysis (EDA): EDA is performed to understand the data, find patterns, spot anomalies, test hypotheses, and check assumptions. The main goal of EDA is to provide insight into a dataset, bring important aspects of that dataset into focus for further analysis, and inform model selection and feature engineering.

- Multi-layer Perceptron (MLP) Implementation: MLPs are a class of feedforward artificial neural network. The MLP consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. MLPs are used for classification or regression tasks, and they can handle non-linear separable data.

# Project Structure

This project repository has the following structure:

- `src/`: This folder contains the source code for the Neural Network architecture and custom optimizers implemented in this project. In particular, you will find the implementation of the MLP (Multi-Layer Perceptron) model and self-implemented optimizers including Stochastic Gradient Descent (SGD) with Momentum, AdaGrad, and Adam.

- `notebooks/`: This folder contains Jupyter notebooks that walk through the various stages of the project. These stages include:

     - Exploratory Data Analysis (EDA): The EDA notebook includes a comprehensive analysis of the dataset. It provides an understanding of the data distribution, identifies correlations between different features, and detects any potential outliers.

     - Feature Engineering: The Feature Engineering notebook introduces modifications and additions to the existing features to enhance the model's predictive performance. These could include transformations, binning, or interaction features.

     - Feature Scaling: The Feature Scaling notebook presents different methods to scale the features, which is crucial for many machine learning models, including neural networks. The notebook might explore various scaling techniques such as Min-Max Scaling, Standard Scaling, and Robust Scaling.

     - Implemented MLP from Scratch with backpropagation algorithm 
        
     - Optimizer Comparison: In this notebook, the performance of different optimizers on the dataset is evaluated. Specifically, the custom-implemented SGD with Momentum, AdaGrad, and Adam optimizers are compared to determine which performs best for this specific dataset.
        
     - Hyperparameter Search: The Hyperparameter Search notebook is dedicated to finding the best hyperparameters for the MLP model. This involves trying out different combinations of hyperparameters and selecting the one that provides the best performance based on a specific metric.

     - Model Comparison: The Model Comparison notebook evaluates the performance of various models on the dataset. It compares the MLP model's performance with other popular classifiers and presents a comparison of their accuracies.

## This project requires Python 3.6 or later, and the following Python libraries installed:

    NumPy
    Pandas
    Matplotlib
    Seaborn
    Scikit-Learn
    PyTorch
    
 
