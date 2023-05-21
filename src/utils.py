import warnings
from functools import partial
from typing import Tuple, Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import ndarray
from sklearn.datasets import load_wine
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold

from src.NeuralNetwork import NeuralNetwork
from src.Optimizer import SGD, Adam, AdaGrad

warnings.filterwarnings('ignore')
sns.set()


def load_data():
    '''
    Loads wine data and splits them to X and Y
    '''
    data = load_wine()
    return data['data'], data['target']


def instantialize_from_args(model_args: Dict[str, Any]) -> 'Neural Network':
    """
    Instantiates a Neural Network model from provided arguments.

    :param model_args: dictionary containing arguments for model initialization.
    :return: An instance of the NeuralNetwork class.
    """
    model_args = model_args.copy()
    optim = model_args.pop('optimizer').copy()
    optim_name = optim.pop('name')
    if optim_name == 'SGD':
        opt = SGD(**optim)
    elif optim_name == 'Adam':
        opt = Adam(**optim)
    elif optim_name == 'AdaGrad':
        opt = AdaGrad(**optim)
    else:
        raise ValueError('wrong optimizer')

    return NeuralNetwork(optimizer=opt, **model_args)


def evaluate_model(X: np.ndarray, Y: np.ndarray, model_args: Dict[str, Any], n_splits: int = 5) -> ndarray:
    """
    Evaluates the performance of a model using Stratified K-Fold cross-validation.

    :param X: The feature matrix.
    :param Y: The target vector.
    :param model_args: dictionary containing arguments for model initialization.
    :param n_splits: The number of folds in Stratified K-Fold cross-validation.
    :return: Mean accuracy of the model across all folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=model_args.get('random_seed', 0))
    acc = []
    for i, (train_indices, val_indices) in enumerate(skf.split(X, Y)):
        model = instantialize_from_args(model_args)
        model.fit(X[train_indices], Y[train_indices], X[val_indices], Y[val_indices])
        acc.append(model.history['val_accuracy'][-1])

    return np.mean(acc)


def random_search(X: np.ndarray, Y: np.ndarray, iterations: int, reps: int, pool: Any = __builtins__) -> Tuple[
    Dict[str, Any], float]:
    """
    Performs a random search to find best hyperparameters for the model.
    `iterations` different hyperparameters are randomized, and evaluated `reps` times (with different random seeds)
    returns best hyperparameters and accuracy

    :param X: The feature matrix.
    :param Y: The target vector.
    :param iterations: Number of different hyperparameters to randomize.
    :param reps: Number of repetitions for evaluating each set of hyperparameters.
    :param pool: multiprocessing pool for parallel computation.
    :return: Tuple of best hyperparameters and the corresponding accuracy.
    """
    args = [
        dict(hidden_sizes=[round(2 ** np.random.randint(0, 8)) for _ in range(np.random.randint(0, 4))],
             batch_size=np.random.randint(1, 32),
             num_epochs=np.random.randint(1, 100),
             optimizer=np.random.choice([
                 {
                     'name': 'SGD',
                     'learning_rate': 10 ** np.random.normal(-3, 0),
                     'momentum': np.random.uniform(0, 0.99),
                 },
                 {
                     'name': 'Adam',
                     'learning_rate': 10 ** np.random.normal(-3, 0),
                     'beta1': np.random.uniform(0, 0.99),
                     'beta2': np.random.uniform(0, 0.99),
                 },
                 {
                     'name': 'AdaGrad',
                     'learning_rate': 10 ** np.random.normal(-3, 0),
                 },
             ])
             ) for i in range(iterations)
    ]
    accs = np.array([
        list(pool.map(partial(evaluate_model, X, Y), [a | {'random_seed': j} for a in args]))
        for j in range(reps)
    ])
    accs = accs.mean(0)
    i = accs.argmax()
    return args[i], accs[i]


def eval_and_plot(X: np.ndarray, Y: np.ndarray, model_args: Dict[str, Any], best_args, verbose: bool = True):
    """
    Conducts model evaluation using Stratified K-Fold cross-validation and plots
    the accuracy and loss curves for the training and validation sets.

    :param X: The feature matrix.
    :param Y: The target vector.
    :param model_args: dictionary containing arguments for model initialization.
    :param verbose: If True, prints the best hyperparameters, last training loss,
                    last validation accuracy and confusion matrix.
    """
    y_pred, y_true = [], []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    history = []
    for i, (train_indices, val_indices) in enumerate(skf.split(X, Y)):
        model = instantialize_from_args(model_args)
        model.fit(X[train_indices], Y[train_indices], X[val_indices], Y[val_indices])
        history.append(model.history)

        y_pred.extend(Y[val_indices])
        y_true.extend(model.predict(X[val_indices]))

    history = {key: np.mean([h[key] for h in history], 0) for key in history[0].keys()}

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(pd.DataFrame(history)[['train_accuracy', 'val_accuracy']])
    plt.ylim(0, 1.05)

    plt.subplot(1, 2, 2)
    sns.lineplot(pd.DataFrame(history)[['train_loss', 'val_loss']])
    plt.ylim(0, 1.05)
    plt.show()

    if verbose:
        print(f"Evaluate model on best args : {best_args}")
        print(f"train loss: {history['train_loss'][-1]}")
        print(f"val_accuracy: {history['val_accuracy'][-1]}")

        print(ConfusionMatrixDisplay.from_predictions(y_true, y_pred))
        plt.show()


def plot_factor_dependency(name, values, default_args, X_train, y_train, reps=5, xlog=False, pool=__builtins__, show=True):
    """
    This function evaluates and plots the effect of a particular model parameter (hyperparameter) on
    the accuracy of a given model.

    Parameters:
    name (str): The name of the hyperparameter.
    values (list): A list of values for the hyperparameter to explore.
    default_args (dict): The default arguments for the model.
    reps (int): The number of repetitions for each hyperparameter value.
    xlog (bool): Whether to use log scale for x-axis.
    pool (pool object): A pool object for parallel computing.
    show (bool): Whether to display the plot.

    The function creates multiple versions of a model by varying the given hyperparameter's value.
    Each model's accuracy is evaluated, and the results are plotted to visualize how the
    hyperparameter value affects the model's accuracy. The best value for the hyperparameter
    (yielding the highest accuracy) is highlighted with a red vertical line on the plot.
    """
    res = {name: [], 'accuracy': []}
    for arg in values:
        args = default_args.copy()
        a = args
        for i, x in enumerate(name.split(',')):
            if i == name.count(','):
                a[x] = arg
            a = a[x]
        for accuracy in pool.map(partial(evaluate_model, X_train, y_train),
                                 [args | {'random_seed': i} for i in range(reps)]):
            if isinstance(arg, float):
                res[name].append(arg)
            else:
                res[name].append(str(arg))
            res['accuracy'].append(accuracy)

    df = pd.DataFrame.from_dict(res)
    sns.lineplot(data=df, x=name, y='accuracy', errorbar='sd')
    sns.scatterplot(data=df, x=name, y='accuracy', alpha=0.05)
    if xlog:
        plt.xscale('log')

    # best red line
    aggregated_df = df.groupby(name).mean().reset_index()
    row = aggregated_df[aggregated_df.accuracy == aggregated_df.accuracy.max()].iloc[0]
    plt.axvline(row[name], linestyle='dashed', color='r')
    plt.title(f'{name}\nfound best {name}={row[name]} with average accuracy {row["accuracy"] * 100:.2f}%')
    if show:
        plt.show()
