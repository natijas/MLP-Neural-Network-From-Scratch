from sklearn.model_selection import train_test_split

from src.utils import load_data, eval_and_plot

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)

    best_args = {'hidden_sizes': [128],
                 'batch_size': 7,
                 'num_epochs': 25,
                 'optimizer': {'name': 'Adam',
                               'learning_rate': 0.001,
                               'beta1': 0.4828766300042446,
                               'beta2': 0.9780262776787818}}

    eval_and_plot(X_train, y_train, best_args, best_args, verbose=True)
