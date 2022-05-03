from keras.datasets import mnist
from ann_v3 import ANN
import numpy as np
import pickle
from matplotlib import pyplot as plt
import json

def plot_graphs(mode_msg: str):

    with open('stats.json') as f:
        stats = json.load(f)

    plt.plot(np.arange(len(stats['train_loss'])),
             stats['train_loss'],
             label="Training Loss - " + mode_msg)
    plt.title('Train losses')
    plt.xlabel('Training steps (divided by 100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"plots/train_losses.png")
    plt.show()

    plt.plot(np.arange(len(stats['val_loss'])),
             stats['val_loss'],
             label="Validation Loss - " + mode_msg)

    plt.title('Validation losses')
    plt.xlabel('Training steps (divided by 100)')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"plots/validation_losses.png")
    plt.show()

    plt.plot(np.arange(len(stats['val_acc_last_100_all'])),
             stats['val_acc_last_100_all'],
             label="Mean Validation Accuracy - " + mode_msg)

    plt.title('Mean Validation Accuracy Scores')
    plt.xlabel('Training steps (divided by 100)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"plots/validation_accuracy_scores.png")
    plt.show()


def get_one_hot(targets, nb_classes=10):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def main():
    np.random.seed(42)
    # loading the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (-1, 784)).T
    x_test = np.reshape(x_test, (-1, 784)).T
    # normalize the inputs
    x_train = x_train / 255
    x_test = x_test / 255

    y_tr = get_one_hot(y_train).T
    y_tst = get_one_hot(y_test).T

    ann = ANN()
    layers_dims = [784, 20, 7, 5, 10]
    n_iter = 5000
    batch_size = 64
    lr = 0.009
    # no batch norm
    batch_norm = False
    parameters, costs = ann.l_layer_model(X=x_train, Y=y_tr, layers_dims=layers_dims, learning_rate=lr,
                                          batch_size=batch_size, use_batchnorm=batch_norm)
    f_name = f'ann_parameters_{n_iter}_epochs_{batch_size}_batch_size_{lr}_lr.pkl'
    with open(f_name, 'wb') as fp:
        pickle.dump(parameters, fp)
    print(f'accuracy on train: {ann.predict(x_train, y_tr, parameters)}')
    print(f'accuracy on test: {ann.predict(x_test, y_tst, parameters)}')
    plot_graphs("No Batchnorm")

    # using batch norm
    batch_norm = True
    ann = ANN()
    parameters, costs = ann.l_layer_model(X=x_train, Y=y_tr, layers_dims=layers_dims, learning_rate=lr,
                                          batch_size=batch_size, use_batchnorm=batch_norm)
    f_name = f'ann_parameters_batchnorm_{n_iter}_epochs_{batch_size}_batch_size_{lr}_lr.pkl'
    with open(f_name, 'wb') as fp:
        pickle.dump(parameters, fp)
    plot_graphs("Using Batchnorm")
    print(f'accuracy on train: {ann.predict(x_train, y_tr, parameters, batch_norm)}')
    print(f'accuracy on test: {ann.predict(x_test, y_tst, parameters, batch_norm)}')


if __name__ == '__main__':
    main()