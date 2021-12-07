# This file implements a function to plot the loss history
# of a neural network after training.
#
# Author: Alan Hencey

import matplotlib.pyplot as plt

def plot_loss(history, filename='model_loss.png'):
    # clear the current firgure
    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(filename)
