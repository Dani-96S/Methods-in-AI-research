import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from keras.utils import plot_model


def plot_myhistory(histories, time, epochs):
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8, 10))
    for history in histories:
        ax1.plot(history.history['acc'], color='blue')
        ax1.plot(history.history['val_acc'], color='red')
        ax2.plot(history.history['loss'], color='cyan')
        ax2.plot(history.history['val_loss'], color='magenta')
    ax1.set_title('model accuracy')
    ax2.set_title('model loss')
    ax1.set_ylabel('accuracy')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    plt.xticks(range(0, epochs), range(1, epochs + 1))
    ax1.legend(['train', 'test'], loc='upper left')
    ax2.legend(['train', 'test'], loc='upper left')
    plt.savefig('plots/{}_history.png'.format(time))
    # plt.show()
    plt.close()


def plot_mymodel(model, time):
    plot_model(model, to_file='plots/{}_model.png'.format(time),
               show_shapes=True)


def plot_classifier(histories, model, n_epochs):
    t = time.strftime("%m-%d-%H:%M:%S")
    plot_myhistory(histories, t, n_epochs)
    plot_mymodel(model, t)


def repeat_plot(s, repeats):
    hist = []
    for t in range(1, repeats):
        hist.append(s.history)
        s.fit_model()
    hist.append(s.history)
    plot_classifier(hist, s.model, s.n_epochs)
