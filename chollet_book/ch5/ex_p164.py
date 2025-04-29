from keras.src.datasets import mnist
from keras import models, Input
from keras import layers

from keras import utils

import matplotlib.pyplot as plt


def main():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    model = models.Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

    fig, ax = plt.subplots(1, 1)
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    plt.show()
    plt.close()

    model.summary()

if __name__ == '__main__':
    main()