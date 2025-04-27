from keras.src.datasets import mnist
from keras import models, Input
from keras import layers

from keras import utils
from keras_visualizer import visualizer


def main():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    model = models.Sequential()
    model.add(Input(shape=(28* 28,)))
    model.add(layers.Dense(28 * 28, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))

    x_train = x_train.reshape(60000, 28 * 28).astype('float32') / 255
    x_test = x_test.reshape(10000, 28 * 28).astype('float32') / 255

    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=128)

    print(model.evaluate(x_test, y_test))

    model.summary()

if __name__ == '__main__':
    main()