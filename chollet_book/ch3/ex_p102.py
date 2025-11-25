import numpy as np
from keras import Sequential, Input, losses, optimizers, metrics
from keras.src.datasets import reuters
from keras.api.layers import Dense
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt


def vectorize_sequences(sequences, dimension=10_000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.

    return results


def main():
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10_000)

    vec_train = vectorize_sequences(x_train)
    vec_test = vectorize_sequences(x_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(Input(shape=(vec_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    model.add(Dense(46, activation='softmax'))

    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.RMSprop(), metrics=[metrics.categorical_accuracy])

    history = model.fit(
        vec_train,
        y_train,
        epochs=20,
        validation_data=(vec_test, y_test),
        batch_size=128,
    )

    fig, ax = plt.subplots(1, 1)
    ax.plot(history.epoch, history.history['categorical_accuracy'], label='train')
    ax.plot(history.epoch, history.history['val_categorical_accuracy'], label='validation')
    ax.legend()
    plt.show()
    plt.close()

    print(model.evaluate(vec_test, y_test))

if __name__ == '__main__':
    main()