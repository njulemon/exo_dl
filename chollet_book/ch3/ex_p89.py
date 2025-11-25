import numpy as np
from keras import Sequential, Input, metrics, losses, optimizers
from keras.src.datasets import imdb
from keras.src.layers import Dense
from matplotlib import pyplot as plt


def vectorize_sequences(sequences, dimension=10_000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.

    return results


def main():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10_000)

    vec_train = vectorize_sequences(x_train)
    vec_test = vectorize_sequences(x_test)

    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    x_train_partial = vec_train[10_000:]
    y_train_partial = y_train[10_000:]

    x_validation = vec_train[:10_000]
    y_validation = y_train[:10_000]

    model = Sequential()
    model.add(Input(shape=(vec_train.shape[1],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.RMSprop(), metrics=[metrics.binary_accuracy])

    history = model.fit(
        x_train_partial,
        y_train_partial,
        epochs=10,
        validation_data=(x_validation, y_validation),
        batch_size=512,
    )

    fig, ax = plt.subplots(1, 1)
    ax.plot(history.epoch, history.history['binary_accuracy'], label='train')
    ax.plot(history.epoch, history.history['val_binary_accuracy'], label='validation')
    ax.legend()
    plt.show()
    plt.close()

    print(model.evaluate(vec_test, y_test))


if __name__ == '__main__':
    main()
