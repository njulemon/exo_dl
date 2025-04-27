import numpy as np
from keras import Sequential, Input, losses, optimizers, metrics
from keras.src.datasets import boston_housing
from keras.api.layers import Dense
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt


def vectorize_sequences(sequences, dimension=10_000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.

    return results


def main():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(), metrics=[metrics.mean_absolute_error])

    history = model.fit(
        x_train,
        y_train,
        epochs=400,
        validation_data=(x_test, y_test),
    )

    fig, ax = plt.subplots(1, 1)
    ax.plot(history.epoch, history.history['mean_absolute_error'], label='train')
    ax.plot(history.epoch, history.history['val_mean_absolute_error'], label='validation')
    ax.legend()
    plt.show()
    plt.close()

    print(model.evaluate(x_test, y_test))

if __name__ == '__main__':
    main()