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


def build_model(n_inputs):
    model = Sequential()
    model.add(Input(shape=(n_inputs,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(), metrics=[metrics.mean_absolute_error])

    return model


def k_fold(x_train, y_train, k=4):

    train_len = len(x_train)
    fol_len = int(train_len / k)

    mae_store = []

    for i in range(k):
        temp_x_val = x_train[i * fol_len:(i + 1) * fol_len]
        temp_y_val = y_train[i * fol_len:(i + 1) * fol_len]

        temp_x_train = np.concatenate([x_train[:i * fol_len], x_train[(i + 1) * fol_len:]])
        temp_y_train = np.concatenate([y_train[:i * fol_len], y_train[(i + 1) * fol_len:]])

        model = build_model(temp_x_train.shape[1])

        history = model.fit(
            temp_x_train,
            temp_y_train,
            epochs=500,
            batch_size=1,
            validation_data=(temp_x_val, temp_y_val)
        )

        mae = history.history['val_mean_absolute_error']

        mae_store.append(mae)

    mae_np = np.array(mae_store)
    mean_mae = np.mean(mae_np, axis=0)

    fig, ax = plt.subplots(1, 1)
    for mae in mae_store:
        ax.plot(mae)
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    ax.plot(mean_mae)
    plt.show()
    plt.close()


def main():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    k_fold(x_train, y_train)


if __name__ == '__main__':
    main()
