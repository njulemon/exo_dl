import os
import shutil

from keras import Sequential
from keras import layers
from keras import optimizers
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.src.utils import image_dataset_from_directory
import tensorflow as tf
from matplotlib import pyplot as plt


def copy_sets():

    path_original = '/Users/nicolasjulemont/Documents/DATA/chollet/dogs-vs-cats/train'
    path_exo = '/Users/nicolasjulemont/Documents/DATA/chollet/dogs-vs-cats/exo'

    path_train_dogs = os.path.join(path_exo, 'train/dogs')
    path_train_cats = os.path.join(path_exo, 'train/cats')
    path_val_dogs = os.path.join(path_exo, 'val/dogs')
    path_val_cats = os.path.join(path_exo, 'val/cats')
    path_test_dogs = os.path.join(path_exo, 'test/dogs')
    path_test_cats = os.path.join(path_exo, 'test/cats')

    os.makedirs(path_train_dogs, exist_ok=True)
    os.makedirs(path_train_cats, exist_ok=True)
    os.makedirs(path_val_dogs, exist_ok=True)
    os.makedirs(path_val_cats, exist_ok=True)
    os.makedirs(path_test_dogs, exist_ok=True)
    os.makedirs(path_test_cats, exist_ok=True)

    # train dogs
    f_names = [f'dog.{id}.jpg' for id in range(0, 1600)]
    for file in f_names:
        src = os.path.join(path_original, file)
        dst = os.path.join(path_train_dogs, file)
        shutil.copy(src, dst)

    # train cats
    f_names = [f'cat.{id}.jpg' for id in range(0, 1600)]
    for file in f_names:
        src = os.path.join(path_original, file)
        dst = os.path.join(path_train_cats, file)
        shutil.copy(src, dst)

    # val dogs
    f_names = [f'dog.{id}.jpg' for id in range(1600, 2400)]
    for file in f_names:
        src = os.path.join(path_original, file)
        dst = os.path.join(path_val_dogs, file)
        shutil.copy(src, dst)

    # val cats
    f_names = [f'cat.{id}.jpg' for id in range(1600, 2400)]
    for file in f_names:
        src = os.path.join(path_original, file)
        dst = os.path.join(path_val_cats, file)
        shutil.copy(src, dst)

    # test dogs
    f_names = [f'dog.{id}.jpg' for id in range(2400, 3200)]
    for file in f_names:
        src = os.path.join(path_original, file)
        dst = os.path.join(path_test_dogs, file)
        shutil.copy(src, dst)

    # test cats
    f_names = [f'cat.{id}.jpg' for id in range(2400, 3200)]
    for file in f_names:
        src = os.path.join(path_original, file)
        dst = os.path.join(path_test_cats, file)
        shutil.copy(src, dst)

def build_model():

    model = Sequential()
    model.add(layers.Input(shape=(150, 150, 3)))
    model.add(layers.RandomRotation(factor=1/8, fill_mode='nearest'))
    model.add(layers.RandomTranslation(width_factor=0.2, height_factor=0.2, fill_mode='nearest'))
    model.add(layers.RandomZoom(height_factor=0.2, fill_mode='nearest'))
    model.add(layers.RandomFlip())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='valid', name='conv1'))
    model.add(layers.MaxPooling2D((2, 2), padding='valid', name='pool1'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv2'))
    model.add(layers.MaxPooling2D((2, 2), padding='valid', name='pool2'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv3'))
    model.add(layers.MaxPooling2D((2, 2), padding='valid', name='pool3'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='valid', name='conv4'))
    model.add(layers.MaxPooling2D((2, 2), padding='valid', name='pool4'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(512, activation='relu', name='fc1'))
    model.add(layers.Dense(1, activation='sigmoid', name='fc2'))

    return model

def train_model(train_dir, val_dir):
    model = build_model()
    model.summary()
    model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


    # pre-process images
    # train_datagen = ImageDataGenerator(rescale=1./255)
    # test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator: tf.data.Dataset = image_dataset_from_directory(
        train_dir,
        image_size=(150, 150),
        batch_size=32,
        label_mode='binary'
    )

    train_generator = train_generator.map(lambda x, y : (x / 255, y))

    val_generator: tf.data.Dataset = image_dataset_from_directory(
        val_dir,
        image_size=(150, 150),
        batch_size=32,
        label_mode='binary'
    )

    val_generator = val_generator.map(lambda x, y : (x / 255, y))

    history = model.fit(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=val_generator,
        validation_steps=50
    )

    model.save('ex_p180_1.keras')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax[0].plot(epochs, acc, 'b', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')

    ax[0].legend(loc='best')

    ax[1].plot(epochs, loss, 'b', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')

    ax[1].legend(loc='best')

    plt.show()
    plt.close()




if __name__ == '__main__':
    # copy_sets()

    path_train = '/Users/nicolasjulemont/Documents/DATA/chollet/dogs-vs-cats/exo/train'
    path_val = '/Users/nicolasjulemont/Documents/DATA/chollet/dogs-vs-cats/exo/val'

    train_model(path_train, path_val)

