import os
import shutil

import numpy as np
from keras import Sequential
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras.src.applications.vgg16 import VGG16

from keras.utils import image_dataset_from_directory
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


def store_output():
    conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
    conv_base.summary()

    path_exo = '/Users/nicolasjulemont/Documents/DATA/chollet/dogs-vs-cats/exo'
    train_dir = os.path.join(path_exo, 'train')
    val_dir = os.path.join(path_exo, 'val')

    train_generator: tf.data.Dataset = image_dataset_from_directory(
        train_dir,
        image_size=(150, 150),
        batch_size=1,
        label_mode='binary',
        pad_to_aspect_ratio=True,
    )

    train_generator = train_generator.map(lambda x, y: (x / 255., y))

    train_generator = train_generator.repeat(5)

    val_generator: tf.data.Dataset = image_dataset_from_directory(
        val_dir,
        image_size=(150, 150),
        batch_size=1,
        label_mode='binary',
        pad_to_aspect_ratio=True,
    )

    val_generator = val_generator.map(lambda x, y: (x / 255., y))

    path_ds_output = os.path.join(path_exo, 'vgg16_output')
    os.makedirs(path_ds_output, exist_ok=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Write the records to a file (train)
    # ------------------------------------------------------------------------------------------------------------------

    ds_filename_train = os.path.join(path_ds_output, 'output_train.tfrecords')

    with tf.io.TFRecordWriter(ds_filename_train) as file_writer:

        for idx, sample in enumerate(train_generator):
            if idx >= 3200:
                break
            output = conv_base.predict(sample[0])
            tensor_output = tf.io.serialize_tensor(tf.convert_to_tensor(output.reshape(4, 4, 512))).numpy()
            y_tensor = tf.io.serialize_tensor(tf.convert_to_tensor(np.array([sample[1]]))).numpy()
            record_bytes = tf.train.Example(features=tf.train.Features(feature={
                "x": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_output])),
                "y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_tensor])),
            })).SerializeToString()
            file_writer.write(record_bytes)
            print(idx)


    # ------------------------------------------------------------------------------------------------------------------
    # Write the records to a file (validation)
    # ------------------------------------------------------------------------------------------------------------------

    ds_filename_val = os.path.join(path_ds_output, 'output_val.tfrecords')
    with tf.io.TFRecordWriter(ds_filename_val) as file_writer:

        for idx, sample in enumerate(val_generator):
            if idx >= 1600:
                break
            output = conv_base.predict(sample[0])
            tensor_output = tf.io.serialize_tensor(tf.convert_to_tensor(output.reshape(4, 4, 512))).numpy()
            y_tensor = tf.io.serialize_tensor(tf.convert_to_tensor(np.array([sample[1]]))).numpy()
            record_bytes = tf.train.Example(features=tf.train.Features(feature={
                "x": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_output])),
                "y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_tensor])),
            })).SerializeToString()
            file_writer.write(record_bytes)
            print(idx)



def train_classifier():
    path_exo = '/Users/nicolasjulemont/Documents/DATA/chollet/dogs-vs-cats/exo/vgg16_output'

    train_tfrecord_path = os.path.join(path_exo, 'output_train.tfrecords')
    val_tfrecord_path = os.path.join(path_exo, 'output_val.tfrecords')

    def _parse_numpy_array_function(example_proto):
        tensor_features_description = {
            'x': tf.io.FixedLenFeature([], tf.string),
            'y': tf.io.FixedLenFeature([], tf.string),
        }

        # Parse the input tf.train.Example proto using the dictionary above.
        sample = tf.io.parse_single_example(example_proto, tensor_features_description)

        x = tf.io.parse_tensor(sample['x'], out_type=tf.float32)
        y = tf.io.parse_tensor(sample['y'], out_type=tf.float32)

        x = tf.reshape(x, (4, 4, 512))
        y = tf.reshape(y, (1,))

        print(x)
        print(y)

        return x, y

    train_dataset = tf.data.TFRecordDataset([train_tfrecord_path]).map(lambda ex: _parse_numpy_array_function(ex), num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = tf.data.TFRecordDataset([val_tfrecord_path]).map(lambda ex: _parse_numpy_array_function(ex), num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset.batch(32)
    val_dataset = val_dataset.batch(32)

    model = Sequential()
    model.add(layers.InputLayer(shape=(4, 4, 512)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=[metrics.binary_accuracy])

    history = model.fit(train_dataset, epochs=40, validation_data=val_dataset, validation_batch_size=64)

    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    ax[0].plot(acc, 'b', label='Training accuracy')
    ax[0].plot(val_acc, 'r', label='Validation accuracy')

    ax[0].legend(loc='best')

    ax[1].plot(loss, 'b', label='Training loss')
    ax[1].plot(val_loss, 'r', label='Validation loss')

    ax[1].legend(loc='best')

    plt.show()
    plt.close()

if __name__ == '__main__':
    # store_output()
    train_classifier()
