from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pathlib
import random
import os
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
WIDTH = 192
HEIGHT = 192


def load_and_preprocess_from_path_label(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [WIDTH, HEIGHT])
    image /= 255.0  # normalize to [0,1] range
    return image, label


def get_data():
    data_root_orig = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        fname='flower_photos', untar=True)
    data_root = pathlib.Path(data_root_orig)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    image_label_ds = ds.map(load_and_preprocess_from_path_label)
    ds = image_label_ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds, len(all_image_paths), len(label_names)


def cnn():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(test_acc)


def vgg():
    ds, images, classes = get_data()
    vgg = tf.keras.applications.vgg16.VGG16(include_top=True, weights=None, input_shape=(WIDTH, HEIGHT, 3),
                                            classes=classes)
    vgg.compile(optimizer=tf.keras.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"])
    vgg.summary()
    steps_per_epoch = tf.math.ceil(images / BATCH_SIZE).numpy()
    checkpoint_path = "training_vgg/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, period=5)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if os.path.exists(checkpoint_dir):
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        vgg.load_weights(latest)
    else:
        vgg.save_weights(checkpoint_path.format(epoch=0))
    vgg.fit(ds, epochs=10, steps_per_epoch=steps_per_epoch, callbacks=[cp_callback])


def inception():
    ds, images, classes = get_data()
    inception = tf.keras.applications.inception_v3.InceptionV3(include_top=True, weights=None,
                                                               input_shape=(WIDTH, HEIGHT, 3), classes=classes)
    inception.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='sparse_categorical_crossentropy',
                      metrics=["accuracy"])
    inception.summary()
    steps_per_epoch = tf.math.ceil(images / BATCH_SIZE).numpy()
    checkpoint_path = "training_inception/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, period=5)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if os.path.exists(checkpoint_dir):
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        inception.load_weights(latest)
    else:
        inception.save_weights(checkpoint_path.format(epoch=0))
    inception.fit(ds, epochs=10, steps_per_epoch=steps_per_epoch, callbacks=[cp_callback])


if __name__ == "__main__":
    # load_data()
    # cnn()
    inception()
    vgg()
