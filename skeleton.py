#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template file for Assignment 1 -- Happy, Sad, Surprised, or Mad?

Last updated: 2026-03-23
"""
import os
import time
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

SEED = 2026
class_map = {"Happy": 0,
             "Sad": 1,
             "Surprised": 2,
             "Mad": 3}
cache = True

# You will have to change these two
n_epochs = 2
batch_size = 32

# Directory where the data are stored
data_dir = "/import/course/5dv236/vt26/AffectNet/"
print(f"Loading data from {data_dir}")

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Check GPU availability (it will be very slow without a GPU...)
gpus = tf.config.experimental.list_physical_devices("GPU")
print()
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("GPU(s) available. Training will be lightning fast!")
else:
    print("No GPU(s) available. Training will be very slow ...")

# For pretty plots...
PLOT_SETTINGS = {"text.usetex": True,
                 "font.family": "serif",
                 "figure.figsize": (8.0, 6.0),
                 "font.size": 16,
                 "axes.labelsize": 16,
                 "legend.fontsize": 14,
                 "xtick.labelsize": 14,
                 "ytick.labelsize": 14,
                 "axes.titlesize": 24,
                 "lines.linewidth": 2.0,
                 }
plt.rcParams.update(PLOT_SETTINGS)


class DataLoader(keras.utils.Sequence):
    """A simple data loader for Assignment 1.

    Note: We advice against making changes to the data loader!
    """

    def __init__(self,
                 data_path,
                 class_map,
                 batch_size=32,
                 cache=True,
                 random_state=None,
                 dtype=np.uint8,
                 ):

        self.data_path = data_path
        self.class_map = class_map
        self.batch_size = max(1, int(batch_size))
        self.cache = bool(cache)
        if random_state is None:
            self.random_state = np.random
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)
        self.dtype = dtype

        if self.data_path is None:
            raise ValueError('The data path is not defined.')

        if not os.path.isdir(self.data_path):
            raise ValueError('The data path is incorrectly defined.')

        if not isinstance(self.class_map, dict):
            raise ValueError('The folder map is not a dictionary.')

        # Read the files in all subfolders
        self._file_idx = 0
        self._images = []
        self._labels = []
        for folder in self.class_map:
            path = os.path.join(self.data_path, folder)
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                self._images.append(file_path)
                self._labels.append(folder)

        self._image_cache = dict()

        self.on_epoch_end()

    def __len__(self):
        """Get the number of mini-batches per epoch."""
        return int(len(self._images) / self.batch_size)

    def __getitem__(self, index):
        """Get one batch of data."""
        # Generate indices of the batch
        indices = self._indices[
            index * self.batch_size:(index + 1) * self.batch_size]

        # Find the next set of file indices
        minibatch_files = [self._images[k] for k in indices]
        minibatch_labels = [self.class_map[self._labels[k]] for k in indices]

        # Load up the corresponding minibatch
        minibatch = self.__load_minibatch(minibatch_files)

        return minibatch, minibatch_labels

    def on_epoch_end(self):
        """Update indices after each epoch."""
        self._indices = np.arange(len(self._images))
        self.random_state.shuffle(self._indices)

    def __load_image(self, file):
        """Load a single image from file."""
        im = Image.open(file)
        if im.mode != "RGB":
            im = im.convert("RGB")
        im = np.asarray(im, dtype=self.dtype)

        return im

    def __load_minibatch(self, minibatch_files):
        """Load the next minibatch of samples."""

        try:
            assert self.batch_size == len(minibatch_files)
        except AssertionError:
            print(self.batch_size)
            print(len(minibatch_files))

        minibatch = [None] * self.batch_size
        for i, file in enumerate(minibatch_files):
            if self.cache:
                if file in self._image_cache:
                    im = self._image_cache[file]
                else:
                    im = self.__load_image(file)
                    self._image_cache[file] = im
            else:
                im = self.__load_image(file)

            minibatch[i] = im

        return minibatch

# NOTE: We will use a clean version of the data loader, so you should not rely
#       on any changes to it. Simply, don't change the data loader!


# Create the data loaders
train_ds = DataLoader(os.path.join(data_dir, "train/"),
                      class_map=class_map,
                      batch_size=batch_size,
                      cache=cache,
                      )
val_ds = DataLoader(os.path.join(data_dir, "val/"),
                    class_map=class_map,
                    batch_size=batch_size,
                    cache=cache,
                    )
# Do not use the test data in any way until the very end, when you fill in the
# values in your report just before handing it in!
test_ds = DataLoader(os.path.join(data_dir, "test/"),
                     class_map=class_map,
                     batch_size=batch_size,
                     cache=cache,
                     )

# A quick summary of the data:
print(f"Number of training mini-batches: {len(train_ds)}")
print(f"Number of training images      : {len(train_ds._indices)}")
print(f"Number of validation images    : {len(val_ds._indices)}")
print(f"Number of test images          : {len(test_ds._indices)}")

train_ds = tf.data.Dataset.from_tensors(train_ds)
val_ds = tf.data.Dataset.from_tensors(val_ds)
test_ds = tf.data.Dataset.from_tensors(test_ds)
# Plot a few of the training images
fig = plt.figure(figsize=(12, 5))
fig.subplots_adjust(top=0.995,
                    bottom=0.005,
                    left=0.025,
                    right=0.995,
                    wspace=0.05,
                    hspace=0.0125)
M, N = 4, 10
axs = []
for m in range(M):
    axs.append([])
    for n in range(N):
        ax = plt.subplot2grid((M, N), (m, n), rowspan=1, colspan=1)
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        axs[m].append(ax)

imgs = []
lbls = []
for i in range(3):
    imgs.extend(train_ds[i][0])
    lbls.extend(train_ds[i][1])
indices = [0] * 4
for i in range(len(imgs)):
    y = lbls[i]
    if indices[y] < N:
        axs[y][indices[y]].imshow(imgs[i].astype(int))  # int [0,...,255]
        indices[y] += 1
for m in range(M):
    label = list(train_ds.class_map.keys())[
        list(train_ds.class_map.values()).index(m)]
    axs[m][0].set_ylabel(f"{label}")


model = keras.models.Sequential([keras.layers.Conv2D(32, (3,3), activation='relu'),
                                keras.layers.Flatten(),
                                keras.layers.Dense(4, activation='softmax')
                                 ])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
# Define and compile your model here. Don't forget to use accuracy as a metric.


time_ = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2   # you can increase/decrease based on speed
)
training_loss = []  # Save training loss values here
validation_loss = []  # Save validation loss values here
for epoch in range(n_epochs):

    # Implement the training iterations here.
    # You may need to use model.train_on_batch(...) if you use the data loader
    # above.

    # If you want to do some preprocessing, this may also be where to do it.

    # You can evaluate the model using model.evaluate(...) to measure its
    # performance.

    # Call manually at the end of the epoch to reshuffle the training data.
    train_ds.on_epoch_end()

    #print(f"[{epoch:02d}/{n_epochs}] {training_loss[-1]} "
    #      f"{validation_loss[-1]}")

time_passed = time.time() - time_
print(f"Training done in {int(time_passed // 60):.0f} minutes "
      f"{int(time_passed % 60):.0f} seconds")
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Note: We should never evaluate our model on the test data before we have
#       chosen a _final model_. This means you should not run the below code
#       until you are done and ready to hand in the assignment. It will be
#       tempting, but I repeat: Do not evaluate your model on the test data
#       until you are completely done and have a final model. It is this one
#       _final model_ that you evaluate with the test data. If you do anything
#       to your model and evaluate it again, there is no value in this
#       evaluation any more.
if False:
    test_loss = []
    for idx, (X, y) in enumerate(test_ds):
        pass
        # Your code goes here.
        # If you did preprocessing, don't forget to apply it here as well.

    print(f"Final test results {np.mean(test_loss, axis=0)}")


# Plot the training and validation curves

# Save the model to file using model.save(...)
