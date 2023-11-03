import datetime
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile

"""## Callbacks Helper Functions"""


def create_earlystopping_cb(monitor_value, patience=10):
    """
    Create and returns early-stopping callback.

    Args:
      monitor_value: value to monitor during training (like val_loss, val_accuracy).
      patience: specify patience of our model (for how long it can stops improving).

    Returns:
      Early-stopping callback.
    """
    return tf.keras.callbacks.EarlyStopping(monitor=monitor_value,
                                            patience=patience,
                                            restore_best_weights=True)


c1 = create_earlystopping_cb("val_accuracy")


def create_tensorboard_cb(experiment_name, root_dir):
    """
    Create and returns tensorboard callback. In order to use this
    function we have to import: pathlib & datetime modules.

    Args:
      experiment_name: name for our experiment (unique model name etc...).
      root_dir: directory where logs will be saved.

     Returns:
      Tensorboard callback saved in dynamic directory, every time we
      run this function it generates new directory where metrics
      from training are going to be saved.
    """
    tensorboard_dir = Path(root_dir) / experiment_name / \
        datetime.datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")
    return tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

def create_modelcheckpoint_cb(experiment_name, monitor_value, checkpoints_dir):
    """
    Create and returns model-checkpoint callback.
    Logs from this callback are going to be saved
    in checkpoints_dir/experiment_name directory.

    Args:
      experiment_name: name for our experiment (unique model name etc...).
      monitor_value: value to monitor during training.
      root_dir: directory where logs will be saved.

    Returns:
      ModelCheckpoint callback. Logs are saved in
      experiment_name/root_dir directory.
    """
    modelcheckpoint_dir = Path(checkpoints_dir) / experiment_name
    print(modelcheckpoint_dir)
    return tf.keras.callbacks.ModelCheckpoint(filepath=modelcheckpoint_dir,
                                              monitor=monitor_value,
                                              save_best_only=True)



## Function that unzip our file
def unzip_file(zip_directory, result_directory):
    """
    Unzip file.

    Args:
      zip_directory: directory where we saved zip files.
      result_direcotry: direcotry where we want to save our extracted files.
    """
    zip_ref = zipfile.ZipFile(zip_directory)
    zip_ref.extractall(result_directory)
    zip_ref.close()
    
# Visualize first 30 images
def show_images(images, labels, n_rows=4, n_cols=8):
  """
  This function can be used to visualize any amount
  if images in some particular dataset.

  Args:
      images: images from saome dataset in form
      of numpy arrays.
      labels: labels from some dataset

  Returns:
      n_cols*n_rows images in nice looking figure.
  """

  # Get n_rows*n_cols random indexes
  random_indexes = np.random.randint(0, len(images), size=(n_rows*n_cols))
  images_, labels_ = images[random_indexes], labels[random_indexes]

  fig = plt.figure(figsize=(20, 10))
  for idx, image in enumerate(images_):
    fig.add_subplot(n_rows, n_cols, idx+1)
    plt.imshow(image)
    plt.title(class_names[labels_[idx]])
    plt.axis(False)
  plt.show()


def display_loss_accuracy_curves(history):
  """
  Create two curves that compare losses and accuracy scores
  of training and validation data.

  Args:
      history: history object after training some
      neural network.

  Returns:
      2 seperate plots in one row that show losses and accuracy scores.
  """
  epochs = np.arange(len(history.history["loss"]))
  fig = plt.figure(figsize=(16, 5))
  for idx in range(2):
    fig.add_subplot(1, 2, idx+1)
    if idx == 0:
      # Plot accuracy scores
      plt.plot(epochs, history.history["accuracy"], label="Training Accuracy")
      plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
      plt.title("Accuracy Scores")
      plt.legend()
    else:
      # Plot loss scores
      plt.plot(epochs, history.history["loss"], label="Training Loss")
      plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
      plt.title("Losses")
      plt.legend()

# Some learning rates useful algorithms
K = tf.keras.backend

class OneCycleScheduler(tf.keras.callbacks.Callback):
    def __init__(self, iterations, max_lr=1e-3, start_lr=None,
                 last_iterations=None, last_lr=None):
        self.iterations = iterations
        self.max_lr = max_lr
        self.start_lr = start_lr or max_lr / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_lr = last_lr or self.start_lr / 1000
        self.iteration = 0

    def _interpolate(self, iter1, iter2, lr1, lr2):
        return (lr2 - lr1) * (self.iteration - iter1) / (iter2 - iter1) + lr1

    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            lr = self._interpolate(0, self.half_iteration, self.start_lr,
                                   self.max_lr)
        elif self.iteration < 2 * self.half_iteration:
            lr = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                   self.max_lr, self.start_lr)
        else:
            lr = self._interpolate(2 * self.half_iteration, self.iterations,
                                   self.start_lr, self.last_lr)
        self.iteration += 1
        K.set_value(self.model.optimizer.learning_rate, lr)
