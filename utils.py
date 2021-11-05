import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import zipfile
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing


# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).
    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return img / 255.
    else:
        return img


# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False,
                          xticks_rot="horizontal", yticks_rot="horizontal"):
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.
    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.
    Args:
      y_true: Array of truth labels (must be same shape as y_pred).
      y_pred: Array of predicted labels (must be same shape as y_true).
      classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
      figsize: Size of output figure (default=(10, 10)).
      text_size: Size of output figure text (default=15).
      norm: normalize values or not (default=False).
      savefig: save confusion matrix to file (default=False).
      xticks_rot: rotation of the xticks
      yticks_rot : rotation of the yticks

    Returns:
      A labelled confusion matrix plot comparing y_true and y_pred.
    Example usage:
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10)
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]  # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),  # create enough axis slots for each class
           yticks=np.arange(n_classes),
           xticklabels=labels,  # axes will labeled with class names (if they exist) or ints
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    # Rotation of the labels
    plt.xticks(rotation=xticks_rot)
    plt.yticks(rotation=yticks_rot)

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    # Save the figure to the current working directory
    if savefig:
        fig.savefig("confusion_matrix.png")


# Make a function to predict on images and plot them (works with multi-class)
def pred_and_plot(model, filename, class_names):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    # Import the target image and preprocess it
    img = load_and_prep_image(filename)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1:  # check for multi-class
        pred_class = class_names[pred.argmax()]  # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]  # if only one output, round

    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instance to store log files.
    Stores log files with the filepath:
      "dir_name/experiment_name/current_datetime/"
    Args:
      dir_name: target directory to store TensorBoard log files
      experiment_name: name of experiment directory (e.g. efficientnet_model_1)
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


# Plot the validation and training data separately


def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.
    Args:
      history: TensorFlow model History object
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


def compare_histories(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.

    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here)
    """

    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')  # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


# Create function to unzip a zipfile into current working directory
# (since we're going to be downloading and unzipping a few files)


def unzip_data(filename):
    """
    Unzips filename into the current working directory.
    Args:
      filename (str): a filepath to a target zip folder to be unzipped.
    """
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()


# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str): target directory

    Returns:
      A print out of:
        number of subdirectories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# Function to evaluate: accuracy, precision, recall, f1-score


def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.
    Args:
        y_true: true labels in the form of a 1D array
        y_pred: predicted labels in the form of a 1D array
    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results


def print_dir(path: str, indents=0):
    """
    Prints subdirectories of given path with number of items in each directory
    Args:
        path - path of directory to show
        indents -   ---DO NO TOUCH---
    """
    indent = "    "
    for dir_name in os.listdir(path):
        dir_path = os.path.join(path, dir_name)
        if os.path.isdir(dir_path):
            print(f"{indent * indents}{dir_name} - {len(os.listdir(dir_path))} items")
            print_dir(dir_path, indents + 1)


def create_checkpoint_callback(file_path: str, best_only: bool = False,
                               weights_only: bool = False,
                               metric: str = "val_accuracy") -> tf.keras.callbacks.ModelCheckpoint:
    """
    Create and returns ModelCheckpoint callback.

    Args:
        file_path: checkpoints will be saved in given directory
        best_only: If true only best model's metrics will be save
        weights_only: If true only weights will be stored. If false whole model will be saved.
        metric: models with best given metrics will be saved when best_only set to True.
    """

    return tf.keras.callbacks.ModelCheckpoint(file_path,
                                              save_weights_only=weights_only,
                                              save_best_only=best_only,
                                              save_freq="epoch",
                                              verbose=1,
                                              monitor=metric)


def create_data_aug_layer():
    """
    Create and return data augmentation layer
    """

    return Sequential([
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomRotation(0.2),
        preprocessing.RandomZoom(0.2),
        preprocessing.RandomHeight(0.2),
        preprocessing.RandomWidth(0.2)
    ], name="data_augmentation")


def create_model_for_fine_tuning(base_model, n_layers=10):
    """
    Takes a model and set last n-layers as trainable. where n is n_layers param.

    Args:
        base_model - instance of the base model
        n_layers - number of layers to set as trainable
    """

    base_model.trainable = True
    # Freeze all layers except for the last n
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False


def create_data_loaders(train_path, test_path, image_size: tuple = (224, 224)):
    train = tf.keras.preprocessing.image_dataset_from_directory(train_path,
                                                                label_mode="categorical",
                                                                image_size=image_size)

    test = tf.keras.preprocessing.image_dataset_from_directory(test_path,
                                                               label_mode="categorical",
                                                               image_size=image_size,
                                                               shuffle=False)

    return train, test


def unravel_databatch(batch) -> tuple:
    """
    Takes BatchDataset and returns data and its label in a tuple.

    Args:
        batch - TensorFlow's data batch
    """
    data_ = []
    labels = []
    for data, label in batch.unbatch():
        data_.append(data)
        labels.append(label.numpy().argmax())

    return data_, labels


def report_into_df(class_rep_dict, labels) -> pd.DataFrame:
    """
    Takes classification report dict and turn it to DataFrame.
    Changes numerical indexes to labels. If you don't want to do that,
    just pass range(0, n) as label.
    """
    new_dict = dict.fromkeys(range(0, len(labels)), "")
    for i in range(0, len(labels)):
        new_dict[str(i)] = labels[i]
    report = pd.DataFrame(class_rep_dict).rename(new_dict, axis=1)
    return report.T


def plot_pred_img(model, filepath, unique_labels, true_label, scale=False):
    """
    Plots image with predicted and true label, title of image is green
    if prediction was true.
    :param model: model to make prediction on
    :param filepath: filepath to image
    :param unique_labels: list of unique labels
    :param true_label: True label of predicted image
    :param scale: Image will be normalized is set to True
    :return: None
    """

    img = load_and_prep_image(filepath, scale=scale)

    img_expand = tf.expand_dims(img, axis=0)

    pred = model.predict(img_expand)

    img = tf.cast(img, dtype=tf.int32)

    # Get the predicted class
    if len(pred[0]) > 1:  # check for multi-class
        pred_class = unique_labels[pred.argmax()]  # if more than one output, take the max
    else:
        pred_class = unique_labels[int(tf.round(pred)[0][0])]  # if only one output, round

    if pred_class == true_label:
        color = "green"
    else:
        color = "red"

    # plot image & remove ticks
    plt.imshow(img)
    plt.axis("off")

    # Change plot title to be predicted, probability of prediction and truth label
    plt.title("Pred: {}   Prob {:2.0f}%   True: {}".format(pred_class,
                                                           np.max(pred) * 100,
                                                           true_label),
              color=color)
