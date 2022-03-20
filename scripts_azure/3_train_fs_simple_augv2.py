import numpy as np
import os
import math
import argparse
import matplotlib.pyplot as plt

from azureml.core.run import Run

import tensorflow as tf

import azureml

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

import albumentations as A

import segmentation_models as sm


# Semgentation-models configuration
sm.set_framework('tf.keras')
sm.framework()


# Paramètres
nb_epochs = 20
batch_size = 32
img_size = (128, 256)
num_cats = 8


# Données
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, dest='data_path', help='data folder mounting point')

args = parser.parse_args()

image_dataset_train_dir = os.path.join(args.data_path, 'images/train/')
mask_dataset_train_dir = os.path.join(args.data_path, 'masks/train/')
image_dataset_val_dir = os.path.join(args.data_path, 'images/val/')
mask_dataset_val_dir = os.path.join(args.data_path, 'masks/val/')


# Augmentation des données
def img_aug():
    transform = A.Compose([
           A.ElasticTransform(p=0.5),
           A.HorizontalFlip(p=0.5),
           A.GridDistortion(p=0.5),
           A.RandomGamma(p=0.5),
           A.Emboss(p=0.5),
           A.Blur(p=0.5, blur_limit = 7)
    ])
    return transform


# Génération des données
class DataGeneratorKeras(Sequence):

    def __init__(self, batch_size, img_size, input_img_dir, target_img_dir, augmentation_sequence=None):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_dir = input_img_dir
        self.input_img_paths = os.listdir(input_img_dir)
        self.target_img_dir = target_img_dir
        self.target_img_paths = os.listdir(target_img_dir)
        self.augmentation_sequence = augmentation_sequence

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
                
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (8,), dtype="float32")
        
        if self.augmentation_sequence is not None:
            for j in range(len(batch_input_img_paths)):
                #traitement des images
                input_path = batch_input_img_paths[j]
                img = load_img(os.path.join(self.input_img_dir, input_path), target_size=self.img_size)
                img_array = img_to_array(img)

                #traitement des masques
                target_path = batch_target_img_paths[j]
                mask = load_img(os.path.join(self.target_img_dir, target_path), target_size=self.img_size, color_mode="grayscale")
                mask_array = img_to_array(mask)
                
                augmented_data = self.augmentation_sequence(image=img_array, mask=mask_array)

                x[j] = augmented_data['image'] / 255. #normalisation des données
                y[j] = tf.one_hot(np.squeeze(augmented_data['mask']), 8, on_value=1.0, off_value=0.0, axis=-1)
        
        else:
            #traitement des images
            for j, input_path in enumerate(batch_input_img_paths):
                img = load_img(os.path.join(self.input_img_dir, input_path), target_size=self.img_size)
                img_array = img_to_array(img)
                x[j] = img_array / 255. #normalisation des données
        
            #traitement des masques
            for j, target_path in enumerate(batch_target_img_paths):
                mask = load_img(os.path.join(self.target_img_dir, target_path), target_size=self.img_size, color_mode="grayscale")
                mask_array = img_to_array(mask)
                y[j] = tf.one_hot(np.squeeze(mask_array), 8, on_value=1.0, off_value=0.0, axis=-1)
        
        return x, y

train_gen = DataGeneratorKeras(batch_size, img_size, image_dataset_train_dir, mask_dataset_train_dir, augmentation_sequence=img_aug())
val_gen = DataGeneratorKeras(batch_size, img_size, image_dataset_val_dir, mask_dataset_val_dir, augmentation_sequence=None)


# Modèle
def get_simple_model(img_size, num_classes):
    inputs = Input(shape=img_size + (3,))

    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(inputs)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)

    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same", strides=2)(x)

    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    model = Model(inputs, outputs)
    return model

model = get_simple_model(img_size=img_size, num_classes=num_cats)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[sm.metrics.iou_score])

nb_params = model.count_params()


# Run Azure
run = Run.get_context()


# Logs
class LogRunMetrics(Callback):
    def on_epoch_end(self, epoch, log):
        run.log('Loss', log['loss'])
        run.log('Validation Loss', log['val_loss'])
        run.log('IoU Score', log['iou_score'])
        run.log('Validation IoU Score', log['val_iou_score'])
        run.log('Nombre de paramètres', nb_params)


# Callbacks
callbacks=[
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('cityscape-fs-simple-aug.h5', save_best_only=True, verbose=1),
    LogRunMetrics()
]

        
# Entrainement        
history = model.fit(train_gen, epochs=nb_epochs, validation_data=val_gen, callbacks=callbacks)


# Sauvegarde du modèle
model.save('./outputs/cityscape-fs-simple-aug.h5')
run.upload_file('./outputs/cityscape-fs-simple-aug.h5', './outputs/cityscape-fs-simple-aug.h5')


# Enregistrement du modèle
run.register_model(model_name='cityscape-fs-simple-aug.h5', model_path='./outputs/cityscape-fs-simple-aug.h5')


# Graphiques
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Validation'])
plt.title("Evolution de la Loss en fonction du nombre d'epochs")
plt.show()
run.log_image('Loss en fonction du nombre epochs', plot=plt)
plt.figure()
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.legend(['Train', 'Validation'])
plt.title("Evolution du Score IoU en fonction du nombre d'epochs")
plt.show()
run.log_image('IoU en fonction du nombre epochs', plot=plt)