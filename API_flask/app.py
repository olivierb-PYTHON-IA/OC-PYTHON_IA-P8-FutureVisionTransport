import numpy as np
import os
import json

from flask import Flask
from flask import render_template

from azureml.core import Workspace
from azureml.core.datastore import Datastore
from azureml.core.dataset import Dataset
from azureml.data.dataset_type_definitions import FileType
from azureml.core.model import Model
from azureml.core.authentication import ServicePrincipalAuthentication

import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import save_img
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.models import load_model

import albumentations as A

import segmentation_models as sm

sm.set_framework('tf.keras')
sm.framework()

BACKBONE = 'efficientnetb3'
preprocess_input = sm.get_preprocessing(BACKBONE)


app = Flask(__name__)


authenticated = False


#Fonction de preprocessing
def get_preprocessing(preprocessing_fn):   
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


# Fonction permettant l'authentification auprès de Azure et la récupération du Workspace
# (Authentification par ServicePrincipalAuthentication de Azure)
def authentication():
    global ws
    tenant_id = '894ad120-c276-4dfa-b218-d82b3fece6a7'
    application_id = '21cf902f-1dc0-459d-b352-b7490946f6c6'
    svc_pr_password = os.environ['CITYSCAPE_SPA_PASSWORD']
    svc_pr = ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=application_id,
            service_principal_password=svc_pr_password)

    subscription_id = 'dc0050bb-8e50-4b60-8aac-034371ba1a2a'
    resource_group = 'OC-IA-P8-GPU'
    workspace_name = 'WS-IA-P8-GPU'
    ws = Workspace(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            auth=svc_pr)

    
# Route affichant la page d'accueil    
@app.route('/')
def index():
    return render_template("index.html")    
    

# Route affichant la liste des Ids des images disponibles pour la ségmentation      
@app.route('/list')
def list():
    
    #Etapes :
    
    #1 - Authentification auprès de Azure si ce n'est pas déjà le cas
    global authenticated
    
    if (authenticated == False):
        authentication()
        authenticated = True

    #2 - Récupération de la liste des images disponibles pour ségmentation
    datastore_prepared_name = "datastore_cityscape_prepared"
    datastore_prepared = Datastore.get(ws, datastore_prepared_name)
    
    path_prepared_data_images = '/prepared_data/images/train'
    dataset = Dataset.File.from_files(path=[(datastore_prepared, ('/'))])
    
    dataset_images = Dataset.File.from_files(path=[(datastore_prepared, (path_prepared_data_images))])
    list_images = dataset_images.to_path()
    list_images_ids = [s.strip('/') for s in list_images]
    list_images_ids = [s.strip('.png') for s in list_images_ids]
    list_images_ids = [int(s) for s in list_images_ids]
    list_images_ids = sorted(list_images_ids) 
    
    return render_template("list.html", list_images_ids = list_images_ids)


# Route affichant le résultat : 
# - prend en paramètre l'id de l'image à segmenter
# - retourne l'image avec les segments identifiés par le modèle et l'image avec les segments identifiés annotés dans le jeu de données
@app.route('/request/<id_img>')
def request(id_img):
    
    #Etapes :
    
    #1 - Authentification auprès de Azure si ce n'est pas déjà le cas
    global authenticated
   
    if (authenticated == False):
        authentication()
        authenticated = True
   
    
    #2 - Chargement du meilleur modèle préalablement enregistré dans Azure
    model_name = 'cityscape-best-model.h5'
    existing_models = os.listdir('./models')
    
    #si le modèle a déjà été chargé on le récupère
    if model_name in existing_models:
        custom_objects = {'iou_score': sm.metrics.iou_score, 'dice_loss': sm.losses.dice_loss}
        model_path = os.path.join('./models', model_name)
        model = load_model(model_path, custom_objects=custom_objects)
    #si le modèle n'a pas déjà été chargé, on le charge
    else:
        custom_objects = {'iou_score': sm.metrics.iou_score, 'dice_loss': sm.losses.dice_loss}
        model_obj = Model(ws, model_name)
        model_path = model_obj.download(target_dir='./models', exist_ok = True)
        model = load_model(model_path, custom_objects=custom_objects)
    
    
    #3 - Récupération l'id de l'image à prédire fournie par l'utilisateur
    id_img_to_predict = id_img
    
    
    #4 - Récupération de l'image correspondant à l'id et le masque à prédire
    datastore_prepared_name = "datastore_cityscape_prepared"
    datastore_prepared = Datastore.get(ws, datastore_prepared_name)
    
    path_prepared_data_images = '/prepared_data/images/train'
    path_prepared_data_masks = '/prepared_data/masks/train'
    dataset = Dataset.File.from_files(path=[(datastore_prepared, ('/'))])
    
    #image
    dataset_images = Dataset.File.from_files(path=[(datastore_prepared, (path_prepared_data_images))])
    list_images = dataset_images.to_path()
    list_images_names = [s.strip('/') for s in list_images]
    list_images_names = [s.strip('.png') for s in list_images_names]
    
    idx_image_to_predict = list_images_names.index(id_img_to_predict)
    
    image_to_predict_path_dataset = path_prepared_data_images + dataset_images.to_path()[idx_image_to_predict]
    dataset_image_to_predict = Dataset.File.from_files(path=[(datastore_prepared, (image_to_predict_path_dataset))])
    image_to_predict_path_local = dataset_image_to_predict.download(target_path='./static', overwrite=True)
    image_to_predict = load_img(image_to_predict_path_local[0], target_size=(256,512))
    image_to_predict_array = img_to_array(image_to_predict)
    
    #masque
    dataset_masks = Dataset.File.from_files(path=[(datastore_prepared, (path_prepared_data_masks))])
    list_masks = dataset_masks.to_path()
    list_masks_names = [s.strip('/') for s in list_masks]
    list_masks_names = [s.strip('.png') for s in list_masks_names]
    
    mask_to_predict_path_dataset = path_prepared_data_masks + dataset_masks.to_path()[idx_image_to_predict]
    dataset_mask_to_predict = Dataset.File.from_files(path=[(datastore_prepared, (mask_to_predict_path_dataset))])
    mask_to_predict_path_local = dataset_mask_to_predict.download(target_path='./static', overwrite=True)
    mask_to_predict = load_img(mask_to_predict_path_local[0], target_size=(256,512))
    mask_to_predict_array = img_to_array(mask_to_predict)
    
    
    #5 - Preprocessing de l'image
    preprocessed_image = get_preprocessing(preprocess_input)(image=image_to_predict_array)
    image_to_predict_array_pp = preprocessed_image['image']
    image_to_predict_array_pp = np.expand_dims(image_to_predict_array_pp, axis=0)
    
    
    #6 - Prédiction du masque correspondant à l'image
    mask_predicted_array = model.predict(image_to_predict_array_pp)[0]
    mask_predicted_array = np.argmax(mask_predicted_array, axis=-1)
    mask_predicted_array = np.expand_dims(mask_predicted_array, axis=-1)
    
    #Colorisation
    color_map = {
         0: [0, 0, 0],
         1: [153, 153, 0],
         2: [255, 204, 204],
         3: [255, 0, 127],
         4: [0, 255, 0],
         5: [0, 204, 204],
         6: [255, 0, 0],
         7: [0, 0, 255]
    }
    mask_to_predict_array_colors = np.zeros([mask_to_predict_array.shape[0], mask_to_predict_array.shape[1], 3], dtype="float32")
    mask_predicted_array_colors = np.zeros([mask_to_predict_array.shape[0], mask_to_predict_array.shape[1], 3], dtype="float32")
    for i in range(mask_to_predict_array.shape[0]):
        for j in range(mask_to_predict_array.shape[1]):
            mask_to_predict_array_colors[i,j,:] = color_map[int(mask_to_predict_array[i,j][0])]
            mask_predicted_array_colors[i,j,:] = color_map[int(mask_predicted_array[i,j][0])]

    
    #7 - Superposition images + masques
    added_image_mask_to_predict_array = 0.5 * img_to_array(image_to_predict) + 0.5 * np.squeeze(mask_to_predict_array_colors)
    added_image_mask_predicted_array = 0.5 * img_to_array(image_to_predict) + 0.5 * mask_predicted_array_colors
  
    
    #8 - Sérialisation
    image_mask_to_predict_name = 'image_mask_to_predict_'+id_img_to_predict+'.png'
    image_mask_to_predict_path = './static/image_mask_to_predict_'+id_img_to_predict+'.png'
    save_img(image_mask_to_predict_path, added_image_mask_to_predict_array)
    
    image_mask_predicted_name = 'image_mask_predicted_'+id_img_to_predict+'.png'
    image_mask_predicted_path = './static/image_mask_predicted_'+id_img_to_predict+'.png'
    save_img(image_mask_predicted_path, added_image_mask_predicted_array)

    img_to_predict_name = 'img_to_predict_'+id_img_to_predict+'.png'
    img_to_predict_path = './static/img_to_predict_'+id_img_to_predict+'.png'
    save_img(img_to_predict_path, np.squeeze(image_to_predict_array))

     
    #9 - Retourne le résultat à l'utilisateur
    return render_template("result.html",
                           id_image = id_img_to_predict,
                           image_mask_to_predict_url = image_mask_to_predict_name,
                           image_mask_predicted_url = image_mask_predicted_name)

                               
if __name__ == "__main__":
    app.run()