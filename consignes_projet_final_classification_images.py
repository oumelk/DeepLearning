#!/usr/bin/env python
# coding: utf-8

# #  Classification avec des images réelles: dogs vs cats
# 
# 
# Le dataset complet provient d'une compétition Kaggle: https://www.kaggle.com/c/dogs-vs-cats
# 
# Le dataset en question contient 2 folders: dogs - cats
# 
# Soit un total de 2000 images.
# 
# L'objectif est de capitaliser sur les notions du cours pour développer un réseau CNN qui arrive à prédire avec la meilleure performance possible les images de chiens et de chats.
# 
# Ce notebook va permettre de structurer l'approche et la construction du modèle.

# ## 1- Importer des librairies pertinentes:

# In[2]:


import numpy as np
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout





# ## 2- Localiser le path où se trouvent toutes les images

# In[39]:


# Define a variable as the directory path
my_data_dir = "C:/Users/ASUS/Downloads/final_project-M7_Deep-Learning/final_project/nom_de_votre_env/data_cats_and_dogs"


# ### 2.1 - Vérifier que la commande ci-dessous retourne ['train', 'validation']

# In[40]:


os.listdir(my_data_dir)



# ### 2.2 - Définir les variables train_path et val_path:

# In[41]:


# train and test paths (\ for windows, / for mac)
train = "train"
validation= "validation"
train_path = os.path.join(my_data_dir, train)
test_path = os.path.join(my_data_dir, validation)


# ### 2.3 - Print le nombre d'images pour chaque class (cats & dogs) dans le dossier train et validation:

# In[42]:


# Vérifier le nombre d'images de chaque classe pour le train_path et val_path
# Fonction pour compter le nombre d'images dans un dossier
def count_images_in_directory(directory):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):  
            count += 1
    return count

# Compter le nombre d'images dans les dossiers "train" et "validation" pour les chiens et les chats
train_dogs_count = count_images_in_directory(os.path.join(train_path, "dogs"))
train_cats_count = count_images_in_directory(os.path.join(train_path, "cats"))
val_dogs_count = count_images_in_directory(os.path.join(test_path, "dogs"))
val_cats_count = count_images_in_directory(os.path.join(test_path, "cats"))

# Afficher les résultats
print("Nombre d'images de chiens dans le dossier 'train':", train_dogs_count)
print("Nombre d'images de chats dans le dossier 'train':", train_cats_count)
print("Nombre d'images de chiens dans le dossier 'validation':", val_dogs_count)
print("Nombre d'images de chats dans le dossier 'validation':", val_cats_count)


# Vérifier que vous avez bien 2000 images au total !
total_images = train_dogs_count + train_cats_count + val_dogs_count + val_cats_count

if total_images == 2000:
    print("Vous avez bien 2000 images au total.")
else:
    print("Le nombre total d'images n'est pas égal à 2000.")



# ## 3) Analyse d'exemples d'images dogs and cats

# ### 3.1 - Choisir au hasard une image de dog dans le train_path

# In[43]:


# Définir le chemin du dossier "train" pour les chiens
train_dogs_path = os.path.join(train_path, "dogs")

# Liste des noms de fichiers d'images de chiens
dog_image_files = os.listdir(train_dogs_path)

# Choisir au hasard une image de chien
random_dog_image = random.choice(dog_image_files)

# Chemin complet de l'image sélectionnée
random_dog_image_path = os.path.join(train_dogs_path, random_dog_image)

# Afficher le nom de l'image sélectionnée
print("Image de chien choisie au hasard:", random_dog_image)


# ### 3.2 - Transformer cette image en numpy array

# In[44]:


# Charger l'image à partir du chemin
image = Image.open(random_dog_image_path)

# Convertir l'image en un tableau NumPy
image_array = np.array(image)

# Assurez-vous que l'image est en couleur (3 canaux) et non en noir et blanc (1 canal)
if len(image_array.shape) == 2:
    # L'image est en noir et blanc, convertissez-la en couleur en dupliquant les canaux
    image_array = np.stack((image_array,) * 3, axis=-1)

# Afficher la forme du tableau NumPy
print("Forme du tableau NumPy de l'image:", image_array.shape)


# ### 3.3 - Vérifier les dimensions de cette image

# In[45]:


# Vérifier les dimensions de l'image
hauteur, largeur, canaux = image_array.shape

print("Largeur de l'image :", largeur)
print("Hauteur de l'image :", hauteur)
print("Nombre de canaux (RVB) de l'image :", canaux)


# ### 3.4 -Plot cette image via 'imshow'

# In[46]:


# Afficher l'image
plt.imshow(image_array)

# Titre de l'image
plt.title("Image de chien")

# Supprimer les axes
plt.axis('off')

# Afficher l'image
plt.show()


# ### 3.5 - Refaire le même travail avec l'image d'un cat depuis le dossier train 

# In[47]:


# Définir le chemin du dossier "train" pour les chats
train_cats_path = os.path.join(train_path, "cats")

# Liste des noms de fichiers d'images de chats
cat_image_files = os.listdir(train_cats_path)

# Choisir au hasard une image de chat
random_cat_image = random.choice(cat_image_files)

# Chemin complet de l'image sélectionnée
random_cat_image_path = os.path.join(train_cats_path, random_cat_image)

# Charger l'image à partir du chemin
cat_image = Image.open(random_cat_image_path)

# Convertir l'image en un tableau NumPy
cat_image_array = np.array(cat_image)

# Vérifier les dimensions de l'image
hauteur, largeur, canaux = cat_image_array.shape

print("Largeur de l'image de chat :", largeur)
print("Hauteur de l'image de chat :", hauteur)
print("Nombre de canaux (RVB) de l'image de chat :", canaux)

# Afficher l'image de chat
plt.imshow(cat_image_array)
plt.title("Image de chat")
plt.axis('off')
plt.show()


# ## 4) Créer un ImageDataGenerator qui effectue un retraitement "pertinent" de ces images:

# In[48]:


# Créer un générateur d'images avec des prétraitements
datagen = ImageDataGenerator(
    rescale=1./255,        # Mise à l'échelle des pixels (conversion en valeurs entre 0 et 1)
    rotation_range=20,     # Augmentation : rotation aléatoire de 20 degrés
    width_shift_range=0.2, # Augmentation : déplacement horizontal aléatoire de 20% de la largeur de l'image
    height_shift_range=0.2,# Augmentation : déplacement vertical aléatoire de 20% de la hauteur de l'image
    shear_range=0.2,       # Augmentation : cisaillement aléatoire
    zoom_range=0.2,        # Augmentation : zoom aléatoire
    horizontal_flip=True,  # Augmentation : retournement horizontal aléatoire
    fill_mode='nearest'    # Mode de remplissage pour les pixels créés lors de l'augmentation
)

# Configurer le répertoire source des images
source_directory = train_path  

# Créer un générateur d'images à partir du répertoire source
image_generator = datagen.flow_from_directory(
    source_directory,
    target_size=(150, 150),  # Redimensionner les images à la taille spécifiée
    batch_size=32,           # Taille du lot (batch size)
    class_mode='binary'      # Mode de classification, 'binary' pour deux classes (chien et chat)
)


# In[ ]:





# ## 5) Construire un modèle CNN from scratch pour la classification binaire de ces images:
# 
# 
# **- Utiliser à minima les types layers suivants: Conv2D, MaxPooling2D, Dense.**

# **- Utiliser également la technique du Dropout.**

# **- Prendre un input_shape arbitraire fixe et approprié**

# **- Print le model summary**

# **- Ne pas hésiter à ajouter des techniques ou des méthodes sur les données ou le modèle pour améliorer la performance !**

# **L'objectif est de maximiser l'accuracy sur les données de test**

# In[ ]:


# Définir le modèle CNN
model = Sequential()

# Ajouter une couche de convolution 2D avec activation ReLU
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

# Ajouter une couche de MaxPooling 2D
model.add(MaxPooling2D((2, 2)))

# Ajouter une deuxième couche de convolution 2D avec activation ReLU
model.add(Conv2D(64, (3, 3), activation='relu'))

# Ajouter une deuxième couche de MaxPooling 2D
model.add(MaxPooling2D((2, 2)))

# Ajouter une troisième couche de convolution 2D avec activation ReLU
model.add(Conv2D(128, (3, 3), activation='relu'))

# Ajouter une troisième couche de MaxPooling 2D
model.add(MaxPooling2D((2, 2))

# Aplatir les données pour les passer à la couche Dense
model.add(Flatten())

# Ajouter une couche Dense avec activation ReLU
model.add(Dense(512, activation='relu'))

# Ajouter une couche de Dropout pour la régularisation
model.add(Dropout(0.5))

# Couche de sortie avec une seule unité et activation sigmoid (classification binaire)
model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Afficher un résumé du modèle
model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:











# In[ ]:





# ### 5.2 Créer une instance de EarlyStopping

# In[ ]:


# Créer une instance de EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',       # Mesure à surveiller (perte sur l'ensemble de validation)
    patience=5,               # Nombre d'époques sans amélioration avant de s'arrêter
    restore_best_weights=True  # Restaurer les poids du modèle à la meilleure époque
)

# Ajouter l'early stopping lors de l'entraînement du modèle
history = model.fit(
    train_generator,
    epochs=50,                   # Nombre d'époques d'entraînement
    validation_data=validation_generator,
    callbacks=[early_stopping]  # Ajouter l'early stopping ici
)



# ### 5.3 Créer un generator pour le train et validation set: 

# In[ ]:


# Définir les transformations de données pour le jeu d'entraînement (train set)
train_datagen = ImageDataGenerator(
    rescale=1./255,       # Mise à l'échelle des pixels (conversion en valeurs entre 0 et 1)
    rotation_range=40,    # Augmentation : rotation aléatoire de 40 degrés
    width_shift_range=0.2,# Augmentation : déplacement horizontal aléatoire de 20% de la largeur de l'image
    height_shift_range=0.2,# Augmentation : déplacement vertical aléatoire de 20% de la hauteur de l'image
    shear_range=0.2,      # Augmentation : cisaillement aléatoire
    zoom_range=0.2,       # Augmentation : zoom aléatoire
    horizontal_flip=True  # Augmentation : retournement horizontal aléatoire
)

# Définir les transformations de données pour le jeu de validation (validation set)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Créer les générateurs pour le jeu d'entraînement et de validation
batch_size = 32  # Taille du lot (batch size)

# Chemins des répertoires contenant les images
train_dir = 'chemin_vers_le_dossier_train'
validation_dir = 'chemin_vers_le_dossier_validation'

# Créer le générateur du jeu d'entraînement
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Redimensionner les images à la taille spécifiée
    batch_size=batch_size,
    class_mode='binary'       # Mode de classification, 'binary' pour deux classes (chien et chat)
)

# Créer le générateur du jeu de validation
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),  # Redimensionner les images à la taille spécifiée
    batch_size=batch_size,
    class_mode='binary'       # Mode de classification, 'binary' pour deux classes (chien et chat)
)


# In[ ]:





# In[ ]:





# ### 5.3 Entrainer le modèle à partir du train_image_generator et utiliser le EarlyStopping

# In[ ]:


# Créer une instance de EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',       # Mesure à surveiller (perte sur l'ensemble de validation)
    patience=5,               # Nombre d'époques sans amélioration avant de s'arrêter
    restore_best_weights=True  # Restaurer les poids du modèle à la meilleure époque
)

# Entraîner le modèle en utilisant le générateur de données du jeu d'entraînement
history = model.fit(
    train_generator,
    epochs=50,                   # Nombre d'époques d'entraînement
    validation_data=validation_generator,
    callbacks=[early_stopping]  
)


# ## 8) Evaluation du modèle

# ### 8.1 Sauvegarder les losses dans un dataframe

# In[ ]:


# Créer un DataFrame pour stocker les pertes
loss_df = pd.DataFrame(history.history)

# Sauvegarder le DataFrame dans un fichier CSV
loss_df.to_csv('losses.csv', index=False)


# ### 8.2 Plot le training et validation loss 

# In[ ]:


# Charger les pertes à partir du fichier CSV
loss_df = pd.read_csv('losses.csv')

# Extraire les pertes d'entraînement et de validation
training_loss = loss_df['loss']
validation_loss = loss_df['val_loss']

# Créer un graphique pour les pertes
plt.figure(figsize=(10, 6))
plt.plot(training_loss, label='Training Loss', color='blue')
plt.plot(validation_loss, label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# ### 8.3 Calculer les probabilités pour le validation image generator 

# In[ ]:


# Utiliser le modèle pour prédire les probabilités sur l'ensemble de validation
y_pred_proba = model.predict(validation_generator)

# y_pred_proba contient maintenant les probabilités prédites pour chaque image de l'ensemble de validation


# ### 8.4 Transformer ces probabilités en classes en prenant un threshold de 0.5

# In[ ]:


# Seuil (threshold) pour la classification
threshold = 0.5

# Transformer les probabilités en classes
y_pred_classes = (y_pred_proba >= threshold).astype(int)


# ### 8.5 Récupérer le vecteur des true labels à partir du validation image generator

# In[ ]:


# Récupérer le vecteur des vraies étiquettes à partir du générateur de validation
true_labels = validation_generator.labels


# ### 8.6 Afficher le classification report et la matrice de confusion

# In[ ]:


# Calculer la matrice de confusion
confusion = confusion_matrix(true_labels, y_pred_classes)

# Afficher la matrice de confusion sous forme de heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Afficher le classification report
report = classification_report(true_labels, y_pred_classes, target_names=['Cat', 'Dog'])
print(report)


# ### 8.7 KPI final: quel est l'accuracy du model sur les données de test ? Etes-vous satisfaits de la performance de votre modèle ?

# In[ ]:


# Calculer l'exactitude du modèle sur les données de test
accuracy = accuracy_score(true_labels, y_pred_classes)

print(f"L'exactitude du modèle sur les données de test est : {accuracy:.2f}")


# In[ ]:





# # 9) Prédictions sur des cas particuliers

# ### 9.1 Afficher quelques images des données de test où le modèle s'est trompé.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Trouver les indices des échantillons mal classés
misclassified_indices = np.where(y_pred_classes != true_labels)[0]

# Choisir un nombre d'échantillons mal classés à afficher
num_samples_to_display = 5

# Afficher les échantillons mal classés
for i in range(num_samples_to_display):
    index = misclassified_indices[i]
    
    # Récupérer l'image mal classée
    misclassified_image = validation_generator[index][0][0]

    # Récupérer la vraie étiquette et la prédiction
    true_label = true_labels[index]
    predicted_label = y_pred_classes[index][0]

    # Afficher l'image et les étiquettes
    plt.figure(figsize=(6, 6))
    plt.imshow(misclassified_image)
    plt.title(f"True Label: {true_label} Predicted Label: {predicted_label}")
    plt.axis('off')
    plt.show()


# In[ ]:





# In[ ]:





# ### 9.2 Ces images ont-elles des patterns en commun ?

# In[ ]:





# In[ ]:





# # 10) Data augmentation (optionnel)
# 
# Utiliser des techniques de Data augmentation. L'objectif est d'enrichir le training set à partir des images initiales afin d'améliorer la performance du modèle.
# 
# Votre accuracy s'améliore t-elle post votre data augmentation ?
# 
# Vous êtes libre de structurer cette partie comme vous le jugez pertinent.

# In[ ]:





# In[ ]:





# In[ ]:





# # 11) Transfer learning
# 
# Utiliser la technique de transfer learning à partir d'un modèle open source qui vous semble pertinent. Justifier le choix du modèle ? 
# 
# Au final, votre accuracy s'est elle améliorée significativement ?
# 
# Vous êtes libre de structurer cette partie comme vous le jugez pertinent.

# In[ ]:





# In[ ]:





# In[ ]:





# # 12) Déploiement
# 
# À partir du modèle définitif que vous aurez construit et sauvegardé sur Keras, vous devez créer une webapp où l'utilisateur peut se connecter à l'URL, charger une image de chien ou chat et obtenir en retour la prédiction du modèle.
# 
# Stack recommandé: streamlit pour le développement de la webapp & render pour le hosting

# In[ ]:


#URL de git de l'application est : 


# In[ ]:





# In[ ]:





# # Fin du projet!
# 
# Au final, vous devez remettre à l'équipe pédagogique: 
# 
# ### 1 - Ce notebook rempli avec les outputs visibles
# ### 2 - Le lien URL de votre application web
