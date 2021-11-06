
from keras import applications
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow import keras
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
# la position des fichiers 
train_dir = 'train' 
validation_dir = 'valid'
test_dir= 'test'  

IMG_h, IMG_w = 48, 48 # la taille de images qu'on veut

# charger les images en les ajustant entre 0 et 1 et les redimmensionner en taille 48x48 en 12 lot avec label associé
train_datagen = ImageDataGenerator(rescale=1. / 255)  
valid_datagen = ImageDataGenerator(rescale=1. / 255)  

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(48, 48), batch_size=12)

validation_generator = valid_datagen.flow_from_directory(validation_dir, target_size=(48, 48), batch_size=12)

# charger un réseau VGG16 tronqué s’arrêtant juste après les dernières couches de convolution avec une entrée de dimension (48,48) en 3 couches
base_model = VGG16( include_top=False , weights= 'imagenet',input_shape=(IMG_w, IMG_h, 3)) 

top_model = Sequential()  
top_model.add(Flatten(input_shape=base_model.output_shape[1:])) #ajouter une couche d'aplaissement des données entrées
top_model.add(Dense(512, activation='relu')) # ajouter une couche dense de 512 neurones avec avtivation RELU
top_model.add(Dropout(0.3)) # couche dropout de 30%
top_model.add(Dense(2, activation='softmax')) # ajouter une autre couche dense de 2 neurone avec activation softmax
model = Model(inputs=base_model.input,outputs=top_model(base_model.output))
model.summary() #afficher le modèle complèt
top_model.summary() #afficher notre propre modèle qui se situe dans le top du réseau VGG16
keras.utils.plot_model(top_model, show_shapes=True)
# compiler avec l'optimiseur SGD pour calculer l'entropie croisée et la précision
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(learning_rate=0.0001, momentum=0.9),metrics=['accuracy'])

# entrainer le modèle compilé sur 15 époques
hist = model.fit(train_generator,validation_data=validation_generator,epochs=2)

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(48, 48), batch_size=12)

model.save('dechet.h5') # enregistrer le modème dans le fichier dechet.h5
accu = model.evaluate_generator(test_generator,2569) # evaluer le modèle en utilisant les images qui sert à tester

#affichage du taux d'erreur et précision
print("Taux d'erreur est",1-accu[1])
print("Taux de precision est",accu[1])

# tracer la courbe de l'évolution de l'erreur et la précision entrainé et validé en fonction de nombre d'époque 
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epoque = range(len(acc))

plt.plot(epoque, acc , label='Training acc')
plt.plot(epoque, val_acc, label='Validation acc')
plt.title("Precision d'entrainement et validation")

plt.figure()
plt.plot(epoque, loss, label='Training loss')
plt.plot(epoque, val_loss, label='Validation loss')
plt.title("Perte d'entrainement et validation")
plt.legend()
plt.show()
