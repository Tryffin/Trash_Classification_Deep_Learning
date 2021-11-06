from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

model = load_model("dechet.h5")
model.summary()

# Pre-traiter l'image
file_path='test1.jpg'
img = image.load_img(file_path, target_size=(48, 48)) # Redimensionner a 48x48
img_tensor = image.img_to_array(img)
img_tensor=np.expand_dims(img_tensor,axis=0)
img_tensor/=255.
print("Le taille reel d'image1 est：",img_tensor.shape)

file_path2='test2.jpg'
img2 = image.load_img(file_path2, target_size=(48, 48)) # Redimensionner a 48x48
img_tensor2 = image.img_to_array(img2)
img_tensor2 = np.expand_dims(img_tensor2,axis=0)
img_tensor2/=255.
print("Le taille reel d'image2 est：",img_tensor2.shape)

# def decode_predictions_custom(preds, top=5):
#     CLASS_CUSTOM = ["s","w"] # s est le dechet solide et w est le dechet liquide
#     results = []
#     for pred in preds:
#         top_indices = pred.argsort()[-top:][::-1]
#         result = [tuple(CLASS_CUSTOM[i]) + (pred[i]*100,) for i in top_indices]
#         results.append(result)
#     return results

# Prediction
prediction = model.predict(img_tensor)
pre_y = np.argmax(prediction)
print("L'image1 est label: %d" % pre_y)

prediction2 = model.predict(img_tensor2)
pre_y2 = np.argmax(prediction2)
print("L'image2 est label: %d" % pre_y2)


if(pre_y == 0):
    print("Le dechet d'image1 est le dechet solide")
elif(pre_y == 1):
    print("Le dechet d'image1 est le dechet liquide")
    
if(pre_y2 == 0):
    print("Le dechet d'image2 est le dechet solide")
elif(pre_y2 == 1):
    print("Le dechet d'image2 est le dechet liquide")
    
# results = decode_predictions_custom(prediction, top=2)[0]
# for i in results:
#         print(i)