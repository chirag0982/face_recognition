# We first load the necessary libraries, the dataset and reshape its dimensons to the minimum allowed by the VGG16 --> (48,48,3)
import tensorflow as tf
from keras import callbacks
from keras import optimizers
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.applications import VGG16
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np
x = 48
input_shape = (x, x, 3)
Matrix = list()
q = 100
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Adriana Lima/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)

for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Alex Lawther/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Alexandra Daddario/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Alvaro Morte/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_alycia dabnem carey/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)    
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Amanda Crew/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)

for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_amber heard/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)    
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Andy Samberg/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)    
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Anne Hathaway/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)    
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Anthony Mackie/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)    
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Avril Lavigne/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)    
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_barack obama/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)    
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_barbara palvin/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)    
    
X_train = np.array(Matrix)
print(np.shape(X_train))
(1300, 48, 48, 3)
p = 100
q = 15
Matrix1 = list()
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Adriana Lima/A ('+ str(i + 1 + p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)

for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Alex Lawther/A ('+ str(i + 1 + p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Alexandra Daddario/A ('+ str(i + 1 + p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Alvaro Morte/A ('+ str(i + 1 + p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_alycia dabnem carey/A ('+ str(i + 1+ p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)    
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Amanda Crew/A ('+ str(i + 1+ p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)

for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_amber heard/A ('+ str(i + 1+ p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)    
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Andy Samberg/A ('+ str(i + 1+ p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)    
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Anne Hathaway/A ('+ str(i + 1+ p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)    
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Anthony Mackie/A ('+ str(i + 1+ p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)    
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Avril Lavigne/A ('+ str(i + 1+ p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)    
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_barack obama/A ('+ str(i + 1+ p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)    
    
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_barbara palvin/A ('+ str(i + 1+ p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)    
    
X_test = np.array(Matrix1)
print(np.shape(X_test))
(195, 48, 48, 3)
import numpy as np
Total_images_per_person = 100
no_of_people = 13
y_train = np.zeros((Total_images_per_person*no_of_people,no_of_people))

for j in range(no_of_people):
    for i in range(Total_images_per_person):
        y_train[i + j*Total_images_per_person,j] = 1

print(np.shape(y_train))
print(y_train)
(1300, 13)
[[1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]]
import numpy as np
Total_images_per_person = 15
no_of_people = 13
y_train = np.zeros((Total_images_per_person*no_of_people,no_of_people))

for j in range(no_of_people):
    for i in range(Total_images_per_person):
        y_train[i + j*Total_images_per_person,j] = 1

print(np.shape(y_train))
print(y_train)
(195, 13)
[[1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]]
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.summary()
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_5 (InputLayer)         (None, 48, 48, 3)         0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 48, 48, 64)        1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 48, 48, 64)        36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 24, 24, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 24, 24, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 24, 24, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 12, 12, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 12, 12, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 12, 12, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 12, 12, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 6, 6, 256)         0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 6, 6, 512)         1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 6, 6, 512)         2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 6, 6, 512)         2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 3, 3, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 3, 3, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 3, 3, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 3, 3, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 1, 1, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________
for layer in base_model.layers:
    #if layer.name == 'block4_conv1':
    #    break
    #layer.trainable = False
    print('Layer ' + layer.name + ' frozen.')
    
# We add our classificator (top_model) to the last layer of the model
last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(1000, activation='relu', name='fc1')(x)
x = Dropout(0.1)(x)
x = Dense(13, activation='softmax', name='predictions')(x)
model = Model(base_model.input, x)
# We compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# We see the new structure of the model
model.summary()    
Layer input_5 frozen.
Layer block1_conv1 frozen.
Layer block1_conv2 frozen.
Layer block1_pool frozen.
Layer block2_conv1 frozen.
Layer block2_conv2 frozen.
Layer block2_pool frozen.
Layer block3_conv1 frozen.
Layer block3_conv2 frozen.
Layer block3_conv3 frozen.
Layer block3_pool frozen.
Layer block4_conv1 frozen.
Layer block4_conv2 frozen.
Layer block4_conv3 frozen.
Layer block4_pool frozen.
Layer block5_conv1 frozen.
Layer block5_conv2 frozen.
Layer block5_conv3 frozen.
Layer block5_pool frozen.
Model: "model_10"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_5 (InputLayer)         (None, 48, 48, 3)         0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 48, 48, 64)        1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 48, 48, 64)        36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 24, 24, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 24, 24, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 24, 24, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 12, 12, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 12, 12, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 12, 12, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 12, 12, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 6, 6, 256)         0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 6, 6, 512)         1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 6, 6, 512)         2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 6, 6, 512)         2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 3, 3, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 3, 3, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 3, 3, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 3, 3, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 1, 1, 512)         0         
_________________________________________________________________
flatten_10 (Flatten)         (None, 512)               0         
_________________________________________________________________
fc1 (Dense)                  (None, 1000)              513000    
_________________________________________________________________
dropout_5 (Dropout)          (None, 1000)              0         
_________________________________________________________________
predictions (Dense)          (None, 13)                13013     
=================================================================
Total params: 15,240,701
Trainable params: 526,013
Non-trainable params: 14,714,688
_________________________________________________________________
epochs = 15

batch_size = 256
# We train it
model.fit(X_train, Y_train,
          batch_size=batch_size,
          validation_data=(X_test, Y_test),
          epochs=epochs,shuffle=True)
Train on 1300 samples, validate on 195 samples
Epoch 1/15
1300/1300 [==============================] - 18s 14ms/step - loss: 17.3430 - accuracy: 0.0631 - val_loss: 11.3336 - val_accuracy: 0.1795
Epoch 2/15
1300/1300 [==============================] - 17s 13ms/step - loss: 13.0402 - accuracy: 0.0654 - val_loss: 8.3072 - val_accuracy: 0.0718
Epoch 3/15
1300/1300 [==============================] - 16s 12ms/step - loss: 9.6992 - accuracy: 0.0954 - val_loss: 7.8869 - val_accuracy: 0.1077
Epoch 4/15
1300/1300 [==============================] - 10279s 8s/step - loss: 8.4442 - accuracy: 0.0785 - val_loss: 6.2467 - val_accuracy: 0.1487
Epoch 5/15
1300/1300 [==============================] - 13s 10ms/step - loss: 7.8418 - accuracy: 0.0915 - val_loss: 6.0686 - val_accuracy: 0.0513
Epoch 6/15
1300/1300 [==============================] - 12s 9ms/step - loss: 7.2530 - accuracy: 0.0746 - val_loss: 5.9795 - val_accuracy: 0.0769
Epoch 7/15
1300/1300 [==============================] - 12s 9ms/step - loss: 7.4465 - accuracy: 0.0862 - val_loss: 6.1307 - val_accuracy: 0.0564
Epoch 8/15
1300/1300 [==============================] - 11s 9ms/step - loss: 7.2309 - accuracy: 0.0869 - val_loss: 5.5503 - val_accuracy: 0.0564
Epoch 9/15
1300/1300 [==============================] - 11s 9ms/step - loss: 7.4934 - accuracy: 0.0692 - val_loss: 6.0721 - val_accuracy: 0.0974
Epoch 10/15
1300/1300 [==============================] - 11s 9ms/step - loss: 7.4422 - accuracy: 0.0915 - val_loss: 5.8399 - val_accuracy: 0.0103
Epoch 11/15
1300/1300 [==============================] - 12s 9ms/step - loss: 7.9152 - accuracy: 0.0638 - val_loss: 5.9450 - val_accuracy: 0.2051
Epoch 12/15
1300/1300 [==============================] - 14s 11ms/step - loss: 8.1309 - accuracy: 0.0762 - val_loss: 5.7883 - val_accuracy: 0.0667
Epoch 13/15
1300/1300 [==============================] - 13s 10ms/step - loss: 8.2017 - accuracy: 0.1023 - val_loss: 7.7655 - val_accuracy: 0.0103
Epoch 14/15
1300/1300 [==============================] - 12s 9ms/step - loss: 9.0030 - accuracy: 0.0731 - val_loss: 6.5416 - val_accuracy: 0.0256
Epoch 15/15
1300/1300 [==============================] - 12s 9ms/step - loss: 9.3022 - accuracy: 0.0815 - val_loss: 6.5094 - val_accuracy: 0.1128
<keras.callbacks.callbacks.History at 0x29f862f8d88>
 
 