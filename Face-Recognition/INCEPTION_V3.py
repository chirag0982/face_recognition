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
x = 128
input_shape = (x, x, 3)
Using TensorFlow backend.
Matrix = list()
q = 150
for i in range(q):
    # example of using a pre-trained model as a classifier
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.applications.vgg16 import preprocess_input
    from keras.applications.vgg16 import decode_predictions
    from keras.applications.vgg16 import VGG16
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Leonardo DiCaprio/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)

for i in range(q):
   
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
    
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Tom Hardy/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)
    
for i in range(q):
    
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Emma Watson/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)
    
for i in range(q):
   
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Henry Cavil/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)    
    
for i in range(q):
    
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Megan Fox/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)

for i in range(q):
    
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Robert Downey Jr/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix.append(image)    
    
for i in range(q):
    
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Sophie Turner/A ('+ str(i + 1) + ').jpg',target_size=(x,x))
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
(1200, 128, 128, 3)
p = 150
q = 45
Matrix1 = list()
for i in range(q):
    # example of using a pre-trained model as a classifier
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Leonardo DiCaprio/A ('+ str(i + 1 + p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)

for i in range(q):
   
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

    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Tom Hardy/A ('+ str(i + 1 + p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)
    
for i in range(q):
   
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Emma Watson/A ('+ str(i + 1 + p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)
    
for i in range(q):
    
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Henry Cavil/A ('+ str(i + 1+ p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)    
    
for i in range(q):

    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Megan Fox/A ('+ str(i + 1+ p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)

for i in range(q):
    
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Robert Downey Jr/A ('+ str(i + 1+ p) + ').jpg',target_size=(x,x))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    image = image.reshape(image.shape[1],image.shape[2],image.shape[3])
    Matrix1.append(image)    
    
for i in range(q):
   
    # load an image from file
    image = load_img('Desktop/105_classes_pins_dataset/pins_Sophie Turner/A ('+ str(i + 1+ p) + ').jpg',target_size=(x,x))
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
(360, 128, 128, 3)
import numpy as np
Total_images_per_person = 150
no_of_people = 8
Y_train = np.zeros((Total_images_per_person*no_of_people,no_of_people))

for j in range(no_of_people):
    for i in range(Total_images_per_person):
        Y_train[i + j*Total_images_per_person,j] = 1

print(np.shape(Y_train))
print(Y_train)
(1200, 8)
[[1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]]
import numpy as np
Total_images_per_person = 45
no_of_people = 8
Y_test = np.zeros((Total_images_per_person*no_of_people,no_of_people))

for j in range(no_of_people):
    for i in range(Total_images_per_person):
        Y_test[i + j*Total_images_per_person,j] = 1

print(np.shape(Y_test))
print(Y_test)
(360, 8)
[[1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]]
from keras.applications.inception_v3 import InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.summary()
Model: "inception_v3"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            (None, 128, 128, 3)  0                                            
__________________________________________________________________________________________________
conv2d_95 (Conv2D)              (None, 63, 63, 32)   864         input_2[0][0]                    
__________________________________________________________________________________________________
batch_normalization_95 (BatchNo (None, 63, 63, 32)   96          conv2d_95[0][0]                  
__________________________________________________________________________________________________
activation_95 (Activation)      (None, 63, 63, 32)   0           batch_normalization_95[0][0]     
__________________________________________________________________________________________________
conv2d_96 (Conv2D)              (None, 61, 61, 32)   9216        activation_95[0][0]              
__________________________________________________________________________________________________
batch_normalization_96 (BatchNo (None, 61, 61, 32)   96          conv2d_96[0][0]                  
__________________________________________________________________________________________________
activation_96 (Activation)      (None, 61, 61, 32)   0           batch_normalization_96[0][0]     
__________________________________________________________________________________________________
conv2d_97 (Conv2D)              (None, 61, 61, 64)   18432       activation_96[0][0]              
__________________________________________________________________________________________________
batch_normalization_97 (BatchNo (None, 61, 61, 64)   192         conv2d_97[0][0]                  
__________________________________________________________________________________________________
activation_97 (Activation)      (None, 61, 61, 64)   0           batch_normalization_97[0][0]     
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 30, 30, 64)   0           activation_97[0][0]              
__________________________________________________________________________________________________
conv2d_98 (Conv2D)              (None, 30, 30, 80)   5120        max_pooling2d_5[0][0]            
__________________________________________________________________________________________________
batch_normalization_98 (BatchNo (None, 30, 30, 80)   240         conv2d_98[0][0]                  
__________________________________________________________________________________________________
activation_98 (Activation)      (None, 30, 30, 80)   0           batch_normalization_98[0][0]     
__________________________________________________________________________________________________
conv2d_99 (Conv2D)              (None, 28, 28, 192)  138240      activation_98[0][0]              
__________________________________________________________________________________________________
batch_normalization_99 (BatchNo (None, 28, 28, 192)  576         conv2d_99[0][0]                  
__________________________________________________________________________________________________
activation_99 (Activation)      (None, 28, 28, 192)  0           batch_normalization_99[0][0]     
__________________________________________________________________________________________________
max_pooling2d_6 (MaxPooling2D)  (None, 13, 13, 192)  0           activation_99[0][0]              
__________________________________________________________________________________________________
conv2d_103 (Conv2D)             (None, 13, 13, 64)   12288       max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
batch_normalization_103 (BatchN (None, 13, 13, 64)   192         conv2d_103[0][0]                 
__________________________________________________________________________________________________
activation_103 (Activation)     (None, 13, 13, 64)   0           batch_normalization_103[0][0]    
__________________________________________________________________________________________________
conv2d_101 (Conv2D)             (None, 13, 13, 48)   9216        max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
conv2d_104 (Conv2D)             (None, 13, 13, 96)   55296       activation_103[0][0]             
__________________________________________________________________________________________________
batch_normalization_101 (BatchN (None, 13, 13, 48)   144         conv2d_101[0][0]                 
__________________________________________________________________________________________________
batch_normalization_104 (BatchN (None, 13, 13, 96)   288         conv2d_104[0][0]                 
__________________________________________________________________________________________________
activation_101 (Activation)     (None, 13, 13, 48)   0           batch_normalization_101[0][0]    
__________________________________________________________________________________________________
activation_104 (Activation)     (None, 13, 13, 96)   0           batch_normalization_104[0][0]    
__________________________________________________________________________________________________
average_pooling2d_10 (AveragePo (None, 13, 13, 192)  0           max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
conv2d_100 (Conv2D)             (None, 13, 13, 64)   12288       max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
conv2d_102 (Conv2D)             (None, 13, 13, 64)   76800       activation_101[0][0]             
__________________________________________________________________________________________________
conv2d_105 (Conv2D)             (None, 13, 13, 96)   82944       activation_104[0][0]             
__________________________________________________________________________________________________
conv2d_106 (Conv2D)             (None, 13, 13, 32)   6144        average_pooling2d_10[0][0]       
__________________________________________________________________________________________________
batch_normalization_100 (BatchN (None, 13, 13, 64)   192         conv2d_100[0][0]                 
__________________________________________________________________________________________________
batch_normalization_102 (BatchN (None, 13, 13, 64)   192         conv2d_102[0][0]                 
__________________________________________________________________________________________________
batch_normalization_105 (BatchN (None, 13, 13, 96)   288         conv2d_105[0][0]                 
__________________________________________________________________________________________________
batch_normalization_106 (BatchN (None, 13, 13, 32)   96          conv2d_106[0][0]                 
__________________________________________________________________________________________________
activation_100 (Activation)     (None, 13, 13, 64)   0           batch_normalization_100[0][0]    
__________________________________________________________________________________________________
activation_102 (Activation)     (None, 13, 13, 64)   0           batch_normalization_102[0][0]    
__________________________________________________________________________________________________
activation_105 (Activation)     (None, 13, 13, 96)   0           batch_normalization_105[0][0]    
__________________________________________________________________________________________________
activation_106 (Activation)     (None, 13, 13, 32)   0           batch_normalization_106[0][0]    
__________________________________________________________________________________________________
mixed0 (Concatenate)            (None, 13, 13, 256)  0           activation_100[0][0]             
                                                                 activation_102[0][0]             
                                                                 activation_105[0][0]             
                                                                 activation_106[0][0]             
__________________________________________________________________________________________________
conv2d_110 (Conv2D)             (None, 13, 13, 64)   16384       mixed0[0][0]                     
__________________________________________________________________________________________________
batch_normalization_110 (BatchN (None, 13, 13, 64)   192         conv2d_110[0][0]                 
__________________________________________________________________________________________________
activation_110 (Activation)     (None, 13, 13, 64)   0           batch_normalization_110[0][0]    
__________________________________________________________________________________________________
conv2d_108 (Conv2D)             (None, 13, 13, 48)   12288       mixed0[0][0]                     
__________________________________________________________________________________________________
conv2d_111 (Conv2D)             (None, 13, 13, 96)   55296       activation_110[0][0]             
__________________________________________________________________________________________________
batch_normalization_108 (BatchN (None, 13, 13, 48)   144         conv2d_108[0][0]                 
__________________________________________________________________________________________________
batch_normalization_111 (BatchN (None, 13, 13, 96)   288         conv2d_111[0][0]                 
__________________________________________________________________________________________________
activation_108 (Activation)     (None, 13, 13, 48)   0           batch_normalization_108[0][0]    
__________________________________________________________________________________________________
activation_111 (Activation)     (None, 13, 13, 96)   0           batch_normalization_111[0][0]    
__________________________________________________________________________________________________
average_pooling2d_11 (AveragePo (None, 13, 13, 256)  0           mixed0[0][0]                     
__________________________________________________________________________________________________
conv2d_107 (Conv2D)             (None, 13, 13, 64)   16384       mixed0[0][0]                     
__________________________________________________________________________________________________
conv2d_109 (Conv2D)             (None, 13, 13, 64)   76800       activation_108[0][0]             
__________________________________________________________________________________________________
conv2d_112 (Conv2D)             (None, 13, 13, 96)   82944       activation_111[0][0]             
__________________________________________________________________________________________________
conv2d_113 (Conv2D)             (None, 13, 13, 64)   16384       average_pooling2d_11[0][0]       
__________________________________________________________________________________________________
batch_normalization_107 (BatchN (None, 13, 13, 64)   192         conv2d_107[0][0]                 
__________________________________________________________________________________________________
batch_normalization_109 (BatchN (None, 13, 13, 64)   192         conv2d_109[0][0]                 
__________________________________________________________________________________________________
batch_normalization_112 (BatchN (None, 13, 13, 96)   288         conv2d_112[0][0]                 
__________________________________________________________________________________________________
batch_normalization_113 (BatchN (None, 13, 13, 64)   192         conv2d_113[0][0]                 
__________________________________________________________________________________________________
activation_107 (Activation)     (None, 13, 13, 64)   0           batch_normalization_107[0][0]    
__________________________________________________________________________________________________
activation_109 (Activation)     (None, 13, 13, 64)   0           batch_normalization_109[0][0]    
__________________________________________________________________________________________________
activation_112 (Activation)     (None, 13, 13, 96)   0           batch_normalization_112[0][0]    
__________________________________________________________________________________________________
activation_113 (Activation)     (None, 13, 13, 64)   0           batch_normalization_113[0][0]    
__________________________________________________________________________________________________
mixed1 (Concatenate)            (None, 13, 13, 288)  0           activation_107[0][0]             
                                                                 activation_109[0][0]             
                                                                 activation_112[0][0]             
                                                                 activation_113[0][0]             
__________________________________________________________________________________________________
conv2d_117 (Conv2D)             (None, 13, 13, 64)   18432       mixed1[0][0]                     
__________________________________________________________________________________________________
batch_normalization_117 (BatchN (None, 13, 13, 64)   192         conv2d_117[0][0]                 
__________________________________________________________________________________________________
activation_117 (Activation)     (None, 13, 13, 64)   0           batch_normalization_117[0][0]    
__________________________________________________________________________________________________
conv2d_115 (Conv2D)             (None, 13, 13, 48)   13824       mixed1[0][0]                     
__________________________________________________________________________________________________
conv2d_118 (Conv2D)             (None, 13, 13, 96)   55296       activation_117[0][0]             
__________________________________________________________________________________________________
batch_normalization_115 (BatchN (None, 13, 13, 48)   144         conv2d_115[0][0]                 
__________________________________________________________________________________________________
batch_normalization_118 (BatchN (None, 13, 13, 96)   288         conv2d_118[0][0]                 
__________________________________________________________________________________________________
activation_115 (Activation)     (None, 13, 13, 48)   0           batch_normalization_115[0][0]    
__________________________________________________________________________________________________
activation_118 (Activation)     (None, 13, 13, 96)   0           batch_normalization_118[0][0]    
__________________________________________________________________________________________________
average_pooling2d_12 (AveragePo (None, 13, 13, 288)  0           mixed1[0][0]                     
__________________________________________________________________________________________________
conv2d_114 (Conv2D)             (None, 13, 13, 64)   18432       mixed1[0][0]                     
__________________________________________________________________________________________________
conv2d_116 (Conv2D)             (None, 13, 13, 64)   76800       activation_115[0][0]             
__________________________________________________________________________________________________
conv2d_119 (Conv2D)             (None, 13, 13, 96)   82944       activation_118[0][0]             
__________________________________________________________________________________________________
conv2d_120 (Conv2D)             (None, 13, 13, 64)   18432       average_pooling2d_12[0][0]       
__________________________________________________________________________________________________
batch_normalization_114 (BatchN (None, 13, 13, 64)   192         conv2d_114[0][0]                 
__________________________________________________________________________________________________
batch_normalization_116 (BatchN (None, 13, 13, 64)   192         conv2d_116[0][0]                 
__________________________________________________________________________________________________
batch_normalization_119 (BatchN (None, 13, 13, 96)   288         conv2d_119[0][0]                 
__________________________________________________________________________________________________
batch_normalization_120 (BatchN (None, 13, 13, 64)   192         conv2d_120[0][0]                 
__________________________________________________________________________________________________
activation_114 (Activation)     (None, 13, 13, 64)   0           batch_normalization_114[0][0]    
__________________________________________________________________________________________________
activation_116 (Activation)     (None, 13, 13, 64)   0           batch_normalization_116[0][0]    
__________________________________________________________________________________________________
activation_119 (Activation)     (None, 13, 13, 96)   0           batch_normalization_119[0][0]    
__________________________________________________________________________________________________
activation_120 (Activation)     (None, 13, 13, 64)   0           batch_normalization_120[0][0]    
__________________________________________________________________________________________________
mixed2 (Concatenate)            (None, 13, 13, 288)  0           activation_114[0][0]             
                                                                 activation_116[0][0]             
                                                                 activation_119[0][0]             
                                                                 activation_120[0][0]             
__________________________________________________________________________________________________
conv2d_122 (Conv2D)             (None, 13, 13, 64)   18432       mixed2[0][0]                     
__________________________________________________________________________________________________
batch_normalization_122 (BatchN (None, 13, 13, 64)   192         conv2d_122[0][0]                 
__________________________________________________________________________________________________
activation_122 (Activation)     (None, 13, 13, 64)   0           batch_normalization_122[0][0]    
__________________________________________________________________________________________________
conv2d_123 (Conv2D)             (None, 13, 13, 96)   55296       activation_122[0][0]             
__________________________________________________________________________________________________
batch_normalization_123 (BatchN (None, 13, 13, 96)   288         conv2d_123[0][0]                 
__________________________________________________________________________________________________
activation_123 (Activation)     (None, 13, 13, 96)   0           batch_normalization_123[0][0]    
__________________________________________________________________________________________________
conv2d_121 (Conv2D)             (None, 6, 6, 384)    995328      mixed2[0][0]                     
__________________________________________________________________________________________________
conv2d_124 (Conv2D)             (None, 6, 6, 96)     82944       activation_123[0][0]             
__________________________________________________________________________________________________
batch_normalization_121 (BatchN (None, 6, 6, 384)    1152        conv2d_121[0][0]                 
__________________________________________________________________________________________________
batch_normalization_124 (BatchN (None, 6, 6, 96)     288         conv2d_124[0][0]                 
__________________________________________________________________________________________________
activation_121 (Activation)     (None, 6, 6, 384)    0           batch_normalization_121[0][0]    
__________________________________________________________________________________________________
activation_124 (Activation)     (None, 6, 6, 96)     0           batch_normalization_124[0][0]    
__________________________________________________________________________________________________
max_pooling2d_7 (MaxPooling2D)  (None, 6, 6, 288)    0           mixed2[0][0]                     
__________________________________________________________________________________________________
mixed3 (Concatenate)            (None, 6, 6, 768)    0           activation_121[0][0]             
                                                                 activation_124[0][0]             
                                                                 max_pooling2d_7[0][0]            
__________________________________________________________________________________________________
conv2d_129 (Conv2D)             (None, 6, 6, 128)    98304       mixed3[0][0]                     
__________________________________________________________________________________________________
batch_normalization_129 (BatchN (None, 6, 6, 128)    384         conv2d_129[0][0]                 
__________________________________________________________________________________________________
activation_129 (Activation)     (None, 6, 6, 128)    0           batch_normalization_129[0][0]    
__________________________________________________________________________________________________
conv2d_130 (Conv2D)             (None, 6, 6, 128)    114688      activation_129[0][0]             
__________________________________________________________________________________________________
batch_normalization_130 (BatchN (None, 6, 6, 128)    384         conv2d_130[0][0]                 
__________________________________________________________________________________________________
activation_130 (Activation)     (None, 6, 6, 128)    0           batch_normalization_130[0][0]    
__________________________________________________________________________________________________
conv2d_126 (Conv2D)             (None, 6, 6, 128)    98304       mixed3[0][0]                     
__________________________________________________________________________________________________
conv2d_131 (Conv2D)             (None, 6, 6, 128)    114688      activation_130[0][0]             
__________________________________________________________________________________________________
batch_normalization_126 (BatchN (None, 6, 6, 128)    384         conv2d_126[0][0]                 
__________________________________________________________________________________________________
batch_normalization_131 (BatchN (None, 6, 6, 128)    384         conv2d_131[0][0]                 
__________________________________________________________________________________________________
activation_126 (Activation)     (None, 6, 6, 128)    0           batch_normalization_126[0][0]    
__________________________________________________________________________________________________
activation_131 (Activation)     (None, 6, 6, 128)    0           batch_normalization_131[0][0]    
__________________________________________________________________________________________________
conv2d_127 (Conv2D)             (None, 6, 6, 128)    114688      activation_126[0][0]             
__________________________________________________________________________________________________
conv2d_132 (Conv2D)             (None, 6, 6, 128)    114688      activation_131[0][0]             
__________________________________________________________________________________________________
batch_normalization_127 (BatchN (None, 6, 6, 128)    384         conv2d_127[0][0]                 
__________________________________________________________________________________________________
batch_normalization_132 (BatchN (None, 6, 6, 128)    384         conv2d_132[0][0]                 
__________________________________________________________________________________________________
activation_127 (Activation)     (None, 6, 6, 128)    0           batch_normalization_127[0][0]    
__________________________________________________________________________________________________
activation_132 (Activation)     (None, 6, 6, 128)    0           batch_normalization_132[0][0]    
__________________________________________________________________________________________________
average_pooling2d_13 (AveragePo (None, 6, 6, 768)    0           mixed3[0][0]                     
__________________________________________________________________________________________________
conv2d_125 (Conv2D)             (None, 6, 6, 192)    147456      mixed3[0][0]                     
__________________________________________________________________________________________________
conv2d_128 (Conv2D)             (None, 6, 6, 192)    172032      activation_127[0][0]             
__________________________________________________________________________________________________
conv2d_133 (Conv2D)             (None, 6, 6, 192)    172032      activation_132[0][0]             
__________________________________________________________________________________________________
conv2d_134 (Conv2D)             (None, 6, 6, 192)    147456      average_pooling2d_13[0][0]       
__________________________________________________________________________________________________
batch_normalization_125 (BatchN (None, 6, 6, 192)    576         conv2d_125[0][0]                 
__________________________________________________________________________________________________
batch_normalization_128 (BatchN (None, 6, 6, 192)    576         conv2d_128[0][0]                 
__________________________________________________________________________________________________
batch_normalization_133 (BatchN (None, 6, 6, 192)    576         conv2d_133[0][0]                 
__________________________________________________________________________________________________
batch_normalization_134 (BatchN (None, 6, 6, 192)    576         conv2d_134[0][0]                 
__________________________________________________________________________________________________
activation_125 (Activation)     (None, 6, 6, 192)    0           batch_normalization_125[0][0]    
__________________________________________________________________________________________________
activation_128 (Activation)     (None, 6, 6, 192)    0           batch_normalization_128[0][0]    
__________________________________________________________________________________________________
activation_133 (Activation)     (None, 6, 6, 192)    0           batch_normalization_133[0][0]    
__________________________________________________________________________________________________
activation_134 (Activation)     (None, 6, 6, 192)    0           batch_normalization_134[0][0]    
__________________________________________________________________________________________________
mixed4 (Concatenate)            (None, 6, 6, 768)    0           activation_125[0][0]             
                                                                 activation_128[0][0]             
                                                                 activation_133[0][0]             
                                                                 activation_134[0][0]             
__________________________________________________________________________________________________
conv2d_139 (Conv2D)             (None, 6, 6, 160)    122880      mixed4[0][0]                     
__________________________________________________________________________________________________
batch_normalization_139 (BatchN (None, 6, 6, 160)    480         conv2d_139[0][0]                 
__________________________________________________________________________________________________
activation_139 (Activation)     (None, 6, 6, 160)    0           batch_normalization_139[0][0]    
__________________________________________________________________________________________________
conv2d_140 (Conv2D)             (None, 6, 6, 160)    179200      activation_139[0][0]             
__________________________________________________________________________________________________
batch_normalization_140 (BatchN (None, 6, 6, 160)    480         conv2d_140[0][0]                 
__________________________________________________________________________________________________
activation_140 (Activation)     (None, 6, 6, 160)    0           batch_normalization_140[0][0]    
__________________________________________________________________________________________________
conv2d_136 (Conv2D)             (None, 6, 6, 160)    122880      mixed4[0][0]                     
__________________________________________________________________________________________________
conv2d_141 (Conv2D)             (None, 6, 6, 160)    179200      activation_140[0][0]             
__________________________________________________________________________________________________
batch_normalization_136 (BatchN (None, 6, 6, 160)    480         conv2d_136[0][0]                 
__________________________________________________________________________________________________
batch_normalization_141 (BatchN (None, 6, 6, 160)    480         conv2d_141[0][0]                 
__________________________________________________________________________________________________
activation_136 (Activation)     (None, 6, 6, 160)    0           batch_normalization_136[0][0]    
__________________________________________________________________________________________________
activation_141 (Activation)     (None, 6, 6, 160)    0           batch_normalization_141[0][0]    
__________________________________________________________________________________________________
conv2d_137 (Conv2D)             (None, 6, 6, 160)    179200      activation_136[0][0]             
__________________________________________________________________________________________________
conv2d_142 (Conv2D)             (None, 6, 6, 160)    179200      activation_141[0][0]             
__________________________________________________________________________________________________
batch_normalization_137 (BatchN (None, 6, 6, 160)    480         conv2d_137[0][0]                 
__________________________________________________________________________________________________
batch_normalization_142 (BatchN (None, 6, 6, 160)    480         conv2d_142[0][0]                 
__________________________________________________________________________________________________
activation_137 (Activation)     (None, 6, 6, 160)    0           batch_normalization_137[0][0]    
__________________________________________________________________________________________________
activation_142 (Activation)     (None, 6, 6, 160)    0           batch_normalization_142[0][0]    
__________________________________________________________________________________________________
average_pooling2d_14 (AveragePo (None, 6, 6, 768)    0           mixed4[0][0]                     
__________________________________________________________________________________________________
conv2d_135 (Conv2D)             (None, 6, 6, 192)    147456      mixed4[0][0]                     
__________________________________________________________________________________________________
conv2d_138 (Conv2D)             (None, 6, 6, 192)    215040      activation_137[0][0]             
__________________________________________________________________________________________________
conv2d_143 (Conv2D)             (None, 6, 6, 192)    215040      activation_142[0][0]             
__________________________________________________________________________________________________
conv2d_144 (Conv2D)             (None, 6, 6, 192)    147456      average_pooling2d_14[0][0]       
__________________________________________________________________________________________________
batch_normalization_135 (BatchN (None, 6, 6, 192)    576         conv2d_135[0][0]                 
__________________________________________________________________________________________________
batch_normalization_138 (BatchN (None, 6, 6, 192)    576         conv2d_138[0][0]                 
__________________________________________________________________________________________________
batch_normalization_143 (BatchN (None, 6, 6, 192)    576         conv2d_143[0][0]                 
__________________________________________________________________________________________________
batch_normalization_144 (BatchN (None, 6, 6, 192)    576         conv2d_144[0][0]                 
__________________________________________________________________________________________________
activation_135 (Activation)     (None, 6, 6, 192)    0           batch_normalization_135[0][0]    
__________________________________________________________________________________________________
activation_138 (Activation)     (None, 6, 6, 192)    0           batch_normalization_138[0][0]    
__________________________________________________________________________________________________
activation_143 (Activation)     (None, 6, 6, 192)    0           batch_normalization_143[0][0]    
__________________________________________________________________________________________________
activation_144 (Activation)     (None, 6, 6, 192)    0           batch_normalization_144[0][0]    
__________________________________________________________________________________________________
mixed5 (Concatenate)            (None, 6, 6, 768)    0           activation_135[0][0]             
                                                                 activation_138[0][0]             
                                                                 activation_143[0][0]             
                                                                 activation_144[0][0]             
__________________________________________________________________________________________________
conv2d_149 (Conv2D)             (None, 6, 6, 160)    122880      mixed5[0][0]                     
__________________________________________________________________________________________________
batch_normalization_149 (BatchN (None, 6, 6, 160)    480         conv2d_149[0][0]                 
__________________________________________________________________________________________________
activation_149 (Activation)     (None, 6, 6, 160)    0           batch_normalization_149[0][0]    
__________________________________________________________________________________________________
conv2d_150 (Conv2D)             (None, 6, 6, 160)    179200      activation_149[0][0]             
__________________________________________________________________________________________________
batch_normalization_150 (BatchN (None, 6, 6, 160)    480         conv2d_150[0][0]                 
__________________________________________________________________________________________________
activation_150 (Activation)     (None, 6, 6, 160)    0           batch_normalization_150[0][0]    
__________________________________________________________________________________________________
conv2d_146 (Conv2D)             (None, 6, 6, 160)    122880      mixed5[0][0]                     
__________________________________________________________________________________________________
conv2d_151 (Conv2D)             (None, 6, 6, 160)    179200      activation_150[0][0]             
__________________________________________________________________________________________________
batch_normalization_146 (BatchN (None, 6, 6, 160)    480         conv2d_146[0][0]                 
__________________________________________________________________________________________________
batch_normalization_151 (BatchN (None, 6, 6, 160)    480         conv2d_151[0][0]                 
__________________________________________________________________________________________________
activation_146 (Activation)     (None, 6, 6, 160)    0           batch_normalization_146[0][0]    
__________________________________________________________________________________________________
activation_151 (Activation)     (None, 6, 6, 160)    0           batch_normalization_151[0][0]    
__________________________________________________________________________________________________
conv2d_147 (Conv2D)             (None, 6, 6, 160)    179200      activation_146[0][0]             
__________________________________________________________________________________________________
conv2d_152 (Conv2D)             (None, 6, 6, 160)    179200      activation_151[0][0]             
__________________________________________________________________________________________________
batch_normalization_147 (BatchN (None, 6, 6, 160)    480         conv2d_147[0][0]                 
__________________________________________________________________________________________________
batch_normalization_152 (BatchN (None, 6, 6, 160)    480         conv2d_152[0][0]                 
__________________________________________________________________________________________________
activation_147 (Activation)     (None, 6, 6, 160)    0           batch_normalization_147[0][0]    
__________________________________________________________________________________________________
activation_152 (Activation)     (None, 6, 6, 160)    0           batch_normalization_152[0][0]    
__________________________________________________________________________________________________
average_pooling2d_15 (AveragePo (None, 6, 6, 768)    0           mixed5[0][0]                     
__________________________________________________________________________________________________
conv2d_145 (Conv2D)             (None, 6, 6, 192)    147456      mixed5[0][0]                     
__________________________________________________________________________________________________
conv2d_148 (Conv2D)             (None, 6, 6, 192)    215040      activation_147[0][0]             
__________________________________________________________________________________________________
conv2d_153 (Conv2D)             (None, 6, 6, 192)    215040      activation_152[0][0]             
__________________________________________________________________________________________________
conv2d_154 (Conv2D)             (None, 6, 6, 192)    147456      average_pooling2d_15[0][0]       
__________________________________________________________________________________________________
batch_normalization_145 (BatchN (None, 6, 6, 192)    576         conv2d_145[0][0]                 
__________________________________________________________________________________________________
batch_normalization_148 (BatchN (None, 6, 6, 192)    576         conv2d_148[0][0]                 
__________________________________________________________________________________________________
batch_normalization_153 (BatchN (None, 6, 6, 192)    576         conv2d_153[0][0]                 
__________________________________________________________________________________________________
batch_normalization_154 (BatchN (None, 6, 6, 192)    576         conv2d_154[0][0]                 
__________________________________________________________________________________________________
activation_145 (Activation)     (None, 6, 6, 192)    0           batch_normalization_145[0][0]    
__________________________________________________________________________________________________
activation_148 (Activation)     (None, 6, 6, 192)    0           batch_normalization_148[0][0]    
__________________________________________________________________________________________________
activation_153 (Activation)     (None, 6, 6, 192)    0           batch_normalization_153[0][0]    
__________________________________________________________________________________________________
activation_154 (Activation)     (None, 6, 6, 192)    0           batch_normalization_154[0][0]    
__________________________________________________________________________________________________
mixed6 (Concatenate)            (None, 6, 6, 768)    0           activation_145[0][0]             
                                                                 activation_148[0][0]             
                                                                 activation_153[0][0]             
                                                                 activation_154[0][0]             
__________________________________________________________________________________________________
conv2d_159 (Conv2D)             (None, 6, 6, 192)    147456      mixed6[0][0]                     
__________________________________________________________________________________________________
batch_normalization_159 (BatchN (None, 6, 6, 192)    576         conv2d_159[0][0]                 
__________________________________________________________________________________________________
activation_159 (Activation)     (None, 6, 6, 192)    0           batch_normalization_159[0][0]    
__________________________________________________________________________________________________
conv2d_160 (Conv2D)             (None, 6, 6, 192)    258048      activation_159[0][0]             
__________________________________________________________________________________________________
batch_normalization_160 (BatchN (None, 6, 6, 192)    576         conv2d_160[0][0]                 
__________________________________________________________________________________________________
activation_160 (Activation)     (None, 6, 6, 192)    0           batch_normalization_160[0][0]    
__________________________________________________________________________________________________
conv2d_156 (Conv2D)             (None, 6, 6, 192)    147456      mixed6[0][0]                     
__________________________________________________________________________________________________
conv2d_161 (Conv2D)             (None, 6, 6, 192)    258048      activation_160[0][0]             
__________________________________________________________________________________________________
batch_normalization_156 (BatchN (None, 6, 6, 192)    576         conv2d_156[0][0]                 
__________________________________________________________________________________________________
batch_normalization_161 (BatchN (None, 6, 6, 192)    576         conv2d_161[0][0]                 
__________________________________________________________________________________________________
activation_156 (Activation)     (None, 6, 6, 192)    0           batch_normalization_156[0][0]    
__________________________________________________________________________________________________
activation_161 (Activation)     (None, 6, 6, 192)    0           batch_normalization_161[0][0]    
__________________________________________________________________________________________________
conv2d_157 (Conv2D)             (None, 6, 6, 192)    258048      activation_156[0][0]             
__________________________________________________________________________________________________
conv2d_162 (Conv2D)             (None, 6, 6, 192)    258048      activation_161[0][0]             
__________________________________________________________________________________________________
batch_normalization_157 (BatchN (None, 6, 6, 192)    576         conv2d_157[0][0]                 
__________________________________________________________________________________________________
batch_normalization_162 (BatchN (None, 6, 6, 192)    576         conv2d_162[0][0]                 
__________________________________________________________________________________________________
activation_157 (Activation)     (None, 6, 6, 192)    0           batch_normalization_157[0][0]    
__________________________________________________________________________________________________
activation_162 (Activation)     (None, 6, 6, 192)    0           batch_normalization_162[0][0]    
__________________________________________________________________________________________________
average_pooling2d_16 (AveragePo (None, 6, 6, 768)    0           mixed6[0][0]                     
__________________________________________________________________________________________________
conv2d_155 (Conv2D)             (None, 6, 6, 192)    147456      mixed6[0][0]                     
__________________________________________________________________________________________________
conv2d_158 (Conv2D)             (None, 6, 6, 192)    258048      activation_157[0][0]             
__________________________________________________________________________________________________
conv2d_163 (Conv2D)             (None, 6, 6, 192)    258048      activation_162[0][0]             
__________________________________________________________________________________________________
conv2d_164 (Conv2D)             (None, 6, 6, 192)    147456      average_pooling2d_16[0][0]       
__________________________________________________________________________________________________
batch_normalization_155 (BatchN (None, 6, 6, 192)    576         conv2d_155[0][0]                 
__________________________________________________________________________________________________
batch_normalization_158 (BatchN (None, 6, 6, 192)    576         conv2d_158[0][0]                 
__________________________________________________________________________________________________
batch_normalization_163 (BatchN (None, 6, 6, 192)    576         conv2d_163[0][0]                 
__________________________________________________________________________________________________
batch_normalization_164 (BatchN (None, 6, 6, 192)    576         conv2d_164[0][0]                 
__________________________________________________________________________________________________
activation_155 (Activation)     (None, 6, 6, 192)    0           batch_normalization_155[0][0]    
__________________________________________________________________________________________________
activation_158 (Activation)     (None, 6, 6, 192)    0           batch_normalization_158[0][0]    
__________________________________________________________________________________________________
activation_163 (Activation)     (None, 6, 6, 192)    0           batch_normalization_163[0][0]    
__________________________________________________________________________________________________
activation_164 (Activation)     (None, 6, 6, 192)    0           batch_normalization_164[0][0]    
__________________________________________________________________________________________________
mixed7 (Concatenate)            (None, 6, 6, 768)    0           activation_155[0][0]             
                                                                 activation_158[0][0]             
                                                                 activation_163[0][0]             
                                                                 activation_164[0][0]             
__________________________________________________________________________________________________
conv2d_167 (Conv2D)             (None, 6, 6, 192)    147456      mixed7[0][0]                     
__________________________________________________________________________________________________
batch_normalization_167 (BatchN (None, 6, 6, 192)    576         conv2d_167[0][0]                 
__________________________________________________________________________________________________
activation_167 (Activation)     (None, 6, 6, 192)    0           batch_normalization_167[0][0]    
__________________________________________________________________________________________________
conv2d_168 (Conv2D)             (None, 6, 6, 192)    258048      activation_167[0][0]             
__________________________________________________________________________________________________
batch_normalization_168 (BatchN (None, 6, 6, 192)    576         conv2d_168[0][0]                 
__________________________________________________________________________________________________
activation_168 (Activation)     (None, 6, 6, 192)    0           batch_normalization_168[0][0]    
__________________________________________________________________________________________________
conv2d_165 (Conv2D)             (None, 6, 6, 192)    147456      mixed7[0][0]                     
__________________________________________________________________________________________________
conv2d_169 (Conv2D)             (None, 6, 6, 192)    258048      activation_168[0][0]             
__________________________________________________________________________________________________
batch_normalization_165 (BatchN (None, 6, 6, 192)    576         conv2d_165[0][0]                 
__________________________________________________________________________________________________
batch_normalization_169 (BatchN (None, 6, 6, 192)    576         conv2d_169[0][0]                 
__________________________________________________________________________________________________
activation_165 (Activation)     (None, 6, 6, 192)    0           batch_normalization_165[0][0]    
__________________________________________________________________________________________________
activation_169 (Activation)     (None, 6, 6, 192)    0           batch_normalization_169[0][0]    
__________________________________________________________________________________________________
conv2d_166 (Conv2D)             (None, 2, 2, 320)    552960      activation_165[0][0]             
__________________________________________________________________________________________________
conv2d_170 (Conv2D)             (None, 2, 2, 192)    331776      activation_169[0][0]             
__________________________________________________________________________________________________
batch_normalization_166 (BatchN (None, 2, 2, 320)    960         conv2d_166[0][0]                 
__________________________________________________________________________________________________
batch_normalization_170 (BatchN (None, 2, 2, 192)    576         conv2d_170[0][0]                 
__________________________________________________________________________________________________
activation_166 (Activation)     (None, 2, 2, 320)    0           batch_normalization_166[0][0]    
__________________________________________________________________________________________________
activation_170 (Activation)     (None, 2, 2, 192)    0           batch_normalization_170[0][0]    
__________________________________________________________________________________________________
max_pooling2d_8 (MaxPooling2D)  (None, 2, 2, 768)    0           mixed7[0][0]                     
__________________________________________________________________________________________________
mixed8 (Concatenate)            (None, 2, 2, 1280)   0           activation_166[0][0]             
                                                                 activation_170[0][0]             
                                                                 max_pooling2d_8[0][0]            
__________________________________________________________________________________________________
conv2d_175 (Conv2D)             (None, 2, 2, 448)    573440      mixed8[0][0]                     
__________________________________________________________________________________________________
batch_normalization_175 (BatchN (None, 2, 2, 448)    1344        conv2d_175[0][0]                 
__________________________________________________________________________________________________
activation_175 (Activation)     (None, 2, 2, 448)    0           batch_normalization_175[0][0]    
__________________________________________________________________________________________________
conv2d_172 (Conv2D)             (None, 2, 2, 384)    491520      mixed8[0][0]                     
__________________________________________________________________________________________________
conv2d_176 (Conv2D)             (None, 2, 2, 384)    1548288     activation_175[0][0]             
__________________________________________________________________________________________________
batch_normalization_172 (BatchN (None, 2, 2, 384)    1152        conv2d_172[0][0]                 
__________________________________________________________________________________________________
batch_normalization_176 (BatchN (None, 2, 2, 384)    1152        conv2d_176[0][0]                 
__________________________________________________________________________________________________
activation_172 (Activation)     (None, 2, 2, 384)    0           batch_normalization_172[0][0]    
__________________________________________________________________________________________________
activation_176 (Activation)     (None, 2, 2, 384)    0           batch_normalization_176[0][0]    
__________________________________________________________________________________________________
conv2d_173 (Conv2D)             (None, 2, 2, 384)    442368      activation_172[0][0]             
__________________________________________________________________________________________________
conv2d_174 (Conv2D)             (None, 2, 2, 384)    442368      activation_172[0][0]             
__________________________________________________________________________________________________
conv2d_177 (Conv2D)             (None, 2, 2, 384)    442368      activation_176[0][0]             
__________________________________________________________________________________________________
conv2d_178 (Conv2D)             (None, 2, 2, 384)    442368      activation_176[0][0]             
__________________________________________________________________________________________________
average_pooling2d_17 (AveragePo (None, 2, 2, 1280)   0           mixed8[0][0]                     
__________________________________________________________________________________________________
conv2d_171 (Conv2D)             (None, 2, 2, 320)    409600      mixed8[0][0]                     
__________________________________________________________________________________________________
batch_normalization_173 (BatchN (None, 2, 2, 384)    1152        conv2d_173[0][0]                 
__________________________________________________________________________________________________
batch_normalization_174 (BatchN (None, 2, 2, 384)    1152        conv2d_174[0][0]                 
__________________________________________________________________________________________________
batch_normalization_177 (BatchN (None, 2, 2, 384)    1152        conv2d_177[0][0]                 
__________________________________________________________________________________________________
batch_normalization_178 (BatchN (None, 2, 2, 384)    1152        conv2d_178[0][0]                 
__________________________________________________________________________________________________
conv2d_179 (Conv2D)             (None, 2, 2, 192)    245760      average_pooling2d_17[0][0]       
__________________________________________________________________________________________________
batch_normalization_171 (BatchN (None, 2, 2, 320)    960         conv2d_171[0][0]                 
__________________________________________________________________________________________________
activation_173 (Activation)     (None, 2, 2, 384)    0           batch_normalization_173[0][0]    
__________________________________________________________________________________________________
activation_174 (Activation)     (None, 2, 2, 384)    0           batch_normalization_174[0][0]    
__________________________________________________________________________________________________
activation_177 (Activation)     (None, 2, 2, 384)    0           batch_normalization_177[0][0]    
__________________________________________________________________________________________________
activation_178 (Activation)     (None, 2, 2, 384)    0           batch_normalization_178[0][0]    
__________________________________________________________________________________________________
batch_normalization_179 (BatchN (None, 2, 2, 192)    576         conv2d_179[0][0]                 
__________________________________________________________________________________________________
activation_171 (Activation)     (None, 2, 2, 320)    0           batch_normalization_171[0][0]    
__________________________________________________________________________________________________
mixed9_0 (Concatenate)          (None, 2, 2, 768)    0           activation_173[0][0]             
                                                                 activation_174[0][0]             
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 2, 2, 768)    0           activation_177[0][0]             
                                                                 activation_178[0][0]             
__________________________________________________________________________________________________
activation_179 (Activation)     (None, 2, 2, 192)    0           batch_normalization_179[0][0]    
__________________________________________________________________________________________________
mixed9 (Concatenate)            (None, 2, 2, 2048)   0           activation_171[0][0]             
                                                                 mixed9_0[0][0]                   
                                                                 concatenate_3[0][0]              
                                                                 activation_179[0][0]             
__________________________________________________________________________________________________
conv2d_184 (Conv2D)             (None, 2, 2, 448)    917504      mixed9[0][0]                     
__________________________________________________________________________________________________
batch_normalization_184 (BatchN (None, 2, 2, 448)    1344        conv2d_184[0][0]                 
__________________________________________________________________________________________________
activation_184 (Activation)     (None, 2, 2, 448)    0           batch_normalization_184[0][0]    
__________________________________________________________________________________________________
conv2d_181 (Conv2D)             (None, 2, 2, 384)    786432      mixed9[0][0]                     
__________________________________________________________________________________________________
conv2d_185 (Conv2D)             (None, 2, 2, 384)    1548288     activation_184[0][0]             
__________________________________________________________________________________________________
batch_normalization_181 (BatchN (None, 2, 2, 384)    1152        conv2d_181[0][0]                 
__________________________________________________________________________________________________
batch_normalization_185 (BatchN (None, 2, 2, 384)    1152        conv2d_185[0][0]                 
__________________________________________________________________________________________________
activation_181 (Activation)     (None, 2, 2, 384)    0           batch_normalization_181[0][0]    
__________________________________________________________________________________________________
activation_185 (Activation)     (None, 2, 2, 384)    0           batch_normalization_185[0][0]    
__________________________________________________________________________________________________
conv2d_182 (Conv2D)             (None, 2, 2, 384)    442368      activation_181[0][0]             
__________________________________________________________________________________________________
conv2d_183 (Conv2D)             (None, 2, 2, 384)    442368      activation_181[0][0]             
__________________________________________________________________________________________________
conv2d_186 (Conv2D)             (None, 2, 2, 384)    442368      activation_185[0][0]             
__________________________________________________________________________________________________
conv2d_187 (Conv2D)             (None, 2, 2, 384)    442368      activation_185[0][0]             
__________________________________________________________________________________________________
average_pooling2d_18 (AveragePo (None, 2, 2, 2048)   0           mixed9[0][0]                     
__________________________________________________________________________________________________
conv2d_180 (Conv2D)             (None, 2, 2, 320)    655360      mixed9[0][0]                     
__________________________________________________________________________________________________
batch_normalization_182 (BatchN (None, 2, 2, 384)    1152        conv2d_182[0][0]                 
__________________________________________________________________________________________________
batch_normalization_183 (BatchN (None, 2, 2, 384)    1152        conv2d_183[0][0]                 
__________________________________________________________________________________________________
batch_normalization_186 (BatchN (None, 2, 2, 384)    1152        conv2d_186[0][0]                 
__________________________________________________________________________________________________
batch_normalization_187 (BatchN (None, 2, 2, 384)    1152        conv2d_187[0][0]                 
__________________________________________________________________________________________________
conv2d_188 (Conv2D)             (None, 2, 2, 192)    393216      average_pooling2d_18[0][0]       
__________________________________________________________________________________________________
batch_normalization_180 (BatchN (None, 2, 2, 320)    960         conv2d_180[0][0]                 
__________________________________________________________________________________________________
activation_182 (Activation)     (None, 2, 2, 384)    0           batch_normalization_182[0][0]    
__________________________________________________________________________________________________
activation_183 (Activation)     (None, 2, 2, 384)    0           batch_normalization_183[0][0]    
__________________________________________________________________________________________________
activation_186 (Activation)     (None, 2, 2, 384)    0           batch_normalization_186[0][0]    
__________________________________________________________________________________________________
activation_187 (Activation)     (None, 2, 2, 384)    0           batch_normalization_187[0][0]    
__________________________________________________________________________________________________
batch_normalization_188 (BatchN (None, 2, 2, 192)    576         conv2d_188[0][0]                 
__________________________________________________________________________________________________
activation_180 (Activation)     (None, 2, 2, 320)    0           batch_normalization_180[0][0]    
__________________________________________________________________________________________________
mixed9_1 (Concatenate)          (None, 2, 2, 768)    0           activation_182[0][0]             
                                                                 activation_183[0][0]             
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 2, 2, 768)    0           activation_186[0][0]             
                                                                 activation_187[0][0]             
__________________________________________________________________________________________________
activation_188 (Activation)     (None, 2, 2, 192)    0           batch_normalization_188[0][0]    
__________________________________________________________________________________________________
mixed10 (Concatenate)           (None, 2, 2, 2048)   0           activation_180[0][0]             
                                                                 mixed9_1[0][0]                   
                                                                 concatenate_4[0][0]              
                                                                 activation_188[0][0]             
==================================================================================================
Total params: 21,802,784
Trainable params: 21,768,352
Non-trainable params: 34,432
__________________________________________________________________________________________________
for layer in base_model.layers:
    if layer.name == 'conv2d_470':
        break
    layer.trainable = False
    print('Layer ' + layer.name + ' frozen.')
    
# We add our classificator (top_model) to the last layer of the model
last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(1024, activation='relu', name='fc1')(x)
#x = Dropout(0.3)(x)
x = Dense(512, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', name='fc3')(x)
#x = Dropout(0.5)(x)
x = Dense(128, activation='relu', name='fc4')(x)
x = Dense(64, activation='relu', name='fc5')(x)
x = Dense(8, activation='softmax', name='predictions')(x)
model = Model(base_model.input, x)
# We compile the model
model.compile(optimizer=Adam(lr=0.003), loss='categorical_crossentropy', metrics=['accuracy'])
# We see the new structure of the model
model.summary()    
Layer input_2 frozen.
Layer conv2d_95 frozen.
Layer batch_normalization_95 frozen.
Layer activation_95 frozen.
Layer conv2d_96 frozen.
Layer batch_normalization_96 frozen.
Layer activation_96 frozen.
Layer conv2d_97 frozen.
Layer batch_normalization_97 frozen.
Layer activation_97 frozen.
Layer max_pooling2d_5 frozen.
Layer conv2d_98 frozen.
Layer batch_normalization_98 frozen.
Layer activation_98 frozen.
Layer conv2d_99 frozen.
Layer batch_normalization_99 frozen.
Layer activation_99 frozen.
Layer max_pooling2d_6 frozen.
Layer conv2d_103 frozen.
Layer batch_normalization_103 frozen.
Layer activation_103 frozen.
Layer conv2d_101 frozen.
Layer conv2d_104 frozen.
Layer batch_normalization_101 frozen.
Layer batch_normalization_104 frozen.
Layer activation_101 frozen.
Layer activation_104 frozen.
Layer average_pooling2d_10 frozen.
Layer conv2d_100 frozen.
Layer conv2d_102 frozen.
Layer conv2d_105 frozen.
Layer conv2d_106 frozen.
Layer batch_normalization_100 frozen.
Layer batch_normalization_102 frozen.
Layer batch_normalization_105 frozen.
Layer batch_normalization_106 frozen.
Layer activation_100 frozen.
Layer activation_102 frozen.
Layer activation_105 frozen.
Layer activation_106 frozen.
Layer mixed0 frozen.
Layer conv2d_110 frozen.
Layer batch_normalization_110 frozen.
Layer activation_110 frozen.
Layer conv2d_108 frozen.
Layer conv2d_111 frozen.
Layer batch_normalization_108 frozen.
Layer batch_normalization_111 frozen.
Layer activation_108 frozen.
Layer activation_111 frozen.
Layer average_pooling2d_11 frozen.
Layer conv2d_107 frozen.
Layer conv2d_109 frozen.
Layer conv2d_112 frozen.
Layer conv2d_113 frozen.
Layer batch_normalization_107 frozen.
Layer batch_normalization_109 frozen.
Layer batch_normalization_112 frozen.
Layer batch_normalization_113 frozen.
Layer activation_107 frozen.
Layer activation_109 frozen.
Layer activation_112 frozen.
Layer activation_113 frozen.
Layer mixed1 frozen.
Layer conv2d_117 frozen.
Layer batch_normalization_117 frozen.
Layer activation_117 frozen.
Layer conv2d_115 frozen.
Layer conv2d_118 frozen.
Layer batch_normalization_115 frozen.
Layer batch_normalization_118 frozen.
Layer activation_115 frozen.
Layer activation_118 frozen.
Layer average_pooling2d_12 frozen.
Layer conv2d_114 frozen.
Layer conv2d_116 frozen.
Layer conv2d_119 frozen.
Layer conv2d_120 frozen.
Layer batch_normalization_114 frozen.
Layer batch_normalization_116 frozen.
Layer batch_normalization_119 frozen.
Layer batch_normalization_120 frozen.
Layer activation_114 frozen.
Layer activation_116 frozen.
Layer activation_119 frozen.
Layer activation_120 frozen.
Layer mixed2 frozen.
Layer conv2d_122 frozen.
Layer batch_normalization_122 frozen.
Layer activation_122 frozen.
Layer conv2d_123 frozen.
Layer batch_normalization_123 frozen.
Layer activation_123 frozen.
Layer conv2d_121 frozen.
Layer conv2d_124 frozen.
Layer batch_normalization_121 frozen.
Layer batch_normalization_124 frozen.
Layer activation_121 frozen.
Layer activation_124 frozen.
Layer max_pooling2d_7 frozen.
Layer mixed3 frozen.
Layer conv2d_129 frozen.
Layer batch_normalization_129 frozen.
Layer activation_129 frozen.
Layer conv2d_130 frozen.
Layer batch_normalization_130 frozen.
Layer activation_130 frozen.
Layer conv2d_126 frozen.
Layer conv2d_131 frozen.
Layer batch_normalization_126 frozen.
Layer batch_normalization_131 frozen.
Layer activation_126 frozen.
Layer activation_131 frozen.
Layer conv2d_127 frozen.
Layer conv2d_132 frozen.
Layer batch_normalization_127 frozen.
Layer batch_normalization_132 frozen.
Layer activation_127 frozen.
Layer activation_132 frozen.
Layer average_pooling2d_13 frozen.
Layer conv2d_125 frozen.
Layer conv2d_128 frozen.
Layer conv2d_133 frozen.
Layer conv2d_134 frozen.
Layer batch_normalization_125 frozen.
Layer batch_normalization_128 frozen.
Layer batch_normalization_133 frozen.
Layer batch_normalization_134 frozen.
Layer activation_125 frozen.
Layer activation_128 frozen.
Layer activation_133 frozen.
Layer activation_134 frozen.
Layer mixed4 frozen.
Layer conv2d_139 frozen.
Layer batch_normalization_139 frozen.
Layer activation_139 frozen.
Layer conv2d_140 frozen.
Layer batch_normalization_140 frozen.
Layer activation_140 frozen.
Layer conv2d_136 frozen.
Layer conv2d_141 frozen.
Layer batch_normalization_136 frozen.
Layer batch_normalization_141 frozen.
Layer activation_136 frozen.
Layer activation_141 frozen.
Layer conv2d_137 frozen.
Layer conv2d_142 frozen.
Layer batch_normalization_137 frozen.
Layer batch_normalization_142 frozen.
Layer activation_137 frozen.
Layer activation_142 frozen.
Layer average_pooling2d_14 frozen.
Layer conv2d_135 frozen.
Layer conv2d_138 frozen.
Layer conv2d_143 frozen.
Layer conv2d_144 frozen.
Layer batch_normalization_135 frozen.
Layer batch_normalization_138 frozen.
Layer batch_normalization_143 frozen.
Layer batch_normalization_144 frozen.
Layer activation_135 frozen.
Layer activation_138 frozen.
Layer activation_143 frozen.
Layer activation_144 frozen.
Layer mixed5 frozen.
Layer conv2d_149 frozen.
Layer batch_normalization_149 frozen.
Layer activation_149 frozen.
Layer conv2d_150 frozen.
Layer batch_normalization_150 frozen.
Layer activation_150 frozen.
Layer conv2d_146 frozen.
Layer conv2d_151 frozen.
Layer batch_normalization_146 frozen.
Layer batch_normalization_151 frozen.
Layer activation_146 frozen.
Layer activation_151 frozen.
Layer conv2d_147 frozen.
Layer conv2d_152 frozen.
Layer batch_normalization_147 frozen.
Layer batch_normalization_152 frozen.
Layer activation_147 frozen.
Layer activation_152 frozen.
Layer average_pooling2d_15 frozen.
Layer conv2d_145 frozen.
Layer conv2d_148 frozen.
Layer conv2d_153 frozen.
Layer conv2d_154 frozen.
Layer batch_normalization_145 frozen.
Layer batch_normalization_148 frozen.
Layer batch_normalization_153 frozen.
Layer batch_normalization_154 frozen.
Layer activation_145 frozen.
Layer activation_148 frozen.
Layer activation_153 frozen.
Layer activation_154 frozen.
Layer mixed6 frozen.
Layer conv2d_159 frozen.
Layer batch_normalization_159 frozen.
Layer activation_159 frozen.
Layer conv2d_160 frozen.
Layer batch_normalization_160 frozen.
Layer activation_160 frozen.
Layer conv2d_156 frozen.
Layer conv2d_161 frozen.
Layer batch_normalization_156 frozen.
Layer batch_normalization_161 frozen.
Layer activation_156 frozen.
Layer activation_161 frozen.
Layer conv2d_157 frozen.
Layer conv2d_162 frozen.
Layer batch_normalization_157 frozen.
Layer batch_normalization_162 frozen.
Layer activation_157 frozen.
Layer activation_162 frozen.
Layer average_pooling2d_16 frozen.
Layer conv2d_155 frozen.
Layer conv2d_158 frozen.
Layer conv2d_163 frozen.
Layer conv2d_164 frozen.
Layer batch_normalization_155 frozen.
Layer batch_normalization_158 frozen.
Layer batch_normalization_163 frozen.
Layer batch_normalization_164 frozen.
Layer activation_155 frozen.
Layer activation_158 frozen.
Layer activation_163 frozen.
Layer activation_164 frozen.
Layer mixed7 frozen.
Layer conv2d_167 frozen.
Layer batch_normalization_167 frozen.
Layer activation_167 frozen.
Layer conv2d_168 frozen.
Layer batch_normalization_168 frozen.
Layer activation_168 frozen.
Layer conv2d_165 frozen.
Layer conv2d_169 frozen.
Layer batch_normalization_165 frozen.
Layer batch_normalization_169 frozen.
Layer activation_165 frozen.
Layer activation_169 frozen.
Layer conv2d_166 frozen.
Layer conv2d_170 frozen.
Layer batch_normalization_166 frozen.
Layer batch_normalization_170 frozen.
Layer activation_166 frozen.
Layer activation_170 frozen.
Layer max_pooling2d_8 frozen.
Layer mixed8 frozen.
Layer conv2d_175 frozen.
Layer batch_normalization_175 frozen.
Layer activation_175 frozen.
Layer conv2d_172 frozen.
Layer conv2d_176 frozen.
Layer batch_normalization_172 frozen.
Layer batch_normalization_176 frozen.
Layer activation_172 frozen.
Layer activation_176 frozen.
Layer conv2d_173 frozen.
Layer conv2d_174 frozen.
Layer conv2d_177 frozen.
Layer conv2d_178 frozen.
Layer average_pooling2d_17 frozen.
Layer conv2d_171 frozen.
Layer batch_normalization_173 frozen.
Layer batch_normalization_174 frozen.
Layer batch_normalization_177 frozen.
Layer batch_normalization_178 frozen.
Layer conv2d_179 frozen.
Layer batch_normalization_171 frozen.
Layer activation_173 frozen.
Layer activation_174 frozen.
Layer activation_177 frozen.
Layer activation_178 frozen.
Layer batch_normalization_179 frozen.
Layer activation_171 frozen.
Layer mixed9_0 frozen.
Layer concatenate_3 frozen.
Layer activation_179 frozen.
Layer mixed9 frozen.
Layer conv2d_184 frozen.
Layer batch_normalization_184 frozen.
Layer activation_184 frozen.
Layer conv2d_181 frozen.
Layer conv2d_185 frozen.
Layer batch_normalization_181 frozen.
Layer batch_normalization_185 frozen.
Layer activation_181 frozen.
Layer activation_185 frozen.
Layer conv2d_182 frozen.
Layer conv2d_183 frozen.
Layer conv2d_186 frozen.
Layer conv2d_187 frozen.
Layer average_pooling2d_18 frozen.
Layer conv2d_180 frozen.
Layer batch_normalization_182 frozen.
Layer batch_normalization_183 frozen.
Layer batch_normalization_186 frozen.
Layer batch_normalization_187 frozen.
Layer conv2d_188 frozen.
Layer batch_normalization_180 frozen.
Layer activation_182 frozen.
Layer activation_183 frozen.
Layer activation_186 frozen.
Layer activation_187 frozen.
Layer batch_normalization_188 frozen.
Layer activation_180 frozen.
Layer mixed9_1 frozen.
Layer concatenate_4 frozen.
Layer activation_188 frozen.
Layer mixed10 frozen.
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            (None, 128, 128, 3)  0                                            
__________________________________________________________________________________________________
conv2d_95 (Conv2D)              (None, 63, 63, 32)   864         input_2[0][0]                    
__________________________________________________________________________________________________
batch_normalization_95 (BatchNo (None, 63, 63, 32)   96          conv2d_95[0][0]                  
__________________________________________________________________________________________________
activation_95 (Activation)      (None, 63, 63, 32)   0           batch_normalization_95[0][0]     
__________________________________________________________________________________________________
conv2d_96 (Conv2D)              (None, 61, 61, 32)   9216        activation_95[0][0]              
__________________________________________________________________________________________________
batch_normalization_96 (BatchNo (None, 61, 61, 32)   96          conv2d_96[0][0]                  
__________________________________________________________________________________________________
activation_96 (Activation)      (None, 61, 61, 32)   0           batch_normalization_96[0][0]     
__________________________________________________________________________________________________
conv2d_97 (Conv2D)              (None, 61, 61, 64)   18432       activation_96[0][0]              
__________________________________________________________________________________________________
batch_normalization_97 (BatchNo (None, 61, 61, 64)   192         conv2d_97[0][0]                  
__________________________________________________________________________________________________
activation_97 (Activation)      (None, 61, 61, 64)   0           batch_normalization_97[0][0]     
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 30, 30, 64)   0           activation_97[0][0]              
__________________________________________________________________________________________________
conv2d_98 (Conv2D)              (None, 30, 30, 80)   5120        max_pooling2d_5[0][0]            
__________________________________________________________________________________________________
batch_normalization_98 (BatchNo (None, 30, 30, 80)   240         conv2d_98[0][0]                  
__________________________________________________________________________________________________
activation_98 (Activation)      (None, 30, 30, 80)   0           batch_normalization_98[0][0]     
__________________________________________________________________________________________________
conv2d_99 (Conv2D)              (None, 28, 28, 192)  138240      activation_98[0][0]              
__________________________________________________________________________________________________
batch_normalization_99 (BatchNo (None, 28, 28, 192)  576         conv2d_99[0][0]                  
__________________________________________________________________________________________________
activation_99 (Activation)      (None, 28, 28, 192)  0           batch_normalization_99[0][0]     
__________________________________________________________________________________________________
max_pooling2d_6 (MaxPooling2D)  (None, 13, 13, 192)  0           activation_99[0][0]              
__________________________________________________________________________________________________
conv2d_103 (Conv2D)             (None, 13, 13, 64)   12288       max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
batch_normalization_103 (BatchN (None, 13, 13, 64)   192         conv2d_103[0][0]                 
__________________________________________________________________________________________________
activation_103 (Activation)     (None, 13, 13, 64)   0           batch_normalization_103[0][0]    
__________________________________________________________________________________________________
conv2d_101 (Conv2D)             (None, 13, 13, 48)   9216        max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
conv2d_104 (Conv2D)             (None, 13, 13, 96)   55296       activation_103[0][0]             
__________________________________________________________________________________________________
batch_normalization_101 (BatchN (None, 13, 13, 48)   144         conv2d_101[0][0]                 
__________________________________________________________________________________________________
batch_normalization_104 (BatchN (None, 13, 13, 96)   288         conv2d_104[0][0]                 
__________________________________________________________________________________________________
activation_101 (Activation)     (None, 13, 13, 48)   0           batch_normalization_101[0][0]    
__________________________________________________________________________________________________
activation_104 (Activation)     (None, 13, 13, 96)   0           batch_normalization_104[0][0]    
__________________________________________________________________________________________________
average_pooling2d_10 (AveragePo (None, 13, 13, 192)  0           max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
conv2d_100 (Conv2D)             (None, 13, 13, 64)   12288       max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
conv2d_102 (Conv2D)             (None, 13, 13, 64)   76800       activation_101[0][0]             
__________________________________________________________________________________________________
conv2d_105 (Conv2D)             (None, 13, 13, 96)   82944       activation_104[0][0]             
__________________________________________________________________________________________________
conv2d_106 (Conv2D)             (None, 13, 13, 32)   6144        average_pooling2d_10[0][0]       
__________________________________________________________________________________________________
batch_normalization_100 (BatchN (None, 13, 13, 64)   192         conv2d_100[0][0]                 
__________________________________________________________________________________________________
batch_normalization_102 (BatchN (None, 13, 13, 64)   192         conv2d_102[0][0]                 
__________________________________________________________________________________________________
batch_normalization_105 (BatchN (None, 13, 13, 96)   288         conv2d_105[0][0]                 
__________________________________________________________________________________________________
batch_normalization_106 (BatchN (None, 13, 13, 32)   96          conv2d_106[0][0]                 
__________________________________________________________________________________________________
activation_100 (Activation)     (None, 13, 13, 64)   0           batch_normalization_100[0][0]    
__________________________________________________________________________________________________
activation_102 (Activation)     (None, 13, 13, 64)   0           batch_normalization_102[0][0]    
__________________________________________________________________________________________________
activation_105 (Activation)     (None, 13, 13, 96)   0           batch_normalization_105[0][0]    
__________________________________________________________________________________________________
activation_106 (Activation)     (None, 13, 13, 32)   0           batch_normalization_106[0][0]    
__________________________________________________________________________________________________
mixed0 (Concatenate)            (None, 13, 13, 256)  0           activation_100[0][0]             
                                                                 activation_102[0][0]             
                                                                 activation_105[0][0]             
                                                                 activation_106[0][0]             
__________________________________________________________________________________________________
conv2d_110 (Conv2D)             (None, 13, 13, 64)   16384       mixed0[0][0]                     
__________________________________________________________________________________________________
batch_normalization_110 (BatchN (None, 13, 13, 64)   192         conv2d_110[0][0]                 
__________________________________________________________________________________________________
activation_110 (Activation)     (None, 13, 13, 64)   0           batch_normalization_110[0][0]    
__________________________________________________________________________________________________
conv2d_108 (Conv2D)             (None, 13, 13, 48)   12288       mixed0[0][0]                     
__________________________________________________________________________________________________
conv2d_111 (Conv2D)             (None, 13, 13, 96)   55296       activation_110[0][0]             
__________________________________________________________________________________________________
batch_normalization_108 (BatchN (None, 13, 13, 48)   144         conv2d_108[0][0]                 
__________________________________________________________________________________________________
batch_normalization_111 (BatchN (None, 13, 13, 96)   288         conv2d_111[0][0]                 
__________________________________________________________________________________________________
activation_108 (Activation)     (None, 13, 13, 48)   0           batch_normalization_108[0][0]    
__________________________________________________________________________________________________
activation_111 (Activation)     (None, 13, 13, 96)   0           batch_normalization_111[0][0]    
__________________________________________________________________________________________________
average_pooling2d_11 (AveragePo (None, 13, 13, 256)  0           mixed0[0][0]                     
__________________________________________________________________________________________________
conv2d_107 (Conv2D)             (None, 13, 13, 64)   16384       mixed0[0][0]                     
__________________________________________________________________________________________________
conv2d_109 (Conv2D)             (None, 13, 13, 64)   76800       activation_108[0][0]             
__________________________________________________________________________________________________
conv2d_112 (Conv2D)             (None, 13, 13, 96)   82944       activation_111[0][0]             
__________________________________________________________________________________________________
conv2d_113 (Conv2D)             (None, 13, 13, 64)   16384       average_pooling2d_11[0][0]       
__________________________________________________________________________________________________
batch_normalization_107 (BatchN (None, 13, 13, 64)   192         conv2d_107[0][0]                 
__________________________________________________________________________________________________
batch_normalization_109 (BatchN (None, 13, 13, 64)   192         conv2d_109[0][0]                 
__________________________________________________________________________________________________
batch_normalization_112 (BatchN (None, 13, 13, 96)   288         conv2d_112[0][0]                 
__________________________________________________________________________________________________
batch_normalization_113 (BatchN (None, 13, 13, 64)   192         conv2d_113[0][0]                 
__________________________________________________________________________________________________
activation_107 (Activation)     (None, 13, 13, 64)   0           batch_normalization_107[0][0]    
__________________________________________________________________________________________________
activation_109 (Activation)     (None, 13, 13, 64)   0           batch_normalization_109[0][0]    
__________________________________________________________________________________________________
activation_112 (Activation)     (None, 13, 13, 96)   0           batch_normalization_112[0][0]    
__________________________________________________________________________________________________
activation_113 (Activation)     (None, 13, 13, 64)   0           batch_normalization_113[0][0]    
__________________________________________________________________________________________________
mixed1 (Concatenate)            (None, 13, 13, 288)  0           activation_107[0][0]             
                                                                 activation_109[0][0]             
                                                                 activation_112[0][0]             
                                                                 activation_113[0][0]             
__________________________________________________________________________________________________
conv2d_117 (Conv2D)             (None, 13, 13, 64)   18432       mixed1[0][0]                     
__________________________________________________________________________________________________
batch_normalization_117 (BatchN (None, 13, 13, 64)   192         conv2d_117[0][0]                 
__________________________________________________________________________________________________
activation_117 (Activation)     (None, 13, 13, 64)   0           batch_normalization_117[0][0]    
__________________________________________________________________________________________________
conv2d_115 (Conv2D)             (None, 13, 13, 48)   13824       mixed1[0][0]                     
__________________________________________________________________________________________________
conv2d_118 (Conv2D)             (None, 13, 13, 96)   55296       activation_117[0][0]             
__________________________________________________________________________________________________
batch_normalization_115 (BatchN (None, 13, 13, 48)   144         conv2d_115[0][0]                 
__________________________________________________________________________________________________
batch_normalization_118 (BatchN (None, 13, 13, 96)   288         conv2d_118[0][0]                 
__________________________________________________________________________________________________
activation_115 (Activation)     (None, 13, 13, 48)   0           batch_normalization_115[0][0]    
__________________________________________________________________________________________________
activation_118 (Activation)     (None, 13, 13, 96)   0           batch_normalization_118[0][0]    
__________________________________________________________________________________________________
average_pooling2d_12 (AveragePo (None, 13, 13, 288)  0           mixed1[0][0]                     
__________________________________________________________________________________________________
conv2d_114 (Conv2D)             (None, 13, 13, 64)   18432       mixed1[0][0]                     
__________________________________________________________________________________________________
conv2d_116 (Conv2D)             (None, 13, 13, 64)   76800       activation_115[0][0]             
__________________________________________________________________________________________________
conv2d_119 (Conv2D)             (None, 13, 13, 96)   82944       activation_118[0][0]             
__________________________________________________________________________________________________
conv2d_120 (Conv2D)             (None, 13, 13, 64)   18432       average_pooling2d_12[0][0]       
__________________________________________________________________________________________________
batch_normalization_114 (BatchN (None, 13, 13, 64)   192         conv2d_114[0][0]                 
__________________________________________________________________________________________________
batch_normalization_116 (BatchN (None, 13, 13, 64)   192         conv2d_116[0][0]                 
__________________________________________________________________________________________________
batch_normalization_119 (BatchN (None, 13, 13, 96)   288         conv2d_119[0][0]                 
__________________________________________________________________________________________________
batch_normalization_120 (BatchN (None, 13, 13, 64)   192         conv2d_120[0][0]                 
__________________________________________________________________________________________________
activation_114 (Activation)     (None, 13, 13, 64)   0           batch_normalization_114[0][0]    
__________________________________________________________________________________________________
activation_116 (Activation)     (None, 13, 13, 64)   0           batch_normalization_116[0][0]    
__________________________________________________________________________________________________
activation_119 (Activation)     (None, 13, 13, 96)   0           batch_normalization_119[0][0]    
__________________________________________________________________________________________________
activation_120 (Activation)     (None, 13, 13, 64)   0           batch_normalization_120[0][0]    
__________________________________________________________________________________________________
mixed2 (Concatenate)            (None, 13, 13, 288)  0           activation_114[0][0]             
                                                                 activation_116[0][0]             
                                                                 activation_119[0][0]             
                                                                 activation_120[0][0]             
__________________________________________________________________________________________________
conv2d_122 (Conv2D)             (None, 13, 13, 64)   18432       mixed2[0][0]                     
__________________________________________________________________________________________________
batch_normalization_122 (BatchN (None, 13, 13, 64)   192         conv2d_122[0][0]                 
__________________________________________________________________________________________________
activation_122 (Activation)     (None, 13, 13, 64)   0           batch_normalization_122[0][0]    
__________________________________________________________________________________________________
conv2d_123 (Conv2D)             (None, 13, 13, 96)   55296       activation_122[0][0]             
__________________________________________________________________________________________________
batch_normalization_123 (BatchN (None, 13, 13, 96)   288         conv2d_123[0][0]                 
__________________________________________________________________________________________________
activation_123 (Activation)     (None, 13, 13, 96)   0           batch_normalization_123[0][0]    
__________________________________________________________________________________________________
conv2d_121 (Conv2D)             (None, 6, 6, 384)    995328      mixed2[0][0]                     
__________________________________________________________________________________________________
conv2d_124 (Conv2D)             (None, 6, 6, 96)     82944       activation_123[0][0]             
__________________________________________________________________________________________________
batch_normalization_121 (BatchN (None, 6, 6, 384)    1152        conv2d_121[0][0]                 
__________________________________________________________________________________________________
batch_normalization_124 (BatchN (None, 6, 6, 96)     288         conv2d_124[0][0]                 
__________________________________________________________________________________________________
activation_121 (Activation)     (None, 6, 6, 384)    0           batch_normalization_121[0][0]    
__________________________________________________________________________________________________
activation_124 (Activation)     (None, 6, 6, 96)     0           batch_normalization_124[0][0]    
__________________________________________________________________________________________________
max_pooling2d_7 (MaxPooling2D)  (None, 6, 6, 288)    0           mixed2[0][0]                     
__________________________________________________________________________________________________
mixed3 (Concatenate)            (None, 6, 6, 768)    0           activation_121[0][0]             
                                                                 activation_124[0][0]             
                                                                 max_pooling2d_7[0][0]            
__________________________________________________________________________________________________
conv2d_129 (Conv2D)             (None, 6, 6, 128)    98304       mixed3[0][0]                     
__________________________________________________________________________________________________
batch_normalization_129 (BatchN (None, 6, 6, 128)    384         conv2d_129[0][0]                 
__________________________________________________________________________________________________
activation_129 (Activation)     (None, 6, 6, 128)    0           batch_normalization_129[0][0]    
__________________________________________________________________________________________________
conv2d_130 (Conv2D)             (None, 6, 6, 128)    114688      activation_129[0][0]             
__________________________________________________________________________________________________
batch_normalization_130 (BatchN (None, 6, 6, 128)    384         conv2d_130[0][0]                 
__________________________________________________________________________________________________
activation_130 (Activation)     (None, 6, 6, 128)    0           batch_normalization_130[0][0]    
__________________________________________________________________________________________________
conv2d_126 (Conv2D)             (None, 6, 6, 128)    98304       mixed3[0][0]                     
__________________________________________________________________________________________________
conv2d_131 (Conv2D)             (None, 6, 6, 128)    114688      activation_130[0][0]             
__________________________________________________________________________________________________
batch_normalization_126 (BatchN (None, 6, 6, 128)    384         conv2d_126[0][0]                 
__________________________________________________________________________________________________
batch_normalization_131 (BatchN (None, 6, 6, 128)    384         conv2d_131[0][0]                 
__________________________________________________________________________________________________
activation_126 (Activation)     (None, 6, 6, 128)    0           batch_normalization_126[0][0]    
__________________________________________________________________________________________________
activation_131 (Activation)     (None, 6, 6, 128)    0           batch_normalization_131[0][0]    
__________________________________________________________________________________________________
conv2d_127 (Conv2D)             (None, 6, 6, 128)    114688      activation_126[0][0]             
__________________________________________________________________________________________________
conv2d_132 (Conv2D)             (None, 6, 6, 128)    114688      activation_131[0][0]             
__________________________________________________________________________________________________
batch_normalization_127 (BatchN (None, 6, 6, 128)    384         conv2d_127[0][0]                 
__________________________________________________________________________________________________
batch_normalization_132 (BatchN (None, 6, 6, 128)    384         conv2d_132[0][0]                 
__________________________________________________________________________________________________
activation_127 (Activation)     (None, 6, 6, 128)    0           batch_normalization_127[0][0]    
__________________________________________________________________________________________________
activation_132 (Activation)     (None, 6, 6, 128)    0           batch_normalization_132[0][0]    
__________________________________________________________________________________________________
average_pooling2d_13 (AveragePo (None, 6, 6, 768)    0           mixed3[0][0]                     
__________________________________________________________________________________________________
conv2d_125 (Conv2D)             (None, 6, 6, 192)    147456      mixed3[0][0]                     
__________________________________________________________________________________________________
conv2d_128 (Conv2D)             (None, 6, 6, 192)    172032      activation_127[0][0]             
__________________________________________________________________________________________________
conv2d_133 (Conv2D)             (None, 6, 6, 192)    172032      activation_132[0][0]             
__________________________________________________________________________________________________
conv2d_134 (Conv2D)             (None, 6, 6, 192)    147456      average_pooling2d_13[0][0]       
__________________________________________________________________________________________________
batch_normalization_125 (BatchN (None, 6, 6, 192)    576         conv2d_125[0][0]                 
__________________________________________________________________________________________________
batch_normalization_128 (BatchN (None, 6, 6, 192)    576         conv2d_128[0][0]                 
__________________________________________________________________________________________________
batch_normalization_133 (BatchN (None, 6, 6, 192)    576         conv2d_133[0][0]                 
__________________________________________________________________________________________________
batch_normalization_134 (BatchN (None, 6, 6, 192)    576         conv2d_134[0][0]                 
__________________________________________________________________________________________________
activation_125 (Activation)     (None, 6, 6, 192)    0           batch_normalization_125[0][0]    
__________________________________________________________________________________________________
activation_128 (Activation)     (None, 6, 6, 192)    0           batch_normalization_128[0][0]    
__________________________________________________________________________________________________
activation_133 (Activation)     (None, 6, 6, 192)    0           batch_normalization_133[0][0]    
__________________________________________________________________________________________________
activation_134 (Activation)     (None, 6, 6, 192)    0           batch_normalization_134[0][0]    
__________________________________________________________________________________________________
mixed4 (Concatenate)            (None, 6, 6, 768)    0           activation_125[0][0]             
                                                                 activation_128[0][0]             
                                                                 activation_133[0][0]             
                                                                 activation_134[0][0]             
__________________________________________________________________________________________________
conv2d_139 (Conv2D)             (None, 6, 6, 160)    122880      mixed4[0][0]                     
__________________________________________________________________________________________________
batch_normalization_139 (BatchN (None, 6, 6, 160)    480         conv2d_139[0][0]                 
__________________________________________________________________________________________________
activation_139 (Activation)     (None, 6, 6, 160)    0           batch_normalization_139[0][0]    
__________________________________________________________________________________________________
conv2d_140 (Conv2D)             (None, 6, 6, 160)    179200      activation_139[0][0]             
__________________________________________________________________________________________________
batch_normalization_140 (BatchN (None, 6, 6, 160)    480         conv2d_140[0][0]                 
__________________________________________________________________________________________________
activation_140 (Activation)     (None, 6, 6, 160)    0           batch_normalization_140[0][0]    
__________________________________________________________________________________________________
conv2d_136 (Conv2D)             (None, 6, 6, 160)    122880      mixed4[0][0]                     
__________________________________________________________________________________________________
conv2d_141 (Conv2D)             (None, 6, 6, 160)    179200      activation_140[0][0]             
__________________________________________________________________________________________________
batch_normalization_136 (BatchN (None, 6, 6, 160)    480         conv2d_136[0][0]                 
__________________________________________________________________________________________________
batch_normalization_141 (BatchN (None, 6, 6, 160)    480         conv2d_141[0][0]                 
__________________________________________________________________________________________________
activation_136 (Activation)     (None, 6, 6, 160)    0           batch_normalization_136[0][0]    
__________________________________________________________________________________________________
activation_141 (Activation)     (None, 6, 6, 160)    0           batch_normalization_141[0][0]    
__________________________________________________________________________________________________
conv2d_137 (Conv2D)             (None, 6, 6, 160)    179200      activation_136[0][0]             
__________________________________________________________________________________________________
conv2d_142 (Conv2D)             (None, 6, 6, 160)    179200      activation_141[0][0]             
__________________________________________________________________________________________________
batch_normalization_137 (BatchN (None, 6, 6, 160)    480         conv2d_137[0][0]                 
__________________________________________________________________________________________________
batch_normalization_142 (BatchN (None, 6, 6, 160)    480         conv2d_142[0][0]                 
__________________________________________________________________________________________________
activation_137 (Activation)     (None, 6, 6, 160)    0           batch_normalization_137[0][0]    
__________________________________________________________________________________________________
activation_142 (Activation)     (None, 6, 6, 160)    0           batch_normalization_142[0][0]    
__________________________________________________________________________________________________
average_pooling2d_14 (AveragePo (None, 6, 6, 768)    0           mixed4[0][0]                     
__________________________________________________________________________________________________
conv2d_135 (Conv2D)             (None, 6, 6, 192)    147456      mixed4[0][0]                     
__________________________________________________________________________________________________
conv2d_138 (Conv2D)             (None, 6, 6, 192)    215040      activation_137[0][0]             
__________________________________________________________________________________________________
conv2d_143 (Conv2D)             (None, 6, 6, 192)    215040      activation_142[0][0]             
__________________________________________________________________________________________________
conv2d_144 (Conv2D)             (None, 6, 6, 192)    147456      average_pooling2d_14[0][0]       
__________________________________________________________________________________________________
batch_normalization_135 (BatchN (None, 6, 6, 192)    576         conv2d_135[0][0]                 
__________________________________________________________________________________________________
batch_normalization_138 (BatchN (None, 6, 6, 192)    576         conv2d_138[0][0]                 
__________________________________________________________________________________________________
batch_normalization_143 (BatchN (None, 6, 6, 192)    576         conv2d_143[0][0]                 
__________________________________________________________________________________________________
batch_normalization_144 (BatchN (None, 6, 6, 192)    576         conv2d_144[0][0]                 
__________________________________________________________________________________________________
activation_135 (Activation)     (None, 6, 6, 192)    0           batch_normalization_135[0][0]    
__________________________________________________________________________________________________
activation_138 (Activation)     (None, 6, 6, 192)    0           batch_normalization_138[0][0]    
__________________________________________________________________________________________________
activation_143 (Activation)     (None, 6, 6, 192)    0           batch_normalization_143[0][0]    
__________________________________________________________________________________________________
activation_144 (Activation)     (None, 6, 6, 192)    0           batch_normalization_144[0][0]    
__________________________________________________________________________________________________
mixed5 (Concatenate)            (None, 6, 6, 768)    0           activation_135[0][0]             
                                                                 activation_138[0][0]             
                                                                 activation_143[0][0]             
                                                                 activation_144[0][0]             
__________________________________________________________________________________________________
conv2d_149 (Conv2D)             (None, 6, 6, 160)    122880      mixed5[0][0]                     
__________________________________________________________________________________________________
batch_normalization_149 (BatchN (None, 6, 6, 160)    480         conv2d_149[0][0]                 
__________________________________________________________________________________________________
activation_149 (Activation)     (None, 6, 6, 160)    0           batch_normalization_149[0][0]    
__________________________________________________________________________________________________
conv2d_150 (Conv2D)             (None, 6, 6, 160)    179200      activation_149[0][0]             
__________________________________________________________________________________________________
batch_normalization_150 (BatchN (None, 6, 6, 160)    480         conv2d_150[0][0]                 
__________________________________________________________________________________________________
activation_150 (Activation)     (None, 6, 6, 160)    0           batch_normalization_150[0][0]    
__________________________________________________________________________________________________
conv2d_146 (Conv2D)             (None, 6, 6, 160)    122880      mixed5[0][0]                     
__________________________________________________________________________________________________
conv2d_151 (Conv2D)             (None, 6, 6, 160)    179200      activation_150[0][0]             
__________________________________________________________________________________________________
batch_normalization_146 (BatchN (None, 6, 6, 160)    480         conv2d_146[0][0]                 
__________________________________________________________________________________________________
batch_normalization_151 (BatchN (None, 6, 6, 160)    480         conv2d_151[0][0]                 
__________________________________________________________________________________________________
activation_146 (Activation)     (None, 6, 6, 160)    0           batch_normalization_146[0][0]    
__________________________________________________________________________________________________
activation_151 (Activation)     (None, 6, 6, 160)    0           batch_normalization_151[0][0]    
__________________________________________________________________________________________________
conv2d_147 (Conv2D)             (None, 6, 6, 160)    179200      activation_146[0][0]             
__________________________________________________________________________________________________
conv2d_152 (Conv2D)             (None, 6, 6, 160)    179200      activation_151[0][0]             
__________________________________________________________________________________________________
batch_normalization_147 (BatchN (None, 6, 6, 160)    480         conv2d_147[0][0]                 
__________________________________________________________________________________________________
batch_normalization_152 (BatchN (None, 6, 6, 160)    480         conv2d_152[0][0]                 
__________________________________________________________________________________________________
activation_147 (Activation)     (None, 6, 6, 160)    0           batch_normalization_147[0][0]    
__________________________________________________________________________________________________
activation_152 (Activation)     (None, 6, 6, 160)    0           batch_normalization_152[0][0]    
__________________________________________________________________________________________________
average_pooling2d_15 (AveragePo (None, 6, 6, 768)    0           mixed5[0][0]                     
__________________________________________________________________________________________________
conv2d_145 (Conv2D)             (None, 6, 6, 192)    147456      mixed5[0][0]                     
__________________________________________________________________________________________________
conv2d_148 (Conv2D)             (None, 6, 6, 192)    215040      activation_147[0][0]             
__________________________________________________________________________________________________
conv2d_153 (Conv2D)             (None, 6, 6, 192)    215040      activation_152[0][0]             
__________________________________________________________________________________________________
conv2d_154 (Conv2D)             (None, 6, 6, 192)    147456      average_pooling2d_15[0][0]       
__________________________________________________________________________________________________
batch_normalization_145 (BatchN (None, 6, 6, 192)    576         conv2d_145[0][0]                 
__________________________________________________________________________________________________
batch_normalization_148 (BatchN (None, 6, 6, 192)    576         conv2d_148[0][0]                 
__________________________________________________________________________________________________
batch_normalization_153 (BatchN (None, 6, 6, 192)    576         conv2d_153[0][0]                 
__________________________________________________________________________________________________
batch_normalization_154 (BatchN (None, 6, 6, 192)    576         conv2d_154[0][0]                 
__________________________________________________________________________________________________
activation_145 (Activation)     (None, 6, 6, 192)    0           batch_normalization_145[0][0]    
__________________________________________________________________________________________________
activation_148 (Activation)     (None, 6, 6, 192)    0           batch_normalization_148[0][0]    
__________________________________________________________________________________________________
activation_153 (Activation)     (None, 6, 6, 192)    0           batch_normalization_153[0][0]    
__________________________________________________________________________________________________
activation_154 (Activation)     (None, 6, 6, 192)    0           batch_normalization_154[0][0]    
__________________________________________________________________________________________________
mixed6 (Concatenate)            (None, 6, 6, 768)    0           activation_145[0][0]             
                                                                 activation_148[0][0]             
                                                                 activation_153[0][0]             
                                                                 activation_154[0][0]             
__________________________________________________________________________________________________
conv2d_159 (Conv2D)             (None, 6, 6, 192)    147456      mixed6[0][0]                     
__________________________________________________________________________________________________
batch_normalization_159 (BatchN (None, 6, 6, 192)    576         conv2d_159[0][0]                 
__________________________________________________________________________________________________
activation_159 (Activation)     (None, 6, 6, 192)    0           batch_normalization_159[0][0]    
__________________________________________________________________________________________________
conv2d_160 (Conv2D)             (None, 6, 6, 192)    258048      activation_159[0][0]             
__________________________________________________________________________________________________
batch_normalization_160 (BatchN (None, 6, 6, 192)    576         conv2d_160[0][0]                 
__________________________________________________________________________________________________
activation_160 (Activation)     (None, 6, 6, 192)    0           batch_normalization_160[0][0]    
__________________________________________________________________________________________________
conv2d_156 (Conv2D)             (None, 6, 6, 192)    147456      mixed6[0][0]                     
__________________________________________________________________________________________________
conv2d_161 (Conv2D)             (None, 6, 6, 192)    258048      activation_160[0][0]             
__________________________________________________________________________________________________
batch_normalization_156 (BatchN (None, 6, 6, 192)    576         conv2d_156[0][0]                 
__________________________________________________________________________________________________
batch_normalization_161 (BatchN (None, 6, 6, 192)    576         conv2d_161[0][0]                 
__________________________________________________________________________________________________
activation_156 (Activation)     (None, 6, 6, 192)    0           batch_normalization_156[0][0]    
__________________________________________________________________________________________________
activation_161 (Activation)     (None, 6, 6, 192)    0           batch_normalization_161[0][0]    
__________________________________________________________________________________________________
conv2d_157 (Conv2D)             (None, 6, 6, 192)    258048      activation_156[0][0]             
__________________________________________________________________________________________________
conv2d_162 (Conv2D)             (None, 6, 6, 192)    258048      activation_161[0][0]             
__________________________________________________________________________________________________
batch_normalization_157 (BatchN (None, 6, 6, 192)    576         conv2d_157[0][0]                 
__________________________________________________________________________________________________
batch_normalization_162 (BatchN (None, 6, 6, 192)    576         conv2d_162[0][0]                 
__________________________________________________________________________________________________
activation_157 (Activation)     (None, 6, 6, 192)    0           batch_normalization_157[0][0]    
__________________________________________________________________________________________________
activation_162 (Activation)     (None, 6, 6, 192)    0           batch_normalization_162[0][0]    
__________________________________________________________________________________________________
average_pooling2d_16 (AveragePo (None, 6, 6, 768)    0           mixed6[0][0]                     
__________________________________________________________________________________________________
conv2d_155 (Conv2D)             (None, 6, 6, 192)    147456      mixed6[0][0]                     
__________________________________________________________________________________________________
conv2d_158 (Conv2D)             (None, 6, 6, 192)    258048      activation_157[0][0]             
__________________________________________________________________________________________________
conv2d_163 (Conv2D)             (None, 6, 6, 192)    258048      activation_162[0][0]             
__________________________________________________________________________________________________
conv2d_164 (Conv2D)             (None, 6, 6, 192)    147456      average_pooling2d_16[0][0]       
__________________________________________________________________________________________________
batch_normalization_155 (BatchN (None, 6, 6, 192)    576         conv2d_155[0][0]                 
__________________________________________________________________________________________________
batch_normalization_158 (BatchN (None, 6, 6, 192)    576         conv2d_158[0][0]                 
__________________________________________________________________________________________________
batch_normalization_163 (BatchN (None, 6, 6, 192)    576         conv2d_163[0][0]                 
__________________________________________________________________________________________________
batch_normalization_164 (BatchN (None, 6, 6, 192)    576         conv2d_164[0][0]                 
__________________________________________________________________________________________________
activation_155 (Activation)     (None, 6, 6, 192)    0           batch_normalization_155[0][0]    
__________________________________________________________________________________________________
activation_158 (Activation)     (None, 6, 6, 192)    0           batch_normalization_158[0][0]    
__________________________________________________________________________________________________
activation_163 (Activation)     (None, 6, 6, 192)    0           batch_normalization_163[0][0]    
__________________________________________________________________________________________________
activation_164 (Activation)     (None, 6, 6, 192)    0           batch_normalization_164[0][0]    
__________________________________________________________________________________________________
mixed7 (Concatenate)            (None, 6, 6, 768)    0           activation_155[0][0]             
                                                                 activation_158[0][0]             
                                                                 activation_163[0][0]             
                                                                 activation_164[0][0]             
__________________________________________________________________________________________________
conv2d_167 (Conv2D)             (None, 6, 6, 192)    147456      mixed7[0][0]                     
__________________________________________________________________________________________________
batch_normalization_167 (BatchN (None, 6, 6, 192)    576         conv2d_167[0][0]                 
__________________________________________________________________________________________________
activation_167 (Activation)     (None, 6, 6, 192)    0           batch_normalization_167[0][0]    
__________________________________________________________________________________________________
conv2d_168 (Conv2D)             (None, 6, 6, 192)    258048      activation_167[0][0]             
__________________________________________________________________________________________________
batch_normalization_168 (BatchN (None, 6, 6, 192)    576         conv2d_168[0][0]                 
__________________________________________________________________________________________________
activation_168 (Activation)     (None, 6, 6, 192)    0           batch_normalization_168[0][0]    
__________________________________________________________________________________________________
conv2d_165 (Conv2D)             (None, 6, 6, 192)    147456      mixed7[0][0]                     
__________________________________________________________________________________________________
conv2d_169 (Conv2D)             (None, 6, 6, 192)    258048      activation_168[0][0]             
__________________________________________________________________________________________________
batch_normalization_165 (BatchN (None, 6, 6, 192)    576         conv2d_165[0][0]                 
__________________________________________________________________________________________________
batch_normalization_169 (BatchN (None, 6, 6, 192)    576         conv2d_169[0][0]                 
__________________________________________________________________________________________________
activation_165 (Activation)     (None, 6, 6, 192)    0           batch_normalization_165[0][0]    
__________________________________________________________________________________________________
activation_169 (Activation)     (None, 6, 6, 192)    0           batch_normalization_169[0][0]    
__________________________________________________________________________________________________
conv2d_166 (Conv2D)             (None, 2, 2, 320)    552960      activation_165[0][0]             
__________________________________________________________________________________________________
conv2d_170 (Conv2D)             (None, 2, 2, 192)    331776      activation_169[0][0]             
__________________________________________________________________________________________________
batch_normalization_166 (BatchN (None, 2, 2, 320)    960         conv2d_166[0][0]                 
__________________________________________________________________________________________________
batch_normalization_170 (BatchN (None, 2, 2, 192)    576         conv2d_170[0][0]                 
__________________________________________________________________________________________________
activation_166 (Activation)     (None, 2, 2, 320)    0           batch_normalization_166[0][0]    
__________________________________________________________________________________________________
activation_170 (Activation)     (None, 2, 2, 192)    0           batch_normalization_170[0][0]    
__________________________________________________________________________________________________
max_pooling2d_8 (MaxPooling2D)  (None, 2, 2, 768)    0           mixed7[0][0]                     
__________________________________________________________________________________________________
mixed8 (Concatenate)            (None, 2, 2, 1280)   0           activation_166[0][0]             
                                                                 activation_170[0][0]             
                                                                 max_pooling2d_8[0][0]            
__________________________________________________________________________________________________
conv2d_175 (Conv2D)             (None, 2, 2, 448)    573440      mixed8[0][0]                     
__________________________________________________________________________________________________
batch_normalization_175 (BatchN (None, 2, 2, 448)    1344        conv2d_175[0][0]                 
__________________________________________________________________________________________________
activation_175 (Activation)     (None, 2, 2, 448)    0           batch_normalization_175[0][0]    
__________________________________________________________________________________________________
conv2d_172 (Conv2D)             (None, 2, 2, 384)    491520      mixed8[0][0]                     
__________________________________________________________________________________________________
conv2d_176 (Conv2D)             (None, 2, 2, 384)    1548288     activation_175[0][0]             
__________________________________________________________________________________________________
batch_normalization_172 (BatchN (None, 2, 2, 384)    1152        conv2d_172[0][0]                 
__________________________________________________________________________________________________
batch_normalization_176 (BatchN (None, 2, 2, 384)    1152        conv2d_176[0][0]                 
__________________________________________________________________________________________________
activation_172 (Activation)     (None, 2, 2, 384)    0           batch_normalization_172[0][0]    
__________________________________________________________________________________________________
activation_176 (Activation)     (None, 2, 2, 384)    0           batch_normalization_176[0][0]    
__________________________________________________________________________________________________
conv2d_173 (Conv2D)             (None, 2, 2, 384)    442368      activation_172[0][0]             
__________________________________________________________________________________________________
conv2d_174 (Conv2D)             (None, 2, 2, 384)    442368      activation_172[0][0]             
__________________________________________________________________________________________________
conv2d_177 (Conv2D)             (None, 2, 2, 384)    442368      activation_176[0][0]             
__________________________________________________________________________________________________
conv2d_178 (Conv2D)             (None, 2, 2, 384)    442368      activation_176[0][0]             
__________________________________________________________________________________________________
average_pooling2d_17 (AveragePo (None, 2, 2, 1280)   0           mixed8[0][0]                     
__________________________________________________________________________________________________
conv2d_171 (Conv2D)             (None, 2, 2, 320)    409600      mixed8[0][0]                     
__________________________________________________________________________________________________
batch_normalization_173 (BatchN (None, 2, 2, 384)    1152        conv2d_173[0][0]                 
__________________________________________________________________________________________________
batch_normalization_174 (BatchN (None, 2, 2, 384)    1152        conv2d_174[0][0]                 
__________________________________________________________________________________________________
batch_normalization_177 (BatchN (None, 2, 2, 384)    1152        conv2d_177[0][0]                 
__________________________________________________________________________________________________
batch_normalization_178 (BatchN (None, 2, 2, 384)    1152        conv2d_178[0][0]                 
__________________________________________________________________________________________________
conv2d_179 (Conv2D)             (None, 2, 2, 192)    245760      average_pooling2d_17[0][0]       
__________________________________________________________________________________________________
batch_normalization_171 (BatchN (None, 2, 2, 320)    960         conv2d_171[0][0]                 
__________________________________________________________________________________________________
activation_173 (Activation)     (None, 2, 2, 384)    0           batch_normalization_173[0][0]    
__________________________________________________________________________________________________
activation_174 (Activation)     (None, 2, 2, 384)    0           batch_normalization_174[0][0]    
__________________________________________________________________________________________________
activation_177 (Activation)     (None, 2, 2, 384)    0           batch_normalization_177[0][0]    
__________________________________________________________________________________________________
activation_178 (Activation)     (None, 2, 2, 384)    0           batch_normalization_178[0][0]    
__________________________________________________________________________________________________
batch_normalization_179 (BatchN (None, 2, 2, 192)    576         conv2d_179[0][0]                 
__________________________________________________________________________________________________
activation_171 (Activation)     (None, 2, 2, 320)    0           batch_normalization_171[0][0]    
__________________________________________________________________________________________________
mixed9_0 (Concatenate)          (None, 2, 2, 768)    0           activation_173[0][0]             
                                                                 activation_174[0][0]             
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 2, 2, 768)    0           activation_177[0][0]             
                                                                 activation_178[0][0]             
__________________________________________________________________________________________________
activation_179 (Activation)     (None, 2, 2, 192)    0           batch_normalization_179[0][0]    
__________________________________________________________________________________________________
mixed9 (Concatenate)            (None, 2, 2, 2048)   0           activation_171[0][0]             
                                                                 mixed9_0[0][0]                   
                                                                 concatenate_3[0][0]              
                                                                 activation_179[0][0]             
__________________________________________________________________________________________________
conv2d_184 (Conv2D)             (None, 2, 2, 448)    917504      mixed9[0][0]                     
__________________________________________________________________________________________________
batch_normalization_184 (BatchN (None, 2, 2, 448)    1344        conv2d_184[0][0]                 
__________________________________________________________________________________________________
activation_184 (Activation)     (None, 2, 2, 448)    0           batch_normalization_184[0][0]    
__________________________________________________________________________________________________
conv2d_181 (Conv2D)             (None, 2, 2, 384)    786432      mixed9[0][0]                     
__________________________________________________________________________________________________
conv2d_185 (Conv2D)             (None, 2, 2, 384)    1548288     activation_184[0][0]             
__________________________________________________________________________________________________
batch_normalization_181 (BatchN (None, 2, 2, 384)    1152        conv2d_181[0][0]                 
__________________________________________________________________________________________________
batch_normalization_185 (BatchN (None, 2, 2, 384)    1152        conv2d_185[0][0]                 
__________________________________________________________________________________________________
activation_181 (Activation)     (None, 2, 2, 384)    0           batch_normalization_181[0][0]    
__________________________________________________________________________________________________
activation_185 (Activation)     (None, 2, 2, 384)    0           batch_normalization_185[0][0]    
__________________________________________________________________________________________________
conv2d_182 (Conv2D)             (None, 2, 2, 384)    442368      activation_181[0][0]             
__________________________________________________________________________________________________
conv2d_183 (Conv2D)             (None, 2, 2, 384)    442368      activation_181[0][0]             
__________________________________________________________________________________________________
conv2d_186 (Conv2D)             (None, 2, 2, 384)    442368      activation_185[0][0]             
__________________________________________________________________________________________________
conv2d_187 (Conv2D)             (None, 2, 2, 384)    442368      activation_185[0][0]             
__________________________________________________________________________________________________
average_pooling2d_18 (AveragePo (None, 2, 2, 2048)   0           mixed9[0][0]                     
__________________________________________________________________________________________________
conv2d_180 (Conv2D)             (None, 2, 2, 320)    655360      mixed9[0][0]                     
__________________________________________________________________________________________________
batch_normalization_182 (BatchN (None, 2, 2, 384)    1152        conv2d_182[0][0]                 
__________________________________________________________________________________________________
batch_normalization_183 (BatchN (None, 2, 2, 384)    1152        conv2d_183[0][0]                 
__________________________________________________________________________________________________
batch_normalization_186 (BatchN (None, 2, 2, 384)    1152        conv2d_186[0][0]                 
__________________________________________________________________________________________________
batch_normalization_187 (BatchN (None, 2, 2, 384)    1152        conv2d_187[0][0]                 
__________________________________________________________________________________________________
conv2d_188 (Conv2D)             (None, 2, 2, 192)    393216      average_pooling2d_18[0][0]       
__________________________________________________________________________________________________
batch_normalization_180 (BatchN (None, 2, 2, 320)    960         conv2d_180[0][0]                 
__________________________________________________________________________________________________
activation_182 (Activation)     (None, 2, 2, 384)    0           batch_normalization_182[0][0]    
__________________________________________________________________________________________________
activation_183 (Activation)     (None, 2, 2, 384)    0           batch_normalization_183[0][0]    
__________________________________________________________________________________________________
activation_186 (Activation)     (None, 2, 2, 384)    0           batch_normalization_186[0][0]    
__________________________________________________________________________________________________
activation_187 (Activation)     (None, 2, 2, 384)    0           batch_normalization_187[0][0]    
__________________________________________________________________________________________________
batch_normalization_188 (BatchN (None, 2, 2, 192)    576         conv2d_188[0][0]                 
__________________________________________________________________________________________________
activation_180 (Activation)     (None, 2, 2, 320)    0           batch_normalization_180[0][0]    
__________________________________________________________________________________________________
mixed9_1 (Concatenate)          (None, 2, 2, 768)    0           activation_182[0][0]             
                                                                 activation_183[0][0]             
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 2, 2, 768)    0           activation_186[0][0]             
                                                                 activation_187[0][0]             
__________________________________________________________________________________________________
activation_188 (Activation)     (None, 2, 2, 192)    0           batch_normalization_188[0][0]    
__________________________________________________________________________________________________
mixed10 (Concatenate)           (None, 2, 2, 2048)   0           activation_180[0][0]             
                                                                 mixed9_1[0][0]                   
                                                                 concatenate_4[0][0]              
                                                                 activation_188[0][0]             
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 8192)         0           mixed10[0][0]                    
__________________________________________________________________________________________________
fc1 (Dense)                     (None, 1024)         8389632     flatten_2[0][0]                  
__________________________________________________________________________________________________
fc2 (Dense)                     (None, 512)          524800      fc1[0][0]                        
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 512)          0           fc2[0][0]                        
__________________________________________________________________________________________________
fc3 (Dense)                     (None, 512)          262656      dropout_2[0][0]                  
__________________________________________________________________________________________________
fc4 (Dense)                     (None, 128)          65664       fc3[0][0]                        
__________________________________________________________________________________________________
fc5 (Dense)                     (None, 64)           8256        fc4[0][0]                        
__________________________________________________________________________________________________
predictions (Dense)             (None, 8)            520         fc5[0][0]                        
==================================================================================================
Total params: 31,054,312
Trainable params: 9,251,528
Non-trainable params: 21,802,784
__________________________________________________________________________________________________
epochs = 15

batch_size = 64
# We train it
model.fit(X_train, Y_train,
          batch_size=batch_size,
          validation_data=(X_test, Y_test),
          epochs=epochs,shuffle=True)
Train on 1200 samples, validate on 360 samples
Epoch 1/15
1200/1200 [==============================] - 32s 27ms/step - loss: 3.4204 - accuracy: 0.1283 - val_loss: 15.1587 - val_accuracy: 0.1250
Epoch 2/15
1200/1200 [==============================] - 28s 23ms/step - loss: 2.0880 - accuracy: 0.1417 - val_loss: 5.7659 - val_accuracy: 0.1139
Epoch 3/15
1200/1200 [==============================] - 28s 23ms/step - loss: 2.0338 - accuracy: 0.1900 - val_loss: 17.5912 - val_accuracy: 0.1139
Epoch 4/15
1200/1200 [==============================] - 28s 23ms/step - loss: 1.8787 - accuracy: 0.2558 - val_loss: 39.2187 - val_accuracy: 0.1222
Epoch 5/15
1200/1200 [==============================] - 28s 23ms/step - loss: 1.7239 - accuracy: 0.2992 - val_loss: 21.2076 - val_accuracy: 0.0861
Epoch 6/15
1200/1200 [==============================] - 28s 23ms/step - loss: 1.5872 - accuracy: 0.3642 - val_loss: 49.5472 - val_accuracy: 0.0833
Epoch 7/15
1200/1200 [==============================] - 28s 23ms/step - loss: 1.4617 - accuracy: 0.4167 - val_loss: 41.7643 - val_accuracy: 0.1306
Epoch 8/15
1200/1200 [==============================] - 28s 23ms/step - loss: 1.4780 - accuracy: 0.4050 - val_loss: 43.7105 - val_accuracy: 0.1000
Epoch 9/15
1200/1200 [==============================] - 28s 23ms/step - loss: 1.3503 - accuracy: 0.4675 - val_loss: 47.9649 - val_accuracy: 0.1361
Epoch 10/15
1200/1200 [==============================] - 28s 24ms/step - loss: 1.3008 - accuracy: 0.5092 - val_loss: 47.1353 - val_accuracy: 0.1083
Epoch 11/15
1200/1200 [==============================] - 27s 23ms/step - loss: 1.2868 - accuracy: 0.4942 - val_loss: 69.8077 - val_accuracy: 0.0944
Epoch 12/15
1200/1200 [==============================] - 28s 23ms/step - loss: 1.1645 - accuracy: 0.5525 - val_loss: 49.5942 - val_accuracy: 0.0722
Epoch 13/15
1200/1200 [==============================] - 28s 23ms/step - loss: 1.0783 - accuracy: 0.5933 - val_loss: 90.5733 - val_accuracy: 0.1000
Epoch 14/15
1200/1200 [==============================] - 29s 24ms/step - loss: 1.0821 - accuracy: 0.6233 - val_loss: 76.6727 - val_accuracy: 0.0861
Epoch 15/15
1200/1200 [==============================] - 29s 24ms/step - loss: 0.9863 - accuracy: 0.6375 - val_loss: 89.5056 - val_accuracy: 0.1389
<keras.callbacks.callbacks.History at 0x257dd049388>
 
 