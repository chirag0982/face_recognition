import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model, Sequential
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))
from keras.models import model_from_json
model.load_weights('vgg_face_weights.h5')
vgg_face_descriptor = Model(inputs=model.layers[0].input
, outputs=model.layers[-1].output)
def only_face(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('img',img)
        ##cv2.waitKey(0)
    return img   
def preprocess_image(image):
    ##img = load_img(image_path, target_size=(224, 224))
    img = cv2.resize(image, dsize =(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
def accuracy(data , actualdata, data1, actualdata1):
    j=0
    k=0
    l=0
    for i in range(len(data)):
        if data1[i]==actualdata1[i]=="Not in the database":
            j = j+1
            k = k + 1
        if data1[i]==actualdata1[i]=="In the data base":
            l = l + 1
            if data[i]==actualdata[i]:
                j = j+1
        
    return 100*(j/len(data)) , (100*k)/60 , (100*l)/140
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
 
 
 
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
def verifyFace(img1_representation, img2_representation):
    ##EuclideanDistance = findEuclideanDistance(img1_representation,img2_representation)
    ##return EuclideanDistance
    cosine_distance = findCosineDistance(img1_representation, img2_representation)
    return cosine_distance
 
def who_is_it(image_path, database):
    min_dist = 100
    img1_representation = vgg_face_descriptor.predict(preprocess_image(only_face(image_path)))[0,:]
    for name in database :
        dist = verifyFace(img1_representation, database[name])
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.8:
        return "Not in the database" ,identity ,min_dist
    
    else:
        return   "In the data base" ,identity , min_dist
        
    
database = {}
##database["Adriana Lima"] = vgg_face_descriptor.predict(preprocess_image("Desktop/Data Base/Adriana Lima.jpg"))[0,:]
database["Emma Watson"] =  vgg_face_descriptor.predict(preprocess_image(only_face("Desktop/Data Base/Emma Watson.jpg")))[0,:]
database["grant gustin"] = vgg_face_descriptor.predict(preprocess_image(only_face("Desktop/Data Base/grant gustin.jpg")))[0,:]
##database["Inbar Lavi"] = vgg_face_descriptor.predict(preprocess_image("Desktop/Data Base/Inbar Lavi.jpg"))[0,:]
database["Natalie Dormer"] =vgg_face_descriptor.predict(preprocess_image(only_face("Desktop/Data Base/Natalie Dormer.jpg")))[0,:]
##database["margot robbie"] =vgg_face_descriptor.predict(preprocess_image("Desktop/Data Base/margot robbie.jpg"))[0,:]
##database["Tom Holland"] =vgg_face_descriptor.predict(preprocess_image("Desktop/Data Base/Tom Holland.jpg"))[0,:]
##database["Tom Cruise"] = vgg_face_descriptor.predict(preprocess_image("Desktop/Data Base/Tom Cruise.jpg"))[0,:]
##database["amber heard"] =vgg_face_descriptor.predict(preprocess_image("Desktop/Data Base/amber heard.jpg"))[0,:]
database["Anthony Mackie"] = vgg_face_descriptor.predict(preprocess_image(only_face("Desktop/Data Base/Anthony Mackie.jpg")))[0,:]
##database["barack obama"] =vgg_face_descriptor.predict(preprocess_image( "Desktop/Data Base/barack obama.jpg"))[0,:]
##database["barbara palvin"] =vgg_face_descriptor.predict(preprocess_image("Desktop/Data Base/barbara palvin.jpg"))[0,:]
database["Bill Gates"] =vgg_face_descriptor.predict(preprocess_image(only_face("Desktop/Data Base/Bill Gates.jpg")))[0,:]
##database["Cristiano Ronaldo"] =vgg_face_descriptor.predict(preprocess_image("Desktop/Data Base/Cristiano Ronaldo.jpg"))[0,:]
database["Jessica Barden"] =vgg_face_descriptor.predict(preprocess_image(only_face("Desktop/Data Base/Jessica Barden.jpg")))[0,:]
##database["Sanskar"]=vgg_face_descriptor.predict(preprocess_image("Desktop/Data Base/Sanskar.jpeg"))[0,:]
database["Sophie Turner"]=vgg_face_descriptor.predict(preprocess_image(only_face("Desktop/Data Base/Sophie Turner.jpg")))[0,:]
print(who_is_it("Desktop/WhatsApp Image 2020-06-23 at 02.56.57.jpeg",database))
     
data = []
data1 = []
for i in range(20):
    a ,b, c = who_is_it("Desktop/DataSet1/105_classes_pins_dataset/pins_Anthony Mackie/Anthony ("+str(i+1)+").jpg",database)
    data.append(b)
    data1.append(a)
for i in range(20):
    a ,b, c = who_is_it("Desktop/DataSet1/105_classes_pins_dataset/pins_Jessica Barden/Jessica Barden ("+str(i+1)+").jpg",database)
    data.append(b)
    data1.append(a)
for i in range(20):
    a ,b, c = who_is_it("Desktop/DataSet1/105_classes_pins_dataset/pins_Natalie Dormer/Natalie Dormer ("+str(i+1)+").jpg",database)
    data.append(b)
    data1.append(a)
for i in range(20):
    a ,b, c = who_is_it("Desktop/DataSet1/105_classes_pins_dataset/pins_Emma Watson/Emma Watson ("+str(i+1)+").jpg",database)
    data.append(b)
    data1.append(a)
for i in range(20):
    a ,b, c = who_is_it("Desktop/DataSet1/105_classes_pins_dataset/pins_grant gustin/grant gustin ("+str(i+1)+").jpg",database)
    data.append(b)
    data1.append(a)
for i in range(20):
    a ,b, c = who_is_it("Desktop/DataSet1/105_classes_pins_dataset/pins_Sophie Turner/Sophie Turner ("+str(i+1)+").jpg",database)
    data.append(b)
    data1.append(a)
for i in range(20):
    a ,b, c = who_is_it("Desktop/DataSet1/105_classes_pins_dataset/pins_Bill Gates/Bill Gates ("+str(i+1)+").jpg",database)
    data.append(b)
    data1.append(a)
for i in range(20):
    a ,b, c = who_is_it("Desktop/DataSet1/105_classes_pins_dataset/pins_Tom Holland/Tom Holland ("+str(i+1)+").jpg",database)
    data.append(b)
    data1.append(a)
for i in range(20):
    a ,b, c = who_is_it("Desktop/DataSet1/105_classes_pins_dataset/pins_Cristiano Ronaldo/Cristiano Ronaldo ("+str(i+1)+").jpg",database)
    data.append(b)
    data1.append(a)
for i in range(20):
    a ,b, c = who_is_it("Desktop/DataSet1/105_classes_pins_dataset/pins_margot robbie/margot robbie ("+str(i+1)+").jpg",database)
    data.append(b)
    data1.append(a)
##def accuracy(data , actualdata):
    ##j=0
    ##for i in range(len(data)):
        ##if data[i]==actualdata[i]:
            ##j = j+1
    ##return 100*(j/len(data))        
 
(69.5, 100.0, 58.57142857142857)
actualdata = []
for i in range(20):
    actualdata.append("Anthony Mackie")
for i in range(20):
    actualdata.append("Jessica Barden")
for i in range(20):
    actualdata.append("Natalie Dormer")
for i in range(20):
    actualdata.append("Emma Watson")
for i in range(20):
    actualdata.append("grant gustin")
for i in range(20):
    actualdata.append("Sophie Turner")
for i in range(20):
    actualdata.append("Bill Gates")
for i in range(20):
    actualdata.append("Tom Holland")
for i in range(20):
    actualdata.append("Cristiano Ronaldo")
for i in range(20):
    actualdata.append("margot robbie")
actualdata1 = []
for i in range(len(data1)-60):
    actualdata1.append("In the data base")
for i in range(60):
    actualdata1.append("Not in the database")
print(accuracy(data , actualdata, data1, actualdata1))
#(72.5, 98.33333333333333, 62.142857142857146)
 
 
 
 
 
 