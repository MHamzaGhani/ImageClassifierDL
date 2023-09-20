import os
from PIL import Image
import imghdr
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


class data_ingestion():
    def __init__(self,directory):
        self.directory=directory

    def data_ingest(self):
        for folder in os.listdir(self.directory):
            for image in os.listdir(os.path.join(self.directory,folder)):
                image_path = os.path.join(self.directory,folder,image)
                
                try:
                    img = Image.open(image_path)
                    img.verify()
                    tip=imghdr.what(image_path)
                    if tip not in ['jpeg','jpg','bmp','png']:
                        print('image is not in extension list {}'.format(image_path))
                        os.remove(image_path)
                except Exception as e:
                    print("issue with the image {}" .format(image_path))
                    os.remove(image_path)        

        data = tf.keras.utils.image_dataset_from_directory('data')
        data = data.map(lambda x,y : (x/255 ,y))
        return data


class train_test_split():
    def __init__(self,data):
        self.data=data

    def tts(self):
        train_size = int(len(self.data)*.7)
        test_size = int(len(self.data)* .1)+1
        val_size = int(len(self.data)* .2)

        train_data = self.data.take(train_size)
        val_data = self.data.skip(train_size).take(val_size)
        test_data = self.data.skip(train_size+val_size).take(test_size)

        return train_data, val_data, test_data
    

class model_building():
    def model_compile(self):
        model =Sequential()

        model.add(Conv2D(16,(3,3),1,activation = 'relu' , input_shape = (256,256,3)))
        model.add(MaxPooling2D())

        model.add(Conv2D(32,(3,3),1,activation='relu'))
        model.add(MaxPooling2D())

        model.add(Conv2D(16,(3,3),1,activation ='relu'))
        model.add(MaxPooling2D())

        model.add(Flatten())

        model.add(Dense(256,activation='relu'))
        model.add(Dense(1,activation='sigmoid'))
            
        model.compile('adam',loss = tf.losses.BinaryCrossentropy(),metrics=['accuracy'])
        return model

class model_training():
    def __init__(self, model , train_data , val_data):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

    def trained_model(self):
        hist=self.model.fit(self.train_data,epochs=20,validation_data=self.val_data)

        fig = plt.figure()
        plt.plot(hist.history['loss'], color='teal', label='loss')
        plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()
    
        
        fig = plt.figure()
        plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
        plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()
        self.model.save(os.path.join('models','imageclassifier.h5'))


class prediction():
    def __init__(self,model,path_to_image):
        self.model = model
        self.path =path_to_image

    def predict(self):
        img = cv2.imread(self.path)
        resize = tf.image.resize(img, (256,256))
        yhat=self.model.predict(np.expand_dims(resize/255,0))
        if yhat > 0.5: 
            print(f'Predicted class is Sad')
        else:
            print(f'Predicted class is Happy')


    