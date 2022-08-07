
try:    
    import numpy as np
    import cv2
    import os
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    
except Exception as e:
    print(f"Some dependencies were not found : {e}")

class DLModel():
    def __init__(self):
        print("constructor of DL model called")
        
    
    def setModel(self, weights):
        ip = layers.Input(shape=(48,48,1))
        x = layers.Conv2D(filters=32,kernel_size=3,padding='same', activation='relu' )(ip)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=64,kernel_size=3,padding='same', activation='relu' )(x)
        x = layers.BatchNormalization()(x)  
        x = layers.MaxPooling2D(pool_size=(2,2) )(x)

        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(filters=128,kernel_size=3,padding='same', activation="relu" )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=128,kernel_size=3,padding='same', activation="relu" )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2,2) )(x)

        x = layers.Conv2D(filters=256,kernel_size=3,padding='same', activation="relu" )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters=256,kernel_size=3,padding='same', activation="relu" )(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dropout(0.25)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(516, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        op = layers.Dense(7, activation='sigmoid')(x)



        model3 = keras.Model(inputs=ip, outputs=op)
        model3.load_weights(weights)
        self.classifier = model3


    def predict(self,processed_img):
        
        res = self.classifier.predict(processed_img)
        # print(res)
        # print(res.dtype)
        res = np.squeeze(res)
        # print(res)
        # print(res.shape)
        print(self.classes.get(np.argmax(res)))
        #print(f"probability scores {res}")
        #print(f"detected emotion: {classes.get(np.argmax(res))}")
        return {"scores":res,"emotion": self.classes.get(np.argmax(res))}


class FaceDetector():
    def __init__(self) -> None:
        print("constructor of Face detector called")

    def setDetector(self, filename):
        self.faceDetector = cv2.CascadeClassifier(filename)

    def detect_faces(self, img):
        gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces=self.faceDetector.detectMultiScale(gray_img, 1.3, 6)
        return faces    
