try:
    import cv2
    import numpy as np

except Exception as e:
    print(f"Some dependencies were not found : {e}")

class Image():
    def __init__(self):
        pass

    def load_img(self, path):
        return cv2.imread(path)   

    def setImg(self, img):
        self.img=img 

    def preprocessForModel(self, img):
        img = cv2.resize(img, (48,48))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)/255.0

    def preprocessForDetector(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def getFaceFromImage(self, face):
            x,y,w,h=face
            cropped_img = self.img[y:y+h,x:x+w]
            return cropped_img
