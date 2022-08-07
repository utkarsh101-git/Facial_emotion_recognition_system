try:
    import cv2
    import time
    import ImageClass, DLmodel
    import numpy as np


except Exception as e:
    print(f"Some dependencies were not found : {e}")


class System(ImageClass.Image, DLmodel.DLModel, DLmodel.FaceDetector):
    def __init__(self):
        print("System loading Detector and model")
        self.state=False
        self.refreshRate=7
        self.classes = {0:'angry',
        1:'disgusted',
        2:'fearful',
        3:'happy',
        4:'neutral',
        5:'sad',
        6:'surprised'}
        self.setDetector("haarcascade_frontalface_default.xml")
        print("Detector loaded")
        self.setModel("increased_acc_82_66.h5")
        print("Dl model loaded")
        print("getting camera ready")
        self.cap = cv2.VideoCapture(0)
        print("camera ready")
        print("good to go")

    def release_camera(self):
        self.cap.release()

    def markImage(self, img):
        temp = []
        self.img=img
        faces = self.detect_faces(img)

        for face in faces:
            x,y,w,h=face
            test_img = self.getFaceFromImage(face)

            processed_img = self.preprocessForModel(test_img)
            res = self.predict(processed_img)
            
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (36,255,12), 1)
            cv2.putText(self.img, res["emotion"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)   
            temp.append((face, res["emotion"]))
        return self.img, temp


    def manual_prediction(self, filepath):
        cv2.imshow("emotion",self.markImage(self.load_img(filepath))[0])
        cv2.waitKey(0);
        cv2.destroyAllWindows()

    def runMainLoop(self):
        ctr=0
        flag=0
        while(True):
            ret, frame = self.cap.read()
            if(ret==False):
                break
                          
            if(ctr!=self.refreshRate & flag==1): #new line added
                #flag=1  #new line added
                for cords in temp:
                    x,y,w,h=cords
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 1)
                    cv2.putText(frame, temp[cords], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                   

            else: # if ctr==refresh rate
                frame, temp = self.markImage(frame)
                ctr=0
                flag=1 #new line added

            cv2.imshow("camera", frame)
            ctr+=1 
            
            if(cv2.waitKey(1)==ord('q') ):
                break

        #cap.release()
        cv2.destroyAllWindows() 

    def captureFromCamera(self):
       
        #cap = cv2.VideoCapture(0)
        t1=time.time()

        flag=0
        while(True):
            ret, frame = self.cap.read()
            copy=frame.copy()
            if(ret==False):
                break
    
            t2=time.time()
    
            cv2.putText(frame,str(int(t2-t1)),(250,250), cv2.FONT_HERSHEY_DUPLEX, 4,(255,0,0),2)

            cv2.imshow("camera",frame)
            if(cv2.waitKey(1)==ord('q') ):
                break
    
            if(int(t2-t1)==5):
                recorded_frame=copy
                flag=1
                break
        
        #cap.release()
        cv2.destroyAllWindows()

        if(flag==1):
            cv2.imshow("recorded_image", recorded_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
   
            faces = self.detect_faces(recorded_frame)
            print(f"faces detected: {len(faces)}")
            marked_img, _ = self.markImage(recorded_frame)
            print(_)
            cv2.imshow("marked_img",marked_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
