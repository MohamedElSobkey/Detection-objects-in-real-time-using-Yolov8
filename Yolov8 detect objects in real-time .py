# deeper look into Working with results
import torch
import numpy as np 
from time import time 
import cv2
from ultralytics import YOLO


class ObjectDetection:
    def __init__ (self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda. is_available() else 'cpu'
        print('Using device : ', self.device)
                
        self.model = self.load_model()
        
        
    def load_model(self):
        model = YOLO("yolov8m.pt")
        model.fuse()
        return model
        
    def predict(self, frame):
        results = self.model(frame)
        return results
    
    def plot_boxes(self, results, frame):
        xyxys= []
        confidence = []
        class_ids = []
        
        # extract detections for person class
        for results in results:
            boxes = results.boxes.cpu().numpy()
            
            # xyxys = boxes.xyxy
            
            # for xyxy in xyxys :
                
            #     cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1]) , int(xyxy[2]), int(xyxy[3]) ), (0,255,0))
            
            
            
            xyxys.append(boxes.xyxy)
            confidence.append(boxes.conf)
            class_ids.append(boxes.cls)
            frame = results[0].plot()
            
            
        print(confidence)
        print(class_ids)
            
            
        return frame
         
     
    def __call__(self): 
        
        cap = cv2.VideoCapture(self.capture_index) 
        assert cap.isOpened()
        
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH , 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True :
            start_time = time()
            
            ret,frame = cap.read()
            
            assert ret
            
            results = self.predict(frame)
            frame = self.plot_boxes(results, frame)
            
            end_time = time()
            
            fps = 1/ np.round (end_time - start_time , 2)
            
            cv2.putText(frame, f'FPS : {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection' , frame)
            
            if cv2.waitKey(5) & 0xFF == 27 :
                
                
                break 
            
            
        cap.release() 
        cv2.destroyAllWindows()


detector = ObjectDetection(0)
detector()         