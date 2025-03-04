import os
import cv2
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from keras.api.models import load_model

class CameraVision:
    def __init__(self, path) -> None:
        self.model = load_model(filepath=path)
        self.cam = cv2.VideoCapture(0)
        self.predict = ""

    def show(self):
        while (self.cam.isOpened()):
            ret, frame = self.cam.read()

            if ret == True:
                frame = cv2.flip(frame, 1)
                
                if cv2.waitKey(1) & 0xFF == ord("t"):
                    class_names = ["Dog", "Cat"]  
                    image = cv2.resize(frame, (64, 64))
                    image = np.array(image).reshape(-1, 64, 64, 3) / 255.0     

                    y_proba = self.model.predict(image)
                    y_pred = (y_proba > 0.5).astype("int32")
                    preds = np.array(class_names)[y_pred]
                    
                    self.predict = preds[0][0]
                
                cv2.putText(frame, f'Predict: {self.predict}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Frame", frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CameraVision("Projects/Cat-Dog-Classification/artifacts/model.h5").show()