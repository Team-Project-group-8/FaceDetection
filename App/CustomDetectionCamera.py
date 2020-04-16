from imageai.Detection.Custom import CustomVideoObjectDetection
import os
import cv2

execution_path = os.getcwd()

camera = cv2.VideoCapture(0)

detector = CustomVideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "detection_model-ex-015--loss-0011.676.h5"))
detector.setJsonPath("detection_config.json")
detector.loadModel()

detector.detectObjectsFromVideo(camera_input=camera,
                                        output_file_path=os.path.join(execution_path, "web_face"),
                                        frames_per_second=30,
                                        minimum_percentage_probability=40,
                                        log_progress=True,
                                        frame_detection_interval=15)