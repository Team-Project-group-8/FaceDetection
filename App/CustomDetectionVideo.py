from imageai.Detection.Custom import CustomVideoObjectDetection
import os

execution_path = os.getcwd()

detector = CustomVideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("detection_model-ex-015--loss-0011.676.h5")
detector.setJsonPath("detection_config.json")
detector.loadModel()

detector.detectObjectsFromVideo(input_file_path="video.mp4",
                                        output_file_path=os.path.join(execution_path, "traffic_face"),
                                        frames_per_second=30,
                                        minimum_percentage_probability=40,
                                        log_progress=True)