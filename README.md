# Pedestrian-Walk-Detection
The objective of this project is to develop a pedestrian detection system for video streams. The problem addressed is the need for an automated and efficient method to detect pedestrians in real-time video footage.

The objective of this project is to develop a pedestrian detection system using the YOLO (You Only Look Once) model. The system aims to detect pedestrians in a given video stream and draw bounding boxes around them to aid in their identification and tracking. The YOLO model is employed due to its real-time object detection capabilities.

The implementation involves utilizing pre-trained weights and a configuration file for the YOLO model, along with the COCO dataset's labels. The pedestrian detection algorithm works by processing video frames one by one. For each frame, the YOLO model is used to detect objects, and the algorithm specifically focuses on identifying pedestrians based on their class label.

Once pedestrians are detected, bounding boxes are drawn around them using the OpenCV library. The system applies Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes and keep only the most confident detections. The NMS threshold and minimum confidence parameters are used to control the sensitivity and accuracy of the detections.

The code allows for easy integration with different video sources by specifying the path to the video file. The detection results are displayed in real-time, with bounding boxes drawn around pedestrians. The system provides an interactive interface where the user can stop the execution by pressing the 'Esc' key.

This pedestrian detection system can be applied to various scenarios, such as surveillance, traffic monitoring, or pedestrian safety analysis. It offers a practical solution for automatic pedestrian detection in videos and serves as a foundation for further development and customization.
