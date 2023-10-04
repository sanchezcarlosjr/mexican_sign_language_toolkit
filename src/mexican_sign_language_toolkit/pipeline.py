import numpy as np
from mexican_sign_language_toolkit.neighbors import Bruteforce
from mexican_sign_language_toolkit.pose_landmarker import PoseLandmarker
from mexican_sign_language_toolkit.lexer import tokenize
import mediapipe as mp
import cv2
import time

class Pipeline:
    """
    Base pipeline class for processing and predicting landmarks.
    """
    def __init__(self, space_path='../checkpoints/sign_language_space.npy', regex_path='../checkpoints/regex.npy'):
        space = np.load(space_path, allow_pickle=True)
        regex = str(np.load(regex_path, allow_pickle=True))
        self.brute_force = Bruteforce(space)
        self.match = tokenize(regex)

    def predict(self, *input):
        """
        Predict with landmarks and handle exceptions gracefully.
        """
        try:
            landmarks = self.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=input[0]), *input[1:])
            signal = self.brute_force.classify(landmarks)
            result = self.match(signal)
            return result[0] if result else ""
        except Exception as e:
            return ""


class ImagePipeline(Pipeline):
    def start_landmarker(self):
        self.pose_landmarker = PoseLandmarker(mp.tasks.vision.RunningMode.IMAGE) 
    def detect(self, mp_image):
        return self.pose_landmarker.detect_for_image(mp_image) 


class VideoPipeline(Pipeline):
    def start_landmarker(self):
        self.pose_landmarker = PoseLandmarker(mp.tasks.vision.RunningMode.VIDEO) 
    """
    Pipeline for processing and predicting landmarks from videos.
    """
    def predict(self, input):
        cap = cv2.VideoCapture(input)
        fps = cap.get(cv2.CAP_PROP_FPS)
        predictions = []
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_timestamp_ms = 1000 * frame_index / fps
            prediction = super().predict(rgb_frame, int(frame_timestamp_ms))
            if prediction != "" and (len(predictions) == 0 or prediction != predictions[-1]):
               predictions.append(prediction) 
            frame_index += 1

        cap.release()
        return " ".join(predictions)

    def detect(self, mp_image, frame_timestamp_ms):
        return self.pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
        
def pipeline(pipe = ImagePipeline()):
    pipe.start_landmarker()
    return pipe.predict
