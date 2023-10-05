import numpy as np
from mexican_sign_language_toolkit.neighbors import Bruteforce
from mexican_sign_language_toolkit.pose_landmarker import PoseLandmarker
from mexican_sign_language_toolkit.lexer import tokenize
import mediapipe as mp
import cv2
import time
import requests
import os

def download_file(url, local_filename):
    """
    Download a file from the given URL and save it to the specified local filename.
    :param url: The URL of the file to download
    :param local_filename: The local file name to save the downloaded file
    """
    with requests.get(url, stream=True) as response:
        # Check if the request was successful (status code 200)
        response.raise_for_status()
        # Write the content of the request to a local file
        with open(local_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                
def install(*paths):
    default_http_requests = [
        "https://github.com/sanchezcarlosjr/mexican_sign_language_toolkit/raw/main/checkpoints/regex.npy",
        "https://github.com/sanchezcarlosjr/mexican_sign_language_toolkit/raw/main/checkpoints/sign_language_space.npy"
    ]
    default_names = [
        "regex.npy",
        "sign_language_space.npy"
    ]
    for index, path in enumerate(paths):
        if not os.path.exists(path):
            download_file(default_http_requests[index], default_names[index])

class Pipeline:
    """
    Base pipeline class for processing and predicting landmarks.
    """
    def __init__(self, space_path='sign_language_space.npy', regex_path='regex.npy'):
        install(space_path, regex_path)
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
