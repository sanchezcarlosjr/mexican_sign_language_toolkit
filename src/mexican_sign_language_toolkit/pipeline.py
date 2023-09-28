import numpy as np
from mexican_sign_language_toolkit.neighbors import Bruteforce
from mexican_sign_language_toolkit.pose_landmarker import PoseLandmarker
from mexican_sign_language_toolkit.lexer import tokenize
import mediapipe as mp

def pipeline(space_path='../checkpoints/sign_language_space.npy', regex_path='../checkpoints/regex.npy'):
    space = np.load(space_path, allow_pickle=True)
    regex = str(np.load(regex_path, allow_pickle=True))
    brute_force = Bruteforce(space)
    pose_landmarker = PoseLandmarker()
    match = tokenize(regex)
    # image must be an numpy array
    def predict(image):
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            landmarks = pose_landmarker.detect_from_image(mp_image)
            signal = brute_force.classify(landmarks)
            result = match(signal)
            if result:
                return result[0]
            return ""
        except:
            return ""
    return predict