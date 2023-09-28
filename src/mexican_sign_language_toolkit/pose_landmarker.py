import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
from uuid_extensions import uuid7str
from mexican_sign_language_toolkit.neighbors import standard_normalization
import regex
    
class PoseLandmarker:
    def __init__(self, pose_landmarker_model_asset='pose_landmarker_heavy.task', hand_landmarker_model_asset='hand_landmarker.task'):
        base_options = python.BaseOptions(model_asset_path=pose_landmarker_model_asset)
        options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        base_options = python.BaseOptions(model_asset_path=hand_landmarker_model_asset)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=base_options, num_hands=2))
    
    def create_database_from_images(self, paths):
        space = []
        for path in paths:
            name, _ = os.path.splitext(os.path.basename(path))
            image = mp.Image.create_from_file(path)
            space.append({
                'segment': uuid7str(),
                'name': name,
                'matrix': standard_normalization(self.detect_from_image(image))
            })
        similar_paths = {}
        for element_i in space:
            name_i = regex.search("[A-Za-z]+", element_i['name']).group()
            if name_i not in similar_paths:
                similar_paths[name_i] = set()
            for element_j in space:
                name_j = regex.search("[A-Za-z]+", element_j['name']).group()
                if regex.match("("+name_i+"){e<=1}",name_j, regex.IGNORECASE):
                    similar_paths[name_i].add(element_j['name'])
        regex_expressions = []
        for key,paths in similar_paths.items():
            paths = sorted(list(paths))
            regex_expressions.append(f"(?P<{key}>" + "".join(paths) + ")")
        return [r"(?P<noise>[\n\r\s]+)|"+"|".join(regex_expressions), np.array(space)]
            
    def detect_from_image(self,image):
        pose_world_landmarks = self.pose_landmarker.detect(image).pose_world_landmarks
        hand_landmarker_result = self.hand_landmarker.detect(image)
        handedness = hand_landmarker_result.handedness
        hand_world_landmarks = hand_landmarker_result.hand_world_landmarks
        landmarks = np.zeros((59,3))
        for idx in range(0,15):
            landmarks[idx] = (pose_world_landmarks[0][idx].x,pose_world_landmarks[0][idx].y,pose_world_landmarks[0][idx].z)
        left,right = (
                       0 if len(handedness) >= 1 and len(handedness[0]) >= 1 and handedness[0][0].category_name == 'Left' else 1 if len(handedness) >= 2 and len(handedness[1]) >= 1 and handedness[1][0].category_name == 'Left'  else None,
                       0 if len(handedness) >= 1 and len(handedness[0]) >= 1 and handedness[0][0].category_name == 'Right' else 1 if len(handedness) >= 2 and len(handedness[1]) >= 1 and handedness[1][0].category_name == 'Right'  else None
                     )

        if left != None:
            for idx in range(0,21):
                landmarks[idx+15] = (hand_world_landmarks[left][idx].x,hand_world_landmarks[left][idx].y,hand_world_landmarks[left][idx].z)
        else:
            landmarks[15] = (pose_world_landmarks[0][15].x,pose_world_landmarks[0][15].y,pose_world_landmarks[0][15].z)
            landmarks[15+5] = (pose_world_landmarks[0][19].x,pose_world_landmarks[0][19].y,pose_world_landmarks[0][19].z)
            landmarks[15+4] = (pose_world_landmarks[0][21].x,pose_world_landmarks[0][21].y,pose_world_landmarks[0][21].z)
            landmarks[15+17] = (pose_world_landmarks[0][17].x,pose_world_landmarks[0][17].y,pose_world_landmarks[0][17].z)
                
        if right != None:
            for idx in range(0,21):
                landmarks[idx+15+21] = (hand_world_landmarks[right][idx].x,hand_world_landmarks[right][idx].y,hand_world_landmarks[right][idx].z)
        else:
            landmarks[15+21] = (pose_world_landmarks[0][16].x,pose_world_landmarks[0][16].y,pose_world_landmarks[0][16].z)
            landmarks[15+21+5] = (pose_world_landmarks[0][20].x,pose_world_landmarks[0][20].y,pose_world_landmarks[0][20].z)
            landmarks[15+21+4] = (pose_world_landmarks[0][22].x,pose_world_landmarks[0][22].y,pose_world_landmarks[0][22].z)
            landmarks[15+21+17] = (pose_world_landmarks[0][18].x,pose_world_landmarks[0][18].y,pose_world_landmarks[0][18].z)
        
        for idx in range(23,25):
            landmarks[idx-23+15+42] = (pose_world_landmarks[0][idx].x,pose_world_landmarks[0][idx].y,pose_world_landmarks[0][idx].z)
        return landmarks

def detect_landmarks_from_image(path):
    poseLandmarker = PoseLandmarker()
    image = mp.Image.create_from_file(path)
    return poseLandmarker.detect_from_image(image)