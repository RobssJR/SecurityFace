import os
import pickle
import face_recognition
import numpy as np
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from typing import List

path_to_training_images: str = "C:\APS\Codigo\SecurityFace\Images"

face_encodings: List[np.ndarray] = []
labels: List[str] = []

def LoadFaces():
    for person_name in os.listdir(path_to_training_images):
        person_folder: str = os.path.join(path_to_training_images, person_name)

        if os.path.isdir(person_folder) == False:
            continue

        for image_file in os.listdir(person_folder):
            image_path: str = os.path.join(person_folder, image_file)
            image: np.ndarray = face_recognition.load_image_file(image_path)
            
            face_encoding: List[np.ndarray] = face_recognition.face_encodings(image)
            if len(face_encoding) == 1:
                face_encodings.append(face_encoding[0])
                labels.append(person_name)

    label_encoder: LabelEncoder = LabelEncoder()
    encoded_labels: np.ndarray = label_encoder.fit_transform(labels)
    
    clf: svm.SVC = svm.SVC(kernel='linear')
    clf.fit(face_encodings, encoded_labels)

    with open("classifier_model.pkl", "wb") as clf_file:
        pickle.dump(clf, clf_file)
    
    with open("label_encoder.pkl", "wb") as encoder_file:
        pickle.dump(label_encoder, encoder_file)

if __name__ == "__main__":
    LoadFaces()
