import os
import pickle
import face_recognition
import numpy as np
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

path_to_training_images = "../Images"

face_encodings = []
labels = []

def LoadFaces():
    for person_name in os.listdir(path_to_training_images):
        person_folder = os.path.join(path_to_training_images, person_name)

        for image_file in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_file)
            image = face_recognition.load_image_file(image_path)

            face_encoding = face_recognition.face_encodings(image)
            if len(face_encoding) == 1:
                face_encodings.append(face_encoding[0])
                labels.append(person_name)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    clf = svm.SVC(kernel='linear')
    clf.fit(face_encodings, encoded_labels)

    # Salvar o classificador e o label_encoder em arquivos separados
    with open("classifier_model.pkl", "wb") as clf_file:
        pickle.dump(clf, clf_file)
    
    with open("label_encoder.pkl", "wb") as encoder_file:
        pickle.dump(label_encoder, encoder_file)

if __name__ == "__main__":
    LoadFaces()
