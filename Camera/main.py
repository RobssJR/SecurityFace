import json
import pickle
from typing import Any, List, Tuple
import cv2
import face_recognition

with open("../IA/classifier_model.pkl", "rb") as clf_file:
    clf: Any = pickle.load(clf_file)

with open("../IA/label_encoder.pkl", "rb") as encoder_file:
    label_encoder: Any = pickle.load(encoder_file)

with open('../Images/dados.json', 'r') as json_file:
    data = json.load(json_file)

video_capture: cv2.VideoCapture = cv2.VideoCapture(0)

while True:
    ret: bool
    frame: Any
    ret, frame = video_capture.read()
    rgb_frame: Any = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations: List[Tuple[int, int, int, int]]
    face_encodings: List[Any]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        label_id: Any = clf.predict([face_encoding])[0]
        person_ra: Any = label_encoder.inverse_transform([label_id])[0]

        nome: str = data[person_ra]["Nome"]
        nivel: str = data[person_ra]["Nivel"]

        print("Pessoa reconhecida:", nome, "Nivel:", nivel)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()