import pickle

import cv2
import face_recognition

with open("../IA/classifier_model.pkl", "rb") as clf_file:
    clf = pickle.load(clf_file)

with open("../IA/label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

video_capture: cv2.VideoCapture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        label_id = clf.predict([face_encoding])
        person_name = label_encoder.inverse_transform(label_id)[0]

        print("Pessoa reconhecida:", person_name)
        
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
