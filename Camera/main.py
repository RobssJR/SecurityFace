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

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        label_id = clf.predict([face_encoding])
        person_name = label_encoder.inverse_transform(label_id)[0]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, person_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
