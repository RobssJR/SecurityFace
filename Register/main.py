from ast import List
import PySimpleGUI as sg
import cv2
import os
import datetime
from typing import NoReturn, Tuple
import numpy as np


def count_images_in_folder(folder_path):
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return len(image_files)

def get_frame_and_return_value(cap: cv2.VideoCapture) -> Tuple[bool, np.ndarray]:
    return cap.read()

def SaveImage(path: str, cap: cv2.VideoCapture) -> NoReturn:
    folder_name: str = "../Images/" + path

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    ret, frame = get_frame_and_return_value(cap)
    img_name: str = f"{count_images_in_folder(folder_name)}-{datetime.datetime.now().strftime('%Y-%m-%d')}.png"

    img_path = os.path.join(folder_name, img_name)
    cv2.imwrite(img_path, frame)

def main() -> NoReturn:
    sg.theme('Black')

    layout: List[List[sg.Element]] = [
        [sg.Image(filename='', key='image')],
        [sg.Input('', enable_events=True, key='-INPUT-', font=('Arial Bold', 20), expand_x=True, justification='left'),
         sg.Button('Save', size=(10, 1), font='Helvetica 14')],
    ]

    window: sg.Window = sg.Window('Demo Application - OpenCV Integration',
                       layout, location=(800, 400), finalize=True)

    cap: cv2.VideoCapture = cv2.VideoCapture(0)

    while True:
        event: str
        values: dict
        event, values = window.read(timeout=20)
        if event == sg.WIN_CLOSED:
            break

        if event == 'Save':
            folder_name: str = values['-INPUT-']
            if folder_name:
                SaveImage(folder_name, cap)

        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        imgbytes: bytes = cv2.imencode('.png', frame)[1].tobytes()
        window['image'].update(data=imgbytes)

    cap.release()
    cv2.destroyAllWindows()
    window.close()

if __name__ == '__main__':
    main()