from typing import List, NoReturn, Tuple
import PySimpleGUI as sg
import cv2
import os
import datetime
import numpy as np
import json

niveis: List[str] = ["Geral", "Diretores", "Ministro"]

def CountImagesFolder(folder_path: str) -> int:
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return len(image_files)

def SaveJson(ra: str, nome: str, nivel: str) -> NoReturn:
    try:
        with open('./Images/dados.json', 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {}

    data[ra] = {"Nome": nome, "Nivel": nivel}

    with open('./Images/dados.json', 'w') as json_file:
        json.dump(data, json_file)


def SaveImage(path: str, cap: cv2.VideoCapture) -> NoReturn:
    folderName: str = "./Images/" + path

    ret: bool
    frame: np.ndarray

    if not os.path.exists(folderName):
        os.makedirs(folderName)

    ret, frame = cap.read() 
    imgName: str = f"{CountImagesFolder(folderName)}-{datetime.datetime.now().strftime('%Y-%m-%d')}.png"

    img_path = os.path.join(folderName, imgName)
    cv2.imwrite(img_path, frame)

def main() -> NoReturn:
    sg.theme('Black')

    layout: List[List[sg.Element]] = [
        [sg.Image(filename='', key='image', expand_x=True)],
        [sg.Text(text="RA", font=('Arial Bold', 15)),
         sg.Input('', enable_events=True, key='-INPUT_RA-', font=('Arial Bold', 15), expand_x=True, justification='left', size=(10,100)),
         sg.Text(text="Nome", font=('Arial Bold', 15)),
         sg.Input('', enable_events=True, key='-INPUT_NOME-', font=('Arial Bold', 15), expand_x=True, justification='left', size=(30,100)),
         sg.Combo(values=niveis, font=('Arial Bold', 15), key="-COMBO-", enable_events=True),
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
            ra: str = values['-INPUT_RA-']
            nome: str = values['-INPUT_NOME-']
            nivel_selecionado: str = values["-COMBO-"]

            if not ra:
                sg.popup_error("Preencha o RA")
                continue
            
            if not nivel_selecionado:
                sg.popup_error("Preencha o nivel de acesso")
                continue

            SaveImage(ra, cap)
            SaveJson(ra, nome, nivel_selecionado)

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
