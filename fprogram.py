from PySimpleGUI import PySimpleGUI as sg
import glob
import zipfile
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array
import os
import shutil

def make_direct(caminho):
    directory = "resultado"
    parent_dir = caminho
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    list_emotions = ["happy","sad","fear","angry","disgust","neutral","surprise"]
    dir_emot = caminho + '/resultado'

    directlist = []
    for i in range(len(list_emotions)):
      final_way = os.path.join(dir_emot, list_emotions[i])
      algo = os.mkdir(final_way)
      directlist.append(str(final_way))
    return directlist  
      
def emotins_classification(result):

    labels = []

    for i in range(0,len(result)):
        output = DeepFace.detectFace(result[i],enforce_detection=False)
        if len(output) != 0:
            roi = output
            cinza = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cinza = cv2.resize(cinza, (48, 48))
            cinza = cinza.astype("float") / 255
            cinza = img_to_array(cinza)
            cinza = np.expand_dims(cinza, axis = 0)
            predictions= DeepFace.analyze(result[i],actions = ['emotion'],enforce_detection=False)
            labels.append(predictions['dominant_emotion'])
        else:
            continue
    
    return labels

def export(ogpaste,labels,caminhos):
    for f in range(len(labels)):
        if labels[f] == 'happy': 
            shutil.copy(ogpaste[f],caminhos[0])
        elif labels[f] == 'angry':
            shutil.copy(ogpaste[f],caminhos[3])
        elif labels[f] == 'disgust':
            shutil.copy(ogpaste[f],caminhos[4])
        elif labels[f] == 'fear':
            shutil.copy(ogpaste[f],caminhos[2])
        elif labels[f] == 'neutral':
            shutil.copy(ogpaste[f],caminhos[5])
        elif labels[f] == 'sad':
            shutil.copy(ogpaste[f],caminhos[1])
        else:
            shutil.copy(ogpaste[f],caminhos[6])
            
def main():
    sg.theme('Reddit')

    col_one = [[sg.Text('Pasta de entrada: '), sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()]]
    col_sec = [[sg.Text('Indique o local de saída: '), sg.In(size=(25,1), enable_events=True ,key='-OUTPUT-'), sg.FolderBrowse()]]
    layout = [
        [sg.Column(col_one, element_justification='c')],
        [sg.Column(col_sec, element_justification='c')],
        [sg.Text('Formato das Imagens: ')],
        [sg.Combo(['.jpg','.png'],size=(20, 1),default_value='.jpg',key='formato')],
        [sg.Button('Go')],
        [sg.Text('',key='mensagem')]
    ]  

    janela = sg.Window('Classificador Facial',layout)

    while True:
        eventos,valores = janela.read()
        if eventos == sg.WINDOW_CLOSED:
            break
        if eventos == 'Go':
            if valores['-FOLDER-']:
                exportation = valores.get('-OUTPUT-')#folder output
                pasta = valores.get('-FOLDER-')
                
                if valores.get('formato') == '.png':
                    files1 = pasta + '/*png'
                else:
                    files1 = pasta + '/*jpg'
                    
                a = glob.glob(files1)
                result = [cv2.imread(file) for file in glob.glob(files1)]
                
                b = make_direct(exportation)
                c = emotins_classification(result)
                export(a,c,b)
                janela['mensagem'].update('Classificações Concluídas!')
                
if __name__ == "__main__":
    main()