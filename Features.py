
#Dataset - Global Wheat Head Dataset 2021 - https://zenodo.org/record/5092309#.YiE9SujMI2x
# Farb Features werden als Histogramm im HSL Farbbereich ausgewählt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io
from datetime import datetime
from tqdm import tqdm
#Prozess Zeit
start_time = datetime.now()

path_read = "../../../../1. Data/1. Agri/competition_train.csv"
path_write = "../../../../1. Data/1. Agri/Feature_Farbe.csv"
img_path = "../../../../1. Data/1. Agri/images/"

print("# Einlesen der Dataframes ####################")
df_main = pd.read_csv(path_read, sep=',') #Einlesesen der CSV Datei , header=1

### Only selecting Norway for Testing
df_select_filling = df_main.loc[df_main['domain'] == 'NMBU_1'] # Filling nur aus Norwegen für Test
df_select_ripe = df_main.loc[df_main['domain'] == 'NMBU_2'] # Ripening nur aus Norwegen

df_farbe = pd.DataFrame()
df_H = pd.DataFrame()
df_L = pd.DataFrame()
df_S = pd.DataFrame()
df_label = pd.DataFrame()       # 0=Ripe, 1=Filling
x_range = list(range(0,256))
df_H["ID"] = x_range
df_L["ID"] = x_range
df_S["ID"] = x_range

label = []
### Ripe Select
arr_ripe = df_select_ripe.to_numpy()
a = 0
for column in tqdm(arr_ripe):
    filename = column[0]
    img = io.imread('{}{}'.format(img_path, filename))
    boxes = column[1]
    boxes_list = boxes.split(';')
    if boxes == 'no_box':
        pass
    else:
        for box in boxes_list:
            box_points = box.split(' ')
            #print("box_points: ", box_points)
            left, top, right, bottom = int(box_points[0]), int(box_points[1]), int(box_points[2]), int(box_points[3])
            img1 = img[top:bottom, left:right]

            single_column = np.vstack(np.hsplit(img1, img1.shape[1]))  # Bild wird in eine Spalete(Pixel) gebracht
            img1_hsl = cv2.cvtColor(single_column, cv2.COLOR_RGB2HLS_FULL)
            H, L, S = cv2.split(img1_hsl)  # Bild in Farben/lightness zerlegen
            Hue = pd.Series(H.reshape(-1))  # Reshape für eine Dimensionsreduzierung
            Light = pd.Series(L.reshape(-1))
            Saturation = pd.Series(S.reshape(-1))
            size = img1.shape[0]     #Bildgröße für Normierung
            Hue = Hue.value_counts()
            Light = Light.value_counts()
            Saturation = Saturation.value_counts()

            df_H[a] = Hue / size  # Summe der vorkommenden Farbwerte (0-255) als neue Spalte
            df_L[a] = Light / size
            df_S[a] = Saturation / size

            a = a+1
        label_r = [0]*(a+1)

label_r = [0]*(a+1)
b = a+1
#####################################################################
arr_fill = df_select_filling.to_numpy()
for column in tqdm(arr_fill):
    filename = column[0]
    img = io.imread('{}{}'.format(img_path, filename))
    boxes = column[1]
    boxes_list = boxes.split(';')
    if boxes == 'no_box':
        pass
    else:
        for box in boxes_list:
            box_points = box.split(' ')
            #print("box_points: ", box_points)
            left, top, right, bottom = int(box_points[0]), int(box_points[1]), int(box_points[2]), int(box_points[3])
            img1 = img[top:bottom, left:right]
            single_column = np.vstack(np.hsplit(img1, img1.shape[1]))  # Bild wird in eine Spalete(Pixel) gebracht
            img1_hsl = cv2.cvtColor(single_column, cv2.COLOR_RGB2HLS_FULL)
            H, L, S = cv2.split(img1_hsl)  # Bild in Farben/lightness zerlegen
            Hue = pd.Series(H.reshape(-1))  # Reshape für eine Dimensionsreduzierung
            Light = pd.Series(L.reshape(-1))
            Saturation = pd.Series(S.reshape(-1))
            size = img1.shape[0]     #Bildgröße für Normierung
            Hue = Hue.value_counts()
            Light = Light.value_counts()
            Saturation = Saturation.value_counts()

            df_H[a] = Hue / size  # Summe der vorkommenden Farbwerte (0-255) als neue Spalte
            df_L[a] = Light / size
            df_S[a] = Saturation / size

            a = a+1
label_f = [1]*(a-b)

df_H = df_H.drop(["ID"], axis=1)
df_L = df_L.drop(["ID"], axis=1)
df_S = df_S.drop(["ID"], axis=1)

df_H = df_H.transpose()
df_L = df_L.transpose()
df_S = df_S.transpose()

df_farbe = pd.concat([df_H , df_L, df_S], axis=1)
df_farbe["Label"] = label_r+label_f
df_farbe = df_farbe.fillna(0)
print(df_farbe)
df_farbe.to_csv(path_write, sep='/')

print("ende gelände")