import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import glob
import os
from os import listdir
from datetime import datetime
from skimage import io
import pandas as pd
from PIL import Image
import glob
import os
from os import listdir
from datetime import datetime
from skimage import io
from datetime import datetime
from tqdm import tqdm
#Prozess Zeit
start_time = datetime.now()

path_read = "../../../../1. Data/1. Agri/competition_train.csv"
path_write_ripe = "../../../../1. Data/1. Agri/ripe/"
path_write_fill = "../../../../1. Data/1. Agri/fill/"
img_path = "../../../../1. Data/1. Agri/images/"

print("# Einlesen der Dataframes ####################")
df_main = pd.read_csv(path_read, sep=',') #Einlesesen der CSV Datei , header=1

### Only selecting Norway for Testing
df_select_filling = df_main.loc[df_main['domain'] == 'NMBU_1'] # Filling
df_select_ripe = df_main.loc[df_main['domain'] == 'NMBU_2'] # Ripening
print(df_select_ripe)

### Ripe Select
arr_ripe = df_select_ripe.to_numpy()
a = 0
for column in tqdm(arr_ripe):
    filename = column[0]
    #img = io.imread('{}{}'.format(img_path, filename))
    img = Image.open('{}{}'.format(img_path, filename))
    boxes = column[1]
    boxes_list = boxes.split(';')
    if boxes == 'no_box':
        pass
    else:

        for box in boxes_list:
            box_points = box.split(' ')
            #print("box_points: ", box_points)
            left, top, right, bottom = int(box_points[0]), int(box_points[1]), int(box_points[2]), int(box_points[3])
            img1 = img.crop((left, top, right, bottom))
            a = a+1
            img1.save('{}{}{}'.format(path_write_ripe, a,"-r.png"))
        a = a+1

### Fill Select
arr_fill = df_select_filling.to_numpy()
b = 0
for column in tqdm(arr_fill):
    filename = column[0]
    img = Image.open('{}{}'.format(img_path, filename))
    boxes = column[1]
    boxes_list = boxes.split(';')
    if boxes == 'no_box':
        pass
    else:
        for box in boxes_list:
            box_points = box.split(' ')
            left, top, right, bottom = int(box_points[0]), int(box_points[1]), int(box_points[2]), int(box_points[3])
            img1 = img.crop((left, top, right, bottom))
            b = b+1
            img1.save('{}{}{}'.format(path_write_fill, b,"-f.png"))
        b = b+1

print("Ende Gelände")
