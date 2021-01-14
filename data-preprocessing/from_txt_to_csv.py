




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import glob2
import PIL
from tqdm import tqdm
import numpy as np
import shutil

try:
    import Image
except ImportError:
    from PIL import Image
    

    
txtfiles = glob2.glob('./annotations/*.txt')  #create a list of all txt files

def from_txt_to_csv(txtfiles):
    
    x1 = []  #create an empty list
    y1 = []  #create an empty list
    x2 = []  #create an empty list
    y2 = []  #create an empty list
    x3 = []  #create an empty list
    y3 = []  #create an empty list
    x4 = []  #create an empty list
    y4 = []  #create an empty list
    labels = []  #create an empty list
    names = []  #create an empty list
    X1 = []  #create an empty list
    Y1 = []  #create an empty list
    X2 = []  #create an empty list
    Y2 = []  #create an empty list
    image_width = []  #create an empty list
    image_height = []  #create an empty list


    for txt in tqdm(txtfiles):  #iterate over each txt file

        with open(txt, 'r', encoding='utf-8-sig') as f:  #read through a txt file
            lines = f.readlines()  #store the read lines into a list

            gt_name = txt.split('/')[2].split('.')[0]  #get this part 'gt_img_1'
            name = gt_name.removeprefix('gt_')  #get this part 'img_1'

            #print(len(lines))


            for thing in lines:    #iterate over each line

                x_up_left = int(thing.split(',')[0])  #store the corresponding coordinate
                y_up_left = int(thing.split(',')[1])  #store the corresponding coordinate
                x_up_right = int(thing.split(',')[2]) #store the corresponding coordinate
                y_up_right = int(thing.split(',')[3])  #store the corresponding coordinate
                x_down_right = int(thing.split(',')[4])  #store the corresponding coordinate
                y_down_right = int(thing.split(',')[5])  #store the corresponding coordinate
                x_down_left = int(thing.split(',')[6])  #store the corresponding coordinate
                y_down_left = int(thing.split(',')[7])  #store the corresponding coordinate

                X_up_left = min(x_up_left, x_down_left)  #take the min for the rectangle
                Y_up_left = min(y_up_left, y_up_right)  #take the min for the rectangle
                X_down_right = max(x_up_right, x_down_right)  #take the max for the rectangle
                Y_down_right = max(y_down_right, y_down_left)  #take the max for the rectangle

                image = PIL.Image.open(f'./images/{name}.jpg')  #read coresponding images
                width, height = image.size  #store the dimensions

                image_width.append(image.size[0])  #add the width to the image_width list
                image_height.append(image.size[1])  #add the height to the image_height list


                x1.append(int(x_up_left))  #add to the x1 list
                y1.append(int(y_up_left))  #add to the y1 list
                x2.append(int(x_up_right))  #add to the x2 list
                y2.append(int(y_up_right))  #add to the y2 list
                x3.append(int(x_down_right))  #add to the x3 list
                y3.append(int(y_down_right))  #add to the y3 list
                x4.append(int(x_down_left))  #add to the x4 list
                y4.append(int(y_down_left))  #add to the y4 list

                X1.append(int(X_up_left))  #add to the X1 list
                Y1.append(int(Y_up_left))  #add to the Y1 list
                X2.append(int(X_down_right))  #add to the X2 list
                Y2.append(int(Y_down_right))  #add to the Y2 list

                labels.append(thing.split(',')[8].replace("\n", ""))  #add to the labes list
                names.append(name)

            #x1 = list(map(int, x1))  #transform to a list of int
            #y1 = list(map(int, y1))  #transform to a list of int
            #x2 = list(map(int, x2))  #transform to a list of int
            #y2 = list(map(int, y2))  #transform to a list of int
            #x3 = list(map(int, x3))  #transform to a list of int
            #y3 = list(map(int, y3))  #transform to a list of int
            #x4 = list(map(int, x4))  #transform to a list of int
            #y4 = list(map(int, y4))  #transform to a list of int



    #create a dataframe containig the needed lists
    df = pd.DataFrame(list(zip(names, labels, X1, Y1, X2, Y2, image_width, image_height)), 
                   columns =['imageName', 'bbox', 'X1', 'Y1', 'X2', 'Y2', 'image_width', 'image_height']) 

    return

