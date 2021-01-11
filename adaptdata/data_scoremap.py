'''
@uthor: me_teor21
d@te: 11/01/2021

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import glob2
import PIL
from tqdm import tqdm
import cv2
import numpy as np
import shutil

try:
    import Image
except ImportError:
    from PIL import Image

def data_for_scoremap(csv, images):

    dataset = pd.read_csv(csv)  #load the dataset
    listededico = dataset['labels'].tolist()  #store the labels column to a list

    realdico = []
    height = []
    width = []
    exes = []
    wayy = []

    for i in range(len(listededico)):
        alpha = ast.literal_eval(listededico[i])  #convert a string of dico to a real dico

        realdico.append(alpha)

    for i in range(len(listededico)):
        height.append(int(realdico[i]['height']))    

    for i in range(len(listededico)):
        width.append(int(realdico[i]['width'])) 

    for i in range(len(listededico)):
        exes.append(int(realdico[i]['x']))

    for i in range(len(listededico)):
        wayy.append(int(realdico[i]['y']))

    exes1 = exes
    wayy1 = wayy

    exes2 = [x + y for x, y in zip(exes, width)]  #add up both lists

    wayy2 = [a + b for a, b in zip(wayy, height)]  #add up both lists

    dataset['x1'] = exes1
    dataset['y1'] = wayy1
    dataset['x2'] = exes2
    dataset['y2'] = wayy2


    lesimages = dataset['imageName'].tolist()

    liens = ["./svt/svt1/"]*len(lesimages)

    jointure = []
    for i in range(len(lesimages)):

        joint = liens[i] + lesimages[i]

        jointure.append(joint)

    imagewidthhh = []
    imageheighttt = []

    for thing in jointure:
        image = PIL.Image.open(thing)
        width, height = image.size

        imagewidthhh.append(image.size[0])
        imageheighttt.append(image.size[1])

    dataset['joint'] = jointure
    dataset['imagewidth'] = imagewidthhh
    dataset['imageheight'] = imageheighttt

    les_noms = dataset['imageName'].unique().tolist()

    for noms in tqdm(les_noms):
        #firstimage = (dataset['imageName'] == noms)
        groupe_of_first_image = dataset[dataset['imageName'] == noms]
        groupe = groupe_of_first_image.reset_index(drop=True)

        matrix = (np.zeros(((groupe.iloc[0, 9]), (groupe.iloc[0, 8]))))
        for i in range(len(groupe)):

            #matrix = (np.zeros(((groupe.iloc[i, 9]), (groupe.iloc[i, 8]))))

            block = (np.zeros(((groupe.iloc[i, 9]), (groupe.iloc[i, 8]))))

            #block = (np.ones(((groupe.iloc[i, 6]) - (groupe.iloc[i, 4])), ((groupe.iloc[i, 5]) - (groupe.iloc[i, 3]))))

            block[(groupe.iloc[i, 4]):(groupe.iloc[i, 6]), (groupe.iloc[i, 3]):(groupe.iloc[i, 5])] = 1

            matrix = np.add(matrix, block) #matrix + block

        #plt.imshow(matrix)
        #plt.show()
        img = Image.fromarray(matrix)
        img = img.convert('RGB')
        img.save(f'{images}/{noms}')


    return
    
    
    
def main():
    data_for_scoremap('./train.csv', './images')
    
if __name__ == "__main__":
    main()
    