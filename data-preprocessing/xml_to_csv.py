'''
@uthor: me_teor21
d@te: 24/12/2020 11:50 pm
'''

import os
import xml.etree.ElementTree as ET
import pandas as pd


def xtractxml(xmlpath, csvpath):
    '''This function extracts necessary infos from a xml file and store them into csv (format respecting image annotation )'''
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    
    liste0 = []
    liste2 = []
    lesimages = []
    leslabels = []
    
    for thing in root.iter('imageName'):
        lesimages.append(thing.text)
    
    for i in range(len(lesimages)):
        for f in range(20):
            try:


                liste0.append(root[i][4][f][0].text)
          
            except:
                IndexError

   
        liste2.append(liste0)
        liste0 = []

    
    for thing in root.iter('taggedRectangle'):
        leslabels.append(thing.attrib)
    
    
    dej = {'imageName':lesimages, 'bruh':liste2 }
    defo = pd.DataFrame(data=dej)
    
    fif = defo[['imageName', 'bruh']].set_index(['imageName'])['bruh'].apply(pd.Series).stack().reset_index(level=1, drop=True).reset_index().rename(columns={0:'bruh'})
    
    fif['leslabels'] = leslabels
    fif.columns = ['imageName', 'tags', 'labels']
    
    fif.to_csv(csvpath, sep = ',', index = None)
    
    return fif
    
            
        
        
def main():
    xtractxml('/mekeneocr/myUniverse/svt/svt1/test.xml', '/mekeneocr/myUniverse/svt/svt1/test.csv')
    
if __name__ == "__main__":
    main()