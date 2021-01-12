


'''
@uthor: me_teor21
d@te: 12/01/2021
'''




def dataforQuadGeo(dataset, images):
    '''This function adapt the image dataset to the QUADGEO loss function'''
    
    data = pd.read_csv(dataset)  #load the dataset
    
    les_noms = data['imageName'].unique().tolist()  #make a list with the image names (no repetition)

    for noms in tqdm(les_noms):   #iterate over every images

        split = noms.split('.')[0]  #as image names are like img/00_00.jpg, split and store the 'img/00_00' part
        groupe_of_first_image = data[data['imageName'] == noms]  #dataframe of a each single image
        groupe = groupe_of_first_image.reset_index(drop=True)  #drop index to make it start from 0


        flop = (np.zeros((4, (groupe.iloc[0, 9]), (groupe.iloc[0, 8]))))   #create an empty tensor of size (4, m, n) containing zeros

        for f in range(len(groupe)):  #iterate over lengths of each 'groupe'




            #create an empty block of zeros that will contain the distance from the upleft corner
            upleft = np.zeros((((groupe.iloc[f, 6]) - (groupe.iloc[f, 4])), ((groupe.iloc[f, 5]) - (groupe.iloc[f, 3]))))  
            
            #create an empty block of zeros that will contain the distance from the upright corner
            upright = np.zeros((((groupe.iloc[f, 6]) - (groupe.iloc[f, 4])), ((groupe.iloc[f, 5]) - (groupe.iloc[f, 3]))))
            
            #create an empty block of zeros that will contain the distance from the downright corner
            downright = np.zeros((((groupe.iloc[f, 6]) - (groupe.iloc[f, 4])), ((groupe.iloc[f, 5]) - (groupe.iloc[f, 3]))))
            
            #create an empty block of zeros that will contain the distance from the downleft corner
            downleft = np.zeros((((groupe.iloc[f, 6]) - (groupe.iloc[f, 4])), ((groupe.iloc[f, 5]) - (groupe.iloc[f, 3]))))



            for i in range((groupe.iloc[f, 6]) - (groupe.iloc[f, 4])):  #iterate over the row indexes
                for e in range((groupe.iloc[f, 5]) - (groupe.iloc[f, 3])):  #iterate over the column indexes

                    #coordinate of upleft
                    alpha = np.array((0,0)) 
                    
                    #coordinate of the rest
                    beta = np.array((i,e)) 
                    
                    #coordinate of upright
                    gamma = np.array((((groupe.iloc[f, 5]) - (groupe.iloc[f, 3])),0)) 
                    
                    #coordinate of downright
                    delta = np.array((((groupe.iloc[f, 5]) - (groupe.iloc[f, 3])), ((groupe.iloc[f, 6]) - (groupe.iloc[f, 4])))) 
                    lambdaa = np.array((0, ((groupe.iloc[f, 6]) - (groupe.iloc[f, 4])))) #coordinate of downleft

                    upleft[i][e] = np.linalg.norm(alpha - beta) #fill in the distances
                    upright[i][e] = np.linalg.norm(gamma - beta) #fill in the distances
                    downright[i][e] = np.linalg.norm(delta - beta) #fill in the distances
                    downleft[i][e] = np.linalg.norm(lambdaa - beta) #fill in the distances


            flop[0][(groupe.iloc[f, 4]):(groupe.iloc[f, 6]), (groupe.iloc[f, 3]):(groupe.iloc[f, 5])] = upleft  #fill in tensor
            flop[1][(groupe.iloc[f, 4]):(groupe.iloc[f, 6]), (groupe.iloc[f, 3]):(groupe.iloc[f, 5])] = upright  #fill in tensor
            flop[2][(groupe.iloc[f, 4]):(groupe.iloc[f, 6]), (groupe.iloc[f, 3]):(groupe.iloc[f, 5])] = downright  #fill in tensor
            flop[3][(groupe.iloc[f, 4]):(groupe.iloc[f, 6]), (groupe.iloc[f, 3]):(groupe.iloc[f, 5])] = downleft  #fill in tensor





        np.save(f'{images}/{split}.npy',flop)  #save each numpy array to its corresponding image name



    return




def main():
    dataforQuadGeo('./dataset.csv', './images')
    
if __name__ == "__main__":
    main()