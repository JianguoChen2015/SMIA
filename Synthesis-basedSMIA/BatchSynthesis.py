from PIL import Image
import numpy as np
import os
from os import walk
from os import path
import random
import shutil
from datetime import datetime 

#default path of medical image datasets
basepath="../Datasets/original/"
basepath_synthetic="../Datasets/synthetic/"
#Pixel difference threshold (matching background color)
pixelDifference = 60  
#Proportion of background in the candidate area
bgRatio = 0.3


def batch_synthesis_multiple():
    """
    batch synthesis the medical images with multiple tissues
    """    
    dir_list = os.listdir(basepath)   
    for dir_name in dir_list:    
        a=datetime.now()      
        #directory of background-filled medical images, augmented tissues, 
        #synthetic images, synthetic tissues, and synthetic ground truths
        img_dir = path.join(basepath, dir_name, "background-filled")
        tissue_dir = path.join(basepath, dir_name, "tissues")
        newimg_dir = path.join(basepath_synthetic,  dir_name, "img")        
        newtissue_dir = path.join(basepath_synthetic,  dir_name, "tissues")
        seg_dir = path.join(basepath_synthetic,  dir_name, "ground truths")
            
        #create directory if it does not exist
        for file_dir in [newimg_dir, newtissue_dir, seg_dir]:
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)
    
        #For each medical image with multiple tissues
        #try:
        imgNames = os.listdir(img_dir)
        for i in range(len(imgNames)):
            if (len(imgNames)>0):        
                img_name = imgNames[i]
                img_path = path.join(img_dir, img_name)
                newimg_path = path.join(newimg_dir, img_name)
                #Randomly synthesis a medical image with multiple tissues
                randomsynthesis_multiple(img_path, tissue_dir, newimg_path, newtissue_dir, seg_dir)
        b = datetime.now()
        print("cost microseconds: ", (b-a).microseconds)
        print("cost seconds: ", (b-a).seconds)
    
def randomsynthesis_multiple(img_path, tissue_dir, newimg_path, newtissue_dir, seg_dir):
    """
    Randomly synthesis a medical image with multiple tissues
    """
    #Load medical image
    img = Image.open(img_path)
    img_data = np.array(img)
    height_img,width_img,depth_img = img_data.shape
    
    #Find background color
    bgcolor = find_bgcolor(img_data)
    
    tissueNames =  next(walk(tissue_dir))[2]
    #Shuffle the augmented tissues
    random.shuffle(tissueNames)
    for tissueName in tissueNames:
        print(tissueName)
        tissue_path = path.join(tissue_dir, tissueName)
        tissue = Image.open(tissue_path)   

        tissue_data = np.array(tissue)    
        height_tissue,width_tissue,_ = tissue_data.shape    

        if(height_img > height_tissue and  width_img > width_tissue):
            #Find the optimal insertion position
            random_height, random_width = find_OptmizedLocation2(img_data, bgcolor, tissue_data)
            
            #Defined a new image for the synthetic tissue
            newtissue_data = np.zeros((height_img, width_img, depth_img),dtype = np.uint8)
            newtissue_data = np.where(newtissue_data != 0, newtissue_data, 255)
            
            #Defined a new image for the synthetic segmentation (ground truths)
            seg_data = np.zeros((height_img, width_img, depth_img),dtype = np.uint8)
            seg_data = np.where(seg_data != 0, seg_data, 0)
                        
            #Dxtract tissues
            tissue_data_bw = tissue_data.max(axis=2)
            tissue_data_bw = tissue_data_bw -255
            non_empty_points = np.transpose(np.nonzero(tissue_data_bw))
            
            for point in non_empty_points:
                #Insert the tissue into the synthetic image
                img_data[point[0]+random_height, point[1]+random_width,:] = tissue_data[point[0], point[1],:] 
                newtissue_data[point[0]+random_height, point[1]+random_width,:] = tissue_data[point[0], point[1],:]
                seg_data[point[0]+random_height, point[1]+random_width,:] = [255,255,255]
           
            newtissue_path = path.join(newtissue_dir, tissueName)
            seg_path =  path.join(seg_dir, tissueName)
                
            #save the current tissue segmentation label (ground truths)
            img_seg = Image.fromarray(seg_data).convert('RGB')
            #img_seg.save(seg_path)
            
            #save the current tissue
            img_tissue = Image.fromarray(newtissue_data).convert('RGB')
            #img_tissue.save(newtissue_path)

            
    #save the synthetic medical image        
    img_new = Image.fromarray(img_data).convert('RGB')
    #img_new.save(newimg_path)    


       
def find_OptmizedLocation(img_data, bgcolor, tissue_data):    
    """
    Find the optimal insertion position
    @param img_data: Background-filled image
    @param tissue_data: the current tissue
    """
    height_img,width_img,depth_img = img_data.shape
    height_tissue,width_tissue,_ = tissue_data.shape
    
    tryNo = 10
    tempLocations = []
    
    for no in range(tryNo):
       
        #Get a random position   
        random_height =random.randint(0, height_img - height_tissue-1)
        random_width = random.randint(0, width_img - width_tissue-1)
        
        #Get a candidate insertion area based on the random position
        candidateRegion = img_data[random_height: random_height + height_tissue -1, random_width: random_width+width_tissue-1,:]
        img_new = Image.fromarray(img_data).convert('RGB')
        
        #Calculate the ratio of the background pixels in the current candidate insertion area
        I_R =[]
        for i in range(height_tissue-1):
            for j in range(width_tissue-1):
                if (candidateRegion[i,j,0] >= bgcolor[0]-5) and (candidateRegion[i,j,0] <= bgcolor[0]+5):
                    I_R.append((i,j))
        print("I_R:",I_R)
        
        I_G =[]
        for i in range(height_tissue-1):
            for j in range(width_tissue-1):
                if (candidateRegion[i,j,1] >= bgcolor[1]-pixelDifference) and (candidateRegion[i,j,1] <= bgcolor[1]+pixelDifference):
                    I_G.append((i,j))
                    
        print("I_G:",I_G)
        
        I_B =[]
        for i in range(height_tissue-1):
            for j in range(width_tissue-1):
                if (candidateRegion[i,j,2] >= bgcolor[2]-pixelDifference) and (candidateRegion[i,j,2] <= bgcolor[2]+pixelDifference):
                    I_B.append((i,j))
         
        I_R = set(I_R)
        I_G = set(I_G)
        I_B = set(I_B)
        I_RGB = I_R & I_G & I_B
        
        ratio =round( float(len(I_RGB)) / (height_tissue * width_tissue), 2) 
        print("Background ratio: ", ratio) 
        if ratio > bgRatio: 
            opt_height = random_height  
            opt_width = random_width        
            break
        else:
            tempLocations.append([ratio, random_height, random_width])
    
    #Find the Maximum    
    if len(tempLocations) ==  tryNo: 
        tempLocations.sort(key=lambda x:x[0])
        maxRow = tempLocations[0]
        random_height =maxRow[1] 
        random_width = maxRow[2]   
    #print (random_height, ",", random_width)
    return random_height, random_width        
         

def find_OptmizedLocation2(img_data, bgcolor, tissue_data): 
    #Get a random position   
    height_img,width_img,depth_img = img_data.shape
    height_tissue,width_tissue,_ = tissue_data.shape
    random_height =random.randint(0, height_img - height_tissue-1)
    random_width = random.randint(0, width_img - width_tissue-1)
    return random_height, random_width     

def find_bgcolor(img_data):
    """
    Find background color
    """
    a = np.array([img_data[:,:,0], img_data[:,:,1], img_data[:,:,2]])    
    array_img1 =np.sum(a,0)
    max1 = np.max(array_img1)
    maxindex= np.where(array_img1==max1)
    bg_r = img_data [maxindex[0][0],maxindex[1][0],0]
    bg_g = img_data [maxindex[0][0],maxindex[1][0],1]
    bg_b = img_data [maxindex[0][0],maxindex[1][0],2]
    bgcolor = [int(bg_r), int(bg_g), int(bg_b)]
    return bgcolor
    
    
if __name__ == '__main__':
    #batch synthesis the medical images with multiple tissues
    
    batch_synthesis_multiple()
    
