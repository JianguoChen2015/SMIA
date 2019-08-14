from PIL import Image
import numpy as np
from os import walk
from os import path
import shutil


pixelDifference = 60  
bgRatio = 0.3 

def batchFill_bgcolor(basepath):
    g = walk(basepath)  

    for _,dir_list,file_list in g:  
        for dir_name in dir_list:              
            print(dir_name)
            img_dir = path.join(basepath, dir_name, "img")
            seg_dir = path.join(basepath, dir_name, "seg")
            newimg_dir = path.join(basepath, dir_name, "background-filled")
            imgNames = next(walk(img_dir))[2]
            if (len(imgNames)>0):        
                img_name = imgNames[0]
                print(img_name)
                img_path = path.join(img_dir, img_name)
                newimg_path = path.join(newimg_dir, img_name)
                Fill_bgcolor2(img_path, seg_dir, newimg_path )
        
    
def Fill_bgcolor2(img_path, seg_dir, newimg_path):
    img = Image.open(img_path)
    img_data = np.array(img)
    height,width,_ = img_data.shape

    a = np.array([img_data[:,:,0], img_data[:,:,1], img_data[:,:,2]])

    array_img1 =np.sum(a,0)
    max1 = np.max(array_img1)
    maxindex= np.where(array_img1==max1)
    bg_r = img_data [maxindex[0][0],maxindex[1][0],0]
    bg_g = img_data [maxindex[0][0],maxindex[1][0],1]
    bg_b = img_data [maxindex[0][0],maxindex[1][0],2]
    bgcolor = [int(bg_r), int(bg_g), int(bg_b)]
    #print("background:", bgcolor)   
        
    fileNames = next(walk(seg_dir))[2]
    for file in fileNames:
        print(file)
        seg_path = path.join(seg_dir,file)
        seg = Image.open(seg_path)
        seg_data = np.array(seg)
        non_empty_points = np.transpose(np.nonzero(seg_data))
        
        for point in non_empty_points:
            for i in range(-10,10):
                if (point[0]+i >=0 and point[1]+i >=0 ):
                    if (point[0]+i < height and point[1]+i < width):
                        img_data[point[0]+i, point[1]+i,:] = bgcolor 
        img2 = Image.fromarray(img_data).convert('RGB')
        #img2.show()
        img2.save(newimg_path)
    img2 = Image.fromarray(img_data).convert('RGB')
    #img2.show()
    img2.save(newimg_path)


if __name__ == '__main__':
    basepath="../dataset/BMSs/"
    batchFill_bgcolor(basepath)

    
    
    
    