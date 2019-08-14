from PIL import Image
import cv2 
from os import walk
from os import path 

def batchFill_inpaint(basepath):
    img_dir = path.join(basepath, "img")
    tissue_dir = path.join(basepath, "tissue1")
    newimg_dir = path.join(basepath, "background-filled")
    imgNames = next(walk(img_dir))[2]
    
    for img_name in imgNames:              
        img_path = path.join(img_dir, img_name)
        tissue_path= path.join(tissue_dir, img_name)
        newimg_path =path.join(newimg_dir, img_name)         
        Fill_inpaint(img_path, tissue_path, newimg_path)
        
              
def Fill_inpaint(img_path, tissue_path, newimg_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    tissue = cv2.imread(tissue_path)    
    tissue_data_bw = tissue.max(axis=2)
    mask = tissue_data_bw -255
    
    img_data = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)  
    
    img2 = Image.fromarray(img_data).convert('RGB')  
    #img2.show()
    img2.save(newimg_path)
    
    
if __name__ == '__main__':
    basepath="../dataset/skin/"
    batchFill_inpaint(basepath)