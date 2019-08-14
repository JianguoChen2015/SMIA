import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage, misc

filepath="../dataset/"

#边界检测与叠加
def  edge_Detection(image_path, edge_location, saved_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  #加载灰度图像
    edges = cv2.Canny(img_gray,35,35)  #使用35x35高斯滤波器去除图像中的噪声,阈值可调
    
    #显示边缘效果图
    #plt.subplot(121),plt.imshow(img,cmap = 'gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #plt.show()
    cv2.imwrite(edge_location, edges)
    
    #两张图片叠加 (要求两张图必须是同一个size)
    img_b, img_g, img_r = cv2.split(img) #分别取出三个通道,边界叠加在三个通道都执行
    
    #alpha，beta，gamma可调
    alpha = 0.5
    beta = 1-alpha
    gamma = 0.3
    b_add = cv2.addWeighted(img_b, alpha, edges, beta, gamma)
    g_add = cv2.addWeighted(img_g, alpha, edges, beta, gamma)
    r_add = cv2.addWeighted(img_r, alpha, edges, beta, gamma)
    
    #合并通道
    #opencv对于读进来的图片的通道排列是BGR，而不是主流的RGB
    img_add = cv2.merge((b_add, g_add, r_add))
    
    #显示合并后的图片
    #cv2.namedWindow('addImage')
    #cv2.imshow('img_add',img_add)
    #cv2.waitKey()
    #cv2.destroyAllWindows()    
    
    cv2.imwrite(saved_path, img_add)


#灰度叠加
#mergechannel要合并的通道
def grayScale(image_path, mergechannel, saved_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  #加载灰度图像
    img_b, img_g, img_r = cv2.split(img) #分别取出三个通道
    
    #合并通道, OpenCV是按BGR格式
    if (mergechannel=='R'):
        img_merged = cv2.merge((img_b, img_g, img_gray))     
    elif (mergechannel=='G'):
        img_merged = cv2.merge((img_b, img_gray, img_r))  
    elif (mergechannel=='B'):
        img_merged = cv2.merge((img_gray, img_g, img_r))          
    cv2.imwrite(saved_path, img_merged)
 
''' 
#kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0
def sharpen(image_path, saved_path, ):
    """Return a sharpened version of the image, using an unsharp mask."""
    img = cv2.imread('my-image.jpg')
    kernel = np.array([[-1,-1,-1], 
                   [-1, 9,-1],
                   [-1,-1,-1]])
    sharpened = cv2.filter3D(img, -1, kernel) # applying the sharpening kernel to the input image & displaying it.

    
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
       
    cv2.imwrite(saved_path, sharpened)
'''
def sharpen(image_path, saved_path):
    img = misc.imread(image_path).astype(np.float)  # read as float
    kernel = np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]).reshape((3, 3, 1))
    #kernel = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape((3, 3, 1))
    
    # here we do the convolution with the kernel
    imgsharp = ndimage.convolve(img, kernel, mode='nearest')
    # then we clip (0 to 255) and convert to unsigned int
    imgsharp = np.clip(imgsharp, 0, 255).astype(np.uint8)
    
    imgsharp2 = Image.fromarray(imgsharp)
    imgsharp2.save(saved_path)

if __name__ == '__main__':
    #图像路径及文件名
    file_img = filepath + "cell114"
    cell_loc = filepath + "cell1-015"
    cell_unloc = filepath + "cell1-15"
    
    '''
    #1.2.1 灰度叠加
    grayScale(file_img + '.png',  "R", file_img + '-gray-R.png')
    grayScale(cell_loc + '.png',  "R", cell_loc + '-gray-R.png')
    grayScale(cell_unloc + '.png',  "R", cell_unloc + '-gray-R.png')
    
    grayScale(file_img + '.png',  "G", file_img + '-gray-G.png')
    grayScale(cell_loc + '.png',  "G", cell_loc + '-gray-G.png')
    grayScale(cell_unloc + '.png',  "G", cell_unloc + '-gray-G.png')
    
    grayScale(file_img + '.png',  "B", file_img + '-gray-B.png')
    grayScale(cell_loc + '.png',  "B", cell_loc + '-gray-B.png')
    grayScale(cell_unloc + '.png',  "B", cell_unloc + '-gray-B.png')
    '''
    
    
    #1.2.2 边界检测
    edge_Detection(file_img + '.png',  file_img + '-edge.png', file_img + '-edgeadd.png')
    #edge_Detection(cell_loc + '.png',  cell_loc + '-edge.png', cell_loc + '-edgeadd.png')
    #edge_Detection(cell_unloc + '.png', cell_unloc + '-edge.png', cell_unloc + '-edgeadd.png')
    
    #1.2.3 锐化sharpen
    #sharpen(file_img + '.png', file_img + '-sharpen.png')
    #sharpen(cell_loc + '.png', cell_loc + '-sharpen.png')
    #sharpen(cell_unloc + '.png', cell_unloc + '-sharpen.png')
    
    