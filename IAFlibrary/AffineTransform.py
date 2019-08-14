from PIL import Image
from enum import Enum
import numpy as np
import random
import cv2

SEED = 9
random.seed(SEED)
filepath="../dataset/"


class ImageSize(Enum):
    FULL_SIZE = 1
    HALF_SIZE = 0.5

#仿射变换
def affine_transform_image(image_path, saved_path):
    """
    Perform affine transformation in the image

    @param image_path: The path to the image to edit
    @param saved_path: Path to save the transformed image
    """
    img = cv2.imread(image_path)
    width, height, chanel = img.shape
    src_tri = np.array([[0, 0],
                        [width - 1, 0],
                        [0, height - 1]]).astype(np.float32)
    dst_tri = np.array([[0, height * random.uniform(0, 1)],
                        [width * random.uniform(0, 1), height * random.uniform(0, 1)],
                        [width * random.uniform(0, 1), height * random.uniform(0, 1)]]).astype(np.float32)
    warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
    img_transformed = cv2.warpAffine(img, warp_mat, (height, width))
    cv2.imwrite(saved_path, img_transformed)

#透视变换
def perspective_transform_image(image_path, saved_path):
    """
    Perform perspective transformation in the image

    @param image_path: The path to the image to edit
    @param saved_path: Path to save the transformed image
    """
    img = cv2.imread(image_path)
    width, height, chanel = img.shape
    src_tri = np.array([[0, 0],
                        [width - 1, 0],
                        [0, height - 1],
                        [width - 1, height - 1]]).astype(np.float32)
    dst_tri = np.array([[0, height * random.uniform(0, 1)],
                        [width * random.uniform(0, 1), height * random.uniform(0, 1)],
                        [width * random.uniform(0, 1), height * random.uniform(0, 1)],
                        [width * random.uniform(0, 1), height - 1]]).astype(np.float32)
    warp_mat = cv2.getPerspectiveTransform(src_tri, dst_tri)
    img_transformed = cv2.warpPerspective(img, warp_mat, (height, width))
    cv2.imwrite(saved_path, img_transformed)


if __name__ == '__main__':
    #图像路径及文件名
    file_img = filepath + "cell1"
    cell_loc = filepath + "cell1-015"
    cell_unloc = filepath + "cell1-15"
    
    #1.2.1 仿射变换   
    #affine_transform_image(file_img + '.png', file_img + '-affine-transformed.png')
    #affine_transform_image(cell_loc + '.png', cell_loc + '-affine-transformed.png')
    #affine_transform_image(cell_unloc + '.png', cell_unloc + '-affine-transformed.png')
    
    #1.2.2 透视变换
    perspective_transform_image(file_img + '.png', file_img + '-perspective-transformed.png')
    perspective_transform_image(cell_loc + '.png', cell_loc + '-perspective-transformed.png')
    perspective_transform_image(cell_unloc + '.png', cell_unloc + '-perspective-transformed.png')
