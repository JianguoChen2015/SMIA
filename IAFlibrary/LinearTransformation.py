from PIL import Image
from enum import Enum
import random

filepath="../dataset/"

class FlipDirection(Enum):
    HORIZONTAL = Image.FLIP_LEFT_RIGHT
    VERTICAL = Image.FLIP_TOP_BOTTOM

class LinearTransformation:

    #1.1.1 翻转图像
    def flip_image(self, image_path, flip_direction, saved_path):
        """
        Flip the image
    
        @param image_path: The path to the image to edit
        @param flip_direction: The direction to flip the image
        @param saved_path: Path to save the flipped image
        """
        img = Image.open(image_path)
        flipped_image = img.transpose(flip_direction.value)
        flipped_image.save(saved_path)
    
    
    #1.1.2 裁剪图像
    def crop_image(self, image_path, ratio, saved_path):
        """
        Crop the image
    
        @param image_path: The path to the image to edit
        @param ratio: This defines ratio between cropped image and original image
        @param saved_path: Path to save the cropped image
        """
        img = Image.open(image_path)
        width, height = img.size
        cropped_width = int(width * ratio)
        cropped_height = int(height * ratio)
        left_upper_point_x = random.randint(0, width - cropped_width)
        left_upper_point_y = random.randint(0, height - cropped_height)
        cropped_image = img.crop(
            (left_upper_point_x, left_upper_point_y, left_upper_point_x + cropped_width,
             left_upper_point_y + cropped_height))
        cropped_image.save(saved_path)
    
    
    #1.1.3 旋转图片
    def rotate_image(self, image_path, degrees_to_rotate, saved_path):
        """
        Rotate the given photo the amount of given degree (anti-clockwise),
        and save it
     
        @param image_path: The path to the image to edit
        @param degrees_to_rotate: The number of degrees to rotate the image
        @param saved_path: Path to save the rotated image
        """
        img = Image.open(image_path)
        rotated_image = img.rotate(degrees_to_rotate)
        rotated_image.save(saved_path)
    
    
    #1.1.4 缩放(放大)
    def scale_image(self, image_path, ratio, saved_path):
        """
        Zoom into a region of the image and scale it up to the original size
    
        @param image_path: The path to the image to edit
        @param ratio: This defines ratio between cropped image and original
            image
        @param saved_path: Path to save the scaled image
        """
        img = Image.open(image_path)
        width, height = img.size
        cropped_width = int(width * ratio)
        cropped_height = int(height * ratio)
        left_upper_point_x = random.randint(0, width - cropped_width)
        left_upper_point_y = random.randint(0, height - cropped_height)
        cropped_image = img.crop(
            (left_upper_point_x, left_upper_point_y, left_upper_point_x + cropped_width,
             left_upper_point_y + cropped_height))
        scaled_image = cropped_image.resize((width, height), Image.ANTIALIAS)
        scaled_image.save(saved_path, quality=100)


if __name__ == '__main__':
    #图像路径及文件名
    file_img = filepath + "cell1"
    cell_loc = filepath + "cell1-015"
    cell_unloc = filepath + "cell1-15"
    
    lt = LinearTransformation()
    
    #1.1.1 翻转
    ##水平翻转
    lt.flip_image(file_img + ".png", FlipDirection.HORIZONTAL, file_img + "-flipped-horizontal.png")
    lt.flip_image(cell_loc + ".png", FlipDirection.HORIZONTAL, cell_loc + "-flipped-horizontal.png")
    lt.flip_image(cell_unloc + ".png", FlipDirection.HORIZONTAL, cell_unloc + "-flipped-horizontal.png")
    ##垂直翻转
    lt.flip_image(file_img + ".png", FlipDirection.VERTICAL, file_img + "-flipped-vertical.png")
    lt.flip_image(cell_loc + ".png", FlipDirection.VERTICAL, cell_loc + "-flipped-vertical.png")
    lt.flip_image(cell_unloc + ".png", FlipDirection.VERTICAL, cell_unloc + "-flipped-vertical.png")
    
    
    #1.1.2 裁剪crop
    lt.crop_image(file_img + ".png", 0.8, file_img + '-cropped.png')
    lt.crop_image(cell_loc + ".png", 0.8, cell_loc + '-cropped.png')
    lt.crop_image(cell_unloc + ".png", 0.8, cell_unloc + '-cropped.png')
    
    
    #1.1.3 旋转Rotate
    lt.rotate_image(file_img + ".png", 30, file_img + '-rotated.png')
    lt.rotate_image(cell_loc + '.png', 30, cell_loc + '-rotated.png')
    lt.rotate_image(cell_unloc + '.png', 30, cell_unloc + '-rotated.png')
    
    
    #1.1.4 缩放Scaling
    lt.scale_image(file_img + '.png', 0.8, file_img + '-scaled.png')
    lt.scale_image(cell_loc + '.png', 0.8, cell_loc + '-scaled.png')
    lt.scale_image(cell_unloc + '.png', 0.8, cell_unloc + '-scaled.png')
    
    
    #affine_transform_image('cell1.png', 'cell1-affine-transformed.png')
    #perspective_transform_image('cell1.png', 'cell1-perspective-transformed.png')
