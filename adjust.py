import os
from PIL import Image
import numpy as np


def modify_pixels(image):
    # 将图像转换为numpy数组
    image_np = np.array(image)
    image_np[image_np == 1] = 255


    # 将numpy数组转换回图像
    return Image.fromarray(image_np)


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹A中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            image = Image.open(file_path)

            # 修改图像
            modified_image = modify_pixels(image)

            # 保存到文件夹B
            modified_image.save(os.path.join(output_folder, filename))


# 设置文件夹A和B的路径
folder_a = r''
folder_b = r''
process_folder(folder_a, folder_b)
