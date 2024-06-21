import cv2
import os
import random


def random_crop(image, crop_height, crop_width):
    """
    从图像中随机裁剪出指定大小的区域

    :param image: 输入图像
    :param crop_height: 裁剪区域的高度
    :param crop_width: 裁剪区域的宽度
    :return: 裁剪后的图像
    """
    height, width, _ = image.shape

    # 确保裁剪尺寸不超过原图尺寸
    if crop_height > height or crop_width > width:
        raise ValueError("裁剪尺寸超过原图尺寸")

    # 计算随机起始点
    x_start = random.randint(0, width - crop_width)
    y_start = random.randint(0, height - crop_height)

    # 裁剪图像
    cropped_image = image[y_start:y_start + crop_height, x_start:x_start + crop_width]

    return cropped_image


def process_images(input_folder, output_folder, crop_height, crop_width, num_crops):
    """
    处理指定文件夹中的所有图片，随机裁剪并保存到输出文件夹

    :param input_folder: 输入图片文件夹
    :param output_folder: 输出图片文件夹
    :param crop_height: 裁剪区域的高度
    :param crop_width: 裁剪区域的宽度
    :param num_crops: 每张图片要裁剪出的图像数量
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    crop_count = 0  # 初始化裁剪图像的计数器

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for _ in range(num_crops):
                cropped_image = random_crop(image, crop_height, crop_width)
                crop_count += 1
                output_filename = f"{crop_count}.png"
                output_path = os.path.join(output_folder, output_filename)
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, cropped_image)
                print(f"Processed {output_filename} and saved to {output_path}")


# 设置输入文件夹和输出文件夹路径
input_folder = '../Data/SData/Labelpng'
output_folder = '../Data/PData/Test/Label'

# 设置裁剪尺寸
crop_height = 128  # 设置你想要的高度
crop_width = 128  # 设置你想要的宽度
num_crops = 5  # 每张图片裁剪出的随机图像数量

# 处理图片
process_images(input_folder, output_folder, crop_height, crop_width, num_crops)
