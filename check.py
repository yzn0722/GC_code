# import os
# import shutil
# import re
#
#
# def get_new_filename(base_name, extension, existing_files, start_num):
#     """
#     生成一个新的文件名，确保它在existing_files中不存在。
#     """
#     new_name = f"{start_num:04d}.{extension}"
#     while new_name in existing_files:
#         start_num += 1
#         new_name = f"{start_num:04d}.{extension}"
#     return new_name
#
#
# def rename_and_copy_images(src_dir, dst_dir, pattern=r'.*\.(jpg|jpeg|png|gif|bmp)$', start_num=1):
#     """
#     重命名并复制图片到新的文件夹，保留原始扩展名。
#     """
#     # 确保目标文件夹存在
#     if not os.path.exists(dst_dir):
#         os.makedirs(dst_dir)
#
#         # 获取目标文件夹中已存在的文件名集合
#     existing_files = set(os.listdir(dst_dir))
#
#     # 遍历源文件夹及其所有子文件夹
#     for root, dirs, files in os.walk(src_dir):
#         for file in files:
#             if re.match(pattern, file):
#                 # 提取文件名和扩展名
#                 base_name, extension = os.path.splitext(file)
#                 extension = extension.lower().lstrip('.')  # 确保扩展名是小写且没有前导点
#
#                 # 构建源文件的完整路径
#                 src_file_path = os.path.join(root, file)
#
#                 # 生成新的文件名
#                 new_filename = get_new_filename(base_name, extension, existing_files, start_num)
#                 start_num += 1  # 更新起始数字，为下一个文件做准备
#
#                 # 构建目标文件的完整路径
#                 dst_file_path = os.path.join(dst_dir, new_filename)
#
#                 # 复制并重命名文件
#                 shutil.copy2(src_file_path, dst_file_path)
#                 print(f"Copied and renamed {src_file_path} to {dst_file_path}")
#
#             # 使用示例
#
#
# src_dir = "/home/zhangzifan/MaintoCode/2024-3-12-02/datasets/normal"  # 源文件夹路径
# dst_dir = "/home/zhangzifan/MaintoCode/2024-3-12-02/datasets/normal_image"  # 目标文件夹路径
# rename_and_copy_images(src_dir, dst_dir)


import os
import random


def delete_random_files(directory, percentage=0.5):
    """
    随机删除指定文件夹下指定百分比的文件。

    :param directory: 要处理的文件夹路径。
    :param percentage: 要删除的文件百分比，默认为0.9（90%）。
    """
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        # 筛选出要删除的文件数量
        num_to_delete = int(len(files) * percentage)

        # 如果没有文件或文件数少于要删除的数量，则跳过当前目录
        if num_to_delete == 0 or len(files) < num_to_delete:
            continue

            # 随机选择文件列表中的文件
        files_to_delete = random.sample(files, num_to_delete)

        # 删除选中的文件
        for file in files_to_delete:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

            # 使用示例


directory_to_clean = "/home/zhangzifan/MaintoCode/2024-03-12-02/datasets/abnormal"  # 替换为你要清理的文件夹路径
delete_random_files(directory_to_clean)

# import os
# from PIL import Image
# from multiprocessing import Pool, cpu_count
#
#
# def count_bright_pixels(image_path, threshold=(220, 220, 220)):
#     """
#     统计图片中RGB色值大于给定阈值的像素数量，并计算占比。
#
#     :param image_path: 图片路径
#     :param threshold: RGB色值阈值，默认为(240, 240, 240)
#     :return: 大于阈值的像素占比（百分比）
#     """
#     with Image.open(image_path) as img:
#         if img.mode != 'RGB':
#             img = img.convert('RGB')
#         width, height = img.size
#         total_pixels = width * height
#         count = sum(1 for x in range(width) for y in range(height)
#                     if all(c > threshold[i] for i, c in enumerate(img.getpixel((x, y)))))
#
#     percentage = (count / total_pixels) * 100
#     return percentage
#
#
# def process_image(file_path, threshold_percentage=80):
#     percentage = count_bright_pixels(file_path)
#     if percentage > threshold_percentage:
#         print(f"Deleting file: {file_path} because bright area percentage is {percentage:.2f}%.")
#         os.remove(file_path)
#     else:
#         print(f"File: {file_path}, Bright Area Percentage: {percentage:.2f}%. Keeping the file.")
#
#
# def process_folder(folder_path, threshold_percentage=80):
#     """
#     处理文件夹中的所有图片，删除亮色区域占比大于给定阈值的图片。
#
#     :param folder_path: 文件夹路径
#     :param threshold_percentage: 亮色区域占比的阈值，默认为80%
#     """
#     file_paths = []
#     # 遍历文件夹中的所有文件
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#                 file_paths.append(os.path.join(root, file))
#
#     # 多进程处理图片
#     with Pool(processes=cpu_count()) as pool:
#         pool.map(process_image, file_paths)
#
#
# if __name__ == "__main__":
#     folder_to_process = "/home/zhangzifan/MaintoCode/2024-3-12-02/datasets/abnormal_image_temp"  # 替换为你的图片文件夹路径
#     process_folder(folder_to_process)
