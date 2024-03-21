import os
import random
import shutil


def split_dataset(input_dir, output_train_dir, output_test_dir, split_ratio=0.8):
    # 创建输出文件夹
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    # 遍历每个类别的文件夹
    for class_dir in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_dir)
        if not os.path.isdir(class_path):
            continue

        # 获取该类别下所有图片的路径
        images = [img for img in os.listdir(class_path) if img.endswith('.jpeg') or img.endswith('.png')]

        # 打乱图片顺序
        random.shuffle(images)

        # 计算划分的索引
        split_index = int(len(images) * split_ratio)

        # 划分训练集和测试集
        train_images = images[:split_index]
        test_images = images[split_index:]

        # 将图片复制到对应的训练集和测试集文件夹中
        for img in train_images:
            src_path = os.path.join(class_path, img)
            dest_path = os.path.join(output_train_dir, class_dir, img)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(src_path, dest_path)

        for img in test_images:
            src_path = os.path.join(class_path, img)
            dest_path = os.path.join(output_test_dir, class_dir, img)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(src_path, dest_path)


# 设置输入和输出文件夹路径
input_dir = '/home/zhangzifan/MaintoCode/2024-03-12-02/datasets'
output_train_dir = '/home/zhangzifan/MaintoCode/2024-03-12-02/train'
output_test_dir = '/home/zhangzifan/MaintoCode/2024-03-12-02/test'

# 划分数据集
split_dataset(input_dir, output_train_dir, output_test_dir)
