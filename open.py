

# 打开病理切片文件
import torch
import openslide
from PIL import Image
slide = openslide.OpenSlide('TCGA-CG-4462-01A-01-BS1.8655c0a9-39ee-4e10-ac16-d501a9c096a6.svs')
# 获取病理切片的宽度和高度
width = slide.dimensions[0]
height = slide.dimensions[1]
level_count=slide.level_count



print('lever_count:',level_count)
# 获取病理切片的级别数量
levels = slide.level_count

# 获取病理切片的每个级别的尺寸
for level in range(levels):
    level_width = slide.level_dimensions[level][0]
    level_height = slide.level_dimensions[level][1]
    print(f'Level {level}: Width={level_width}, Height={level_height}')

# 关闭病理切片文件
slide.close()
