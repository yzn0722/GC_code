import os

import pyvips

# svs_split.py
import pyvips
import sys

file_path = "/home/zhangzifan/MaintoCode/2024-3-12-02/datasets/normal"

file_list = os.listdir(file_path)

for file in file_list:
    svs_path = os.path.join(file_path, file)
    img = pyvips.Image.new_from_file(svs_path, access='sequential')
    series = svs_path[0:len(svs_path) - 3]
    img.dzsave(series)
