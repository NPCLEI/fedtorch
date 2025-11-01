
import sys
sys.path.append('./')

# print(__file__)

import os

# def get_folder_name(file_path):
#     # 获取文件所在文件夹的路径
#     folder_path = os.path.dirname(file_path)
#     # 提取文件夹名称
#     folder_name = os.path.basename(folder_path)
#     return folder_name

# # 示例使用
# file_path = __file__
# folder_name = get_folder_name(file_path)
# print(folder_name)

from fedtorchPRO.multitest.printfile import printfile

printfile()