import os
from shutil import copy
import random

PATH = './data/traincase2/0Shape'

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

# 获取 photos 文件夹下除 .txt 文件以外所有文件夹名（即3种分类的类名）
file_path = PATH
flower_class = [cla for cla in os.listdir(file_path) if ".txt" not in cla]

# 创建 训练集train 文件夹，并由3种类名在其目录下创建3个子目录
mkfile('data/train')
for cla in flower_class:
    mkfile('data/train/' + cla)

# 创建 验证集val 文件夹，并由3种类名在其目录下创建3个子目录
mkfile('data/val')
for cla in flower_class:
    mkfile('data/val/' + cla)

# 划分比例，训练集 : 测试集 = 8:2
train_rate = 0.8

# 图像文件的扩展名
img_exts = ['.jpg', '.png', '.jpeg', '.bmp']

for cla in flower_class:
    cla_path = file_path + '/' + cla + '/'  # 某一类别动作的子目录
    images = [img for img in os.listdir(cla_path) if os.path.splitext(img)[-1].lower() in img_exts]  # iamges 列表存储了该目录下所有图像的名称
    num = len(images)
    train_index = random.sample(images, k=int(num * train_rate))  # 从images列表中随机抽取 k 个图像名称
    test_index = list(set(images) - set(train_index))
    for index, image in enumerate(images):
        if image in train_index:
            image_path = cla_path + image
            new_path = 'data/train/' + cla
            copy(image_path, new_path)
        else:
            image_path = cla_path + image
            new_path = 'data/val/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
    print()

print("processing done!")