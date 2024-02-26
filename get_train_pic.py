from crop_tongue import *
import os
from PIL import Image
from tqdm import tqdm

# 指定文件夹路径
FILE_PATH = "D:\\00study\\Artificial_Intelligence\\HACI\\TCM\\Model\\vggnet\\Test3_vggnet\\data\\traincase2\\New_added"

# 获取所有子文件夹的名称
subfolders = [os.path.basename(f.path) for f in os.scandir(FILE_PATH) if f.is_dir()]

# 遍历每个子文件夹
for subfolder in tqdm(subfolders):
    dataset_path = os.path.join(FILE_PATH, 'Processed', subfolder)
    if not os.path.exists(dataset_path):
        print("Creating folder: {}".format(dataset_path))
        os.makedirs(dataset_path)
    output_path_crop = os.path.join(dataset_path, "crop")
    print("output_path_crop: {}".format(output_path_crop))
    output_path_seg = os.path.join(dataset_path, "seg")
    print("output_path_seg: {}".format(output_path_seg))
    get_cropped_model(img_path=os.path.join(FILE_PATH, subfolder), output_path_crop=output_path_crop, output_path_seg=output_path_seg)

print("All images have been processed successfully!😁\n")