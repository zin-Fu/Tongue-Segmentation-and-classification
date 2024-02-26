from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, jaccard_score
# set seeds
torch.manual_seed(2023)
np.random.seed(2023)
from skimage import io
from  utils_metrics import *
from skimage import transform, io, segmentation
from segment.yolox import YOLOX
import random
import warnings

def get_cropped_model(img_path, output_path_seg, output_path_crop):
    # 永久性地忽略指定类型的警告
    warnings.filterwarnings("ignore", category=UserWarning)
    #########################################################################################################
    ts_img_path = img_path
    # output_path = output_path
    model_type = 'vit_b'
    checkpoint = './pretrained_model/tonguesam.pth'
    device = 'cuda:0' 
    path_out_crop = output_path_crop
    path_out_seg = output_path_seg
    segment=YOLOX()
    ##############################################################################################################
    
    if not os.path.exists(path_out_seg):
        os.makedirs(path_out_seg)
    if not os.path.exists(path_out_crop):
        os.makedirs(path_out_crop)
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    sam_model.eval()

    for f in os.listdir(ts_img_path):   
        with torch.no_grad():             
            image_data = io.imread(join(ts_img_path, f))
            ori_image = image_data # 用最初的图像裁剪
        
            if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
                image_data = image_data[:, :, :3]
            if len(image_data.shape) == 2:
                image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
            
            lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0
            image_data_pre[image_data == 0] = 0
            image_data_pre = transform.resize(image_data_pre, (400, 400), order=3, preserve_range=True, mode='constant', anti_aliasing=True)
            image_data_pre = np.uint8(image_data_pre)
            
            H, W, _ = image_data_pre.shape
            sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
            resize_img = sam_transform.apply_image(image_data_pre)
            resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
            input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])        
            ts_img_embedding = sam_model.image_encoder(input_image)      

            img = image_data_pre
            boxes = segment.get_prompt(img)
                    
            if boxes is not None:
                sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)                   
                box = sam_trans.apply_boxes(boxes, (400,400))                                                
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)            
            else:            
                box_torch = None                
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            
            # 使用Mask_Decoder生成分割结果
            medsam_seg_prob, _ = sam_model.mask_decoder(
                image_embeddings=ts_img_embedding.to(device),
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )                        
            medsam_seg_prob =medsam_seg_prob.cpu().detach().numpy().squeeze()        
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            
            medsam_seg=cv2.resize(medsam_seg,(400,400))       
            
            
            pred = cv2.Canny(cv2.resize((medsam_seg != 0).astype(np.uint8) * 255, (400, 400)), 100, 200)

            # 获取分割裁剪后的图
            orin_img = Image.fromarray(img)
            orin_img = orin_img.convert("RGBA")
            orin_img_np = np.array(orin_img)

            # 将原图转换为RGB模式
            orin_img_np = cv2.cvtColor(orin_img_np, cv2.COLOR_RGBA2RGB)

            # 获取pred[i, j] != 0的所有点
            y, x = np.nonzero(pred)

            # 创建一个和原图一样大小的全透明的图像
            mask = np.zeros_like(orin_img_np)

            # 创建一个和原图一样大小的全黑的图像，用于存储掩码
            mask_img = np.zeros((orin_img_np.shape[0], orin_img_np.shape[1]), dtype=np.uint8)

            # 使用这些点创建一个掩码，该掩码在封闭图形内部的点处为True，其他地方为False
            cv2.fillPoly(mask_img, [np.column_stack((x, y))], 1)

            # 将掩码转换为和原图一样的形状
            mask_img = np.stack([mask_img]*3, axis=-1)

            # 使用这个掩码和原图进行按位与操作，得到只有封闭图形有颜色的图像
            mask = np.where(mask_img, orin_img_np, 0)

            # 将mask转换为PIL图像，并转换为RGBA模式
            mask_pil = Image.fromarray(mask).convert('RGB')
            # mask_pil = mask_pil.convert('RGB')
            #保存到 path_out_seg路径
            mask_pil.save(os.path.join(path_out_seg, f'{os.path.splitext(f)[0]}.jpg'))


            image1 = Image.fromarray(medsam_seg)
            image2 = Image.fromarray(img)

            image1 = image1.resize(image2.size).convert("RGBA")
            image2 = image2.convert("RGB")
            data1 = image1.getdata()

            new_image = Image.new("RGBA", image2.size)
            new_data = [(0, 0, 128, 96) if pixel1[0] != 0 else (0, 0, 0, 0) for pixel1 in data1]

            new_image.putdata(new_data)
            if boxes is not None:              
                # draw = ImageDraw.Draw(image2)
                # draw.rectangle([boxes[0],boxes[1],boxes[2],boxes[3]],fill=None, outline=(0, 255, 0), width=1)  # 用红色绘制方框的边框，线宽为2
                # 获取矩形裁剪后的图
                cropped = image2.crop((boxes[0],boxes[1],boxes[2],boxes[3]))
                # cropped = cropped.convert("RGB")
                cropped.save(os.path.join(path_out_crop, f'{os.path.splitext(f)[0]}.jpg'))
            print("Finish processing {} 🚀".format(f))

if __name__ == '__main__':
    get_cropped_model(img_path='./data/from_web/orin', output_path_crop='./data/from_web/crop', output_path_seg='./data/from_web/seg')       