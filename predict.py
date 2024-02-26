import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import ViT
from crop_tongue import * 

def main():
    folder_path = "./data/orin"
    cropped_img_path = "./data/cropped"
    seg_img_path = "./data/seg"
    files = os.listdir(folder_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    get_cropped_model(img_path=folder_path, output_path_crop=cropped_img_path, output_path_seg=seg_img_path) 
    print("All images have been cropped successfully!\n")

    # load image
    # img_path = ""
    for file in files:
        print('\nname:', file)
        img_path = os.path.join(folder_path, file)
        # img_path = os.path.join(folder_path, file)
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)


        # color classification
        json_path = './class_indices_color.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)
    
        # create model_color
        model_color = ViT(image_size=224, patch_size=32, num_classes=5, dim=1024, depth=6, heads=16, mlp_dim=2048, channels=3, dropout=0.1, emb_dropout=0.1).to(device)
        # load model_color weights
        weights_path = "./weight/vit_color.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model_color.load_state_dict(torch.load(weights_path, map_location=device))

        model_color.eval()
        with torch.no_grad():
            # predict class
            output_color = torch.squeeze(model_color(img.to(device))).cpu()
            predict_color = torch.softmax(output_color, dim=0)
            predict_cla_color = torch.argmax(predict_color).numpy()
    
        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla_color)],
        #                                          predict_color[predict_cla_color].numpy())
        # plt.title(print_res)
        for i in range(len(predict_color)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict_color[i].numpy()))
        
        print("\n")

        # shape classification
        json_path = './class_indices_shape.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)
    
        # create model_shape
        model_shape = ViT(image_size=224, patch_size=32, num_classes=3, dim=1024, depth=6, heads=16, mlp_dim=2048, channels=3, dropout=0.1, emb_dropout=0.1).to(device)
        # load model_color weights
        weights_path = "./weight/vit_shape.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model_shape.load_state_dict(torch.load(weights_path, map_location=device))

        model_shape.eval()
        with torch.no_grad():
            # predict class 
            output_shape = torch.squeeze(model_shape(img.to(device))).cpu()
            predict_shape = torch.softmax(output_shape, dim=0)
            predict_cla_shape = torch.argmax(predict_shape).numpy()

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla_shape)],
                                                 predict_shape[predict_cla_shape].numpy())
        plt.title(print_res)
        for i in range(len(predict_shape)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict_shape[i].numpy()))
            


if __name__ == '__main__':
    main()
