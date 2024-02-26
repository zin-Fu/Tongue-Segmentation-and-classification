# Tongue-Segmentation-and-classification

😁This project utilizes the SAM algorithm for segmenting tongues in images, followed by classification using ViT.

## Prediction

1. Place the tongue images you want to recognize in the `data/orin` folder.
2. Run the `predict.py` script. You can choose one of the following folders as the inference image path:
   - To use the cropped tongue bounding box images, set the variable `cropped_img_path` to `./data/cropped`.
   - To use the segmented tongue images, set the variable `seg_img_path` to `./data/seg`.

## Classification Json

- The current classification is based on tongue color and tongue coating status. Details can be found in the 2  `class_indices.json` files.

## Fine-tuning

If you need to perform fine-tuning, you can use the code provided in `train.py`.

## Pretrained Models

- Download the tongue segmentation model (`tonguesam.pth`) from the following link: [Download here](https://pan.baidu.com/s/1zG0jpYshlBs3lcdy4F37dQ?pwd=xtfg) (Place `tonguesam.pth` in the `./pretrained_model/` folder).
- Download the ViT model (`ViT.pth`) from the following link: [Download here](https://drive.google.com/drive/folders/1VfsYQrWYqJkDCaKgNZb22_evbMona_fG?usp=drive_link) (Place `ViT.pth` in the `./weight/` folder).

