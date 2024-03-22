import sys
import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
from train_supervision import *
import cv2
import numpy as np
import torch

from torch import nn
import torchvision.transforms as transforms
import albumentations as albu


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [159, 129, 183]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 195, 128]
    return mask_rgb
from PIL import Image

def img_writer(inp,save_path):
    mask=inp
    mask_tif = label2rgb(mask)
    mask_tif = cv2.cvtColor(mask_tif, cv2.COLOR_RGB2BGR)
    image = Image.fromarray(mask_tif)
    resized_image = image.resize((512, 512))
    mask_tif = np.array(resized_image)
    return mask_tif

def process(img):
    # img = np.array(img)
    aug = albu.Normalize()(image=img)
    img = aug['image']
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    resize = transforms.Resize((1024, 1024), interpolation=2)
    img = resize(img)
    return img

class SegDecetor:
    def __call__(self,image,path):
        device='cuda'
        config = py2cfg("path/to/config.py")
        model = Supervision_Train.load_from_checkpoint("path/to/seg",config=config)
        model.to(device)
        model.eval()
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
        img=process(image).unsqueeze(0)
        with torch.no_grad():
            raw_predictions = model(img.to(device))
            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
        image_np=img_writer(mask,path)
        return image_np

