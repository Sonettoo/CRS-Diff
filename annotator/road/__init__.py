import os
import sys
import cv2
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models import SGCNNet
class RoadDector():
    def __call__(self,img,path='',train=True):
        device=torch.device('cuda')
        model=SGCNNet.SGCN_res50(num_classes=2)
        check_point="path/to/road.pth"
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.2304,0.3295,0.4405],std=[0.1389,0.1316,0.1278])
            ]
        )
        model.load_state_dict(torch.load(check_point, map_location='cpu'))
        model.to(device)
        model.eval()
        image = transform(img).float().cuda()
        resize = transforms.Resize((224, 224), interpolation=2)
        image = resize(image)
        image = image.unsqueeze(0)            

        output = model(image)
        _, pred = output.max(1)
        pred = pred.view(224, 224)
        
        mask_im = pred.cpu().numpy().astype(np.uint8)
        cv2.imwrite(path, mask_im)
        im = cv2.imread(path)
        def translabeltovisual(img, path,num_classes):
            im = img
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            for i in range(im.shape[0]):
                for j in range(im.shape[1]):
                    pred_class=im[i][j][0]
                    if pred_class!=0 and pred_class!=1:
                        pred_class=1
                    im[i][j] = num_classes[pred_class]
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            return im
        num_classes=[[0,0,0], [255,255,255]]
        if train:
            return translabeltovisual(im,path,num_classes)
        else:
            image=translabeltovisual(im,path,num_classes)
            image = Image.fromarray(image)
            resized_image = image.resize((512, 512))
            image = np.array(resized_image)
            return image