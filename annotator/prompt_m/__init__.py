import cv2
from PIL import Image

import torch
from transformers import AutoProcessor, CLIPModel

from annotator.util import annotator_ckpts_path


class PromptDetector:
    def __init__(self):

        model_name = "path/to/clip_model-ft"
        model_name_un="path/to/clip_model"

        model = CLIPModel.from_pretrained(model_name, cache_dir=annotator_ckpts_path).cuda().eval()
        self.processor = AutoProcessor.from_pretrained(model_name_un, cache_dir=annotator_ckpts_path)
        self.model=model
    def __call__(self, text):
            assert isinstance(text, str)
            with torch.no_grad():
                inputs = self.processor(text=text, return_tensors="pt").to('cuda')
                text_features = self.model.get_text_features(**inputs)
                text_feature = text_features[0].detach().cpu().numpy()

            return text_feature
