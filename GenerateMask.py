import os
import sys
import math
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.dwpose import DWposeDetector
import torch
import torch.nn as nn
from src.pose_guider import PoseGuider
from PIL import Image
from src.utils_mask import get_mask_location
import numpy as np
import cv2
import random
import argparse
import logging
from Util.Logging import set_up_logging

def parse_args():
    parser = argparse.ArgumentParser(description="GenerateMask")
    parser.add_argument("--vton_image", type=str, required=True, help="vton image")
    parser.add_argument("--category", type=str, required=True, help="category")
    parser.add_argument("--model_path", type=str, required=True, help="The path of FitDiT model.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use.")
    parser.add_argument("--aggressive_offload", action="store_true", help="Offload model more aggressively to CPU when not in use.")
    parser.add_argument("--debug", action='store_true', help="Debugging logging activated")
    return parser.parse_args()

class MaskGenerator:
    def __init__(self, model_root, offload=False, aggressive_offload=False, device="cuda:0"):
        if offload:
            self.dwprocessor = DWposeDetector(model_root=model_root, device='cpu')
            self.parsing_model = Parsing(model_root=model_root, device='cpu')
        elif aggressive_offload:
            self.dwprocessor = DWposeDetector(model_root=model_root, device='cpu')
            self.parsing_model = Parsing(model_root=model_root, device='cpu')
        else:
            self.dwprocessor = DWposeDetector(model_root=model_root, device=device)
            self.parsing_model = Parsing(model_root=model_root, device=device)
        
    def generate_mask(self, vton_img, category, offset_top, offset_bottom, offset_left, offset_right):
        with torch.inference_mode():
            vton_img = Image.open(vton_img)
            vton_img_det = resize_image(vton_img)
            pose_image, keypoints, _, candidate = self.dwprocessor(np.array(vton_img_det)[:,:,::-1])
            candidate[candidate<0]=0
            candidate = candidate[0]

            candidate[:, 0]*=vton_img_det.width
            candidate[:, 1]*=vton_img_det.height

            pose_image = pose_image[:,:,::-1]
            pose_image = Image.fromarray(pose_image)
            model_parse, _ = self.parsing_model(vton_img_det)

            mask, mask_gray = get_mask_location(category, model_parse, \
                                        candidate, model_parse.width, model_parse.height, \
                                        offset_top, offset_bottom, offset_left, offset_right)
            mask = mask.resize(vton_img.size)
            mask_gray = mask_gray.resize(vton_img.size)
            mask = mask.convert("L")
            mask_gray = mask_gray.convert("L")
            masked_vton_img = Image.composite(mask_gray, vton_img, mask)

            im = {}
            im['background'] = np.array(vton_img.convert("RGBA"))
            im['layers'] = [np.concatenate((np.array(mask_gray.convert("RGB")), np.array(mask)[:,:,np.newaxis]),axis=2)]
            im['composite'] = np.array(masked_vton_img.convert("RGBA"))
            
            return im, pose_image

def resize_image(img, target_size=768):
    width, height = img.size
    
    if width < height:
        scale = target_size / width
    else:
        scale = target_size / height
    
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_img

if __name__ == "__main__":
    args = parse_args()

    DEBUG = args.debug
    level = "debug" if DEBUG else "info"
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    set_up_logging(console_log_output="stdout", console_log_level=level,
                   console_log_color=True, logfile_file=script_name + ".log",
                   logfile_log_level=level, logfile_log_color=False)

    generator = MaskGenerator(args.model_path, offload=args.offload, aggressive_offload=args.aggressive_offload, device=args.device)

    vton_img = args.vton_image
    category = args.category

    offset_top = 0
    offset_bottom = 0
    offset_left = 0
    offset_right = 0

    masked_vton_img, pose_image = generator.generate_mask(vton_img, category, offset_top, offset_bottom, offset_left, offset_right)

    Image.fromarray(masked_vton_img['composite']).save("output_masked.png")
    pose_image.save("output_pose.png")
