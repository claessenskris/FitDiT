import os
import sys
import argparse
import logging
import json
import numpy as np
import torch
import cv2

from PIL import Image

from preprocess.humanparsing.run_parsing import Parsing
from preprocess.dwpose import DWposeDetector
from src.utils_mask import get_mask_location
from Util.Logging import set_up_logging

def parse_args():
    parser = argparse.ArgumentParser(description="GenerateMask")
    parser.add_argument("--model_path", type=str, required=True, help="The path of FitDiT model.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the paired file")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input directory")
    parser.add_argument("--input_cloth", type=str, required=True, help="Path to input directory")
    parser.add_argument("--output_mask", type=str, required=True, help="Path to output directory")
    parser.add_argument("--output_pose", type=str, required=True, help="Path to output directory")
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
            im['composite'] = np.array(masked_vton_img.convert("RGB"))
            
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

    FILE_PATH = args.file_path
    DIR_IN_IMAGE = args.input_image
    DIR_IN_CLOTH = args.input_cloth
    DIR_OUT_MASK = args.output_mask
    DIR_OUT_POSE = args.output_pose

    DEBUG = args.debug
    level = "debug" if DEBUG else "info"
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    set_up_logging(console_log_output="stdout", console_log_level=level,
                   console_log_color=True, logfile_file=script_name + ".log",
                   logfile_log_level=level, logfile_log_color=False)

    generator = MaskGenerator(args.model_path, offload=args.offload, aggressive_offload=args.aggressive_offload, device=args.device)

    logging.info("MaskGenerator initialised on %s", args.device)

    # Read paired list
    with open(FILE_PATH, 'r') as f:lines = f.readlines()

    # Main loop
    for line in lines:
        image_file, cloth_file = line.strip().split()
        full_image_file = os.path.join(DIR_IN_IMAGE, image_file)
        full_cloth_file = os.path.join(DIR_IN_CLOTH_METADATA, cloth_file)
        full_json_file = os.path.splitext(full_cloth_file)[0] + '.json'

        # Parse JSON
        with open(full_json_file, "r") as f:record = json.load(f)
        category = record.get('cloth_type')

        offset_top = 0
        offset_bottom = 0
        offset_left = 0
        offset_right = 0

        logging.info("%s | Generating mask and pose for %s", image_file, category)

        masked_vton_img, pose_image = generator.generate_mask(full_image_file, category, offset_top, offset_bottom, offset_left, offset_right)

        mask = masked_vton_img['composite']
        img = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

        lower = np.array([128] * 3, dtype=np.uint8)
        upper = np.array([128] * 3, dtype=np.uint8)

        # mask will be 255 where pixel is in [lower, upper], else 0
        final_mask = cv2.inRange(img, lower, upper)

        base_name = os.path.splitext(os.path.basename(image_file))[0]
        full_masked_file = os.path.join(DIR_OUT_MASK, f"{base_name}_mask.png")
        full_pose_file = os.path.join(DIR_OUT_POSE, f"{base_name}_pose.png")

        Image.fromarray( final_mask).save(full_masked_file)

        logging.info("%s | Saved generated MASK to %s", image_file, full_masked_file)

        pose_image.save(full_pose_file)

        logging.info("%s | Saved generated POSE to %s", image_file, full_pose_file)
