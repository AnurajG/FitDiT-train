import sys
sys.path.append('/home/user/FitDiT-train')

import os
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.dwpose import DWposeDetector
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
import torch.nn as nn
from PIL import Image
from src.utils_mask import get_mask_location
import numpy as np
import cv2
import pickle 
import random
import json
import time
import math

# Import the updated garment classifier function
from garment_classifier import get_garment_category

# Create required directories
os.makedirs("/home/user/FitDiT-train/train/pose", exist_ok=True)
os.makedirs("/home/user/FitDiT-train/train/mask", exist_ok=True)
os.makedirs("/home/user/FitDiT-train/train/cloth_embeds", exist_ok=True)
os.makedirs("/home/user/FitDiT-train/train/category", exist_ok=True)

# Directory for storing category classifications
category_dir = "/home/user/FitDiT-train/train/category"

vton_img_folder = "/home/user/FitDiT-train/train/model"
garm_img_folder = "/home/user/FitDiT-train/train/garment"
model_root = "/home/user/weights/FitDiT"

print("Loading pre-trained models...")
image_encoder_large = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
vit_processing = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

dwprocessor = DWposeDetector(model_root=model_root, device="cuda:0")
parsing_model = Parsing(model_root=model_root, device='cuda:0')

def resize_image(im, new_width=768, new_height=1024, pad_color=(255, 255, 255), mode=Image.LANCZOS):
    old_width, old_height = im.size
    ratio_w = new_width / old_width
    ratio_h = new_height / old_height

    if ratio_w < ratio_h:
        new_size = (new_width, round(old_height * ratio_w))
    else:
        new_size = (round(old_width * ratio_h), new_height)
    im_resized = im.resize(new_size, mode)
    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)

    new_im = Image.new('RGB', (new_width, new_height), pad_color)
    new_im.paste(im_resized, (pad_w, pad_h))

    return new_im


data_list = []
image_files = os.listdir(vton_img_folder)
total_images = len(image_files)

print(f"Processing {total_images} images...")

for i, item in enumerate(image_files):
    try:
        print(f"Processing image {i+1}/{total_images}: {item}")
        
        vton_img_path = os.path.join(vton_img_folder, item)  
        garm_img_path = vton_img_path.replace("/model", "/garment")
        
        # Skip if garment image doesn't exist
        if not os.path.exists(garm_img_path):
            print(f"Skipping {item}: No matching garment image found")
            continue
        
        vton_img = Image.open(vton_img_path)
        garm_img = Image.open(garm_img_path)
        vton_img_det = resize_image(vton_img)
        
        # Process for pose detection
        pose_image, keypoints, _, candidate = dwprocessor(np.array(vton_img_det)[:,:,::-1])
        candidate[candidate<0] = 0
        candidate = candidate[0]

        candidate[:, 0] *= vton_img_det.width
        candidate[:, 1] *= vton_img_det.height

        pose_image = pose_image[:,:,::-1]  # rgb
        pose_image = Image.fromarray(pose_image)
        pose_path = vton_img_path.replace("/model", "/pose")
        pose_image.save(pose_path)
        
        # Determine garment category using GPT-4o with caching
        # We pass the garment path instead of model_parse
        current_category = get_garment_category(garm_img, garm_img_path, category_dir)
        
        # Run human parsing (needed for mask generation, not for classification)
        model_parse, _ = parsing_model(vton_img_det)
        
        # Generate mask using determined category
        mask, mask_gray = get_mask_location(current_category, model_parse, 
                                        candidate, model_parse.width, model_parse.height)
        
        mask = mask.resize(vton_img.size).convert("L")
        mask_gray = mask_gray.resize(vton_img.size).convert("L")
        mask = np.concatenate((np.array(mask_gray.convert("RGB")), np.array(mask)[:,:,np.newaxis]), axis=2)[:,:,3]
        mask = Image.fromarray(mask).convert("L")
        mask_path = vton_img_path.replace("/model", "/mask")
        mask.save(mask_path)
        
        # Process garment embeddings
        cloth_image_vit = vit_processing(images=garm_img, return_tensors="pt").pixel_values
        
        image_encoder_large = image_encoder_large.to("cuda:0")
        image_encoder_bigG = image_encoder_bigG.to("cuda:0")
        cloth_image_vit = cloth_image_vit.to("cuda:0")
        
        with torch.no_grad():
            image_embeds_large = image_encoder_large(cloth_image_vit).image_embeds
            image_embeds_bigG = image_encoder_bigG(cloth_image_vit).image_embeds
            cloth_image_embeds = torch.cat([image_embeds_large, image_embeds_bigG], dim=1)
        
        cloth_embeds_path = vton_img_path.replace("/model", "/cloth_embeds")[:-4] + ".pkl"
        os.makedirs(os.path.dirname(cloth_embeds_path), exist_ok=True)
        
        with open(cloth_embeds_path, 'wb') as f:  
            pickle.dump(cloth_image_embeds.cpu(), f)  
        
        dict_ = {
            "vton_img_path": vton_img_path, 
            "garm_img_path": garm_img_path, 
            "mask_path": mask_path, 
            "pose_path": pose_path, 
            "cloth_embeds_path": cloth_embeds_path,
            "category": current_category  # Store the determined category
        }
        data_list.append(dict_)
        
    except Exception as e:
        print(f"Error processing {item}: {str(e)}")
        continue

print(f"Successfully processed {len(data_list)} images")
print("Saving data.json...")

with open('./data.json', 'w') as json_file:  
    json.dump(data_list, json_file, indent=4)  

print("Data generation complete!")
