# import os
# import argparse
# import json
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", type=str, help="The path to original data file")
# parser.add_argument("--file_name", type=str, help="File name")
#
# args = parser.parse_args()
#
# json_filename = args.file_name + ".json"
# json_file = os.path.join(args.data_dir, json_filename)
# with open(json_file, 'r', encoding='utf-8') as f:
#     content = json.load(f)
#
# print(content)

#add_axis = {"caption":[]}
# content.update(add_axis)
#
# with open(json_file, 'w') as f_new:
#     json.dump(content, f_new)


import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import skimage.io as io
from PIL import Image
from predict import *
from train import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=str, help="seed name")

args = parser.parse_args()

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

#Load model weights

prefix_length = 10

model = ClipCaptionModel(prefix_length)

model_path = "./coco_train/coco_prefix_latest.pt"

model.load_state_dict(torch.load(model_path, map_location='cpu'))

model = model.eval()
model = model.to(device)

clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#images
use_beam_search = False
data_dir = os.path.join("./data/k_shot/MELD", args.seed)
splits = {"dev","train","test"}
os.makedirs(data_dir, exist_ok=True)


for split in splits:
    split_v = split + "_video"
    split_f = split + "_caption.json"
    img_path = os.path.join(data_dir, split_v)
    img_names = os.listdir(img_path)
    output_f = open(os.path.join(data_dir, split_f), "w")

    #print(img_names)
    for i in range(0, len(img_names)):
        img = os.path.join(img_path, img_names[i])
        image = io.imread(img)
        pil_image = Image.fromarray(image)

        image = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        if use_beam_search:
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)


        print("The caption sentences are storaged in the directory: " + os.path.join(data_dir, "all_caption.txt"))

        row_content = img_names[i] + "," + generated_text_prefix + "\n"

        output_f.write(row_content)

output_f.close()
