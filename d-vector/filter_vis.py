# Filtering for visualization

import os
import json
import shutil
import pandas as pd
from tqdm import tqdm

input_file_path = '/result/meta/filtered_metadata.json'
input_data_dir = '/scratch/d-vector/filtered/'
output_dir = '/scratch/d-vector/filtered3'
os.makedirs(output_dir, exist_ok=True)
metadata = []
cnt = 0

with open(input_file_path, 'r') as file:
    data = json.load(file)

for sentence in data:
    if sentence["emotion"] in ["무감정", "슬픔"]:
        speaker_folder = os.path.join(output_dir, str(sentence["speaker_id"]))
        os.makedirs(speaker_folder, exist_ok=True)
        
        src = os.path.join(input_data_dir+f"/{str(sentence["speaker_id"])}", sentence['filename'])
        dst = os.path.join(speaker_folder, sentence['filename'])

        if os.path.exists(src):
            shutil.copyfile(src, dst)
            metadata.append(sentence)
            cnt+=1                   
print(f"All json files filtered. {cnt} files moved.")

# Save
output_file_path = os.path.join("/result/meta", "filtered_metadata3.json")
with open(output_file_path, 'w') as file:
    json.dump(metadata, file, indent=4, ensure_ascii=False) # ensure_ascii: for Korean

