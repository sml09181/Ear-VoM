#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize speaker embeddings."""

from argparse import ArgumentParser
from pathlib import Path
from warnings import filterwarnings

import os
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchaudio
from librosa.util import find_files
from sklearn.manifold import TSNE
import umap.umap_ as umap
from tqdm import tqdm
from data.wav2mel import Wav2Mel

selected_spkrs = random.sample(range(50), num_speakers)
spkrs_f = ''
for s in sorted(selected_spkrs[:-1]):
    spkrs_f += str(s)
    spkrs_f += '-'
spkrs_f += str(selected_spkrs[-1])
print(spkrs_f)

def sample_by_emotion(data_dirs, num_speakers, wav2mel, model: int):
    n_spkrs = 0
    paths, spkr_names, mels = [], [], []
    input_file_path = f"/result/meta/filtered_metadata1.json"
    with open(input_file_path, 'r') as file:
        data = json.load(file)
    
    for data_dir in data_dirs:
        data_dir_path = Path(data_dir)
        for i, spkr_dir in enumerate([x for x in data_dir_path.iterdir() if x.is_dir()]):
            if int(str(spkr_dir).split('/')[-1]) not in selected_spkrs: continue
            audio_paths = find_files(spkr_dir) 
            n_spkrs += 1
            
            for audio_path in audio_paths:
                spkr_name = spkr_dir.name
                flag = False
                filename = str(audio_path).split('/')[-1]
                for instance in data:
                    if filename == instance["filename"]:
                        if instance["emotion"] == "슬픔":
                            spkr_name = spkr_name + "-Sad"
                        elif instance["emotion"] == "무감정":
                            spkr_name = spkr_name + "-Non"
                        flag=True
                        break
                if flag:
                    paths.append(audio_path)
                    spkr_names.append(spkr_name)
    n_spkrs = n_spkrs *2
                    
    for audio_path in tqdm(paths, ncols=0, desc="Preprocess"):
        wav_tensor, sample_rate = torchaudio.load(audio_path)
        with torch.no_grad():
            mel_tensor = wav2mel(wav_tensor, sample_rate)
        mels.append(mel_tensor)
    return paths, spkr_names, mels, n_spkrs

def sample_by_speaker(data_dirs, num_speakers, wav2mel, model=None): # original setting
    n_spkrs = 0
    paths, spkr_names, mels = [], [], []
    
    for data_dir in data_dirs:
        data_dir_path = Path(data_dir)
        for i, spkr_dir in enumerate([x for x in data_dir_path.iterdir() if x.is_dir()]):
            if int(str(spkr_dir).split('/')[-1]) not in selected_spkrs: continue
            n_spkrs += 1
            audio_paths = find_files(spkr_dir)
            spkr_name = spkr_dir.name
            for audio_path in audio_paths:
                paths.append(audio_path)
                spkr_names.append(spkr_name)
    for audio_path in tqdm(paths, ncols=0, desc="Preprocess"):
        wav_tensor, sample_rate = torchaudio.load(audio_path)
        with torch.no_grad():
            mel_tensor = wav2mel(wav_tensor, sample_rate)
        mels.append(mel_tensor)
    return paths, spkr_names, mels, n_spkrs

def draw_graph(transformed, output_dir, spkr_names, n_spkrs, method: str, model: str, rule: str, c_step:str):
    title = f"[{method}] step{c_step}_{model}_{rule}"
    output_path = f"/images/{method}/step{c_step}_{model}_{rule}"
    print(output_path)
    data = {
        "dim-1": transformed[:, 0],
        "dim-2": transformed[:, 1],
        "label": spkr_names,
    }

    plt.figure()
    sns.scatterplot(
        x="dim-1",
        y="dim-2",
        hue="label",
        palette=sns.color_palette(n_colors=n_spkrs),
        data=data,
        legend="full",
    )
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, format='png')
    print(f"[INFO] {method} image saved.")

def draw_graph_emotion(transformed, output_dir, spkr_names, n_spkrs, method: str, model: str, rule: str, c_step:str):
    title = f"[{method}] step{c_step}_{model}_{rule}"
    output_path = f"/images/{method}/{spkrs_f}/step{c_step}_{model}_{rule}"
    os.makedirs('/'.join(output_path.split('/')[:-1]), exist_ok = True)
    print(output_path)
    data = {
        "dim-1": transformed[:, 0],
        "dim-2": transformed[:, 1],
        "label": spkr_names,
    }
    print("**", len(spkr_names), n_spkrs)
    plt.figure()

    colors = ["#d93327", "#ffb399", "#fce026", "#fff199", "#b3ff66" , "#ccff99" , "#3b90ff", "#99c5ff", "#443bff", "#a39eff" ]
    customPalette = sns.set_palette(sns.color_palette(colors))
    sns.scatterplot(
        x="dim-1",
        y="dim-2",
        hue="label",
        palette=customPalette,
        data=data,
        legend="full",
        markers=['o', 'x']*int(n_spkrs/2),
    )
    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, format='png')
    print(f"[INFO] {method} image saved.")
    

def visualize(data_dirs, wav2mel_path, checkpoint_path, output_dir, model_type, rule_type, num_speakers, gpu_id):
    """Visualize high-dimensional embeddings using t-SNE and umap."""

    c_step = checkpoint_path.split("step")[-1].split(".")[0]
    type_dict = {
        0: ["All", "emotion(non-sad)", sample_by_emotion, draw_graph_emotion],
        1: ["Non-Sad", "age", None, draw_graph],
        2: ["Non", "gender", None, draw_graph],
        3: ["Sad", "speaker", sample_by_speaker, draw_graph]
    }
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id;
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu", 0)
    wav2mel = Wav2Mel()
    dvector = torch.jit.load(checkpoint_path).eval().to(device)
    print("[INFO] model loaded.")


    paths, spkr_names, mels, n_spkrs = type_dict.get(rule_type)[2](data_dirs, num_speakers, wav2mel, model_type)
    print(len(paths), len(spkr_names))
    embs = []
    for mel in tqdm(mels, ncols=0, desc="Embed"):
        with torch.no_grad():
            emb = dvector.embed_utterance(mel.to(device))
            emb = emb.detach().cpu().numpy()
        embs.append(emb)
    embs = np.array(embs)
    
    # TSNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    transformed = tsne.fit_transform(embs)
    print("[INFO] tsne embeddings transformed.")
    if rule_type==0: draw_graph_emotion(transformed, output_dir, spkr_names, n_spkrs, "tsne", type_dict.get(model_type)[0], type_dict.get(rule_type)[1], c_step)
    else: draw_graph(transformed, output_dir, spkr_names, n_spkrs, "tsne", type_dict.get(model_type)[0], type_dict.get(rule_type)[1], c_step)

    # UMAP
    umap_model = umap.UMAP(n_neighbors=n_spkrs, min_dist=0.1, n_components=2, random_state=42)
    transformed = umap_model.fit_transform(embs)
    print("[INFO] umap embeddings transformed.")
    if rule_type==0: draw_graph_emotion(transformed, output_dir, spkr_names, n_spkrs, "umap", type_dict.get(model_type)[0], type_dict.get(rule_type)[1], c_step)
    else: draw_graph(transformed, output_dir, spkr_names, n_spkrs, "umap", type_dict.get(model_type)[0], type_dict.get(rule_type)[1], c_step)


if __name__ == "__main__":
    filterwarnings("ignore")
    PARSER = ArgumentParser()
    PARSER.add_argument("data_dirs", type=str, nargs="+")
    PARSER.add_argument("-w", "--wav2mel_path", required=True)
    PARSER.add_argument("-c", "--checkpoint_path", required=True)
    PARSER.add_argument("-o", "--output_dir", default="/images")
    PARSER.add_argument("-m", "--model_type", type=int, default=0) # 0: all, 1: non/sad, 2: non, 3: sad
    PARSER.add_argument("-r", "--rule_type", type=int, default=0) # 0: emotion(non/sad), 1: age, 2: gender, 3: simple
    PARSER.add_argument("-n", "--num_speakers", type=int, default=5) # # of speakers to draw
    PARSER.add_argument("-g", '--gpu_id', type=str, default = "6")
    visualize(**vars(PARSER.parse_args()))
