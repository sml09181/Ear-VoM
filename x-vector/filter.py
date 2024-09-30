# Create metadata
import os
import json

result = []
src = [
    os.path.join("/scratch/ASVspoof2019/ASVspoof2019_LA_cm_protocols", x) 
    in ["ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019.LA.cm.dev.trl.txt"]
]
paths = [["training.txt", "validation.txt", "testing.txt"], ['train', 'dev', 'eval']]
root = "/x-vector/result/meta"

for i, s in enumerate(src):
    temp = {
    x: [] for x in range(1, 21)
    }
    f = open(s, 'r')
    lines = f.readlines()
    for line in lines:
        if 'bonafide' in line:
            temp[20].append(line.split(' ')[1])
            continue
        splited = line.split(' ')
        filename = splited[1]
        model = int(splited[-2][1:])
        temp.get(model).append(filename)
    result.append(temp)
    
    save_filename = os.path.join(root, paths[0][i])
    w = open(save_filename, 'w')
    for k, v in temp.items():
        for f in v:
            audio_filename =f+ ".flac"
            audio_path = os.path.join(f"/scratch/ASVspoof2019/ASVspoof2019_LA_{paths[1][i]}/flac", audio_filename)
            w.write(audio_path+" "+str(k) + '\n')
    w.close()

with open("/result/meta/metadata.json", 'w') as f:
    json.dump([{'train': result[0], 'valid': result[1], 'eval': result[2]}], f)