# Ear-VoM


## d-vector

### Overview
The d-vector model is designed to learn embeddings that effectively differentiate speakers based on various emotional combinations. The training process aims that even if different emotions are present, the embeddings are capable of distinguishing speakers regardless of emotional variation while maintaining proximity among embeddings of the same speaker regardless of emotional variation.

### Training Data
- **Dataset**: Emotion-tagged free conversations (adults)
- **Source**: [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71631)

### Goals
- Verify that d-vector embeddings can effectively compare different speakers based on their emotional combinations.
- Ensure that embeddings of the same speaker, even with varying emotions, are closely aligned.

### Visualize by t-SNE
<img src="https://github.com/user-attachments/assets/9c98426a-4941-4457-b50c-663b2a1a9837" width=1000>

## x-vector

### Overview
The x-vector model aims to distinguish between different deep voice generation models by analyzing their generated outputs.

### Training Data
- **Dataset**: ASVspoof 2019
- **Reference**: Wang, X., Yamagishi, J., Todisco, M., Delgado, H., Nautsch, A., Evans, N., ... & Ling, Z. H. (2020). ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech. *Computer Speech & Language*, 64, 101114.

### Goals
- Effectively classify and differentiate between the outputs of various deep voice generation models.

---

## Reference
+ dvector: [yistLin](https://github.com/yistLin/dvector)
+ xvector: [KrishnaDN](https://github.com/KrishnaDN/x-vector-pytorch)
