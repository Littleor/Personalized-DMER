# Personalized Dynamic Music Emotion Recognition with Dual-Scale Attention-Based Meta-Learning (DSAML)
<p align="center">
    <a href="https://github.com/Littleor/Personalized-DMER/blob/master/LICENSE" target="blank">
    <img src="https://img.shields.io/github/license/Littleor/Personalized-DMER?style=flat-square" alt="github-profile-readme-generator license" />
    </a>
    <a href="https://github.com/Littleor/Personalized-DMER/fork" target="blank">
    <img src="https://img.shields.io/github/forks/Littleor/Personalized-DMER?style=flat-square" alt="github-profile-readme-generator forks"/>
    </a>
    <a href="https://github.com/Littleor/Personalized-DMER/stargazers" target="blank">
    <img src="https://img.shields.io/github/stars/Littleor/Personalized-DMER?style=flat-square" alt="github-profile-readme-generator stars"/>
    </a>
    <a href="https://github.com/Littleor/Personalized-DMER/issues" target="blank">
    <img src="https://img.shields.io/github/issues/Littleor/Personalized-DMER?style=flat-square" alt="github-profile-readme-generator issues"/>
    </a>
    <a href="https://github.com/Littleor/Personalized-DMER/pulls" target="blank">
    <img src="https://img.shields.io/github/issues-pr/Littleor/Personalized-DMER?style=flat-square" alt="github-profile-readme-generator pull-requests"/>
    </a>
</p>

[[Project Website](https://littleor.github.io/PDMER/)] 

![Model Architecture](./static/images/Model-Architecture.png)

Here is the core implementation of the DSAML model in the paper "Personalized Dynamic Music Emotion Recognition with Dual-Scale Attention-Based Meta-Learning", which is accepted by the AAAI 25.

## Get Start

### Prerequisites

* Python >= 3.8.5, < 3.9
* PyTorch >= 2.2.1

### Installation

```bash
conda env create -f environment.yml
```

### Dataset
We use the [DEAM](https://cvml.unige.ch/databases/DEAM/) and [PMEmo](https://github.com/HuiZhangDB/PMEmo) dataset, so you need to download the dataset and unzip both the audio and annotation files. The final file structure should be like this:
```txt
DEAM
├── DEAM_Annotations
│   ├── annotations
├── DEAM_audio
└── features
    └── features
```
```txt
PMEmo
├── annotations
├── chorus
├── comments
├── EDA
├── features
├── lyrics
├── metadata.csv
├── netease_soundcloud.csv
```

Then we need to preprocess the dataset, code will be released as soon as possible.

## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@inproceedings{zhang2025personalized,
    title={Personalized Dynamic Music Emotion Recognition with Dual-Scale Attention-Based Meta-Learning},
    author={Zhang, Dengming and You, Weitao and Liu, Ziheng and Sun, Lingyun and Chen, Pei},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2025}
}
```