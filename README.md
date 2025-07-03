# Edge-semantic Synergy Fusion and Adaptive Noise-aware for Weakly Supervised Pathological Tissue Segmentation
<details>
<summary>Read full abstract.</summary>
Existing studies on weakly supervised pathological tissue segmentation predominantly rely on class activation maps (CAMs) to generate pixel-level pseudo-masks from image-level labels. However, CAMs tend to emphasize only the most discriminative regions, resulting in boundary noise that undermines the quality of pseudo-masks and degrades segmentation performance. To address these challenges, we propose a novel weakly supervised pathological tissue segmentation framework: Edge-semantic Synergy Fusion and Adaptive Noise-aware (ESFAN) mechanism. In the classification phase, the Edge-semantic Synergy Fusion (ESF) improves the quality of pseudo-masks by incorporating four synergistic components. The hybrid edge-aware transformer refines boundaries, while the pyramid context integrator captures multi-scale context. The context channel amplifier fine-tunes semantic features, and the adaptive fusion gating balances feature map contributions using learnable spatial weights. In the segmentation phase, we propose an Adaptive Noise-aware Mechanism (ANM) that incorporates adaptive weighted cross-entropy, uncertainty regularization, and spatial smoothing constraints to mitigate noise in pseudo-masks and enhance segmentation robustness. Extensive experiments on the LUAD-HistoSeg and BCSS datasets demonstrate that ESFAN significantly outperforms state-of-the-art methods.
</details>

# Framework

![framework](framework.png)

# Usage

## Dataset

<pre>
ESFAN/
├── datasets
│   ├── BCSS-WSSS/
│   │   ├── train/img/
│   │   ├── val/img/
│   │   │   └── mask/
│   │   └── test/img/
│   │       └── mask/
│   └── LUAD-HistoSeg/
│       ├── train/img/
│       ├── val/img/
│       │   └── mask/
│       └── test/img/
│           └── mask/
</pre>


## Pretrained weights , datasets and checkpoints

Download the pretained weight of classification stage via Google Cloud Drive ([Link)](https://drive.google.com/file/d/1Rka2SzqAwxUEFb28tbmiy2anhkkFOnTg/view?usp=drive_link)

Download the datasets via Google Cloud Drive ([Link)](https://drive.google.com/file/d/1lWAeCp6UN30VRVmqv97kA2sJ1Pp2frhC/view?usp=drive_link)([Link)](https://drive.google.com/file/d/178eSM9xs5jITt5P2kjaswDlJzwlU5gps/view?usp=drive_link)

Download the checkpoints of the first and second stages of LUAD-HistoSeg via Goole Cloud Drive ([Link)](https://drive.google.com/file/d/1_dSyEy1JrVEystyjqkoYf6YmWMrxmWNk/view?usp=drive_link)([Link)](https://drive.google.com/file/d/12oLS9aj8oEy1fN_xW8DQZMXkBm4qWsJy/view?usp=drive_link) 

Stage1 mIoU:76.42 Stage2 mIoU:79.55

Download the checkpoints of the first and second stages of WSSS-BCSS via Goole Cloud Drive ([Link)](https://drive.google.com/file/d/19CWs3rYqrJKMyZvxD90tejp-Ot2Nxogh/view?usp=drive_link)([Link)](https://drive.google.com/file/d/1ZDXJ9tlYKYnwlfyg88h_HKL0DWmi1sJD/view?usp=drive_link)

Stage1 mIoU:69.48 Stage2 mIoU:71.51

## Run each step:

1、Train the classification model and generate pesudo masks with the image-level label:

```python
python 1_train_stage1.py
```

2、Train the segmentation model with pesudo masks:

```python
python 2_train_stage2.py
```

3、Inference of unlabeled data using the checkpoint obtained in the segmentation stage:

```python
python inference.py
```

