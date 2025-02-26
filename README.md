![framework](framework.jpg)

# Usage

## Dataset

```python
ESFAN/

|_ datasets
|     |_ BCSS-WSSS/
|         |_ train/img/
|         |_ val/img/

|          |     |/mask/

|         |_ test/img/

|          |     |/mask/

|     |_ LUAD-HistoSeg/
|         |_ train/img/

|         |_ val/img/

|          |     |/mask/

|         |_ test/img/

|          |     |/mask/
```

## Pretrained weights

Download the pretained weight of classification stage from

## Run each step:

1、Train the classification model and generate pesudo masks with the image-level label:

```python
python 1_train_stage1.py
```

2、Train the segmentation model with pesudo masks:

```python
python 2_train_stage2.py
```

3、Inference with the weights obtained from the segmentation stage：

```python
python inference.py
```

