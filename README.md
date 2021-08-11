<div align="center">
  <img src="resources/logo.png" width="400px"/>
</div>

## Introducing

[English](/README-zh.md) | chinese version


ToyOCR is an open-source toolbox based on PyTorch and detectron2 for text detection, text recognition and layout_analysis.


## Requirements

- Python >= 3.7
- PyTorch >= 1.4
- torchvision
- OpenCV
- pycocotools

### Main features

-**Simple but not easy**
    the structure is simple,the training speed is fast and the deploying is convenient.

-**Numerous Utilities**

   The toolbox provides a series set of utilities which can help users to train the detection and recognition model.



Supported Algorithms:

#### Text Detection

- [x] [ToyDet](yamls/text_detection/toydet/README.md) 
- [x] [Mask R-CNN](yamls/text_detection/maskrcnn/README.md) (ICCV'2017)

#### Text Recognition 

- [ ] [CRNN](yamls/text_recognition/crnn/README.md) (TPAMI'2016)
- [ ] [ATTENTION](yamls/text_recognition/attention/README.md) ()

#### Layout Analysis

- [x] [CenterNet](yamls/layout_analysis/centernet_res50_layout_analysis.yaml) ()


## Install

```shell
pip install requirements.txt
```

## Training

```
sh shme.sh
```


## Copyright
[copyright](/LICENSE)

