<div align="center">
  <img src="resources/logo.png" width="400px"/>
</div>

## 简介

[English](/README.md) | 简体中文

该repo继承自detectron2

ToyOCR 是基于 PyTorch 和 detectron2 的开源工具箱，专注于文本检测，文本识别以及相应的下游任务

主分支目前支持 **PyTorch 1.4 以上**的版本。



### 主要特性

-**简单而不简约**
    简单到令人法指，效果还很好.

-**全面**

   该工具箱不仅支持文本检测和文本识别，还支持上游任务，例如版面分析等。

-**多种模型**

  该工具箱支持用于文本检测，文本识别和关键信息提取的各种最新模型。

支持的算法：

#### 文字检测

- [x] [DBNet](yamls/text_detection/dbnet/README.md) (AAAI'2020)
- [x] [Mask R-CNN](yamls/text_detection/maskrcnn/README.md) (ICCV'2017)




#### 文字识别 

- [x] [CRNN](yamls/text_recognition/crnn/README.md) (TPAMI'2016)


## 安装

```shell
pip install requirements.txt
```

## 训练

```
sh shme.sh
```


## 版权
[copyright](/LICENSE)

