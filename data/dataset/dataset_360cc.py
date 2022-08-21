from __future__ import print_function, absolute_import
import torch.utils.data as data
import os
import numpy as np
import cv2

class Dataset_360CC(data.Dataset):
    def __init__(self, config, is_train=True):

        self.root = config.DATASETS.ROOT
        self.is_train = is_train
        self.inp_h = config.MODEL.CRNN.IMAGE_SIZE_H
        self.inp_w = config.MODEL.CRNN.IMAGE_SIZE_W

        self.dataset_name = config.DATASETS.TYPE

        self.mean = np.array(config.DATASETS.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASETS.STD, dtype=np.float32)

        char_file = config.DATASETS.CHAR_FILE
        with open(char_file, 'rb') as file:
            char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}

        txt_file = config.DATASETS.JSON_FILE_TRAIN if is_train else config.DATASETS.JSON_FILE_VAL

        # convert name:indices to name:string
        self.labels = []
        with open(txt_file, 'r', encoding='utf-8') as file:
            contents = file.readlines()
            for c in contents:
                imgname = c.split(' ')[0]
                indices = c.split(' ')[1:]
                string = ''.join([char_dict[int(idx)] for idx in indices])
                self.labels.append({imgname: string})

        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_name = list(self.labels[idx].keys())[0]
        img = cv2.imread(os.path.join(self.root, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_h, img_w = img.shape

        img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))

        img = img.astype(np.float32)
        img = (img/255. - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        return img, idx




