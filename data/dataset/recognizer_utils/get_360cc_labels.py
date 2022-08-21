

def get_360cc_labels(cfg, is_train=True):
        char_file = cfg.DATASETS.CHAR_FILE  # 'lib/dataset/txt/char_std_5990.txt'
        with open(char_file, 'rb') as file:
            char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}

        txt_file = cfg.DATASETS.JSON_FILE_TRAIN if is_train else cfg.DATASETS.JSON_FILE_VAL

        # convert name:indices to name:string
        labels = []
        with open(txt_file, 'r', encoding='utf-8') as file:
            contents = file.readlines()
            for c in contents:
                imgname = c.split(' ')[0]
                indices = c.split(' ')[1:]
                string = ''.join([char_dict[int(idx)] for idx in indices])
                labels.append({imgname: string})

        print("load {} images!".format(len(labels)))
        return labels