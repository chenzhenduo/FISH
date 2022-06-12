import os
import torch.utils.data as data
import PIL.Image as Image
import pandas as pd

class dataset(data.Dataset):
    def __init__(self, set_name, root_dir=None, transforms=None, train=False, test=False):
        self.root_path = root_dir
        self.train = train
        self.test = test
        self.transforms=transforms
        if self.train:
            self.train_anno = pd.read_csv(os.path.join(self.root_path, set_name+'_train.txt'), \
                                      sep=" ", \
                                      header=None, \
                                     names=['ImageName', 'label'])
            self.paths= self.train_anno['ImageName'].tolist()
            #print(self.paths)
            self.labels = self.train_anno['label'].tolist()

        if self.test:
            self.test_anno = pd.read_csv(os.path.join(self.root_path, set_name+'_test.txt'), \
                                      sep=" ", \
                                      header=None, \
                                     names=['ImageName', 'label'])
            self.paths= self.test_anno['ImageName'].tolist()

            self.labels = self.test_anno['label'].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(img_path)
        if self.test:
            img = self.transforms(img)
            label = self.labels[item]
            return img, label, item
        if self.train:
            img = self.transforms(img)
            label = self.labels[item]
            return img, label, item
    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

