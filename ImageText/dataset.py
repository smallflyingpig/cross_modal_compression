# load CUB-200-2011, Oxford-102, COCO(TODO) dataset for Image2Text
import torch, torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import json

class Birds(Dataset):
    def __init__(self, json_path, train=True, img_size=256):
        super(Birds, self).__init__()
        with open(json_path, "r") as fp:
            self.json_data = json.load(fp)
        self.train = train
        self.img_size = 256

    def __len__(self):
        return len(self.json_data['data'])

    def __getitem__(self, index):
        _temp = self.json_data['data']
        img_path = _temp['image']
        text = 
        