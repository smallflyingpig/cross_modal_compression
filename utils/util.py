import torch
import numpy as np
import pickle

class RunningAverage(object):
    def __init__(self):
        self.val_hist = {}
        self.n_hist = {}

    def update(self, data:dict, count:dict):
        assert(isinstance(data, dict))
        assert(isinstance(count, dict))
        for key, value in data.items():
            self.val_hist[key] = self.val_hist.get(key, [])+[value]
            self.n_hist[key] = self.n_hist.get(key, [])+[count[key]]

    def clear(self):
        self.val_hist = {}
        self.n_hist = {}

    def average(self):
        avg = {}
        for key, value in self.val_hist.items():
            n = np.array(self.n_hist[key])
            v = np.array(value)
            avg[key] = (n*v).sum()/float(n.sum())
        return avg

def dict2str(data:dict):
    return ','.join(["{}:{:5.2f}".format(k, v) for k,v in data.items()])


def save_dict_models(models, path):
    with open(path, "wb") as fp:
        torch.save({k:v.state_dict() for k,v in models.items()}, fp)

def load_dict_models(models, path):
    data = torch.load(path)
    assert data.keys()==models.keys()
    for k,v in models.items():
        v.load_state_dict(data.get(k, {}))
    return models

import logging
import os

class Outputer(object):
    def __init__(self, log_folder, print_every, save_every):
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        self.model_path = os.path.join(log_folder, "models")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        logging.basicConfig(level=logging.INFO,format="%(message)s")
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        rootLogger = logging.getLogger()
        
        fileHandler = logging.FileHandler(os.path.join(log_folder, "output.log"))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
        
        self.logger = rootLogger
        self.log_folder = log_folder
        self.print_every = print_every
        self.save_every = save_every
        self.log_counter = 0
        self.save_counter = 0
        self.logger.info("init the outputer!")

    def log_step(self, log_str):
        if self.log_counter % self.print_every == 0:
            self.logger.info(log_str)
        self.log_counter += 1
    def log(self, log_str):
        self.logger.info(log_str)

    def save_step(self, save_file):
        if self.save_counter % self.save_every == 0:
            with open(os.path.join(self.model_path,
                "model_{}.pickle".format(self.save_counter)), 'wb') as fp:
                torch.save(save_file, fp)
        self.save_counter += 1

    def save(self, save_file):
        with open(os.path.join(self.model_path,
            "model_{}.pickle".format(self.save_counter)), 'wb') as fp:
            torch.save(save_file, fp)
    

def calculate_rate_folder(data_folder):
    file_list = []
    for path, folder, files in os.walk(data_folder):
        file_list += [os.path.join(path, _f) for _f in files]
    filesize_list = [os.path.getsize(_f) for _f in file_list]
    filesize_mean = sum(filesize_list)/float(len(filesize_list))
    return filesize_mean

import cv2
import math
def PSNR(data_folder1, data_folder2, ext_set=['.jpg', '.jpeg', '.png']):
    file_list = []
    path1_list = []
    ext1_list = []
    for path, folder, files in os.walk(data_folder1):
        if len(files)<1:
            continue
        file_list.append([_f for _f in files if os.path.splitext(_f)[-1] in ext_set])
        ext1_list.append(os.path.splitext(files[0])[-1])
        path1_list.append(path)
    path2_list = []
    ext2_list = []
    for path, folder, files in os.walk(data_folder2):
        if len(files)<1:
            continue
        ext2_list.append(os.path.splitext(files[0])[-1])
        path2_list.append(path)
    assert len(path1_list) == len(path2_list)
    def _psnr(img1, img2):
        mse = np.mean( (img1 - img2) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    psnr_list = []
    for path1, path2, files, ext1, ext2 in \
        zip(path1_list, path2_list, file_list, ext1_list, ext2_list):
        for _f in files:
            file_path1 = os.path.join(path1, os.path.splitext(_f)[0]+ext1)
            file_path2 = os.path.join(path2, os.path.splitext(_f)[0]+ext2)
            if os.path.exists(file_path1) and os.path.exists(file_path2):
                image1 = cv2.imread(file_path1)
                image2 = cv2.imread(file_path2)
                if image1.shape != image2.shape:
                    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
                _d = _psnr(image1, image2)
                psnr_list.append(_d)
    return sum(psnr_list)/len(psnr_list)


    
