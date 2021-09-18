from __future__ import print_function

from utils.config import cfg, cfg_from_file
from utils.dataset import TextDataset, CaptionDatasetMultisize
from TextImage.trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument("--dataset", choices=['bird', 'coco'], type=str, default='bird', help="")
    parser.add_argument("--output_dir", type=str, default="./output/TextImage", help="")
    parser.add_argument("--train_net_e", type=str, default="", help="path for pretrained text encoder")
    parser.add_argument("--local_rank", type=int, default=0, help="")
    parser.add_argument("--batch_size", type=int, default=0, help="")
    args = parser.parse_args()
    return args


def gen_example(wordtoix, algo, filepath=""):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    if filepath == '':
        filepath = '%s/example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}
    with open(filepath, "rb") as f:
        filenames = f.read()
        if isinstance(filenames, bytes):
            filenames = filenames.decode('utf8')
        filenames = filenames.split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
            with open(filepath, "rb") as f:
                print('Load from:', name)
                sentences = f.read()
                if isinstance(sentences, bytes):
                    sentences = sentences.decode('utf8')
                sentences = sentences.split('\n')
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    
    if args.train_net_e != '':
        cfg.TRAIN.NET_E = args.train_net_e

    if args.batch_size > 0:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
        
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    #output_dir = '../output/%s_%s_%s' % \
    #    (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    output_dir = os.path.join(args.output_dir, "{}_{}_{}".format(
        cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp
    ))

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'val'
    

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if args.dataset == 'bird':
        dataset = TextDataset(cfg.DATA_DIR, split_dir,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform, sample_type=split_dir)
    elif args.dataset == 'coco':
        dataset = CaptionDatasetMultisize(cfg.DATA_DIR, split_dir,
        base_size=cfg.TREE.BASE_SIZE,
        transform=image_transform, sample_type=split_dir)
    else:
        raise ValueError
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        '''generate images from pre-extracted embeddings'''
        if cfg.B_VALIDATION:
            algo.sampling(split_dir)  # generate images for the whole valid dataset
        else:
            gen_example(dataset.wordtoix, algo)  # generate images for customized captions
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
