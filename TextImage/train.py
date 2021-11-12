from __future__ import print_function
from utils.config import cfg, cfg_from_file
from utils.dataset import TextDataset, CaptionDatasetMultisize
from TextImage.trainer import condGANTrainer as trainer
from TextImage.text_image_utils import weights_init
from TextImage.model import G_DCGAN, G_NET
from TextImage.model import RNN_ENCODER, CNN_ENCODER, D_NET64, D_NET128, D_NET256

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


class Evaluator(object):
    def __init__(self, cfg, dataset, gpu):
        self.cfg = cfg
        self.cuda = gpu >=0 and torch.cuda.is_available() 
        cfg.CUDA = self.cuda 
        # Get data loader
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
        image_transform = transforms.Compose([
            transforms.Scale(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
        if dataset == 'bird':
            dataset = TextDataset(cfg.DATA_DIR, 'val',
                                  base_size=cfg.TREE.BASE_SIZE,
                                  transform=image_transform, sample_type='val')
        elif dataset == 'coco':
            dataset = CaptionDatasetMultisize(cfg.DATA_DIR, split_dir,
            base_size=cfg.TREE.BASE_SIZE,
            transform=image_transform, sample_type='val')
        else:
            raise ValueError
        assert dataset
        self.dataset = dataset
        self.n_words = len(dataset.ixtoword)

        self.text_encoder, self.image_encoder, self.netG, self.netDs, _ = self.load_model()

    def load_model(self):
        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            # TODO: elif cfg.TREE.BRANCH_NUM > 3:
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
            # TODO: if cfg.TREE.BRANCH_NUM > 3:
        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda().eval()
            image_encoder = image_encoder.cuda().eval()
            netG.cuda().eval()
            for i in range(len(netsD)):
                netsD[i].cuda().eval()
        return [text_encoder, image_encoder, netG, netsD, epoch]

    def forward_one_sent(self, sent):
        word_list = sent.split(' ')
        word_idx = [self.dataset.wordtoix.get(w, len(self.dataset.wordtoix)-2) for w in word_list]
        caption = torch.LongTensor(word_idx).unsqueeze(0)
        caption_len = torch.LongTensor([1])
        hidden = self.text_encoder.init_hidden(1)
        nz = cfg.GAN.Z_DIM
        noise = torch.FloatTensor(1, nz)
        if self.cuda:
            caption, caption_len, noise = caption.cuda(), caption_len.cuda(), noise.cuda()
        words_emb, sent_emb = self.text_encoder(caption, caption_len, hidden)
        words_emb, sent_emb = words_emb.detach(), sent_emb.detach()
        mask = (caption==0)
        num_words = words_emb.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]
        
        noise.data.normal_(0, 1)
        print(noise.shape, sent_emb.shape, words_emb.shape, mask.shape)
        fake_imgs, _, mu, logvar = self.netG(noise, sent_emb, words_emb, mask)
        out_img = fake_imgs[-1].detach().add(1).mul(128).cpu().numpy().astype(np.uint8).squeeze(0)
        print(out_img.shape)
        return out_img



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
