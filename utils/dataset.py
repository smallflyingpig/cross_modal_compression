from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# fork from: https://raw.githubusercontent.com/smallflyingpig/AttnGAN/master/code/datasets.py

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from utils.config import cfg
import PIL

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys = data
   
    # sort data by the length in a decreasing order

    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)
    if isinstance(imgs, list):
        real_imgs = []
        for i in range(len(imgs)):
            imgs[i] = imgs[i][sorted_cap_indices]
            if cfg.CUDA:
                real_imgs.append(imgs[i].cuda())
            else:
                real_imgs.append(imgs[i])
        imgs_rtn = real_imgs
    else: # array
        imgs_rtn = imgs[sorted_cap_indices]
        if cfg.CUDA:
            imgs_rtn = imgs_rtn.cuda()

    captions = captions[sorted_cap_indices].squeeze()
    sorted_cap_lens = sorted_cap_lens
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = captions.cuda()
        sorted_cap_lens = (sorted_cap_lens).cuda()
    else:
        captions = (captions)
        sorted_cap_lens = (sorted_cap_lens)

    return imgs_rtn, captions, sorted_cap_lens, class_ids, keys

def get_crop_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)
    
    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(len(imsize)):
            # print(imsize[i])
            # if i < (cfg.TREE.BRANCH_NUM - 1):
            #     re_img_size = int(imsize[i]*5/4)
            #     re_img = transforms.Resize((re_img_size, re_img_size))(img)
            # else:
            #     re_img = img
            re_img_size = int(imsize[i]*5/4)
            re_img = transforms.Resize((re_img_size, re_img_size))(img)
            re_img = transforms.ToTensor()(re_img)
            ret.append(normalize(re_img))

    
    for idx, (img_, img_size) in enumerate(zip(ret, imsize)):
        i, j, th, tw = get_crop_params(img_, (img_size, img_size))
        ret[idx] = img_[:, i:i+th, j:j+tw]

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None, sample_type='train'):
        assert split in ('train', 'val', 'test')
        if split != 'train':
            split = 'test'
        self.transform = transform
        self.img_channel = 3
        self.norm = transforms.Compose([
            transforms.Normalize((0.5,)*self.img_channel, (0.5,)*self.img_channel)])
        self.target_transform = target_transform

        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.split = split

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

        self.get_imgs = get_imgs
        self.dataset_size = len(self.filenames)*self.embeddings_num if sample_type=='train' else len(self.filenames)
        self.get_item = self.get_train_item if sample_type=='train' else self.get_test_item

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "rb") as f:
                captions = f.read()
                if isinstance(captions, bytes):
                    captions = captions.decode('utf8')
                captions = captions.split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<pad>'
        wordtoix = {}
        wordtoix['<pad>'] = 0
        
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1
        wordtoix['<unk>'] = ix
        wordtoix['<start>'] = ix+1
        wordtoix['<end>'] = ix+2
        ixtoword[ix] = '<unk>'
        ixtoword[ix+1] = '<start>'
        ixtoword[ix+2] = '<end>'

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in ['<start>'] + t + ['<end>']:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in ['<start>'] + t + ['<end>']:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]
    
    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.textimage.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f, encoding="bytes")
            if isinstance(filenames[0], bytes):
                filenames = [v.decode('utf8') for v in filenames]
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        # if (sent_caption == 0).sum() > 0:
        #     print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words-2))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM-2]
            ix = [_x+1 for _x in ix]
            ix = np.sort(ix)
            x[:, 0] = \
            np.array([sent_caption[0]]+sent_caption[ix].tolist()+[sent_caption[-1]])
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def get_train_item(self, index):
        img_idx = index//self.embeddings_num
        sent_idx = index % self.embeddings_num
        #
        key = self.filenames[img_idx]
        cls_id = self.class_id[img_idx]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, str(key))
        imgs = self.get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        # load 
        # random select a sentence
        # sent_ix = random.randint(0, self.embeddings_num)
        new_sent_idx = img_idx * self.embeddings_num + sent_idx
        caps, cap_len = self.get_caption(new_sent_idx)
        return imgs, caps, cap_len, cls_id, key

    def get_test_item(self, index):
        img_idx = index
        sent_idx = np.random.choice(list(range(self.embeddings_num)))
        #
        key = self.filenames[img_idx]
        cls_id = self.class_id[img_idx]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, str(key))
        imgs = self.get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        # load 
        # random select a sentence
        # sent_ix = random.randint(0, self.embeddings_num)
        new_sent_idx = img_idx * self.embeddings_num + sent_idx
        caps, cap_len = self.get_caption(new_sent_idx)
        return imgs, caps, cap_len, cls_id, key

    def __getitem__(self, index):
        return self.get_item(index)
        
        

    def __len__(self):
        return self.dataset_size

# this is for imagetext model training on CUB200 dataset 
class ImageTextDataset(TextDataset):
    def __init__(self, data_dir, split='train',
                 # base_size=64,
                 transform=None, target_transform=None, sample_type='train'):
        # super(ImageTextDataset, self).__init__(data_dir, split, 64, transform, target_transform)

        assert split in ('train', 'val', 'test')
        if split != 'train':
            split = 'test'
        self.transform = transform
        self.img_channel = 3
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.target_transform = target_transform

        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.split = split

        self.imsize = []
        base_size = cfg.TREE.BASE_SIZE
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.imsize = [self.imsize[-1]]

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

        self.get_imgs = get_imgs
        self.dataset_size = len(self.filenames)*self.embeddings_num if sample_type=='train' else len(self.filenames)
        self.getitem = self.get_train_item if sample_type=='train' else self.get_test_item


    def convert_to_onehot(self, data, n_class):
        onehot = np.eye(n_class)[data.reshape(-1)]
        return onehot.reshape(list(data.shape)+[n_class])

    
    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                # captions = f.read().decode('utf8').split('\n')
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(['<start>']+tokens_new+['<end>'])
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def get_train_item(self, index):
        img_idx = index//self.embeddings_num
        sent_idx = index % self.embeddings_num
        #
        key = self.filenames[img_idx]
        cls_id = self.class_id[img_idx]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, str(key))
        imgs = self.get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        # load 
        # random select a sentence
        # sent_ix = random.randint(0, self.embeddings_num)
        new_sent_idx = img_idx * self.embeddings_num + sent_idx
        caps, cap_len = self.get_caption(new_sent_idx)
        return imgs, caps, cap_len, cls_id, key

    def get_test_item(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = self.get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        caps_all = np.stack([self.get_caption(index * self.embeddings_num +
            _sent_ix_temp)[0].squeeze() for _sent_ix_temp in
            range(self.embeddings_num)], axis=0)

        return imgs[0], caps_all, cls_id, key

    def __getitem__(self, index):
        return self.getitem(index)
    
    def __len__(self):
        return self.dataset_size
            


import h5py
import json
import torch
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
# this dataset is for imagetext model training on COCO dataset
# dataset codes are forked from: https://github.com/smallflyingpig/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/datasets.py
class CaptionDataset(data.Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, transform=None, sample_type='train', coco_data_json='dataset_coco.json'):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        super(CaptionDataset, self).__init__()
        self.data_folder = data_folder
        self.split = split
        assert self.split in {'train', 'val'}
        data_name = 'coco' + '_' + str(cfg.TEXT.CAPTIONS_PER_IMAGE) + '_cap_per_img_' + str(cfg.TEXT.MIN_WORD_FREQ) + '_min_word_freq'
        # img_file = os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '_{}.hdf5'.format(hdf5_idx))
        metadata_file = os.path.join(data_folder, self.split + '_METADATA_' + data_name + '.json')
        wordmap_file = os.path.join(data_folder, 'WORDMAP_'+data_name+'.json')
        if not os.path.exists(metadata_file):
            print("dataset does not exist, create it...")
            create_input_files(
                'coco', os.path.join(data_folder, coco_data_json), 
                data_folder, cfg.TEXT.CAPTIONS_PER_IMAGE, 
                cfg.TEXT.MIN_WORD_FREQ, data_folder, cfg.TEXT.WORDS_NUM, word_map_file=wordmap_file
                )

        # Open hdf5 file where images are stored
        # Captions per image
        self.cpi = cfg.TEXT.CAPTIONS_PER_IMAGE

        # Load encoded captions (completely into memory)
        with open(metadata_file, 'r') as j:
            self.metadata = json.load(j)
        # Load word map
        with open(wordmap_file, 'r') as j:
            self.wordmap = json.load(j)
        self.n_words = len(self.wordmap)
        self.wordtoix = self.wordmap
        self.ixtoword = {v:k for k,v in self.wordtoix.items()}

        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        if self.split == 'train':
            self.dataset_size = len(self.metadata) * self.cpi
        else:
            self.dataset_size = len(self.metadata)

        print("data size:", self.dataset_size)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.getitem = self.get_train_item if sample_type=='train' else self.get_test_item

    def get_train_item(self, i):
        img_idx = i // self.cpi
        cap_idx = i % self.cpi
        img_path = self.metadata[img_idx]['img_path']
        img = imread(img_path)
        # print(img_path)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = imresize(img, (256, 256))
        img = img.transpose(2, 0, 1)
        img = torch.FloatTensor(img / 255.)
        if self.norm is not None:
            img = self.norm(img)

        caption = torch.LongTensor(self.metadata[img_idx]['cap'][cap_idx])

        caplen = torch.LongTensor([self.metadata[img_idx]['cap_len'][cap_idx]])

        return img, caption, caplen, img_idx, str(img_idx)

    def get_test_item(self, i):
        img_idx = i 
        img_path = self.metadata[img_idx]['img_path']
        img = imread(img_path)
        # print(img_path)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = imresize(img, (256, 256))
        img = img.transpose(2, 0, 1)
        img = torch.FloatTensor(img / 255.)
        if self.norm is not None:
            img = self.norm(img)
        # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
        all_captions = torch.LongTensor(self.metadata[img_idx]['cap'])
        return img, all_captions, img_idx, os.path.splitext(img_path)[0]


    def __getitem__(self, i):
        return self.getitem(i)

    def __len__(self):
        return self.dataset_size


class CaptionDatasetMultisize(CaptionDataset):
    def __init__(self, data_folder, split, base_size, transform=None, sample_type='train', coco_data_json='dataset_coco.json'):
        super(CaptionDatasetMultisize, self).__init__(data_folder, split,
                transform, coco_data_json)
        self.brunch_num = cfg.TREE.BRANCH_NUM
        imsize = []
        image_transform = []
        for _ in range(self.brunch_num):
            imsize.append(base_size)
            if split == 'train':
                image_transform.append(
                    transforms.Compose([
                            transforms.Resize(int(base_size * 76 / 64)),
                            transforms.RandomCrop(base_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
                )
            else: # val, test
                image_transform.append(
                    transforms.Compose([
                            transforms.Resize(int(base_size * 76 / 64)),
                            transforms.CenterCrop(base_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
                )
            base_size *= 2

        self.imsize = imsize
        self.image_transform = image_transform
        self.data_folder = data_folder
        self.get_item = self.get_train_item if sample_type=='train' else self.get_test_item

    def get_train_item(self, idx):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img_idx = idx // self.cpi
        cap_idx = idx % self.cpi
        # img = self.imgs[img_idx]
        img_path = self.metadata[img_idx]['img_path']
        # img = imread(img_path)
        # if len(img.shape) == 2:
        #     img = img[:, :, np.newaxis]
        #     img = np.concatenate([img, img, img], axis=2)
        imsize = self.imsize
        real_imgs = []
        img = PIL.Image.open(img_path)
        if img.mode != 'RGB':
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = rgbimg
        # img = PIL.Image.fromarray(img.astype('uint8'), 'RGB')
        for transform in self.image_transform:
            re_img = transform(img)
            real_imgs.append(re_img)

        caption = np.array(self.metadata[img_idx]['cap'][cap_idx])

        caplen = self.metadata[img_idx]['cap_len'][cap_idx]

        image_filename = os.path.split(os.path.splitext(img_path)[0])[-1]

        return real_imgs, caption, caplen, img_idx, image_filename

    def get_test_item(self, idx):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img_idx = idx
        cap_idx = np.random.choice(list(range(self.cpi)))
        # img = self.imgs[img_idx]
        img_path = self.metadata[img_idx]['img_path']
        # img = imread(img_path)
        # if len(img.shape) == 2:
        #     img = img[:, :, np.newaxis]
        #     img = np.concatenate([img, img, img], axis=2)
        imsize = self.imsize
        real_imgs = []
        img = PIL.Image.open(img_path)
        if img.mode != 'RGB':
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = rgbimg
        # img = PIL.Image.fromarray(img.astype('uint8'), 'RGB')
        for transform in self.image_transform:
            re_img = transform(img)
            real_imgs.append(re_img)

        caption = np.array(self.metadata[img_idx]['cap'][cap_idx])

        caplen = self.metadata[img_idx]['cap_len'][cap_idx]

        image_filename = os.path.split(os.path.splitext(img_path)[0])[-1]

        return real_imgs, caption, caplen, img_idx, image_filename
        
    def __getitem__(self, idx):
        return self.get_item(idx)


def create_input_files(
    dataset, karpathy_json_path, image_folder, 
    captions_per_image, min_word_freq, 
    output_folder, max_len=50, word_map_file=""):
    """
    Creates input files for training, validation, and test data.
    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
    if word_map_file != "":
        # read work map
        print("load wordmap from: ", word_map_file)
        with open(word_map_file, 'r') as j:
            word_map = json.load(j)
    else:
        # Create word map
        words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
        word_map = {k: v + 1 for v, k in enumerate(words)}
        word_map['<unk>'] = len(word_map) + 1
        word_map['<start>'] = len(word_map) + 1
        word_map['<end>'] = len(word_map) + 1
        word_map['<pad>'] = 0
        # Save word map to a JSON
        with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
            json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions,'train'),
                                   (val_image_paths, val_image_captions, 'val'),
                                   (test_image_paths, test_image_captions, 'test')]:
        # with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
        # Make a note of the number of captions we are sampling per image
        # h.attrs['captions_per_image'] = captions_per_image
        # Create dataset inside HDF5 file to store images
        # images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')
        print("\nReading %s images and captions, storing to file...\n" % split)
        metadata = []
        for i, path in enumerate(tqdm(impaths)):
            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)
            # Sanity check
            assert len(captions) == captions_per_image
            # Read images
            # img = imread(impaths[i])
            # if len(img.shape) == 2:
            #     img = img[:, :, np.newaxis]
            #     img = np.concatenate([img, img, img], axis=2)
            # img = imresize(img, (256, 256))
            # img = img.transpose(2, 0, 1)
            # assert img.shape == (3, 256, 256)
            # assert np.max(img) <= 255
            # Save image to HDF5 file
            # images[i] = img
            enc_cap_this = []
            enc_cap_len_this = []
            for j, c in enumerate(captions):
                # Encode captions
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                    word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                # Find caption lengths
                c_len = len(c) + 2
                enc_cap_this.append(enc_c)
                enc_cap_len_this.append(c_len)
            metadata.append({'cap':enc_cap_this, 'cap_len':enc_cap_len_this, 'img_path':os.path.abspath(path), 'text':c})

            # Sanity check
            # assert images.shape[0] == len(metadata)

        # Save encoded captions and their lengths to JSON files
        metadata_file = os.path.join(output_folder, split + '_METADATA_' + base_filename + '.json')
        with open(metadata_file, 'w') as j:
            json.dump(metadata, j)
        print("save {} metadata to {}".format(split, metadata_file))


