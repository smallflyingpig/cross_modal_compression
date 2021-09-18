import numpy as np
import argparse, os, sys, json, time
import torch, torchvision
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.transforms as transforms

# sys.path.append(os.path.abspath(".."))
from utils.dataset import ImageTextDataset, CaptionDataset
from utils.config import cfg, cfg_from_file
from models import Encoder, DecoderWithAttention, load_checkpoint
from utils.util import RunningAverage, dict2str, save_dict_models,\
     load_dict_models, Outputer
from nltk.translate.bleu_score import corpus_bleu
import random

import pprint

def get_parser():
    parser = argparse.ArgumentParser("train ImageText Model")
    parser.add_argument("--cfg", type=str, default="./utils/cmc.cfg", help="")
    parser.add_argument("--data_dir", type=str, default="./data/birds", help="")
    parser.add_argument("--output_dir", type=str, default="./output/", help="")
    parser.add_argument("--seed", type=int, default=1234, help="")
    parser.add_argument("--eval", action='store_true', default=False, help="")
    parser.add_argument("--dataset", choices=['bird', 'coco'], type=str, default='bird', help="")
    parser.add_argument("--local_rank", type=int, default=0, help="")
    parser.add_argument("--raw_checkpoint", type=str, default='', help="")
    parser.add_argument("--coco_data_json", type=str, default='dataset_coco.json', help="")
    parser.add_argument("--num_workers", type=int, default=8, help="")
    args = parser.parse_args()
    return args

def load_raw_checkpoint(path:str):
    assert path != ''
    print("load raw checkpoint from: ", path)
    checkpoint = torch.load(path)
    # start_epoch = checkpoint['epoch'] + 1
    # epochs_since_improvement = checkpoint['epochs_since_improvement']
    # best_bleu4 = checkpoint['bleu-4']
    decoder = checkpoint['decoder']
    # decoder_optimizer = checkpoint['decoder_optimizer']
    encoder = checkpoint['encoder']
    # encoder_optimizer = checkpoint['encoder_optimizer']
    return encoder, decoder

def train_one_epoch(epoch_idx:int,
    train_loader:torch.utils.data.DataLoader,
    encoder:torch.nn.Module, decoder:torch.nn.Module,
    optimizer_encoder:torch.optim.Optimizer, 
    optimizer_decoder:torch.optim.Optimizer,
    loss_func, outputer:Outputer):
    encoder.train()
    decoder.train()
    loss_epoch = RunningAverage()
    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        imgs, caps, cap_lens, class_id, filename = data
        if isinstance(imgs, list):
            imgs = imgs[-1]
        imgs, caps, cap_lens = imgs.cuda(), caps.cuda(), cap_lens.cuda()
        # print(caps.shape)
        if len(caps.shape) == 3 and caps.shape[-1]==1:
            caps = caps.squeeze(2)
        # forward
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = \
            decoder(imgs, caps, cap_lens)
        targets = caps_sorted[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths,
                batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths,
                batch_first=True).data

        loss_ce = loss_func(scores, targets) 
        loss_attn = (1.0 - alphas.sum(dim=1)**2).mean() 

        loss_total = loss_ce + cfg.IMAGETEXT.ALPHA_C * loss_attn

        # backward
        if optimizer_encoder is not None:
            encoder.zero_grad()
        decoder.zero_grad()
        loss_total.backward()

        # clip the gradient
        if cfg.IMAGETEXT.GRAD_CLIP_VALUE > 0:
            torch.nn.utils.clip_grad_value_(decoder.parameters(), cfg.IMAGETEXT.GRAD_CLIP_VALUE)
            torch.nn.utils.clip_grad_value_(encoder.parameters(), cfg.IMAGETEXT.GRAD_CLIP_VALUE)
            
        # update weight
        optimizer_decoder.step()
        if optimizer_encoder is not None:
            optimizer_encoder.step()

        loss_dict = {'ce': loss_ce.detach().cpu().item(), 
            'attn': loss_attn.detach().cpu().item(),
            'total': loss_total.detach().cpu().item()}
        loss_epoch.update(loss_dict, {k:imgs.shape[0] for k in
            loss_dict.keys()})
        
        outputer.log_step("[train] epoch:{:3d}, batch:[{:3d}/{:3d}], time:{:4.1f}, loss:[{}]".format(
            epoch_idx, batch_idx, len(train_loader), time.time()-start_time, dict2str(loss_dict)
        ))
        start_time = time.time()
    rtn = loss_epoch.average()
    return rtn, outputer
 

@torch.no_grad()
def validate_one_epoch(epoch_idx:int,
    val_loader:torch.utils.data.DataLoader,
    encoder:torch.nn.Module, decoder:torch.nn.Module,
    loss_func, outputer:Outputer):
    encoder.eval()
    decoder.eval()
    loss_epoch = RunningAverage()
    start_time = time.time()
    ref_list, pred_list = [], []
    word_map = val_loader.dataset.wordtoix
    ixtoword = val_loader.dataset.ixtoword
    for batch_idx, data in enumerate(val_loader):
        imgs, all_caps, class_id, filename = data
        if isinstance(imgs, list):
            imgs = imgs[-1]
        batch_size, cap_num, _ = all_caps.shape
        # rand for caps
        rand_idx = torch.randint(cap_num, (batch_size,))
        caps = [all_caps[_idx][rand_idx[_idx]] for _idx in range(batch_size)]
        cap_lens = [(_cap != 0).sum() for _cap in caps ]
        caps, cap_lens = torch.from_numpy(np.stack(caps)).long(), torch.from_numpy(np.stack(cap_lens)).long()

        imgs, caps, cap_lens = \
            imgs.cuda(), caps.cuda(), cap_lens.cuda()
        if len(caps.shape) == 3 and caps.shape[-1] == 1:
            caps = caps.squeeze(2)
        # forward
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = \
            decoder(imgs, caps, cap_lens)
        targets = caps_sorted[:, 1:]
        scores_copy = scores.clone()
        scores = pack_padded_sequence(scores, decode_lengths,
                batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths,
                batch_first=True).data

        loss_ce = loss_func(scores, targets) 
        loss_attn = (1.0 - alphas.sum(dim=1)**2).mean() 

        loss_total = loss_ce + cfg.IMAGETEXT.ALPHA_C * loss_attn

        loss_dict = {'ce': loss_ce.detach().cpu().item(), 
            'attn': loss_attn.detach().cpu().item(),
            'total': loss_total.detach().cpu().item()}
        loss_epoch.update(loss_dict, {k:imgs.shape[0] for k in
            loss_dict.keys()})
        outputer.log_step("[valid] epoch:{:3d}, batch:[{:3d}/{:3d}], time:{:4.1f}, loss:[{}]".format(
                epoch_idx, batch_idx, len(val_loader), time.time()-start_time, dict2str(loss_dict)
            ))
        # References
        allcaps = all_caps[sort_ind]  # because images were sorted in the decoder
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in [word_map['<start>'],
                    word_map['<pad>']]],
                    img_caps))  # remove <start> and pads
            img_captions = [[ixtoword[idx] for idx in sent] for sent in img_captions]
            ref_list.append(img_captions)
        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append([ixtoword[idx] for idx in p[:decode_lengths[j]]])  # remove pads
        preds = temp_preds
        pred_list.extend(preds)
        assert len(ref_list) == len(pred_list)
        start_time = time.time()

    bleu4 = corpus_bleu(ref_list, pred_list)
    rtn = loss_epoch.average()
    rtn.update({"bleu4":bleu4})
    # save pred result
    with open(os.path.join(outputer.log_folder, "debug.txt"), "w") as fp:
        for ref, pred in zip(ref_list, pred_list):
            ref, pred = [' '.join(r) for r in ref], ' '.join(pred)
            string = '|'.join(ref) + '|||'+pred+'\n'
            fp.write(string)
    return rtn, outputer


def train(args):
    cfg_from_file(args.cfg)
    cfg.WORKERS = args.num_workers
    pprint.pprint(cfg)
    # set the seed manually
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # define outputer
    outputer_train = Outputer(args.output_dir, cfg.IMAGETEXT.PRINT_EVERY, cfg.IMAGETEXT.SAVE_EVERY)
    outputer_val = Outputer(args.output_dir, cfg.IMAGETEXT.PRINT_EVERY,
            cfg.IMAGETEXT.SAVE_EVERY)
    # define the dataset
    split_dir, bshuffle = 'train', True

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 **(cfg.TREE.BRANCH_NUM - 1))
    train_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
    ])
    val_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.CenterCrop(imsize),
    ])
    if args.dataset == 'bird':
        train_dataset = ImageTextDataset(args.data_dir,
            split_dir,
            transform=train_transform, sample_type='train')
        val_dataset = ImageTextDataset(args.data_dir, 'val', 
            transform=val_transform, sample_type='val')
    elif args.dataset == 'coco':
        train_dataset = CaptionDataset(args.data_dir,
            split_dir,
            transform=train_transform, sample_type='train',
            coco_data_json=args.coco_data_json)
        val_dataset = CaptionDataset(args.data_dir, 'val', 
            transform=val_transform, sample_type='val', 
            coco_data_json=args.coco_data_json)
    else:
        raise NotImplementedError

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
            batch_size=cfg.IMAGETEXT.BATCH_SIZE, shuffle=bshuffle,
            num_workers=int(cfg.WORKERS))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
            batch_size=cfg.IMAGETEXT.BATCH_SIZE, shuffle=False,
            num_workers=1)
    # define the model and optimizer
    if args.raw_checkpoint != '':
        encoder, decoder = load_raw_checkpoint(args.raw_checkpoint)
    else:
        encoder = Encoder()
        decoder =  DecoderWithAttention(attention_dim=cfg.IMAGETEXT.ATTENTION_DIM,
                embed_dim=cfg.IMAGETEXT.EMBED_DIM,
                decoder_dim=cfg.IMAGETEXT.DECODER_DIM,
                vocab_size=train_dataset.n_words)
        # load checkpoint
        if cfg.IMAGETEXT.CHECKPOINT != '':
            outputer_val.log("load model from: {}".format(cfg.IMAGETEXT.CHECKPOINT))
            encoder, decoder = load_checkpoint(encoder, decoder,
                    cfg.IMAGETEXT.CHECKPOINT)
    
    encoder.fine_tune(False)
    # to cuda
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    loss_func = torch.nn.CrossEntropyLoss()
    if args.eval: # eval only
        outputer_val.log("only eval the model...")
        assert cfg.IMAGETEXT.CHECKPOINT != ''
        val_rtn_dict, outputer_val = validate_one_epoch(0, val_dataloader, 
                encoder, decoder, loss_func, outputer_val)
        outputer_val.log("\n[valid]: {}\n".format(dict2str(val_rtn_dict)))
        return 

    # define optimizer
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=cfg.IMAGETEXT.ENCODER_LR)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=cfg.IMAGETEXT.DECODER_LR)
    encoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=10, gamma=cfg.IMAGETEXT.LR_GAMMA)
    decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_decoder, step_size=10, gamma=cfg.IMAGETEXT.LR_GAMMA)
    print("train the model...")
    for epoch_idx in range(cfg.IMAGETEXT.EPOCH):
        # val_rtn_dict, outputer_val = validate_one_epoch(epoch_idx, val_dataloader, encoder,
        #         decoder, loss_func, outputer_val)
        # outputer_val.log("\n[valid] epoch: {}, {}".format(epoch_idx, dict2str(val_rtn_dict)))
        train_rtn_dict, outputer_train = train_one_epoch(epoch_idx, train_dataloader, 
            encoder, decoder, 
            optimizer_encoder, optimizer_decoder, 
            loss_func, outputer_train)
        # adjust lr scheduler
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        
        outputer_train.log("\n[train] epoch: {}, {}\n".format(epoch_idx,
            dict2str(train_rtn_dict)))
        val_rtn_dict, outputer_val = validate_one_epoch(epoch_idx, val_dataloader, 
                encoder, decoder, loss_func, outputer_val)
        outputer_val.log("\n[valid] epoch: {}, {}\n".format(epoch_idx,
                dict2str(val_rtn_dict)))

        outputer_val.save_step({"encoder":encoder.state_dict(), "decoder":decoder.state_dict()})
    outputer_val.save({"encoder":encoder.state_dict(), "decoder":decoder.state_dict()})

    
def main(args):
    train(args)


if __name__=="__main__":
    args = get_parser()
    main(args)



