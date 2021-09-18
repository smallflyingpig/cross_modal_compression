# evaluate the pretrained model with beam search
import os, sys, json, pickle, argparse, tqdm
import torch, torchvision
from ImageText.models import Encoder, DecoderWithAttention, load_checkpoint
from utils.util import Outputer, dict2str, RunningAverage
from utils.dataset import ImageTextDataset, CaptionDataset
from utils.config import cfg_from_file, cfg
from pprint import pprint
from torchvision.transforms import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
import numpy as np 



def get_parser():
    parser = argparse.ArgumentParser("evaluation")
    parser.add_argument("--cfg", type=str, default="./cfg/example.cfg")
    parser.add_argument("--data_dir", type=str, default="./data/birds", help="")
    parser.add_argument("--model_path", type=str, default="", help="")
    parser.add_argument("--output_dir", type=str, default="./output/evaluation", help="")
    parser.add_argument("--beam_size", type=int, default=1, help="")
    parser.add_argument("--dataset", choices=['coco', 'bird'], default="bird", help="")
    parser.add_argument("--raw_checkpoint", type=str, default="", help="")
    parser.add_argument("--seed", type=int, default=1234, help="")
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

@torch.no_grad()
def evaluate(val_dataloader:DataLoader, 
            encoder:Encoder, decoder:DecoderWithAttention, 
            outputer:Outputer,
            beam_size:int, pred_file:str):
    encoder.eval()
    decoder.eval()
    word_map = val_dataloader.dataset.wordtoix
    ixtoword = val_dataloader.dataset.ixtoword
    vocab_size = val_dataloader.dataset.n_words
    references, hypotheses = [], []
    batch_num = len(val_dataloader)
    print("batch num:", batch_num)
    pred_result_fp = open(pred_file, 'w')
    key_list = []
    ref_all_list = []
    pred_list = []
    for batch_idx, data in tqdm.tqdm(enumerate(val_dataloader)):
        imgs, all_caps, cls_ids, keys = data
        k = beam_size
        imgs, all_caps = \
            imgs.cuda(), all_caps.cuda()
        encoder_out = encoder(imgs)
        enc_img_size, encoder_dim = encoder_out.size(1), encoder_out.size(3)
        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(beam_size, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * beam_size).cuda()  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(beam_size, 1).cuda()  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        # alphas = []
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
            # alphas.append(alpha)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(beam_size, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(beam_size, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k <= 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                outputer.log("the sentence is too long, break it")
                break
            step += 1
        # save alphas
        # with open('./alphas.pickle', 'wb') as fp:
        #     pickle.dump(alphas, fp)
        #     exit(0)

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = all_caps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hyp = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        hypotheses.append(hyp)
        ref_str_list = [' '.join([ixtoword[_r] for _r in r]) for r in img_captions]
        hyp_str = ' '.join([ixtoword[_h] for _h in hyp])

        write_line = ','.join(ref_str_list + [hyp_str])
        pred_result_fp.write(write_line+'\n')
        assert len(references) == len(hypotheses)
        outputer.log_step("[{}/{}]:{}".format(
            batch_idx, batch_num, '|'.join(ref_str_list)+"|||"+hyp_str
            ))
        key_list.append(keys)
        ref_all_list.append(ref_str_list)
        pred_list.append(hyp_str)
    pred_result_fp.close()
    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return {"bleu4": bleu4, 'filename':key_list, "ref":ref_all_list, "pred":pred_list}, outputer



def main(args):
    cfg_from_file(args.cfg)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    outputer = Outputer(args.output_dir, cfg.IMAGETEXT.PRINT_EVERY, cfg.IMAGETEXT.SAVE_EVERY)
    outputer.log(cfg)
    imsize = cfg.TREE.BASE_SIZE * (2 **(cfg.TREE.BRANCH_NUM - 1))
    val_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.CenterCrop(imsize),
    ])
    if args.dataset == 'bird':
        val_dataset = ImageTextDataset(args.data_dir, 'test', 
            transform=val_transform, sample_type='test')
    elif args.dataset == 'coco':
        val_dataset = CaptionDataset(args.data_dir, 'val',
            transform=val_transform, sample_type='test')
    else:
        raise NotImplementedError
    
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
            batch_size=1, shuffle=False,
            num_workers=int(cfg.WORKERS))
    if args.raw_checkpoint != '':
        outputer.log("load raw checkpoint from {}".format(args.raw_checkpoint))
        encoder, decoder = load_raw_checkpoint(args.raw_checkpoint)
    else:
        # define the model
        encoder = Encoder()
        encoder.fine_tune(False)
        decoder = DecoderWithAttention(attention_dim=cfg.IMAGETEXT.ATTENTION_DIM, 
                                    embed_dim=cfg.IMAGETEXT.EMBED_DIM, 
                                    decoder_dim=cfg.IMAGETEXT.DECODER_DIM, 
                                    vocab_size=val_dataset.n_words)
        assert args.model_path != ""
        outputer.log("load model dict from {}".format(args.model_path))
        encoder, decoder = load_checkpoint(encoder, decoder, args.model_path)
        encoder.fine_tune(False)
    encoder, decoder = encoder.cuda(), decoder.cuda()
    outputer.log("eval the model...")
    pred_file = os.path.join(outputer.log_folder, 'pred_result.csv')
    eval_rtn, outputer = evaluate(val_dataloader, 
                            encoder, decoder, 
                            outputer,
                            args.beam_size, pred_file)
    outputer.log("eval result: {}".format(dict2str(eval_rtn)))
    


if __name__=="__main__":
    args = get_parser()
    main(args)
