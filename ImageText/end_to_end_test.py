# this script is used to evaluate the end to end performance for cross modal compression
import argparse, os, tqdm
import torch, torchvision
from ImageText.models import Encoder, DecoderWithAttention, load_checkpoint
from TextImage.trainer import condGANTrainer as trainer
from utils.dataset import ImageTextDataset, CaptionDataset, TextDataset, CaptionDatasetMultisize
from utils.config import cfg_from_file, cfg
from torchvision.transforms import transforms
import torch.nn.functional as F 
from utils.util import Outputer, dict2str, RunningAverage
from nltk.translate.bleu_score import corpus_bleu
import numpy as np 
from pprint import pprint


def get_parser():
    parser = argparse.ArgumentParser("end to end evaluation")
    parser.add_argument("--cfg", type=str, default="./cfg/example.cfg")
    parser.add_argument("--gpu_id", type=int, default=0, help="")
    parser.add_argument("--data_dir", type=str, default="./data/birds")
    parser.add_argument("--imagetext_model_path", type=str, default="", help="")
    parser.add_argument("--textimage_model_path", type=str, default="", help="")
    parser.add_argument("--textimage_nete_path", type=str, default="")
    parser.add_argument("--dataset", choices=['coco', 'bird'], default="bird", help="")
    parser.add_argument("--seed", type=int, default=1234, help="")
    parser.add_argument("--beam_size", type=int, default=4, help="")
    parser.add_argument("--output_dir", type=str, default="output/end_to_end", help="")
    parser.add_argument("--local_rank", type=int, default=0, help="")
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
def gen_example(wordtoix, algo, text_list, filename_list):
    from nltk.tokenize import RegexpTokenizer
    # a list of indices for a sentence
    captions = []
    cap_lens = []
    for sent in text_list:
        if len(sent) == 0:
            continue
        sent = sent.replace("\ufffd\ufffd", " ")
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sent.lower())
        if len(tokens) == 0:
            print('sent', sent)
            continue

        rev = []
        for t in ['<start>'] + tokens + ['<end>']:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0 and t in wordtoix:
                rev.append(wordtoix[t])
        captions.append(rev)
        cap_lens.append(len(rev))
    max_len = max(cap_lens)
    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    filename_list = [filename_list[_idx] for _idx in sorted_indices]
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    key = "end_to_end_test"
    data_dict = {key:[cap_array, cap_lens, sorted_indices, filename_list]}
    algo.gen_text_example(data_dict)



@torch.no_grad()
def evaluate(val_dataloader:ImageTextDataset, 
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
            # print(complete_inds)

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

        write_line = ','.join(ref_str_list + [hyp_str]+[keys[0]])
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
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    
    if args.textimage_nete_path != '':
        cfg.TRAIN.NET_E = args.textimage_nete_path
    pprint(cfg)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    outputer = Outputer(args.output_dir, cfg.IMAGETEXT.PRINT_EVERY, cfg.IMAGETEXT.SAVE_EVERY)
    outputer.log(cfg)
    imsize = cfg.TREE.BASE_SIZE * (2 **(cfg.TREE.BRANCH_NUM - 1))
    val_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.CenterCrop(imsize),
    ])
    if args.dataset == 'bird':
        imagetext_val_dataset = ImageTextDataset(args.data_dir, 'test', 
            transform=val_transform, sample_type='test')
    elif args.dataset == 'coco':
        imagetext_val_dataset = CaptionDataset(args.data_dir, 'val',
            transform=val_transform, sample_type='test')
    else:
        raise NotImplementedError
    imagetext_val_dataloader = torch.utils.data.DataLoader(imagetext_val_dataset,
        batch_size=1, shuffle=False, num_workers=int(cfg.WORKERS))
        
    # evaluation imagetext model
    if args.imagetext_model_path != '':
        outputer.log("load raw checkpoint from: {}".format(args.imagetext_model_path))
        if os.path.splitext(args.imagetext_model_path)[-1] == '.tar': # raw checkpoint 
            imagetext_encoder, imagetext_decoder = load_raw_checkpoint(args.imagetext_model_path)
        else: #define the model
            imagetext_encoder = Encoder()
            imagetext_encoder.fine_tune(False)
            imagetext_decoder = DecoderWithAttention(attention_dim=cfg.IMAGETEXT.ATTENTION_DIM,
                embed_dim=cfg.IMAGETEXT.EMBED_DIM, decoder_dim=cfg.IMAGETEXT.DECODER_DIM,
                vocab_size=imagetext_val_dataset.n_words)
            imagetext_encoder, imagetext_decoder = load_checkpoint(imagetext_encoder, imagetext_decoder, args.imagetext_model_path)
    else:
        print("the model path for imagetext model is NULL")
        raise ValueError

    imagetext_encoder, imagetext_decoder = imagetext_encoder.cuda(), imagetext_decoder.cuda()
    text_file = os.path.join(outputer.log_folder, "pred_result.csv")
    imagetext_eval_rtn, outputer = evaluate(imagetext_val_dataloader, 
        imagetext_encoder, imagetext_decoder, outputer, args.beam_size, text_file)
    # write the pred to local file
    text_list = []
    filename_list = []
    with open(text_file, 'r') as fp:
        for _l in fp.readlines():
            data_list = _l.strip().split(',')
            text_list.append(data_list[-2])
            filename_list.append(data_list[-1])
        
    #--------- generate images from the text -------#
    if args.dataset == 'bird':
        textimage_val_dataset = TextDataset(args.data_dir, 'test', 
            base_size=cfg.TREE.BASE_SIZE, transform=val_transform)
    elif args.dataset == 'coco':
        textimage_val_dataset = CaptionDatasetMultisize(args.data_dir, 'val',
            base_size=cfg.TREE.BASE_SIZE, transform=val_transform)
    else:
        raise ValueError
    textimage_val_dataloader = torch.utils.data.DataLoader(textimage_val_dataset, 
        batch_size=1)
    image_output_dir = os.path.join(args.output_dir, "images")
    algo = trainer(image_output_dir, textimage_val_dataloader, textimage_val_dataset.n_words, textimage_val_dataset.ixtoword)
    gen_example(textimage_val_dataset.wordtoix, algo, text_list, filename_list)



if __name__=="__main__":
    args = get_parser()
    main(args)
