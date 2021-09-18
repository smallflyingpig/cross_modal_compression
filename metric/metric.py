from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from spice.spice import Spice

def evaluate_captions(res:dict, gts:dict):

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]
        rtn = {}
        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    rtn[m] = sc
            else:
                rtn[method] = score

        return rtn




if __name__=="__main__":
    res = {"1":[
        {"caption":'A bunch of small red flowers in a barnacle encrusted clay vase'},
        # {"caption":'A bunch small red flowers in a barnacle encrusted clay vase'}
        ]}
    gts = {"1":[
        {"caption":"A bunch of small red flowers in a barnacle encrusted clay"},
        {"caption":"A bunch of small flowers in a barnacle encrusted clay"}
        ]}
    rtn = evaluate_captions(res, gts)
    print(rtn)
