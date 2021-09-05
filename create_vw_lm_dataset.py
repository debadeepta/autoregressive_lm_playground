import math
import random
import argparse
from typing import Tuple, Dict, List
from collections import defaultdict
from tqdm import tqdm
import os
import time

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torchtext.datasets import WikiText2
from torchtext.datasets import WikiText103
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from vowpalwabbit import pyvw

def create_char_seqs(text_str:str, context_length:int)->Tuple[List[List[str]], List[str]]:
    chars = list(text_str.rstrip('\n'))
    if len(chars) < context_length + 1:
        return None, None
    else:
        seqs = []
        targets = []
        end = len(chars) - context_length
        for s in range(end):
            seq = chars[s:s+context_length]
            target = chars[s+context_length]
            seqs.append(seq)
            targets.append(target)
        return seqs, targets
            

def create_vw_examples(raw_text_iter: dataset.IterableDataset, 
                        context_length:int, 
                        stoi:Dict[str, int],
                        vw_file_name:str,
                        num_max_examples: int=-1):
    # extract character sequences of context length and the next target character
    
    ex_counter = 0
    with open(vw_file_name, 'w') as f:
        for item in tqdm(raw_text_iter):
            if num_max_examples > 0 and ex_counter > num_max_examples:
                break
            seqs, targets = create_char_seqs(item, context_length)
            if seqs:
                for seq, target in zip(seqs, targets):
                    target_idx = stoi[target]
                    if target_idx == -1:
                        continue
                    feature_str = ''
                    for idx, char in enumerate(seq):
                        token_id = stoi[char]
                        if token_id == 'missing':
                            token_id = stoi['<unk>']
                        feature_str += str(token_id) + ' '
                    ex_str = str(target_idx) + ' ' + '|e ' +  feature_str + '\n'
                    f.write(ex_str)
                    ex_counter += 1


def main():
    parser = argparse.ArgumentParser(description='Report creator')
    parser.add_argument('--output-dir', '-o', type=str,
                        default=r'~/dataroot/vw_dataset',
                        help='full path to output directory')
    parser.add_argument('--vocab-embeddings-file', '-e', type=str,
                        help='full path to character level vocab embeddings')
    parser.add_argument('--context-length', '-l', type=int,
                        help='number of last character embeddings to use as features')
    parser.add_argument('--max-examples', '-m', type=int,
                        help='maximum number of examples to generate in vw format')
    parser.add_argument('--vw-file-name', '-f', type=str,
                        help='name of the vw data file')
    args, extra_args = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load the embeddings
    stoi = defaultdict(lambda: -1)
    itos = defaultdict(lambda: 'missing')
    stoe = defaultdict(lambda: 'missing')
    itoe = defaultdict(lambda: 'missing')
    with open(args.vocab_embeddings_file, 'r') as f:
        embed_lines = f.readlines()
        for line in embed_lines:
            temp = eval(line)
            idx = temp['idx']
            token = temp['token']
            embed = temp['embed']
            stoi[token] = idx
            itos[idx] = token
            stoe[token] = embed
            itoe[idx] = embed


    # create vw examples
    train_iter, val_iter, test_iter = WikiText2()
    vw_file_name = os.path.join(args.output_dir, args.vw_file_name)
    create_vw_examples(train_iter, 
                        args.context_length, 
                        stoi, 
                        vw_file_name, 
                        num_max_examples=args.max_examples)

    # convert the embedding dictionary to VW format
    edict_save_name = os.path.join(args.output_dir, 'embeddings_vw.dict')
    with open(edict_save_name, 'w') as f:
        for k, v in itoe.items():
            estring = str(k) + ' '
            for idx, feat in enumerate(v):
                estring += str(idx) + ':' + str(feat) + ' '
            estring += '\n'
            f.write(estring)




    

if __name__ == '__main__':
    main()


