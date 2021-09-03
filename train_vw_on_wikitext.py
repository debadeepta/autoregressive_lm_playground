import math
import random
import argparse
from typing import Tuple, Dict, List
from collections import defaultdict
from tqdm import tqdm
import os

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
            


def featurize_char_seqs(char_seqs:List[List[str]], 
                        char_targets:List[str], 
                        num_max_examples:int, 
                        stoe:Dict[str, List[float]], 
                        stoi:Dict[str, int])->List[str]:

    # featurize each example
    example_store = []
    
    for char_seq, char_target in tqdm(zip(char_seqs, char_targets)):

        if num_max_examples > 0 and len(example_store) > num_max_examples:
            break

        target_idx = stoi[char_target]
        if target_idx == -1:
            continue
        feature_string = ''
        feature_list = []
        for idx,char in enumerate(char_seq):
            embed = stoe[char]
            if embed == 'missing':
                embed = stoe['<unk>']
            feature_list.append(embed)
        for idx, embed in enumerate(feature_list):
            feature_string += '|' + str(idx) + ' '
            for fidx, feat in enumerate(embed):
                feature_string += str(fidx) + ':' + feat + ' '
        example_string = str(target_idx) + ' ' + feature_string + '\n'
        example_store.append(example_string) 

    return example_store

def create_vw_examples(raw_text_iter: dataset.IterableDataset, 
                        context_length:int, 
                        stoe:Dict[str, List[float]], 
                        stoi:Dict[str, int],
                        num_max_examples: int=-1):
    # extract character sequences of context length and the next target character
    char_seqs = []
    char_targets = []
    for item in raw_text_iter:
        seqs, targets = create_char_seqs(item, context_length)
        if seqs:
            char_seqs.extend(seqs)
            char_targets.extend(targets)

    print(f'created {len(char_seqs)} examples')

    example_store = featurize_char_seqs(char_seqs, 
                                        char_targets, 
                                        num_max_examples, 
                                        stoe, stoi)
    # shuffle the examples
    random.shuffle(example_store)
    return example_store    


def main():
    parser = argparse.ArgumentParser(description='Train VW model')
    parser.add_argument('--output-dir', '-o', type=str,
                        default=r'~/dataroot/vw_dataset',
                        help='full path to output directory')
    parser.add_argument('--vocab-embeddings-file', '-e', type=str,
                        help='full path to character level vocab embeddings')
    parser.add_argument('--context-length', '-l', type=int,
                        help='number of last character embeddings to use as features')
    parser.add_argument('--max-examples', '-m', type=int,
                        help='maximum number of examples to generate in vw format')
    parser.add_argument('--num-passes', '-p', type=int,
                        help='number of passes over the dataset')
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
    example_store = create_vw_examples(train_iter, 
                        args.context_length, 
                        stoe, 
                        stoi, 
                        num_max_examples=args.max_examples)

    vw = pyvw.vw('--oaa 131 -b 24 -f my_best_model')
    for p in range(args.num_passes):
        for ex in example_store:
            # TODO: create example batch on the fly
            vw_ex = vw.example(ex)
            vw.learn(vw_ex)
            #print(f'current prediction = {vw_ex.get_multiclass_prediction()}')
            vw.finish_example(vw_ex)

    
    # Test how to do prediction
    test_ex_splits = [feat for feat in example_store[0].split('|')]
    test_gt = test_ex_splits[0]
    test_ex = '|' + '|'.join(test_ex_splits[1:])
    prediction = vw.predict(vw.example(test_ex))
    print(f'Test gt: {test_gt}, predicted: {prediction}')

    #vw.finish()
    # # Test how to load from file and do prediction
    # vw = pyvw.vw('-i my_best_model')
    # prediction = vw.predict(vw.example(test_ex))
    # print(f'Test gt: {test_gt}, predicted: {prediction}')
    # vw.finish()

    # create validation examples
    val_store = create_vw_examples(val_iter, 
                        args.context_length, 
                        stoe, 
                        stoi, 
                        num_max_examples=args.max_examples)

    cum_acc = 0
    ex_counter = 0
    for ex in val_store:
        val_ex_splits = [feat for feat in ex.split('|')]
        val_gt = val_ex_splits[0]
        val_ex = '|' + '|'.join(val_ex_splits[1:])
        prediction = vw.predict(vw.example(val_ex))
        if prediction == int(val_gt):
            cum_acc += 1
        ex_counter += 1
    
    print(f'Accuracy over {ex_counter} val examples: {cum_acc/ex_counter}')

    
    vw.finish()

                    
    



    

if __name__ == '__main__':
    main()


