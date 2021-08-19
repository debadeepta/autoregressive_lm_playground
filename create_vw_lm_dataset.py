import math
import argparse
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
#from torchtext.datasets import WikiText2
from torchtext.datasets import WikiText103
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator




def main():
    parser = argparse.ArgumentParser(description='Report creator')
    parser.add_argument('--output-dir', '-o', type=str,
                        default=r'~/logdir/vw_dataset',
                        help='full path to output directory')
    parser.add_argument('--vocab-embeddings-file', '-e', type=str,
                        help='full path to character level vocab embeddings')
    parser.add_argument('--context-length', '-l', type=int,
                        help='number of last character embeddings to use as features')
    args, extra_args = parser.parse_known_args()

    # load the embeddings
    with open(args.vocab_embeddings_file, 'r') as f:
        embed_lines = f.readlines()
    
    



    # load the WikiText103 dataset
    train_iter = WikiText103(split='train')    


if __name__ == '__main__':
    main()


