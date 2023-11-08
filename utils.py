import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Generator, List
from pathlib import Path
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import SequenceSummary
from torch import nn
import torch

def set_rand_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_gpt2_tok(train_corpus: Generator[str, None, None],
                   vocab_size: int,
                   special_tokens: List[str],
                   output_dir: Path,
                  ):    
    old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    new_tok = old_tokenizer.train_new_from_iterator(train_corpus,
                                                    vocab_size, 
                                                    length=len(list(train_corpus)),
                                                    new_special_tokens=special_tokens
                                                   )
    new_tok.save_pretrained(output_dir)
        
def load_pkl(file: Path):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    
    return data

def ints_to_binary_matrix(score_seq):  # converts integer sequence to n x 62 matrix
    matrix = []
    for event in score_seq:
        binary_rep = list(np.binary_repr(event, 62))
        matrix.append(binary_rep)
    np_mat = np.array(matrix, dtype=np.uint8)
    #np_mat = np.flip(np_mat, axis=0)  # flip to have least significant bit at the front
    return np_mat

def visualize_bootleg_score(bs, lines = [13, 15, 17, 19, 21, 25, 27, 29, 31, 33]):
    if bs.shape[0] == 62:
        lines = [13, 15, 17, 19, 21, 35, 37, 39, 41, 43]
    plt.figure(figsize = (10,10))
    plt.imshow(1 -bs, cmap = 'gray', origin = 'lower')
    # for l in range(1, bs.shape[0], 2):
    #     plt.axhline(l, c = 'grey')
    for l in lines:
        plt.axhline(l, c = 'black')

def to_binary_mat(str, size):
    events = str.split(" ")
    events = [[ord(c) - 10060 for c in event] for event in events]
    events = [[format(i, f'0{size}b')[::-1] for i in event] for event in events]
    events = ["".join(event) for event in events]
    events = [[int(c) for c in event] for event in events]
    events = np.array(events)
    return events
    
# def morsify(file: str) -> np.array:
#      piece = load_pkl(file)
#      morsified_piece = np.array([]).reshape(-1, 62)
#      for page in piece:
#          bin_mat = ints_to_binary_matrix(page).reshape(-1, 62)
#          bin_mat += 45
#          morsified_piece = np.concatenate([morsified_piece, bin_mat.view('U2')])
#      return morsified_piece

def morsify(bscore):
    bscore = bscore.reshape((-1, 62))
    output = (bscore+45).astype(np.int64).view('U1')
    #output = [[chr(elem) for elem in row] for row in bscore]
    output = [''.join(row) for row in output]
    output = '\n'.join(output)
    if len(output)>1 and output[-1] != '\n':
        output += '\n'
    return output 

def merge_staff_overlaps(bscores):
    """
    Takes in either one binary score or a batch of them and merges the left and right hands
    """
    
    # Lower middle c is index 23
    # Upper middle c is index 33
    lower = 23
    upper = 33
    middle = (lower + upper) // 2
    
    # Total notes is 52
    total = 52
    
    # Pad out upper hand and lower hand and combine them
    padded_lower = np.concatenate([bscores[..., :middle], np.zeros((*bscores.shape[:-1], total-middle))], axis=-1)
    padded_upper = np.concatenate([np.zeros((*bscores.shape[:-1], middle-bscores.shape[-1]+total)), bscores[..., middle:]], axis=-1)
    # Logical or
    merged = padded_lower + padded_upper - padded_lower * padded_upper
    return merged

class GPT2Classifier(nn.Module):
    def __init__(self, transformer_model:PreTrainedModel, config:PretrainedConfig, pad_idx:int, cls_idx:int):
        super(GPT2Classifier,self).__init__()
        self.transformer = transformer_model
        self.head = SequenceSummary(config)
        self.pad_idx = pad_idx
        self.cls_idx = cls_idx

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
    ):

        transformer_outputs = self.transformer(
            input_ids,
            # past_key_values=past,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
        )

        # mc_token_ids = (input_ids==self.cls_idx).nonzero()[:,1]
        # assert mc_token_ids.shape[0] == input_ids.shape[0]
        hidden_states = transformer_outputs[0]
        out = self.head(hidden_states).squeeze(-1)
        return out
