import sys
sys.path.append('../../source')
from utils import merge_staff_overlaps
import numpy as np

# DEFINE THE ENCODERS TO USE FOR SPARSE ENCODINGS



# Character Encoder
characters = [chr(c) for c in np.arange(65, 91)] + [chr(c) for c in np.arange(97, 123)]

def char_encoder(fragment):
    fragment = merge_staff_overlaps(fragment)
    notes = ["".join([characters[i] for i, bit in enumerate(note) if bit==1]) for note in fragment]
    string = " | ".join(notes)
    return string



# Chroma Octave Encoder

# middle C (C4) is index 23
octaves = 9

note_numbers = np.array([[i]*7 for i in range(octaves)]).flatten()
note_letters = ["A", "B", "C", "D", "E", "F", "G"] * octaves

note_names = [letter + str(number) for number, letter in zip(note_numbers, note_letters)]

note_names = note_names[note_names.index("C4")-23:note_names.index("C4")+52-23]

def chroma_octave_encoder(fragment):
    fragment = merge_staff_overlaps(fragment)
    notes = ["-".join([note_names[i] for i, bit in enumerate(note) if bit==1]) for note in fragment]
    string = " | ".join(notes)
    return string



# Chroma Octave Base + Interval Encoder
def interval_encoder(fragment):
    fragment = merge_staff_overlaps(fragment)
    
    notes = []
    
    for note in fragment:
        note_numbers = [i for i, bit in enumerate(note) if bit==1]
        if len(note_numbers) != 0:
            base = min(note_numbers)
            note = "-".join([note_names[base]] + [str(num-base) for num in note_numbers if num != base])
            notes.append(note)
        else:
            notes.append("")
            
    string = " | ".join(notes)
    return string



# Dense Encoder

# Continuous line of 256 unicode characters
start = 10060# 931
dense_characters = [chr(i).encode("utf-8").decode("utf-8") for i in range(start, start+512)]


# This code divides the fragment into blocks (and discards any remaining info at the very edges)
# Then it uses einsum with a filter of powers of 2 to convert from binary to an integer.  Then converts integers into
# unicode characters

def dense_encoder(fragment, block_size=[1, 1]):
    fragment = merge_staff_overlaps(fragment)
    # Rewrote this to be much faster but looks complicated
    # This filter has powers of 2 which is how the binary is turned to ints
    filter_ = np.power(2, np.arange(np.prod(block_size))).reshape(block_size)
    
    # The fragment is split into blocks here
    xblocks = np.stack(np.split(fragment[:, :(fragment.shape[1]//block_size[1])*block_size[1]], fragment.shape[1]//block_size[1], axis=1))
    xyblocks = np.stack(np.split(xblocks[:, :(xblocks.shape[1]//block_size[0])*block_size[0]], xblocks.shape[1]//block_size[0], axis=1))
    
    # The blocks are multiplied so they are ints
    numbers = np.einsum("ijkl,kl->ij", xyblocks, filter_)
    
    # The ints are turned into corresponding characters
    characters = (numbers+start).astype(np.int32).view('U1')
    return " ".join(["".join(t) for t in characters])


sparse_encoders = {
    "char_encoder":char_encoder,
    "chroma_octave_encoder":chroma_octave_encoder,
    "interval_encoder":interval_encoder,
}