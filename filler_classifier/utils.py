import numpy as np
import pickle
import matplotlib.pyplot as plt

def convert_binary_to_int(X):
    mask = np.power(2, np.arange(X.shape[0])).reshape((1,-1))
    ints = np.squeeze(mask @ X).astype(np.uint64)
    return list(ints)

def convert_int_to_char_sequence(int64):
    chars = ''
    for i in range(8):
        numshift = i * 8
        charidx = (int64 >> numshift) & 255
        chars += chr(19968 + charidx) # 19968 ensures that all chars are chinese characters (not newline, space, etc)
    return ''.join(chars)

def convert_line_to_char_sequence(line):
    ints = [int(p) for p in line.split()]
    result = ' '.join([convert_int_to_char_sequence(i) for i in ints])
    return result

def save_to_pickle(d, outfile):
    with open(outfile, 'wb') as f:
        pickle.dump(d, f, protocol=4)

def visualize_bootleg(bs, lines = [13, 15, 17, 19, 21, 35, 37, 39, 41, 43]):
    plt.figure(figsize = (10,10))
    plt.imshow(1 - bs, cmap = 'gray', origin = 'lower')
    for l in range(1, bs.shape[0], 2):
        plt.axhline(l, c = 'grey')
    for l in lines:
        plt.axhline(l, c = 'r')
    plt.show()

def load_pkl(file):
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
