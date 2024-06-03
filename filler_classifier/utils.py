import os
import os.path
import pickle
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from extract_bootleg import *
from utils import *


def convert_binary_to_int(X):
    mask = np.power(2, np.arange(X.shape[0])).reshape((1, -1))
    ints = np.squeeze(mask @ X).astype(np.uint64)
    return list(ints)


def convert_int_to_char_sequence(int64):
    chars = ""
    for i in range(8):
        numshift = i * 8
        charidx = (int64 >> numshift) & 255
        chars += chr(
            19968 + charidx
        )  # 19968 ensures that all chars are chinese characters (not newline, space, etc)
    return "".join(chars)


def convert_line_to_char_sequence(line):
    ints = [int(p) for p in line.split()]
    result = " ".join([convert_int_to_char_sequence(i) for i in ints])
    return result


def save_to_pickle(d, outfile):
    with open(outfile, "wb") as f:
        pickle.dump(d, f, protocol=4)


def visualize_bootleg(bs, lines=[13, 15, 17, 19, 21, 35, 37, 39, 41, 43]):
    plt.figure(dpi=1200)
    plt.axis("off")
    plt.imshow(1 - bs, cmap="gray", origin="lower")
    for l in lines:
        plt.axhline(l, c="black", lw=0.4)
    plt.show()


def load_pkl(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def ints_to_binary_matrix(score_seq):  # converts integer sequence to n x 62 matrix
    matrix = []
    for event in score_seq:
        binary_rep = list(np.binary_repr(event, 62))
        matrix.append(binary_rep)
    np_mat = np.array(matrix, dtype=np.uint8)
    # np_mat = np.flip(np_mat, axis=0)  # flip to have least significant bit at the front
    return np_mat


def PDF2PNG(pdffile, pngfile):
    try:
        if not os.path.exists(os.path.dirname(pngfile)):
            os.makedirs(os.path.dirname(pngfile))
        subprocess.call(
            [
                "magick",
                "-limit",
                "memory",
                "32GiB",
                "-limit",
                "map",
                "8GiB",
                "-limit",
                "disk",
                "16GiB",
                "-density",
                "300",
                pdffile,
                "-alpha",
                "remove",
                "-resize",
                "2550",
                pngfile,
            ]
        )
        return True
    except:
        return False


def hashfcn(array):
    # Encodes bootleg array to int to reduce memory
    hashNum = 0
    for i in array:
        hashNum = 2 * hashNum + i
    return int(hashNum)


def PNG2Bootleg(pngDir, errorfile):
    total_bscore = []
    sortedfiles = []

    # First, sort the pages in the right order
    for subdir, dirs, files in os.walk(pngDir):
        if len(files) == 1:
            sortedFiles = [files[0]]
        else:
            sortedFiles = sorted(files, key=lambda x: int(x.split("-")[-1][:-4]))
    for png in sortedFiles:
        page_bscore = np.array([]).reshape(62, 0)
        imagepath = os.path.join(subdir, png)
        try:
            bscore_query = processImageFile(imagepath, errorfile)
        except Exception as e:
            print(f"Process Image File Failed with {imagepath}: {e}")
            bscore_query = np.array([]).reshape(62, 0)

        try:
            page_bscore = np.concatenate((page_bscore, bscore_query), axis=1)
        except Exception as e:
            print(f"Concatenate Failed: {e}")

        hashArray = []
        if page_bscore.shape[0] == 0:
            pass
        elif page_bscore.shape[0] == 1:
            hashArray = [hashfcn(page_bscore)]
        else:
            for col in page_bscore.T:
                hashArray.append(hashfcn(col))
        total_bscore.append(hashArray)

    return total_bscore


def make_dir(file_path):
    directory = "/".join(file_path.split("/")[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)
