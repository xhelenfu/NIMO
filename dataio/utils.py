import os
import sys
import json
import glob
import numpy as np
import random

random.seed(1000)


def check_path(d):
    if not os.path.exists(d):
        sys.exit("Invalid file path %s" % d)


def read_txt(fp):
    with open(fp) as file:
        lines = [line.rstrip() for line in file]
    return lines


def get_splits(dir, fold_id, mode):
    fps_splits = glob.glob(dir + "/*.json")
    fp_splits = [x for x in fps_splits if f"fold_{fold_id}" in x][0]

    with open(fp_splits, "r") as file:
        splits = json.load(file)

    if mode == "train":
        return splits["train_ids"]
    elif mode == "val":
        return splits["val_ids"]
