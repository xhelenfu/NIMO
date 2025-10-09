import os
import datetime as dt
import json
import collections
import re
import torch
import natsort
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


def get_device(gpu_ids):
    if type(gpu_ids) is list:
        gpu_str = ",".join(str(x) for x in gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        print("Using GPUs: {}".format(gpu_str))
        device = torch.device("cuda")
    else:
        device = torch.device("cuda")

    return device


def sorted_alphanumeric(data):
    """
    Alphanumerically sort a list
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def read_txt(fp):
    with open(fp) as file:
        lines = [line.rstrip() for line in file]
    return lines


def json_file_to_pyobj(filename):
    """
    Read json config file
    """

    def _json_object_hook(d):
        return collections.namedtuple("X", d.keys())(*d.values())

    def json2obj(data):
        return json.loads(data, object_hook=_json_object_hook)

    return json2obj(open(filename).read())


def get_newest_id(exp_dir="experiments", fold_id=1):
    """Get the latest experiment ID based on its timestamp

    Parameters
    ----------
    exp_dir : str, optional
        Name of the directory that contains all the experiment directories, by default 'experiments'

    Returns
    -------
    exp_id : str
        Name of the latest experiment directory
    """
    folders = next(os.walk(exp_dir))[1]
    folders = natsort.natsorted(folders)
    # folders = [x for x in folders if mode in x]
    folders = [x for x in folders if ("fold" + str(fold_id) + "_") in x]
    folder_last = folders[-1]
    exp_id = folder_last.replace("\\", "/")
    return exp_id


def get_experiment_id(make_new, load_dir, fold_id):
    """
    Get timestamp ID of current experiment
    """
    if make_new is False:
        if load_dir == "last":
            timestamp = get_newest_id("experiments", fold_id)
        else:
            timestamp = load_dir
    else:
        timestamp = (
            "fold"
            + str(fold_id)
            + "_"
            + dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        )

    return timestamp


def compute_metrics_per_patient(fp_txt):
    df = pd.read_csv(fp_txt, sep="\t", header=None, skipfooter=2, engine='python')
    df.columns = ['core_id', 'true', 'pred_binary', 'pred']

    # Extract patient ID
    df['patient'] = df['core_id'].str.split('_').str[0]

    # Aggregate predictions and truths per patient
    agg = df.groupby('patient').agg({'true': 'mean', 'pred': 'mean'})

    # Convert to binary labels
    agg['true_binary'] = (agg['true'] >= 0.5).astype(int)
    agg['pred_binary'] = (agg['pred'] >= 0.5).astype(int)

    # Compute metrics
    auc_per_pat = roc_auc_score(agg['true_binary'], agg['pred'])
    acc_per_pat = accuracy_score(agg['true_binary'], agg['pred_binary'])
    f1_per_pat = f1_score(agg['true_binary'], agg['pred_binary'])

    return auc_per_pat, acc_per_pat, f1_per_pat