import argparse
import logging
import os
import sys
import natsort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob

from dataio.dataset_input import DataProcessing
from dataio.utils import get_splits
from model.model import Framework
from model.gated_attn_mil import GMA
from utils.utils import *

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
from scipy.stats import rankdata


def main(config):

    config_file = config.config_file

    opts = json_file_to_pyobj(config_file)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    device = get_device(config.gpu_id)

    # Create experiment directories
    make_new = False
    timestamp = get_experiment_id(
        make_new, opts.experiment_dirs.load_dir, config.fold_id
    )
    experiment_path = f"experiments/{timestamp}"
    model_dir = experiment_path + "/" + opts.experiment_dirs.model_dir
    if config.mode == "predict":
        predict_output_dir = (
            experiment_path + "/" + opts.experiment_dirs.predict_output_dir
        )
    else:
        predict_output_dir = experiment_path + "/" + opts.experiment_dirs.val_output_dir
    os.makedirs(predict_output_dir, exist_ok=True)

    # Set up the model
    logging.info("Initialising model")

    classes = opts.data.classes
    n_classes = len(classes)
    print("Classes:", classes)

    genes = read_txt(opts.data_sources.fp_genes)
    n_genes = len(genes)
    print("Num genes:", n_genes)

    model = Framework(
        n_classes,
        n_genes,
        opts.model.emb_dim,
        opts.model.n_heads_gat,
        device,
    )

    model_gma = GMA(
        dropout=False,
        n_classes=len(opts.data.classes),
        ndim=opts.model.emb_dim * 4,
    )

    # Dataloader
    logging.info("Preparing data")

    # Get list of model files
    if config.epoch in ("all", "last"):
        saved_model_paths = glob.glob(f"{model_dir}/epoch_*.pth")
        saved_model_paths = sorted_alphanumeric(saved_model_paths)
        saved_model_names = [
            os.path.splitext(os.path.basename(x))[0] for x in saved_model_paths
        ]
        saved_model_epochs = [name.split("_")[1] for name in saved_model_names]
        saved_model_epochs = sorted_alphanumeric(list(set(saved_model_epochs)))

        if config.epoch == "all":
            saved_model_epochs = np.array(saved_model_epochs, dtype=int)
        elif config.epoch == "last":
            last_epoch = int(saved_model_epochs[-1])
            saved_model_epochs = [last_epoch]
    else:
        saved_model_epochs = [int(config.epoch)]

    # Dataloader
    logging.info("Preparing data")

    test_dataset = DataProcessing(
        opts.data_sources,
        opts.data_sources_predict,
        opts.data,
        opts.data.sample_size_test,
        config.fold_id,
        mode=config.mode,
        seed=config.seed,
        block=config.sample_block,
    )

    dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=opts.training.batch_size,
        shuffle=False,
        num_workers=opts.data.num_workers,
        drop_last=False,
    )

    n_test_examples = len(dataloader)
    logging.info("Total number of patches: %d" % n_test_examples)

    logging.info("Begin prediction")

    all_epochs_f1 = []
    all_epochs_acc = []
    all_epochs_auc = []

    with torch.no_grad():

        for epoch_idx, test_epoch in enumerate(saved_model_epochs):
            fp_out_prefix = (
                f"{predict_output_dir}/epoch_{test_epoch}_{config.sample_block}/outputs"
            )
            os.makedirs(fp_out_prefix, exist_ok=True)

            load_path = model_dir + "/epoch_%d_model.pth" % (test_epoch)
            checkpoint = torch.load(load_path, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint["epoch"]
            print("Predict using " + load_path)

            load_path = model_dir + "/epoch_%d_model_gma.pth" % (test_epoch)
            checkpoint = torch.load(load_path, weights_only=True)
            model_gma.load_state_dict(checkpoint["model_state_dict"])
            print("Predict using " + load_path)

            model.to(device)
            model_gma.to(device)

            model = model.eval()

            all_gt = []
            all_pr = []
            all_pr_prob = []
            all_fnames_unique = []

            pbar = tqdm(dataloader)

            for (
                fname,
                x_exprr_centre,  # centre nodes raw counts
                x_exprr_neighb,  # neighb nodes norm counts
                x_exprr_summed,  # sum of raw counts
                x_edges,
                x_n_nodes,
                x_n_neighbs,
                y_response_idx,
                _,
                cell_ids_all,
                cell_ids_neighb,
                x_edges_vt,
                valid_sampling,
            ) in pbar:

                valid_sampling = valid_sampling.item()

                if valid_sampling:

                    x_exprr_centre = x_exprr_centre.to(device)[0]
                    x_exprr_neighb = x_exprr_neighb.to(device)[0]
                    x_exprr_summed = x_exprr_summed.to(device)[0]
                    x_edges = x_edges.to(device)[0]
                    x_n_nodes = x_n_nodes.to(device)[0]
                    x_n_neighbs = x_n_neighbs.to(device)[0]
                    y_response_idx = y_response_idx.to(device)[0]
                    x_edges_vt = x_edges_vt.to(device)[0]

                    (
                        x_neighbs,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                    ) = model(
                        x_exprr_centre,
                        x_edges,
                        x_exprr_neighb,
                        x_n_nodes,
                        x_n_neighbs,
                        cell_ids_all,
                        cell_ids_neighb,
                        x_edges_vt,
                    )

                    A_raw, sign, output, _ = model_gma(x_neighbs)

                    fname = fname[0].replace(".csv", "")
                    if fname not in all_fnames_unique:
                        all_fnames_unique.append(fname)

                    cell_ids_all = [item[0] for item in cell_ids_all]

                    all_gt.append(y_response_idx[0].detach().cpu().numpy())
                    output = F.softmax(output, dim=1)
                    all_pr_prob.append(output[0, -1].detach().cpu().numpy())
                    pred = torch.argmax(output, dim=1)
                    pred = pred.detach().cpu().numpy()
                    all_pr.append(pred)

                    # save the embeddings of the cell, cell_id, A_raw, sign
                    if config.save_outputs:
                        df_gma = pd.DataFrame(
                            index=cell_ids_all, columns=["A_raw", "sign"]
                        )
                        df_gma.loc[:, "A_raw"] = A_raw.copy()
                        df_gma.loc[:, "sign"] = sign.copy()

                        # file paths
                        fp_attn = f"{fp_out_prefix}/{fname}_{opts.predict_outs.fp_attn}"
                        fp_embeddings = (
                            f"{fp_out_prefix}/{fname}_{opts.predict_outs.fp_embeddings}"
                        )

                        # save
                        df_gma.to_csv(fp_attn)
                        torch.save(x_neighbs, fp_embeddings)

            # have looped through all cores
            cores_gt = all_gt.copy()
            cores_pr_prob_cls = all_pr.copy()
            cores_pr_prob = all_pr_prob.copy()

            # performance metrics
            f1 = f1_score(cores_gt, cores_pr_prob_cls, average=None)
            f1_mean = np.mean(f1)
            all_epochs_f1.append(f1_mean)
            all_epochs_acc.append(accuracy_score(cores_gt, cores_pr_prob_cls))
            all_epochs_auc.append(roc_auc_score(cores_gt, cores_pr_prob))

            print(
                "Epoch[{}], ACC:{:.4f}, F1:{:.4f}, AUC:{:.4f}".format(
                    epoch, all_epochs_acc[-1], all_epochs_f1[-1], all_epochs_auc[-1]
                )
            )

            # save predictions to txt
            fp_txt = f"{fp_out_prefix.replace('/outputs','')}/predictions.txt"

            assert len(all_fnames_unique) == len(cores_gt)
            assert len(all_fnames_unique) == len(cores_pr_prob_cls)

            with open(fp_txt, "w") as output_file:

                # per core
                for fi in range(len(cores_gt)):
                    output_file.write(
                        f"{all_fnames_unique[fi]}\t{cores_gt[fi]}\t{cores_pr_prob_cls[fi]}\t{cores_pr_prob[fi]}\n"
                    )

                output_file.write("Overall F1, ACC, and AUC - across samples and patients\n")
                output_file.write(
                    f"{np.around(all_epochs_f1[-1], 5)},{np.around(all_epochs_acc[-1], 5)},{np.around(all_epochs_auc[-1], 5)}\n"
                )

            auc_per_pat, acc_per_pat, f1_per_pat = compute_metrics_per_patient(fp_txt)
            with open(fp_txt, "a") as output_file:
                output_file.write(
                    f"{np.around(f1_per_pat, 5)},{np.around(acc_per_pat, 5)},{np.around(auc_per_pat, 5)}\n"
                )

    # best epoch, better performance should have lower ranks
    print("***best epoch***")
    rank_auc = rankdata(-np.array(all_epochs_auc), method="min")
    ranksums = rank_auc
    best_idx = np.argmin(ranksums)
    best_epoch = saved_model_epochs[best_idx]
    print(
        f"Best epoch {best_epoch}: F1 {all_epochs_f1[best_idx]}, ACC {all_epochs_acc[best_idx]}, AUC {all_epochs_auc[best_idx]}"
    )

    # write to file
    fp_be = f"{experiment_path}/best_epoch.txt"
    with open(fp_be, "w") as file:
        file.write(str(best_epoch))


logging.info("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        default="configs/config_demo.json",
        type=str,
        help="config file path",
    )
    parser.add_argument(
        "--epoch",
        type=str,
        default="all",
        help="Epoch to load: 'last' for final checkpoint, 'all' to run all, or a specific number",
    )
    parser.add_argument(
        "--mode",
        default="val",
        type=str,
        help="predict or val",
    )
    parser.add_argument(
        "--fold_id",
        default=1,
        type=int,
        help="which cross-validation fold",
    )
    parser.add_argument(
        "--seed",
        default=1000,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--sample_block",
        default=0,
        type=int,
        help="sample block of randomly shuffled cell list - 0 start",
    )
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int,
        help="which GPU to use",
    )
    parser.add_argument("--save_outputs", action=argparse.BooleanOptionalAction)

    config = parser.parse_args()
    main(config)
