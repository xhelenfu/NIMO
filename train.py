import argparse
import logging
import os
import sys
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import natsort
import glob
import random
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score

from dataio.dataset_input import DataProcessing
from dataio.utils import get_splits
from model.model import Framework
from model.gated_attn_mil import GMA
from utils.utils import *
from model.losses import *

random.seed(1000)


def main(config):
    opts = json_file_to_pyobj(config.config_file)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    device = get_device(opts.model.gpu_ids)

    # Create experiment directories
    if config.resume_epoch is not None:
        make_new = False
    else:
        make_new = True

    timestamp = get_experiment_id(
        make_new, opts.experiment_dirs.load_dir, config.fold_id
    )
    experiment_path = f"experiments/{timestamp}"
    os.makedirs(experiment_path + "/" + opts.experiment_dirs.model_dir, exist_ok=True)

    # Save copy of current config file
    shutil.copyfile(
        config.config_file, experiment_path + "/" + os.path.basename(config.config_file)
    )

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

    train_dataset = DataProcessing(
        opts.data_sources,
        opts.data_sources_predict,
        opts.data,
        opts.data.sample_size_train,
        config.fold_id,
        mode="train",
    )

    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=opts.training.batch_size,
        shuffle=True,
        num_workers=opts.data.num_workers,
        drop_last=True,
        pin_memory=True
    )

    n_train_examples = len(dataloader)
    logging.info("Total number of training batches: %d" % n_train_examples)

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opts.training.learning_rate,
        betas=(opts.training.beta1, opts.training.beta2),
        weight_decay=opts.training.weight_decay,
        eps=opts.training.eps,
    )

    global_step = 0

    # Starting epoch
    if config.resume_epoch is not None:
        initial_epoch = config.resume_epoch
    else:
        initial_epoch = 0

    # Restore saved model
    if config.resume_epoch is not None:
        logging.info("Resume training")

        load_path = (
            experiment_path
            + "/"
            + opts.experiment_dirs.model_dir
            + "/epoch_%d_model.pth" % (config.resume_epoch)
        )
        checkpoint = torch.load(load_path, weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
        print("Loaded " + load_path)

        model.to(device)

        load_path = (
            experiment_path
            + "/"
            + opts.experiment_dirs.model_dir
            + "/epoch_%d_model_gma.pth" % (config.resume_epoch)
        )
        checkpoint = torch.load(load_path, weights_only=True)

        model_gma.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]
        print("Loaded " + load_path)

        model_gma.to(device)

        load_path = (
            experiment_path
            + "/"
            + opts.experiment_dirs.model_dir
            + "/epoch_%d_optim.pth" % (config.resume_epoch)
        )
        checkpoint = torch.load(load_path, weights_only=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Loaded " + load_path)

    else:
        model.to(device)
        model_gma.to(device)

    logging.info("Begin training")

    # define losses
    loss_response = nn.CrossEntropyLoss(reduction="mean")
    loss_A_raw = ResistanceScoreLoss()

    for epoch in range(initial_epoch, opts.training.total_epochs):
        print(f"Epoch: {epoch+1}")
        model.train()

        optimizer.param_groups[0]["lr"] = opts.training.learning_rate * (
            1 - epoch / opts.training.total_epochs
        )

        loss_epoch = 0
        loss_epoch_response = 0
        loss_epoch_A_raw = 0

        all_gt = []
        all_pr = []

        pbar = tqdm(dataloader)

        loss_total = None

        for (
            _,
            x_exprr_centre,
            x_exprr_neighb,
            _,
            x_edges,
            x_n_nodes,
            x_n_neighbs,
            y_response_idx,
            _,
            cell_ids_all,
            cell_ids_neighb,
            x_edges_vt,
            _
        ) in pbar:
            optimizer.zero_grad()

            # orch.Size([1, 480, 480, 4])
            # torch.Size([1, 480, 480, 56])
            # torch.Size([1, 2, 172])
            # torch.Size([1, 4])
            # torch.Size([1, 4])
            # tensor([[1]])
            # print(x_exprr_centre.shape)
            # print(x_exprr_neighb.shape)
            # print(x_edges.shape)
            # print(x_n_nodes.shape)
            # print(x_n_neighbs.shape)
            # print(y_response_idx)
            # exit()

            x_exprr_centre = x_exprr_centre.to(device)[0]
            x_exprr_neighb = x_exprr_neighb.to(device)[0]
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
                _
            ) = model(
                x_exprr_centre,
                x_edges,
                x_exprr_neighb,
                x_n_nodes,
                x_n_neighbs,
                cell_ids_all,
                cell_ids_neighb,
                x_edges_vt
            )

            _, _, output, A_raw_tensor = model_gma(x_neighbs)

            # R -> negative A_raw, vice versa
            loss_A_raw_val = loss_A_raw(y_response_idx.repeat(opts.data.sample_size_train), A_raw_tensor)

            all_gt.append(y_response_idx[0].detach().cpu().numpy())
            pred = torch.argmax(output, dim=1)
            pred = pred.detach().cpu().numpy()
            all_pr.append(pred)

            loss_response_val = loss_response(output, torch.tensor([y_response_idx[0]]).to(device))

            loss = loss_response_val + loss_A_raw_val

            loss.backward()

            loss_total = loss.item()

            loss_epoch += loss.mean().item()

            loss_epoch_response += loss_response_val.mean().item()
            loss_epoch_A_raw += loss_A_raw_val.mean().item()

            pbar.set_description(f"loss: {loss_total:.4f}")

            optimizer.step()

        acc_epoch = accuracy_score(all_gt, all_pr)

        print(
            "Epoch[{}/{}], Loss:{:.4f}".format(
                epoch + 1, opts.training.total_epochs, loss_epoch
            )
        )
        print(
            "L_Response:{:.4f}, Acc:{:.4f}".format(
                loss_epoch_response, acc_epoch
            )
        )
        print(
            "L_A_raw:{:.4f}".format(
                loss_epoch_A_raw
            )
        )
        
        # Save model
        if (epoch % opts.save_freqs.model_freq) == 0:
            save_path = f"{experiment_path}/{opts.experiment_dirs.model_dir}/epoch_{epoch+1}_model.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                },
                save_path,
            )
            logging.info("Model saved: %s" % save_path)
            save_path = f"{experiment_path}/{opts.experiment_dirs.model_dir}/epoch_{epoch+1}_model_gma.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model_gma.state_dict(),
                },
                save_path,
            )
            logging.info("Model saved: %s" % save_path)
            save_path = f"{experiment_path}/{opts.experiment_dirs.model_dir}/epoch_{epoch+1}_optim.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Optimiser saved: %s" % save_path)

        global_step += 1

    logging.info("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        default="configs/config.json",
        type=str,
        help="config file path",
    )
    parser.add_argument(
        "--resume_epoch",
        default=None,
        type=int,
        help="resume training from this epoch, set to None for new training",
    )
    parser.add_argument(
        "--fold_id",
        default=1,
        type=int,
        help="which cross-validation fold",
    )

    config = parser.parse_args()
    main(config)
