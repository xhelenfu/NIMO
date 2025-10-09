import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import os
import glob
import torchvision
import random
import scipy

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
import torch.nn.functional as F

from .utils import *


class DataProcessing(data.Dataset):
    def __init__(
        self,
        opts_data_sources,
        opts_data_sources_predict,
        opts_data,
        sample_size,
        fold_id=1,
        mode="train",
        seed=1000,
        block=1,
        patch_centres=None,
    ):

        self.classes = opts_data.classes
        self.fold_id = fold_id
        self.sample_size = sample_size
        self.mode = mode
        self.seed = seed
        self.block = block

        check_path(opts_data_sources.fp_patient_labels)
        check_path(opts_data_sources.dir_expr)
        check_path(opts_data_sources.dir_centroids)
        check_path(opts_data_sources.dir_splits)
        check_path(opts_data_sources.fp_genes)

        self.genes = read_txt(opts_data_sources.fp_genes)
        self.n_genes = len(self.genes)

        if self.mode != "predict":
            # int
            self.split_ids = get_splits(
                opts_data_sources.dir_splits, self.fold_id, self.mode
            )
            assert len(self.split_ids) == len(set(self.split_ids))

            fps_expr = glob.glob(opts_data_sources.dir_expr + "/*.csv")

            self.fps_expr = [
                x
                for x in fps_expr
                if int(os.path.basename(x).split("_")[0]) in self.split_ids
            ]

            df_tma_info = pd.read_csv(opts_data_sources.fp_patient_labels)
            self.df_tma_info = df_tma_info[df_tma_info["patient_id"].isin(self.split_ids)]

            self.fps_centroids = [
                x.replace(opts_data_sources.dir_expr, opts_data_sources.dir_centroids)
                for x in self.fps_expr
            ]

            self.fps_edges = [
                x.replace(opts_data_sources.dir_expr, opts_data_sources.dir_edges)
                for x in self.fps_expr
            ]

        else:
            fps_expr = glob.glob(opts_data_sources_predict.dir_expr + "/*.csv")[:3]
            self.fps_expr = fps_expr.copy()

            self.split_ids = [int(os.path.basename(x).split("_")[0]) for x in self.fps_expr]

            df_tma_info = pd.read_csv(opts_data_sources_predict.fp_patient_labels)
            self.df_tma_info = df_tma_info[df_tma_info["patient_id"].isin(self.split_ids)]

            self.fps_centroids = [
                x.replace(opts_data_sources_predict.dir_expr, opts_data_sources_predict.dir_centroids)
                for x in self.fps_expr
            ]

            self.fps_edges = [
                x.replace(opts_data_sources_predict.dir_expr, opts_data_sources_predict.dir_edges)
                for x in self.fps_expr
            ]

    def get_delaunay_edges(self, points, max_distance=40):
        # Perform Delaunay triangulation
        delaunay = scipy.spatial.Delaunay(points)

        edges = set()  # Use a set to avoid duplicate edges (since edges are bidirectional)s")

        # Plot the Delaunay edges (these are edges between points)
        for simplex in delaunay.simplices:
            for i in range(3):  # There are 3 edges in a triangle
                p1 = points[simplex[i]]
                p2 = points[simplex[(i + 1) % 3]]

                distance = np.linalg.norm(p1 - p2)

                if distance <= max_distance:  # Check if the distance is within the threshold

                    index1 = simplex[i]
                    index2 = simplex[(i + 1) % 3]
                    # Ensure the edges are stored in sorted order to avoid duplicates
                    edges.add(tuple(sorted([index1, index2])))

        edges = list(edges)

        n_neighbs = len(points)

        edges_vt_from = [x[0] for x in edges]  # First element of each tuple
        edges_vt_to = [x[1] for x in edges]  # Second element of each tuple

        nodes_all = np.arange(n_neighbs)

        edge_from = np.vstack((edges_vt_from, edges_vt_to))
        edge_to = np.vstack((edges_vt_to, edges_vt_from))
        edges_self = np.vstack((nodes_all, nodes_all))

        edges_vt = np.hstack((edge_from, edge_to))
        edges_vt = np.hstack((edges_vt, edges_self))

        return edges_vt

    def sample_patches(
        self, df_centroids_core, df_edges_core, df_expr_core
    ):

        edges = None

        # sample centre cells
        cells_with_edges = df_edges_core.index.unique().tolist()

        random.seed(self.seed)

        # sample in blocks of sample_size
        random.shuffle(cells_with_edges)
        block_start = self.block * self.sample_size
        block_end = block_start + self.sample_size
        if len(cells_with_edges) < block_end:
            block_end = len(cells_with_edges)
        cells_sampled = cells_with_edges[block_start:block_end]

        exprr_centre = None
        exprr_neighb = None
        exprr_summed = None

        n_nodes = []
        n_neighbs = []

        cell_ids_all = []
        cell_ids_neighb = []

        centre_xy = []

        # concat the relative expr matrices channelwise
        for ic, index in enumerate(cells_sampled):
            # print("getting cell", index)
            cell_ids_all.append(index)

            centroid_x = df_centroids_core.loc[index, "x_centroid"]
            centroid_y = df_centroids_core.loc[index, "y_centroid"]
            centre_xy.append([centroid_x, centroid_y])

            row = df_expr_core.loc[index, :]
            values = row.to_numpy()

            values = np.expand_dims(values, 0) + 1
            if exprr_centre is None:
                exprr_centre = values.copy()
            else:
                exprr_centre = np.concatenate((exprr_centre, values), 0)

            df_edges_cell = df_edges_core.loc[index, :]

            if len(df_edges_cell.shape) == 1:
                n_neighb = 1
            else:
                n_neighb = len(df_edges_cell["to_cell_id_code"])

            n_nodes.append(n_neighb + 1)
            n_neighbs.append(n_neighb)

            if n_neighb == 1:
                to_cell_list = [df_edges_cell["to_cell_id_code"]]
            else:
                to_cell_list = df_edges_cell["to_cell_id_code"].tolist()

            cell_ids_neighb.extend(to_cell_list)

            # extract the values for all 'to_cell' indices at once
            to_values = df_expr_core.loc[to_cell_list, :].to_numpy()  # Shape: (len(to_cell_list), num_columns)

            # add 1 to all to_values
            to_values += 1

            # summed expr for neighbourhood
            summed_values = np.vstack((values, to_values)) - 1
            summed_values = np.sum(summed_values, 0)
            summed_values = np.expand_dims(summed_values, 0)
            if exprr_summed is None:
                exprr_summed = summed_values.copy()
            else:
                exprr_summed = np.concatenate((exprr_summed, summed_values), 0)

            # use broadcasting to divide by 'values' and handle division by zero
            # Assuming 'values' has a shape that can be broadcasted with 'to_values'
            to_values = np.divide(
                to_values,
                values,
                where=values != 0,
                out=np.zeros_like(to_values, dtype=float),
            )

            # combine the results into the final array (exprr_neighb)
            if len(to_values.shape) < 2:
                to_values = np.expand_dims(to_values, 0)

            if exprr_neighb is None:
                exprr_neighb = to_values.copy()
            else:
                exprr_neighb = np.concatenate((exprr_neighb, to_values), 0)

        # 'batchifying' edges for the graph Data
        # star shaped to and from central node - per neighbourhood
        for i_n, n_nodes_current in enumerate(n_nodes):

            # n_nodes_current includes current central node
            central_node_val = sum(n_nodes[:i_n])
            next_central_node_val = central_node_val + n_nodes_current

            # central node
            nodes_central = np.ones(n_nodes_current - 1) * central_node_val
            nodes_neighb = np.arange(central_node_val + 1, next_central_node_val)
            nodes_neighb_all = np.arange(central_node_val, next_central_node_val)

            edge_from = np.vstack((nodes_neighb, nodes_central))
            edge_to = np.vstack((nodes_central, nodes_neighb))
            edges_self = np.vstack((nodes_neighb_all, nodes_neighb_all))

            edges_neighb = np.hstack((edge_from, edge_to))
            edges_neighb = np.hstack((edges_neighb, edges_self))

            if edges is None:
                edges = edges_neighb.copy()
            else:
                edges = np.hstack((edges, edges_neighb))

        return edges, n_nodes, n_neighbs, cell_ids_all, cell_ids_neighb, exprr_centre, exprr_neighb, exprr_summed, centre_xy

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.fps_expr)

    def __getitem__(self, index):
        "Generates one sample of data"

        fp_expr = self.fps_expr[index]

        fname = os.path.basename(fp_expr)
        # print(fname)

        fp_centroids = [x for x in self.fps_centroids if os.path.basename(x) == fname][
            0
        ]
        fp_edges = [x for x in self.fps_edges if os.path.basename(x) == fname][0]

        patient_id = int(os.path.basename(fp_expr).split("_")[0])

        response = self.df_tma_info.loc[
            self.df_tma_info["patient_id"] == patient_id, "response"
        ].values[0]
        response_idx = self.classes.index(response)
        response_idx = torch.from_numpy(np.array([response_idx])).long()

        df_expr = pd.read_csv(fp_expr, index_col="cell_id")
        df_expr = df_expr[self.genes]
        df_expr = df_expr.loc[df_expr.sum(axis=1) != 0]
        valid_cells = df_expr.index.tolist()

        # to check the idx of cell_idx
        df_centroids = pd.read_csv(fp_centroids, index_col="cell_id")
        df_centroids = df_centroids.loc[valid_cells, :]

        # edges
        df_edges = pd.read_csv(fp_edges, index_col="from_cell_id_code")
        df_edges = df_edges[df_edges.index.isin(valid_cells)]
        df_edges = df_edges[df_edges["to_cell_id_code"].isin(valid_cells)]

        cells_with_edges = df_edges.index.unique().tolist()
        block_start = self.block * self.sample_size

        if block_start < len(cells_with_edges):
            try:
                edges, n_nodes, n_neighbs, cell_ids_all, cell_ids_neighb, exprr_centre, exprr_neighb, exprr_summed, centre_xy = self.sample_patches(
                    df_centroids, df_edges, df_expr
                )

                centre_xy = np.array(centre_xy)
                edges_vt = self.get_delaunay_edges(centre_xy)

                exprr_centre = torch.from_numpy(exprr_centre).float()
                exprr_neighb = torch.from_numpy(exprr_neighb).float()
                exprr_summed = torch.from_numpy(exprr_summed).float()
                edges = torch.from_numpy(edges).long()
                edges = edges.type(torch.int64)
                n_nodes = torch.from_numpy(np.array(n_nodes)).long()
                n_neighbs = torch.from_numpy(np.array(n_neighbs)).long()
                edges_vt = torch.from_numpy(edges_vt).long()
                edges_vt = edges_vt.type(torch.int64)

                valid_sampling = True
            
            except:
                valid_sampling = False

        # sampling hit limit (cannot exceed number of cells in the core)
        else:
            # dummy variables
            centre_xy = np.zeros((10, 2))
            edges_vt = np.zeros((20, 2), dtype=np.int64)
            exprr_centre = torch.zeros((10, 16))
            exprr_neighb = torch.zeros((10, 16))  # same shape
            exprr_summed = torch.zeros((10, 16))  # same shape
            edges = torch.zeros((20, 2), dtype=torch.int64)
            n_nodes = torch.tensor(10, dtype=torch.long)
            n_neighbs = torch.tensor(5, dtype=torch.long)
            edges_vt = torch.from_numpy(edges_vt).type(torch.int64)
            cell_ids_all = np.zeros((10, 1))
            cell_ids_neighb = np.zeros((10, 1))

            valid_sampling = False

        return (
            fname,
            exprr_centre,
            exprr_neighb,
            exprr_summed,
            edges,
            n_nodes,
            n_neighbs,
            response_idx,
            fp_expr,
            cell_ids_all,
            cell_ids_neighb,
            edges_vt,
            valid_sampling
        )
