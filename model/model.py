import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import *


class Framework(nn.Module):

    def __init__(
        self,
        n_classes,
        n_genes,
        emb_dim,
        n_heads_gat,
        device,
    ):
        super(Framework, self).__init__()

        self.n_genes = n_genes

        self.c_attn = CrossAttentionCentre(n_genes, emb_dim)
        self.n_attn = CrossAttentionNeighb(n_genes)

        self.gat_model = GATWithFeatureAttention(
            n_genes, emb_dim, emb_dim, n_heads_gat
        )

        self.gat_model_vt = GATWithFeatureAttentionVT(
            emb_dim, emb_dim
        )

        self.device = device

    def forward(
        self,
        exprr_centre_in,
        edges,
        exprr_neighb_in,
        n_nodes,
        n_neighbs,
        cell_ids_all,
        cell_ids_neighb,
        edges_vt
    ):

        exprr_centre_in_norm = exprr_centre_in.clone()
        exprr_neighb_in_norm = exprr_neighb_in.clone()
        # print(exprr_centre_in_norm.shape, exprr_neighb_in_norm.shape)
        # exit()

        # assign node features
        node_attr = None
        neighb_expr = torch.zeros((len(n_nodes), self.n_genes))
        neighb_expr_in = torch.zeros((len(n_nodes), self.n_genes))

        cell_ids_ordered = []
        cell_neighb_ordered = []

        exprr_centre_attn_all = None
        exprr_neighb_attn_all = None

        for idx, (n_nodes_i, n_neighbs_i) in enumerate(zip(n_nodes, n_neighbs)):
            start_idx = torch.sum(n_nodes[:idx])
            start_idx_neighb = torch.sum(n_neighbs[:idx]).item()

            current_c = exprr_centre_in_norm[idx, :]
            current_n = exprr_neighb_in_norm[
                start_idx_neighb : start_idx_neighb + n_neighbs_i, :
            ]

            # # centre attention - calculate attn per gene and make as relative matrix
            exprr_centre_attn, exprr_centre_attn_vals = self.c_attn(current_n, current_c)
            # exprr_centre_attn = F.softmax(exprr_centre_attn, dim=-1)
            exprr_centre_attn = torch.tanh(exprr_centre_attn)
            current_c_adj = torch.multiply(current_c, exprr_centre_attn)

            # # relative attention - calculate attn per neighbour gene (applied to sum over rows of matrix)
            exprr_neighb_attn = self.n_attn(current_n, current_c, current_c)
            exprr_neighb_attn = torch.tanh(exprr_neighb_attn)
            # print(exprr_neighb_attn.shape, exprr_centre_attn.shape)
            # exit()
            current_n_adj = torch.multiply(current_n, exprr_neighb_attn)
            current_n_adj = current_n_adj[0]

            exprr_centre_attn = exprr_centre_attn.unsqueeze(0)

            exprr_neighb_attn = torch.squeeze(exprr_neighb_attn)
            if len(exprr_neighb_attn.shape) < 2:
                exprr_neighb_attn = exprr_neighb_attn.unsqueeze(0)
            else:
                exprr_neighb_attn = exprr_neighb_attn[0, :].unsqueeze(0)

            # centre node and then the neighbours
            if node_attr is None:
                node_attr = current_c_adj.unsqueeze(0)
                node_attr = torch.cat(
                    (
                        node_attr,
                        current_n_adj,
                    ),
                    0,
                )
                exprr_centre_attn_all = exprr_centre_attn.clone()
                exprr_neighb_attn_all = exprr_neighb_attn.clone()

            else:
                node_attr = torch.cat(
                    (
                        node_attr,
                        current_c_adj.unsqueeze(0),
                    ),
                    0,
                )
                node_attr = torch.cat(
                    (
                        node_attr,
                        current_n_adj,
                    ),
                    0,
                )
                exprr_centre_attn_all = torch.cat(
                    (
                        exprr_centre_attn_all,
                        exprr_centre_attn,
                    ),
                    0,
                )
                exprr_neighb_attn_all = torch.cat(
                    (
                        exprr_neighb_attn_all,
                        exprr_neighb_attn,
                    ),
                    0,
                )

            # all cells including centre
            cell_ids_ordered.extend(cell_ids_all[idx])
            flattened_cell_ids = [item[0] if isinstance(item, tuple) else item for item in cell_ids_neighb[start_idx_neighb: start_idx_neighb + n_neighbs_i]]
            cell_ids_ordered.extend(flattened_cell_ids)

            # centre_neighb
            flattened_cell_ids = [f"{cell_ids_all[idx]}_{item}" for item in flattened_cell_ids]
            cell_neighb_ordered.extend(flattened_cell_ids)

            neighb_expr[idx, :] = (torch.sum(
                current_n_adj, 0
            )) / (n_neighbs_i)

            neighb_expr_in[idx, :] = (exprr_centre_in[idx, :] + torch.sum(
                exprr_neighb_in[start_idx_neighb : start_idx_neighb + n_neighbs_i, :], 0
            ))

        # perm contains the indices of the original nodes that were kept
        x, alpha1_idx, alpha1, alpha2_idx, alpha2 = self.gat_model(node_attr, edges)

        alpha1 = torch.mean(alpha1, 1)
        alpha2 = torch.squeeze(alpha2)

        alpha1_idx = alpha1_idx.detach().cpu().numpy()
        alpha1 = alpha1.detach().cpu().numpy()
        alpha2 = alpha2.detach().cpu().numpy()

        edges_weights = np.column_stack(
            (alpha1_idx[0, :], alpha1_idx[1, :], alpha1, alpha2)
        )

        x_neighbs = None
        exprr_neighb_attn_avg = None

        for idx, n_nodes_i in enumerate(n_nodes):
            start_idx = torch.sum(n_nodes[:idx]).item()
            end_idx = start_idx + n_nodes_i

            x_centre = x[start_idx, :].unsqueeze(0)
            exprr_neighb_attn_avg_node = torch.mean(
                exprr_neighb_attn[start_idx:end_idx, :], 0
            ).unsqueeze(0)
            if x_neighbs is None:
                x_neighbs = x_centre.clone()
                exprr_neighb_attn_avg = exprr_neighb_attn_avg_node.clone()
            else:
                x_neighbs = torch.cat((x_neighbs, x_centre), 0)
                exprr_neighb_attn_avg = torch.cat(
                    (exprr_neighb_attn_avg, exprr_neighb_attn_avg_node), 0
                )

        # global relation by voronoi tessellation
        x_neighbs, _, alpha1_vt = self.gat_model_vt(x_neighbs, edges_vt)

        alpha1_vt_avg = torch.mean(alpha1_vt, -1)

        # print(x_neighbs.shape)
        # print(edges_vt.shape)
        # print(alpha1_vt_avg.shape)
        # exit()

        return (
            x_neighbs,
            exprr_centre_attn_all,
            exprr_neighb_attn_all,
            neighb_expr,
            neighb_expr_in,
            edges_weights,
            cell_ids_ordered,
            cell_neighb_ordered,
            edges_vt,
            alpha1_vt_avg
        )
