import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGPooling


class GATWithFeatureAttentionVT(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(GATWithFeatureAttentionVT, self).__init__()

        self.conv1 = GATConv(
            in_channels,
            out_channels,
            heads=4,
            dropout=0,
            add_self_loops=False,
        )

    def forward(self, x, edge_index, edge_attr=None):

        x, (alpha1_idx, alpha1) = self.conv1(
            x, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )

        return x, alpha1_idx, alpha1


class GATWithFeatureAttention(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_heads,
    ):
        super(GATWithFeatureAttention, self).__init__()

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=num_heads,
            dropout=0,
            add_self_loops=False,
        )
        self.conv2 = GATConv(
            hidden_channels * num_heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
        )

    def forward(self, x, edge_index, edge_attr=None):

        x, (alpha1_idx, alpha1) = self.conv1(
            x, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )
        x = F.elu(x)
        x, (alpha2_idx, alpha2) = self.conv2(
            x, edge_index, edge_attr=edge_attr, return_attention_weights=True
        )

        return x, alpha1_idx, alpha1, alpha2_idx, alpha2


class CrossAttentionCentre(nn.Module):
    def __init__(self, n_genes, hidden_dim):
        super(CrossAttentionCentre, self).__init__()
        # Learnable weights for Query, Key, and Value projections
        self.query_projection = nn.Linear(n_genes, hidden_dim)
        self.key_projection = nn.Linear(n_genes, hidden_dim)
        self.value_projection = nn.Linear(n_genes, hidden_dim)

        # Output projection to map the attended value back to gene space
        self.output_projection = nn.Linear(hidden_dim, n_genes)

    def forward(self, neighbours_genes, current_cell_genes):
        # Step 1: Compute Query, Key, and Value (using linear projections)
        Q = self.query_projection(current_cell_genes)  # (1, hidden_dim)
        K = self.key_projection(neighbours_genes)  # (n_neighbours, hidden_dim)
        V = self.value_projection(neighbours_genes)  # (n_neighbours, hidden_dim)

        # Step 2: Compute attention scores (scaled dot-product)
        attention_scores = torch.matmul(Q, K.T)  # (1, n_neighbours)

        # Scale the attention scores by the square root of the hidden dimension (not n_genes)
        scale_factor = torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float32))
        attention_scores = attention_scores / scale_factor

        # Step 3: Apply softmax to get the attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (1, n_neighbours)

        # Step 4: Compute the weighted sum of values (attention-weighted sum)
        weighted_sum = torch.matmul(attention_weights, V)  # (1, hidden_dim)

        # Step 5: Apply the output projection (map the result back to the original gene space)
        output = self.output_projection(weighted_sum)  # (1, n_genes)

        return output, attention_weights


class CrossAttentionNeighb(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super(CrossAttentionNeighb, self).__init__()

        self.d_model = d_model

        # Linear transformations for Q, K, V
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass for cross-attention with trainable parameters.

        Args:
        - Q: Query tensor (batch_size, num_queries, d_model)
        - K: Key tensor (batch_size, K, d_model)
        - V: Value tensor (batch_size, K, d_model)
        - mask: Optional mask tensor (batch_size, num_queries, K)

        Returns:
        - Output tensor after applying cross-attention (batch_size, num_queries, d_model)
        """
        # batch_size, num_queries, d_model = Q.size()
        # _, K, _ = K.size()

        Q = Q.unsqueeze(0)
        K = K.unsqueeze(0)
        V = V.unsqueeze(0)

        # Apply linear transformations to Q, K, and V
        Q = self.query_linear(Q)  # Shape: (batch_size, num_queries, d_model)
        K = self.key_linear(K)  # Shape: (batch_size, K, d_model)
        V = self.value_linear(V)  # Shape: (batch_size, K, d_model)

        K = K.repeat(Q.shape[1], 1).unsqueeze(0)

        # Compute Q * K^T for attention scores
        attn_scores = torch.bmm(
            Q, K.transpose(1, 2)
        )  # Shape: (batch_size, num_queries, K)

        if mask is not None:
            # Apply mask to attention scores
            attn_scores += mask

        # Scale the attention scores
        attn_scores = attn_scores / (self.d_model**0.5)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(
            attn_scores, dim=-1
        )  # Shape: (batch_size, num_queries, K)

        V = V.repeat(1, attn_weights.shape[1], 1)

        # Compute weighted sum of values
        output = torch.bmm(attn_weights, V)  # Shape: (batch_size, num_queries, d_model)

        return output
