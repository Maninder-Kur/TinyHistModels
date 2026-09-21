import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import scipy.sparse as sp

from build_graph_tensor import build_edge_feature_tensor,normalize_tensor


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True):
        super(GraphConvolution, self).__init__()
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v

        if node_layer:
            self.node_layer = True
            self.weight = Parameter(torch.FloatTensor(in_features_v, out_features_v))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_e))).float())
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_v))
            else:
                self.register_parameter('bias', None)
        else:
            self.node_layer = False
            self.weight = Parameter(torch.FloatTensor(in_features_e, out_features_e))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_v))).float())
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_e))
            else:
                self.register_parameter('bias', None)
        self.activation=nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward_single_batch(self, H_v, H_e, adj_e, adj_v, T):
        if self.node_layer:
            multiplier1 = torch.mm(T, torch.diag((H_e @ self.p.t()).t()[0])) @ T.to_dense().t() # to dense
            dev1 = multiplier1.device
            mask1 = torch.eye(multiplier1.shape[0], device=dev1)
            M1 = mask1 * torch.ones_like(mask1) + (1. - mask1) * multiplier1
            adjusted_A = torch.mul(M1, adj_v)
            output = torch.mm(adjusted_A, torch.mm(H_v, self.weight))
            ret=self.activation(output)
            if self.bias is not None:
                ret = output + self.bias
            return ret, H_e

        else:
            multiplier2 = torch.spmm(T.t(), torch.diag((H_v @ self.p.t()).t()[0])) @ T.to_dense()
            dev2 = multiplier2.device
            mask2 = torch.eye(multiplier2.shape[0], device=dev2)
            M3 = mask2 * torch.ones_like(mask2) + (1. - mask2) * multiplier2
            adjusted_A = torch.mul(M3, adj_e)
            normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0]
            output = torch.mm(normalized_adjusted_A, torch.mm(H_e, self.weight))
            ret=self.activation(output)
            if self.bias is not None:
                ret = output + self.bias
            return H_v, ret

    def forward(self, H_v, H_e, adj_e, adj_v, T):
        if self.node_layer:
            ret=torch.stack([self.forward_single_batch(H_v[i], H_e[i], adj_e[i], adj_v[i], T[i])[0] for i in range(H_v.shape[0])])
            return ret,H_e
        else:
            ret=torch.stack([self.forward_single_batch(H_v[i], H_e[i], adj_e[i], adj_v[i], T[i])[1] for i in range(H_v.shape[0])])
            return H_v,ret


class GCN(nn.Module):
    def __init__(self,nfeat_v, nfeat_e, nhid, nfeat_v_out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat_v, nhid, nfeat_e, nfeat_e, node_layer=True)
        self.gc2 = GraphConvolution(nhid, nhid, nfeat_e, nfeat_e*2, node_layer=False)
        self.gc3 = GraphConvolution(nhid, nfeat_v_out, nfeat_e*2, nfeat_e*2, node_layer=True)
        self.dropout = dropout

    def forward(self, X,graph):
        Z, adj_e, adj_v, T,topk=process_feature(X,graph)
        Z = Z.to(X.device)
        adj_e = adj_e.to(X.device)
        adj_v = adj_v.to(X.device)
        T = T.to(X.device)
        # print x
        gc1 = self.gc1(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc1[0]), F.relu(gc1[1])

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        gc2 = self.gc2(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc2[0]), F.relu(gc2[1])

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        X, Z = self.gc3(X, Z, adj_e, adj_v, T)
        return X,Z,topk,T

class GCN_big(nn.Module):
    def __init__(self,nfeat_v, nfeat_e, nhid, nfeat_v_out, dropout):
        super(GCN_big, self).__init__()
        self.gc1 = GraphConvolution(nfeat_v, nhid, nfeat_e, nfeat_e, node_layer=True)
        self.gc2 = GraphConvolution(nhid, nhid, nfeat_e, nfeat_e*2, node_layer=False)
        self.gc3 = GraphConvolution(nhid, nhid, nfeat_e*2, nfeat_e*2, node_layer=True)
        self.gc4=GraphConvolution(nhid, nhid, nfeat_e*2, nfeat_e*2, node_layer=False)
        self.gc5 = GraphConvolution(nhid, nfeat_v_out, nfeat_e * 2, nfeat_e*2 , node_layer=True)
        self.dropout = dropout

    def forward(self, X,graph):
        Z, adj_e, adj_v, T,topk=process_feature(X,graph)
        Z = Z.to(X.device)
        adj_e = adj_e.to(X.device)
        adj_v = adj_v.to(X.device)
        T = T.to(X.device)
        # print x
        gc1 = self.gc1(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc1[0]), F.relu(gc1[1])

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        gc2 = self.gc2(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc2[0]), F.relu(gc2[1])

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        X, Z = self.gc3(X, Z, adj_e, adj_v, T)

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        X, Z = self.gc4(X, Z, adj_e, adj_v, T)

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        X, Z = self.gc5(X, Z, adj_e, adj_v, T)
        return X,Z,topk,T

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    with torch.no_grad():
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

def create_transition_matrix_new(edge_pairs):
    """
    Build transition matrices T from edge_pairs.

    Input:
      edge_pairs: torch.LongTensor, shape (B, N, K, 2) with values in [0, N-1]
    Output:
      T: torch.sparse.FloatTensor, shape (B, N, E) where E = N*K
         (T[b] is sparse with rows=node, cols=edge_index)
    """
    import numpy as np
    from scipy import sparse as sp

    # allow numpy input too
    if isinstance(edge_pairs, np.ndarray):
        edge_pairs = torch.from_numpy(edge_pairs)

    edge_pairs = edge_pairs.long()
    B, N, K, _ = edge_pairs.shape
    E = N * K

    all_T = []
    device = edge_pairs.device

    for b in range(B):
        pairs = edge_pairs[b].cpu().numpy().reshape(E, 2)  # (E,2)
        row_idx = []
        col_idx = []
        for j in range(E):
            s = int(pairs[j, 0])
            e = int(pairs[j, 1])
            # each edge connects start and end -> mark both nodes in column j
            row_idx.append(s)
            col_idx.append(j)
            row_idx.append(e)
            col_idx.append(j)

        data = np.ones(len(row_idx), dtype=np.float32)
        T_csr = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, E))
        T_torch = sparse_mx_to_torch_sparse_tensor(T_csr)  # returns CPU sparse tensor
        T_torch = T_torch.to(device)                       # move to same device as edge_pairs
        all_T.append(T_torch.unsqueeze(0))

    # concatenate batch dimension
    return torch.cat(all_T, dim=0)  # (B, N, E) sparse tensors


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)).astype("float")
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def create_edge_adj_tensor(edge_pairs, topk=None, flag_normalize=True):
    """
    edge_pairs: (B, N, K, 2)
    returns: edge_adj (B, E, E) where E = N*K
    """
    with torch.no_grad():
        B, N, K, _ = edge_pairs.shape
        E = N * K
        edge_pairs = edge_pairs.long()

        start = edge_pairs[..., 0].view(B, E, 1)  # (B, E, 1)
        end   = edge_pairs[..., 1].view(B, E, 1)  # (B, E, 1)

        edge_adj = torch.zeros((B, E, E), dtype=torch.float32, device=edge_pairs.device)

        # mark connections where edges share start or end nodes
        # (start==start'), (start==end'), (end==start'), (end==end')
        s = start
        e = end
        edge_adj = edge_adj.masked_fill((s == s.transpose(1,2)), 1.0)
        edge_adj = edge_adj.masked_fill((s == e.transpose(1,2)), 1.0)
        edge_adj = edge_adj.masked_fill((e == s.transpose(1,2)), 1.0)
        edge_adj = edge_adj.masked_fill((e == e.transpose(1,2)), 1.0)

        if flag_normalize:
            edge_adj = normalize_tensor(edge_adj, plus_one=True)

        return edge_adj


'''number=0'''
def process_feature(x, graph):
    """
    Returns:
      edge_features: (B, E, F)
      edge_adj:     (B, E, E)
      adj_v_norm:   (B, N, N) normalized vertex adjacency (distance_adj)
      transition_matrix: (B, N, E) sparse matrices in torch.sparse format
      edge_pairs:   (B, N, K, 2)
    """
    # build_edge_feature_tensor must return (distance_adj, edge_features, edge_pairs)
    distance_adj, edge_features, edge_pairs = build_edge_feature_tensor(x, graph, topk=4)

    # build transition matrix using edge_pairs (B, N, K, 2)
    transition_matrix = create_transition_matrix_new(edge_pairs)

    # build edge adjacency (edge-to-edge adjacency) using edge_pairs
    edge_adj = create_edge_adj_tensor(edge_pairs, topk=edge_pairs.shape[2])

    # normalize the distance_adj (vertex adjacency) for later use
    adj_v_norm = normalize_tensor(distance_adj.to(torch.float32))

    return edge_features, edge_adj, adj_v_norm, transition_matrix, edge_pairs


