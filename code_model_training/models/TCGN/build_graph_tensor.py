import torch
import numpy as np


class Global_Adj():
    def __init__(self):
        self.adj_v_196 = None
        self.adj_v_49 = None
        self.batch_size=0

    def initialize(self,batch_size, device):
        self.init_global_adj(batch_size, device)
        self.batch_size=batch_size

    @torch.no_grad()
    def init_global_adj(self,batch_size, device):
        # 196
        adj196 = np.zeros((196, 196), dtype=np.float32)
        num_nodes = adj196.shape[0]
        height = int(num_nodes ** 0.5)
        for i in range(num_nodes):
            position1 = [i // height, i % height]
            for j in range(num_nodes):
                position2 = [j // height, j % height]
                distancei = ((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2) ** 0.5
                distancei = 1 / (distancei / 3 + 1)
                adj196[i][j] = distancei
        self.adj_v_196 = torch.from_numpy(adj196).unsqueeze(0).repeat(batch_size,1,1).to(torch.float32).to(device)

        # 49
        adj49 = np.zeros((49, 49), dtype=np.float32)
        num_nodes = adj49.shape[0]
        height = int(num_nodes ** 0.5)
        for i in range(num_nodes):
            position1 = [i // height, i % height]
            for j in range(num_nodes):
                position2 = [j // height, j % height]
                distancei = ((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2) ** 0.5
                distancei = 1 / (distancei / 3 + 1)
                adj49[i][j] = distancei
        self.adj_v_49 = torch.from_numpy(adj49).unsqueeze(0).repeat(batch_size,1,1).to(torch.float32).to(device)

global_adj = Global_Adj()


def build_edge_feature_tensor(x_input, graph_input, topk=4):
    """
    x: (B, N, C) on some device
    graph: (B, H, N, N) on some device
    returns:
        adj_v: (B, N, N)             (float tensor, device=x.device)
        edge_features: (B, N*K, H+3) (float tensor, device=x.device)
        edge_pairs: (B, N, K, 2)     (long tensor, device=x.device)
    """
    device = x_input.device
    x = x_input
    graph = graph_input
    B, N, C = x.shape
    H = graph.shape[1]

    if global_adj.batch_size != B or global_adj.adj_v_196.device != device:
        # initialize on the same device
        global_adj.initialize(B, device)

    # adjacency of patches based on global spatial distances (float)
    if N == 196:
        adj_v = global_adj.adj_v_196.clone()
    elif N == 49:
        adj_v = global_adj.adj_v_49.clone()
    else:
        raise ValueError(f"Unsupported N: {N}")

    # 1) mean over heads -> attention matrix A
    A = graph.mean(dim=1)  # (B, N, N) on device

    # 2) top-k neighbors for each node (per-row topk)
    # returns values and indices: topk_vals: (B,N,K) topk_idx: (B,N,K)
    topk_vals, topk_idx = torch.topk(A, k=topk, dim=-1)

    # build edge pair tensor (start, end)
    # start indices shape: (B, N, K)
    start_idx = torch.arange(N, device=device).view(1, N, 1).repeat(B, 1, topk)
    end_idx = topk_idx  # already on correct device
    edge_pairs = torch.stack([start_idx, end_idx], dim=-1).long()  # (B, N, K, 2) on device

    # build adjacency mask (B,N,N)
    adj = torch.zeros((B, N, N), device=device, dtype=torch.float32)
    # scatter: set ones at (b, i, end_idx[b,i,k])
    # need an index for batch dimension
    batch_idx = torch.arange(B, device=device).view(B,1,1).repeat(1,N,topk)
    node_idx  = torch.arange(N, device=device).view(1,N,1).repeat(B,1,topk)
    adj[batch_idx, node_idx, end_idx] = 1.0
    # remove self loops
    eye = torch.eye(N, device=device).unsqueeze(0)
    adj = adj * (1 - eye)

    # 3) compute L2 and cosine similarities for all pairs, then pick edge entries
    x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = torch.matmul(x_norm, x_norm.transpose(1,2))   # (B,N,N)
    L2 = torch.cdist(x, x)                                   # (B,N,N)
    L2 = 1.0 / (1.0 + L2)                                   # normalized into (0,1]

    # flatten index helpers for extracting edge scalars
    B_idx = torch.arange(B, device=device).repeat_interleave(N*topk)
    flat_start = start_idx.reshape(B*N*topk)
    flat_end   = end_idx.reshape(B*N*topk)

    # attention A feature (the mean attention value)
    A_feat = A[B_idx, flat_start, flat_end].view(B, N*topk, 1)

    # L2 and cosine features
    L2_feat = L2[B_idx, flat_start, flat_end].view(B, N*topk, 1)
    cos_feat = cos_sim[B_idx, flat_start, flat_end].view(B, N*topk, 1)

    # per-head features: collect H features for each edge
    graph_heads = []
    for h in range(H):
        g = graph[:, h]  # (B, N, N)
        gh = g[B_idx, flat_start, flat_end].view(B, N*topk, 1)
        graph_heads.append(gh)

    # final edge_features: concat [head1, head2, ..., A, L2, cos] -> (B, N*K, H+3)
    # final edge_features: concat [head1...headH, A, L2, cos, dummy] -> (B, N*K, H+4)
    dummy = torch.ones((B, N*topk, 1), device=device)

    edge_features = torch.cat(
        graph_heads + [A_feat, L2_feat, cos_feat, dummy],
        dim=-1
    )


    return adj, edge_features, edge_pairs


def normalize_tensor(x, plus_one=False):
    with torch.no_grad():
        if plus_one:
            rowsum = torch.sum(x + torch.diag_embed(torch.ones(x.shape[0], x.shape[1], device=x.device)), dim=-1)
        else:
            rowsum = torch.sum(x, dim=-1)

        r_inv_sqrt = rowsum ** (-0.5)
        r_inv_sqrt = torch.where(torch.isinf(r_inv_sqrt), torch.zeros_like(r_inv_sqrt), r_inv_sqrt)
        r_inv_sqrt = torch.diag_embed(r_inv_sqrt)

    return r_inv_sqrt @ x @ r_inv_sqrt.transpose(-2, -1)


