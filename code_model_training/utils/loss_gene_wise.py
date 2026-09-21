import torch.nn.functional as F
import torch.nn as nn


def gene_weighted_mse_loss(y_pred, y_true, eps=1e-4):
    # Compute variance of each gene across batch
    gene_var = y_true.var(dim=0, unbiased=False)  # shape: [num_genes]

    # Inverse-variance weights (to emphasize under-represented genes)
    gene_weights = 1.0 / (gene_var + eps)  # shape: [num_genes]
    gene_weights = gene_weights / gene_weights.sum()  # normalize to sum=1

    # Squared error per gene per sample
    mse_per_gene = (y_pred - y_true) ** 2  # shape: [B, G]

    # Weight and average over genes and batch
    weighted_mse = (mse_per_gene * gene_weights).sum(dim=1).mean()  # [B] → scalar
    return weighted_mse

def kd_loss(student_pred, teacher_pred, target, alpha=0.6, temperature=2.0):
    # Soft targets
    T = temperature
    kd_loss = F.kl_div(
        F.log_softmax(student_pred / T, dim=1),
        F.softmax(teacher_pred / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)
    
    mse = F.mse_loss(student_pred, target)
    return alpha * mse + (1 - alpha) * kd_loss


def distil_loss(y_pred, y_teacher,y_true, lambda_sup=1.0, lambda_kd=1.0):
    loss_sup = F.mse_loss(y_pred, y_true)
    loss_kd  = F.mse_loss(y_pred, y_teacher)
    return lambda_sup * loss_sup + lambda_kd * loss_kd