import torch
import torch.nn as nn

class KDLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, student_out, teacher_out, targets):

        # ---- Ground truth loss ----
        loss_gt = self.mse(student_out, targets)

        # ---- KD loss (safe) ----
        if teacher_out is None:
            return loss_gt

        if not torch.isfinite(teacher_out).all():
            return loss_gt

        if teacher_out.shape != student_out.shape:
            raise ValueError(
                f"Shape mismatch: student {student_out.shape}, teacher {teacher_out.shape}"
            )

        loss_kd = self.mse(
            (student_out - student_out.mean(dim=1, keepdim=True)),
            (teacher_out - teacher_out.mean(dim=1, keepdim=True))
        )

        return self.alpha * loss_gt + self.beta * loss_kd