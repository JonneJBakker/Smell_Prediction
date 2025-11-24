import torch
import torch.nn as nn
import torch.nn.functional as F

class ChemicallyInformedLoss(nn.Module):
    """
    Implementation of CIL (Chemically-Informed Loss) from:
    'Molecular Odor Prediction with Harmonic Modulated Feature Mapping and Chemically-Informed Loss'
    https://arxiv.org/pdf/2502.01296

    Expects:
        logits: (N, M) raw model outputs
        y_true: (N, M) {0,1} labels
        features: (N, F) molecular features for structural similarity term

    Uses dataset-level pos/neg counts for class weights.
    Uses batch-level approximations for energy / correlation terms.
    """
    def __init__(
        self,
        pos_counts,        # 1D tensor (M,)
        neg_counts,        # 1D tensor (M,)
        lambda1=0.3,
        lambda2=0.3,
        lambda3=0.5,
        lambda4=0.3,
        c=0.2,             # co-occurrence energy hyperparameter
        e1=1.0,
        e2=1.0,
        sim_tau=0.8,       # similarity threshold τ
        device="cpu"
    ):
        super().__init__()

        pos_counts = torch.as_tensor(pos_counts, dtype=torch.float32, device=device)
        neg_counts = torch.as_tensor(neg_counts, dtype=torch.float32, device=device)

        # w_j = Wneg_j / Wpos_j, clamped to [0.1, 10]
        class_weights = neg_counts / (pos_counts + 1e-8)
        class_weights = class_weights.clamp(0.1, 10.0)

        self.register_buffer("class_weights", class_weights)

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.c = c
        self.e1 = e1
        self.e2 = e2
        self.sim_tau = sim_tau

    def forward(self, logits, y_true, features):
        """
        logits: (N, M)
        y_true: (N, M)
        features: (N, F)
        """
        y_true = y_true.float()
        y_pred = torch.sigmoid(logits)
        N, M = y_true.shape

        # ----- 1) Weighted BCE (Lbasis) -----
        # Broadcast class weights over batch dimension
        weight = self.class_weights.unsqueeze(0)  # (1, M)
        Lbasis = F.binary_cross_entropy_with_logits(
            logits,
            y_true,
            weight=weight,
            reduction="mean"
        )

        # ----- 2) Structural similarity loss (Lstt) -----
        # Normalize features and compute cosine similarity matrix
        eps = 1e-8
        f_norm = features / (features.norm(dim=1, keepdim=True) + eps)  # (N, F)
        sim_matrix = f_norm @ f_norm.t()  # (N, N)

        # Mask for similar pairs
        sim_mask = (sim_matrix > self.sim_tau).float()  # (N, N)

        # Pairwise squared L2 distance in label space
        # Compute Y_pred_i - Y_pred_j via broadcasting
        diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)  # (N, N, M)
        dist2 = (diff ** 2).sum(dim=-1)  # (N, N)

        # Only penalize similar pairs
        # Note: we can divide by N^2 or by number of positive pairs; here N^2 as in paper
        if sim_mask.sum() > 0:
            Lstt = (sim_mask * dist2).sum() / (N * N)
        else:
            Lstt = logits.new_tensor(0.0)

        # ----- 3) Class "energy" loss (Lclass) -----
        # Ej = mean predicted probability across batch for each class
        Ej = y_pred.mean(dim=0)  # (M,)

        # Co-occurrence matrix approximated from batch
        # Yi are rows of y_true, shape (N, M)
        # C = (Y^T @ Y) / N
        co_mat = (y_true.t() @ y_true) / max(N, 1)  # (M, M)

        # For min: use diag of Y^T Y
        co_diag_pos = torch.diag(co_mat)  # (M,)
        min_target = 1.0 + self.c * co_diag_pos

        # For mout: use co-occurrence of (1 - Y)
        y_neg = 1.0 - y_true
        co_mat_neg = (y_neg.t() @ y_neg) / max(N, 1)
        co_diag_neg = torch.diag(co_mat_neg)  # (M,)
        mout_target = self.c * co_diag_neg

        # Number of positives and negatives in THIS batch
        batch_pos_counts = y_true.sum(dim=0)      # (M,)
        batch_neg_counts = y_neg.sum(dim=0)       # (M,)

        # Eq. (14) style: sum over samples with Yi,j = 1 or 0.
        pos_term = torch.relu(Ej - min_target) ** 2  # (M,)
        neg_term = torch.relu(mout_target - Ej) ** 2  # (M,)

        Lclass = (batch_pos_counts * pos_term + batch_neg_counts * neg_term).sum() / max(N, 1)

        # ----- 4) Sample-level multi-label constraint (Lsample) -----
        # Expected energy for sample i: e1 + e2 * (# labels)
        label_counts = y_true.sum(dim=1)  # (N,)
        E_expected = self.e1 + self.e2 * label_counts  # (N,)

        # Predicted energy per sample = sum_j Yhat_i,j
        E_pred_sample = y_pred.sum(dim=1)  # (N,)

        Lsample = torch.relu(E_expected - E_pred_sample) ** 2
        Lsample = Lsample.mean()

        # ----- 5) Label correlation loss (Lcol) -----
        # Predicted correlation: (Yhat^T Yhat) / N
        corr_pred = (y_pred.t() @ y_pred) / max(N, 1)  # (M, M)
        corr_true = (y_true.t() @ y_true) / max(N, 1)  # (M, M)

        Lcol = F.mse_loss(corr_pred, corr_true)

        # ----- 6) Total loss -----
        Ltotal = Lbasis \
                 + self.lambda1 * Lstt \
                 + self.lambda2 * Lclass \
                 + self.lambda3 * Lsample \
                 + self.lambda4 * Lcol

        # For logging you might want the components as well
        return Ltotal, {
            "Lbasis": Lbasis.detach(),
            "Lstt": Lstt.detach(),
            "Lclass": Lclass.detach(),
            "Lsample": Lsample.detach(),
            "Lcol": Lcol.detach(),
        }
