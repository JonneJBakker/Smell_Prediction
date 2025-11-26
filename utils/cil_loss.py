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
    Uses batch-level or dataset-level (configurable) approximations
    for energy / correlation terms.
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
        device="cpu",
        # NEW: dataset-level stats (optional)
        use_dataset_cooccur=False,
        co_diag_pos_dataset=None,  # 1D (M,), diag of Y^T Y / N_dataset
        co_diag_neg_dataset=None,  # 1D (M,), diag of (1-Y)^T(1-Y) / N_dataset
        corr_true_dataset=None,    # 2D (M,M), Y^T Y / N_dataset
    ):
        super().__init__()

        # ---- Class weights (unchanged) ----
        pos_counts = torch.as_tensor(pos_counts, dtype=torch.float32, device=device)
        neg_counts = torch.as_tensor(neg_counts, dtype=torch.float32, device=device)

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

        # ---- New: dataset-level co-occurrence / correlation ----
        self.use_dataset_cooccur = use_dataset_cooccur

        if use_dataset_cooccur:
            if co_diag_pos_dataset is None or co_diag_neg_dataset is None or corr_true_dataset is None:
                raise ValueError(
                    "use_dataset_cooccur=True requires co_diag_pos_dataset, "
                    "co_diag_neg_dataset, and corr_true_dataset to be provided."
                )

            co_diag_pos_dataset = torch.as_tensor(
                co_diag_pos_dataset, dtype=torch.float32, device=device
            )
            co_diag_neg_dataset = torch.as_tensor(
                co_diag_neg_dataset, dtype=torch.float32, device=device
            )
            corr_true_dataset = torch.as_tensor(
                corr_true_dataset, dtype=torch.float32, device=device
            )

            self.register_buffer("co_diag_pos_dataset", co_diag_pos_dataset)
            self.register_buffer("co_diag_neg_dataset", co_diag_neg_dataset)
            self.register_buffer("corr_true_dataset", corr_true_dataset)

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
        weight = self.class_weights.unsqueeze(0)  # (1, M)
        Lbasis = F.binary_cross_entropy_with_logits(
            logits,
            y_true,
            weight=weight,
            reduction="mean"
        )

        # ----- 2) Structural similarity loss (Lstt) -----
        eps = 1e-8
        f_norm = features / (features.norm(dim=1, keepdim=True) + eps)  # (N, F)
        sim_matrix = f_norm @ f_norm.t()  # (N, N)
        sim_mask = (sim_matrix > self.sim_tau).float()  # (N, N)

        diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)  # (N, N, M)
        dist2 = (diff ** 2).sum(dim=-1)  # (N, N)

        if sim_mask.sum() > 0:
            Lstt = (sim_mask * dist2).sum() / (N * N)
        else:
            Lstt = logits.new_tensor(0.0)

        # ----- 3) Class "energy" loss (Lclass) -----
        Ej = y_pred.mean(dim=0)  # (M,)

        # Batch-level co-occurrence (still needed if we don't use dataset stats)
        co_mat = (y_true.t() @ y_true) / max(N, 1)  # (M, M)
        y_neg = 1.0 - y_true
        co_mat_neg = (y_neg.t() @ y_neg) / max(N, 1)

        if self.use_dataset_cooccur:
            # Use precomputed dataset-level diagonal entries
            co_diag_pos = self.co_diag_pos_dataset  # (M,)
            co_diag_neg = self.co_diag_neg_dataset  # (M,)
        else:
            co_diag_pos = torch.diag(co_mat)       # (M,)
            co_diag_neg = torch.diag(co_mat_neg)   # (M,)

        min_target = 1.0 + self.c * co_diag_pos
        mout_target = self.c * co_diag_neg

        batch_pos_counts = y_true.sum(dim=0)  # (M,)
        batch_neg_counts = y_neg.sum(dim=0)   # (M,)

        pos_term = torch.relu(Ej - min_target) ** 2
        neg_term = torch.relu(mout_target - Ej) ** 2

        Lclass = (batch_pos_counts * pos_term + batch_neg_counts * neg_term).sum() / max(N, 1)

        # ----- 4) Sample-level multi-label constraint (Lsample) -----
        label_counts = y_true.sum(dim=1)  # (N,)
        E_expected = self.e1 + self.e2 * label_counts  # (N,)
        E_pred_sample = y_pred.sum(dim=1)  # (N,)

        Lsample = torch.relu(E_expected - E_pred_sample) ** 2
        Lsample = Lsample.mean()

        # ----- 5) Label correlation loss (Lcol) -----
        corr_pred = (y_pred.t() @ y_pred) / max(N, 1)  # (M, M)

        if self.use_dataset_cooccur:
            corr_true = self.corr_true_dataset        # (M, M)
        else:
            corr_true = (y_true.t() @ y_true) / max(N, 1)  # (M, M)

        Lcol = F.mse_loss(corr_pred, corr_true)

        # ----- 6) Total loss -----
        Ltotal = (
            Lbasis
            + self.lambda1 * Lstt
            + self.lambda2 * Lclass
            + self.lambda3 * Lsample
            + self.lambda4 * Lcol
        )

        return Ltotal

