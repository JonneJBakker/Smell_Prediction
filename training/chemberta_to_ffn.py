# chemberta_trainer.py
import os, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

class FrozenChemBERTaMultiLabel:
    """
    Frozen ChemBERTa encoder + GELU FFN head for multi-label classification.
    - Uses CLS token (index 0) as embedding (h=384 for DeepChem/ChemBERTa-77M-MLM).
    - Handles class imbalance via BCEWithLogitsLoss(pos_weight).
    - Tracks micro/macro AUROC and mAP on validation.
    """

    def __init__(
        self,
        csv_path="multi_labelled_smiles_odors.csv",
        smiles_col="nonStereoSMILES",
        exclude_cols = ["descriptors"],
        backbone="DeepChem/ChemBERTa-77M-MLM",
        n_labels=138,
        max_len=512,
        batch_train=32,
        batch_val=64,
        epochs=10,
        seed=1999,
        lr_head=1e-3,
        weight_decay=1e-2,
        dropout=0.3,
        warmup_frac=0.1,
        device=None,
    ):
        self.csv_path = csv_path
        self.smiles_col = smiles_col
        self.exclude_cols = exclude_cols
        self.backbone = backbone
        self.n_labels = n_labels
        self.max_len = max_len
        self.batch_train = batch_train
        self.batch_val = batch_val
        self.epochs = epochs
        self.seed = seed
        self.lr_head = lr_head
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.warmup_frac = warmup_frac
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tb_dir = "runs/chemberta_frozen"  # change if you like
        self.tb = SummaryWriter(log_dir=self.tb_dir)

        # Will be set later
        self.label_cols = None
        self.tok = None
        self.model = None
        self.tr_dl = None
        self.va_dl = None
        self.pos_weight = None
        self.thresholds = None  # per-label thresholds for hard predictions

        # Reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    # --------------------- Data ---------------------
    def _load_data(self):
        df = pd.read_csv(self.csv_path).dropna(subset=[self.smiles_col]).reset_index(drop=True)
        label_cols = [c for c in df.columns if c != self.smiles_col and df[c].dropna().isin([0,1]).all()]
        assert len(label_cols) == self.n_labels, f"Expected {self.n_labels} labels, found {len(label_cols)}"
        self.label_cols = label_cols
        Y = df[label_cols].astype(np.float32).values
        S = df[self.smiles_col].astype(str).tolist()
        return S, Y

    def _split(self, S, Y):
        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        train_idx, val_idx = next(mskf.split(np.zeros(len(S)), Y))
        S_tr, S_va = [S[i] for i in train_idx], [S[i] for i in val_idx]
        Y_tr, Y_va = Y[train_idx], Y[val_idx]
        return S_tr, Y_tr, S_va, Y_va

    def _tokenizer(self):
        if self.tok is None:
            self.tok = AutoTokenizer.from_pretrained(self.backbone)
        return self.tok

    def _tokenize(self, smiles_list):
        tok = self._tokenizer()
        return tok(smiles_list, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")

    class _SmilesMLDataset(Dataset):
        def __init__(self, parent, smiles, labels):
            self.parent, self.s, self.y = parent, smiles, labels
        def __len__(self): return len(self.s)
        def __getitem__(self, i):
            t = self.parent._tokenize([self.s[i]])
            item = {k: v.squeeze(0) for k,v in t.items()}
            item["labels"] = torch.tensor(self.y[i], dtype=torch.float32)
            return item

    def _make_dataloaders(self, S_tr, Y_tr, S_va, Y_va):
        tr_ds = self._SmilesMLDataset(self, S_tr, Y_tr)
        va_ds = self._SmilesMLDataset(self, S_va, Y_va)
        self.tr_dl = DataLoader(tr_ds, batch_size=self.batch_train, shuffle=True,  pin_memory=True)
        self.va_dl = DataLoader(va_ds, batch_size=self.batch_val,   shuffle=False, pin_memory=True)

        # pos_weight for BCEWithLogitsLoss
        pos_freq = Y_tr.mean(axis=0) + 1e-6
        self.pos_weight = torch.tensor((1 - pos_freq) / pos_freq, dtype=torch.float32, device=self.device)

    # --------------------- Model ---------------------
    class _FrozenHead(nn.Module):
        def __init__(self, backbone, n_labels, dropout):
            super().__init__()
            self.enc = AutoModel.from_pretrained(backbone)
            for p in self.enc.parameters(): p.requires_grad = False
            h = self.enc.config.hidden_size  # 384 for ChemBERTa-77M-MLM
            self.head = nn.Sequential(
                nn.LayerNorm(h),
                nn.Linear(h, 512), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(256, n_labels)
            )
        def forward(self, input_ids, attention_mask):
            with torch.no_grad():
                hs = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            z0 = hs[:, 0, :]  # CLS
            return self.head(z0)  # logits (B, n_labels)

    def _build_model(self):
        self.model = self._FrozenHead(self.backbone, self.n_labels, self.dropout).to(self.device)

    # ------------------- Training -------------------
    def fit(self):
        # Data
        S, Y = self._load_data()
        S_tr, Y_tr, S_va, Y_va = self._split(S, Y)
        self._make_dataloaders(S_tr, Y_tr, S_va, Y_va)

        # Model
        self._build_model()

        # Loss / Optim / Scheduler
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        opt = torch.optim.AdamW(self.model.head.parameters(), lr=self.lr_head, weight_decay=self.weight_decay)
        steps = self.epochs * len(self.tr_dl)
        sched = get_linear_schedule_with_warmup(opt, int(self.warmup_frac * steps), steps)

        best_map, best_state = -1.0, None

        for ep in range(1, self.epochs + 1):
            # ---------- Training ----------
            self.model.train()
            train_loss = 0.0
            for b in self.tr_dl:
                ids = b["input_ids"].to(self.device)
                mask = b["attention_mask"].to(self.device)
                y = b["labels"].to(self.device)

                logits = self.model(ids, mask)
                loss = criterion(logits, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()

                train_loss += loss.item()

            train_loss /= len(self.tr_dl)

            # ---------- Validation ----------
            metrics, P, Yval, val_loss = self.evaluate(return_raw=True, criterion=criterion)
            print(
                f"Epoch {ep:02d} | Train Loss={train_loss:.4f} | "
                f"Val Loss={val_loss:.4f} | AUCμ={metrics['AUC_micro']:.4f} | "
                f"mAPμ={metrics['mAP_micro']:.4f}"
            )

            self.tb.add_scalar("Loss/train", train_loss, ep)
            self.tb.add_scalar("Loss/val", val_loss, ep)
            self.tb.add_scalar("AUC/micro", metrics["AUC_micro"], ep)
            self.tb.add_scalar("AUC/macro", metrics["AUC_macro"], ep)
            self.tb.add_scalar("mAP/micro", metrics["mAP_micro"], ep)
            self.tb.add_scalar("mAP/macro", metrics["mAP_macro"], ep)
            self.tb.flush()

            if metrics["mAP_micro"] > best_map:
                best_map = metrics["mAP_micro"]
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                np.save("val_probs.npy", P)
                np.save("val_labels.npy", Yval)



        # ---------- Save best ----------
        if best_state is not None:
            self.model.load_state_dict(best_state)
            torch.save(self.model.state_dict(), "chemberta_frozen_head.pt")
            print(f"✔ Saved best head with micro-mAP={best_map:.4f}")

        return best_map

    def evaluate(self, return_raw=False, criterion=None):
        """Computes validation metrics and optionally returns raw probs + loss."""
        self.model.eval()
        all_logits, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for b in self.va_dl:
                ids = b["input_ids"].to(self.device)
                mask = b["attention_mask"].to(self.device)
                y = b["labels"].to(self.device)
                logits = self.model(ids, mask)
                all_logits.append(logits.cpu())
                all_labels.append(b["labels"])
                if criterion is not None:
                    val_loss += criterion(logits, y).item()

        val_loss /= len(self.va_dl)
        P = torch.sigmoid(torch.cat(all_logits)).numpy()
        Y = torch.cat(all_labels).numpy()

        try:
            auc_micro = roc_auc_score(Y, P, average="micro")
            auc_macro = roc_auc_score(Y, P, average="macro")
        except ValueError:
            auc_micro = auc_macro = float("nan")
        ap_micro = average_precision_score(Y, P, average="micro")
        ap_macro = average_precision_score(Y, P, average="macro")

        metrics = {
            "AUC_micro": auc_micro,
            "AUC_macro": auc_macro,
            "mAP_micro": ap_micro,
            "mAP_macro": ap_macro,
        }

        return (metrics, P, Y, val_loss) if return_raw else (metrics, val_loss)

    # ------------------- Inference -------------------
    @torch.no_grad()
    def predict_probs(self, smiles_list, ckpt_path="chemberta_frozen_head.pt"):
        # If you want to load from disk; by default, use current model
        if os.path.exists(ckpt_path):
            self._ensure_model_loaded()
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()
        t = self._tokenize(smiles_list)
        logits = self.model(t["input_ids"].to(self.device), t["attention_mask"].to(self.device))
        return torch.sigmoid(logits).cpu().numpy()  # (N, n_labels)

    @torch.no_grad()
    def predict_labels(self, smiles_list, thresholds_path="thresholds.npy", return_probs=False):
        """Predicts label names (and optionally probabilities) for given SMILES."""


        # --- predict probabilities ---
        probs = self.predict_probs(smiles_list)

        # --- load thresholds ---
        if self.thresholds is not None:
            T = self.thresholds
        elif os.path.exists(thresholds_path):
            T = np.load(thresholds_path)
        else:
            print("⚠️ No thresholds found — using 0.5 for all labels.")
            T = np.full(self.n_labels, 0.5)

        # --- binarize ---
        preds = (probs >= T[None, :]).astype(np.uint8)

        # --- convert to label names ---
        results = []
        for i, row in enumerate(preds):
            active = [self.label_cols[j] for j, v in enumerate(row) if v == 1]
            results.append(active)

        if return_probs:
            # Optional: return both names + probabilities for inspection
            return results, probs
        return results

    # ------------------- Utils -------------------
    def save(self, path="chemberta_frozen_head.pt"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="chemberta_frozen_head.pt"):
        self._ensure_model_loaded()
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def _ensure_model_loaded(self):
        if self.model is None:
            self._build_model()

