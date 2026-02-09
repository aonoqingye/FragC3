import random
import argparse
import numpy as np

from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import KFold, GroupShuffleSplit, ShuffleSplit
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    accuracy_score,
    precision_score,
)

from utils import *
from model.FragC3 import *
from datas.dataset import PairDataset
from torch_geometric.loader import DataLoader
from tools.process_folds import process_folds

# -----------------------------
# tqdm configuration & logging
# -----------------------------
TQDM_KW = dict(dynamic_ncols=True, mininterval=1.0, smoothing=0.0)


def log(msg: str):
    """Unified logging entry-point to avoid interfering with tqdm."""
    tqdm.write(str(msg))


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    """Fix random seeds for reproducibility (deterministic settings where applicable)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_device(device_arg: str):
    """Select compute device based on CLI argument (prefer cuda:0 when available)."""
    if device_arg and device_arg.lower() in {"cpu", "cuda"}:
        if device_arg.lower() == "cuda" and torch.cuda.is_available():
            log("The code uses GPU...")
            return torch.device("cuda:0")
        else:
            log("The code uses CPU!!!")
            return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        log("The code uses GPU...")
        return torch.device("cuda:0")
    log("The code uses CPU!!!")
    return torch.device("cpu")


def save_checkpoint(save_path: str, model, optimizer, epoch: int, best_val: float, args, extra: dict = None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_val": float(best_val),
        "args": vars(args),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, save_path)

def compute_performance_classification(T, S, Y, best_auc, file, epoch):
    AUC = roc_auc_score(T, S)
    precision, recall, _ = metrics.precision_recall_curve(T, S)
    PR_AUC = metrics.auc(recall, precision)
    BACC = balanced_accuracy_score(T, Y)
    tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
    TPR = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    PREC = precision_score(T, Y, zero_division=0)
    ACC = accuracy_score(T, Y)
    KAPPA = cohen_kappa_score(T, Y)
    REC = recall_score(T, Y, zero_division=0)
    F1 = f1_score(T, Y, zero_division=0)

    row = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, REC, F1]
    save_AUCs(row, file)
    if best_auc < AUC:
        best_auc = AUC
    return best_auc, AUC


def compute_performance_regression(T, P, best_mse, file, epoch):
    mse_val = np.mean((np.array(T) - np.array(P)) ** 2)
    rmse_val = rmse(P, T)
    mae_val = mae(P, T)
    row = [epoch, float(mse_val), float(rmse_val), float(mae_val)]
    save_AUCs(row, file)  # uses utils.save_AUCs (kept consistent with the original pipeline)
    if best_mse > mse_val:
        best_mse = mse_val
    return best_mse, mse_val



def train(
        # Training loop for one epoch.

        args,
        model,
        device,
        task,
        loader_train,
        optimizer,
        loss_fn,
        epoch: int,
        log_interval: int,
        show_batch_pbar: bool = False,
):
    """Run one training epoch and return the average loss."""
    model.train()
    total_loss = 0.0

    total_batches = len(loader_train)
    iterator = loader_train

    if show_batch_pbar:
        iterator = tqdm(
            iterator,
            total=total_batches,
            desc=f"Train epoch {epoch}",
            position=2,
            leave=False,
            **TQDM_KW,
        )

    for batch_idx, data in enumerate(iterator):
        data = data.to(device)
        if task == "classification":
            y = data.y.view(-1, 1).long().to(device).squeeze(1)
        else:
            y = data.y.view(-1, 1).to(device).float()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if show_batch_pbar and (batch_idx % max(1, log_interval) == 0):
            iterator.set_postfix(loss=f"{loss.item():.6f}")

    if show_batch_pbar and hasattr(iterator, "clear"):
        iterator.clear()

    avg_loss = total_loss / max(1, total_batches)
    return avg_loss


@torch.no_grad()
def predicting(
        args,
        model,
        device,
        task,
        loader_test,
        show_batch_pbar: bool = False,
        epoch: int = None,
):
    """Run model inference and return (y_true, y_score, y_pred)."""
    model.eval()
    total_scores = []
    total_labels = []
    total_pred_labels = []

    total_batches = len(loader_test)
    iterator = loader_test

    if show_batch_pbar:
        tag = f"Eval epoch {epoch}" if epoch is not None else "Eval"
        iterator = tqdm(
            iterator,
            total=total_batches,
            desc=tag,
            position=3,
            leave=False,
            **TQDM_KW,
        )

    for data in iterator:
        data = data.to(device)
        output = model(data)

        if task == "classification":
            ys = F.softmax(output, dim=1).to("cpu").data.numpy()
            pred_labels = np.argmax(ys, axis=1).tolist()
            pred_scores = [row[1] for row in ys]
        else:
            ys = output.squeeze(1).detach().cpu().numpy()
            pred_labels = ys
            pred_scores = ys

        total_scores.extend(pred_scores)
        total_pred_labels.extend(pred_labels)
        total_labels.extend(data.y.view(-1, 1).cpu().numpy().flatten().tolist())

    if show_batch_pbar and hasattr(iterator, "clear"):
        iterator.clear()

    return (
        np.asarray(total_labels).flatten(),
        np.asarray(total_scores).flatten(),
        np.asarray(total_pred_labels).flatten(),
    )



def main():
    global out_info
    args = parse_args()

    set_seed(args.seed)

    device = build_device(args.device)

    work_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(work_dir, "datas")
    dataset = args.dataset
    pairs_data = PairDataset(root=data_dir, dataset=dataset)

    cell_dim = pairs_data.data.cell1.shape[1]

    length = len(pairs_data)
    log(f"Parameters:  {args}")

    if args.groups == 'Cell':
        groups = np.asarray(pairs_data.data.cell_id)
        cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=args.seed)
    elif args.groups == 'Drug':
        groups = np.asarray(pairs_data.data.drug1_id)
        cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=args.seed)
    else:
        groups = None
        cv = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        log("Grouping strategy not implemented; falling back to KFold.")

    out_dir = os.path.join(work_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    fold_iter = tqdm(enumerate(cv.split(np.zeros(length), None, groups), 1),
                     desc=f"{args.folds}-Fold CV", position=0, leave=True, **TQDM_KW)

    for fold, (trainval_idx, test_idx) in fold_iter:
        if args.only_fold:
            if fold != args.only_fold:
                continue
        if args.groups != "none":
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed + fold)
            tv_groups = groups[trainval_idx]
            tv_train_rel, tv_valid_rel = next(gss.split(np.arange(len(trainval_idx)), None, groups=tv_groups))
        else:
            ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed + fold)
            tv_train_rel, tv_valid_rel = next(ss.split(np.arange(len(trainval_idx)), None, None))
        train_idx = trainval_idx[tv_train_rel]
        valid_idx = trainval_idx[tv_valid_rel]
        train_idx = train_idx.astype(np.int64)
        valid_idx = valid_idx.astype(np.int64)
        test_idx = test_idx.astype(np.int64)



        data_train = pairs_data[train_idx]
        data_valid = pairs_data[valid_idx]
        data_test = pairs_data[test_idx]

        loader_train = DataLoader(data_train, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
        loader_valid = DataLoader(data_valid, batch_size=args.test_batch_size)
        loader_test = DataLoader(data_test, batch_size=args.test_batch_size)

        task_type = ("classification" if args.dataset.lower().startswith("o") else "regression")
        if task_type == "classification":
            loss_fn = nn.CrossEntropyLoss()
            n_output = 2
        else:
            loss_fn = nn.MSELoss()
            n_output = 1

        model = FragC3(
            n_output=n_output,
            cell_dim=cell_dim,
            hid_dim=args.hidden,
            heads=args.heads,
            ffn_expansion=args.ffn_expansion,
            use_C3Attn=args.use_C3Attn,
            tri_attn=args.tri_attn,
            tri_variant=args.tri_variant,
            cv_mode=args.cv_mode,
            tokenizer=args.tokenizer,
            dropout=args.dropout,
            cell_agg=args.cell_agg,
            cell_pred=args.cell_pred,
            Lc=args.Lc,
            frag_list=args.frag_list,
            frag_agg=args.frag_agg,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        out_info = (f'{args.dataset}_Group{args.groups}_Frags{"_".join(args.frag_list)}_Agg{args.frag_agg}'
                    f'_C3Attn{args.use_C3Attn}_tri{args.tri_attn}_tokenizer{args.tokenizer}_Lc{args.Lc}')
        csv_path = os.path.join(out_dir, f"{out_info}_fold_{fold}.csv")

        with open(csv_path, "w") as f:
            if task_type == "classification":
                f.write("Epoch,AUC_dev,PR_AUC,ACC,BACC,PREC,TPR,KAPPA,RECALL,F1\n")
            else:
                f.write("Epoch,MSE,RMSE,MAE\n")

        best_val = (-np.inf if task_type == "classification" else np.inf)
        patience = int(args.early_stopping)
        bad_epochs = 0
        ckpt_path = os.path.join(args.ckpt_dir, f"{out_info}_fold_{fold}_best.pt")

        epoch_iter = tqdm(
            range(1, args.epochs + 1),
            desc=f"Fold {fold}/{args.folds}",
            position=1,
            leave=False,
            **TQDM_KW,
        )
        for epoch in epoch_iter:
            avg_train_loss = train(
                args=args,
                model=model,
                device=device,
                task=task_type,
                loader_train=loader_train,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epoch=epoch,
                log_interval=args.log_interval,
                show_batch_pbar=args.show_batch_pbar,
            )

            if task_type == "classification":
                T, S, Y = predicting(
                    args, model, device, task_type, loader_valid,
                    show_batch_pbar=args.show_batch_pbar, epoch=epoch
                )
                prev_best = best_val
                best_val, current_val = compute_performance_classification(
                    T=T, S=S, Y=Y, best_auc=best_val, file=csv_path, epoch=epoch
                )
                improved = best_val > prev_best + 1e-12
                epoch_iter.set_postfix(
                    auc=f"{current_val:.4f}",
                    best_auc=f"{best_val:.4f}",
                    train_loss=f"{avg_train_loss:.4f}",
                )
                if improved:
                    bad_epochs = 0
                    save_checkpoint(
                        save_path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_val=best_val,
                        args=args,
                        extra={"fold": fold, "out_info": out_info, "task_type": task_type},
                    )
                else:
                    bad_epochs += 1
                if patience > 0 and bad_epochs >= patience:
                    log(f"[Fold {fold}] Early stopping at epoch {epoch}: "
                        f"no improvement for {patience} epochs. best_auc={best_val:.6f}")
                    break
            else:
                T, P, _ = predicting(
                    args, model, device, task_type, loader_valid,
                    show_batch_pbar=args.show_batch_pbar, epoch=epoch
                )
                prev_best = best_val
                best_val, current_val = compute_performance_regression(
                    T, P, best_val, file=csv_path, epoch=epoch
                )
                improved = best_val < prev_best - 1e-12
                epoch_iter.set_postfix(
                    mse=f"{current_val:.4f}",
                    best_mse=f"{best_val:.4f}",
                    train_loss=f"{avg_train_loss:.4f}"
                )
                if improved:
                    bad_epochs = 0
                    save_checkpoint(
                        save_path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_val=best_val,
                        args=args,
                        extra={"fold": fold, "out_info": out_info, "task_type": task_type},
                    )
                else:
                    bad_epochs += 1
                if patience > 0 and bad_epochs >= patience:
                    log(f"[Fold {fold}] Early stopping at epoch {epoch}: "
                        f"no improvement for {patience} epochs. best_mse={best_val:.6f}")
                    break

        if hasattr(epoch_iter, "clear"):
            epoch_iter.clear()

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            best_val_loaded = float(ckpt.get("best_val", best_val))
            if task_type == "classification":
                log(f"[Fold {fold}] Loaded best checkpoint from: {ckpt_path} (best_auc={best_val_loaded:.6f})")
            else:
                log(f"[Fold {fold}] Loaded best checkpoint from: {ckpt_path} (best_mse={best_val_loaded:.6f})")
        else:
            raise RuntimeError(f"[Fold {fold}] best checkpoint not found: {ckpt_path}")

        if task_type == "classification":
            T, S, Y = predicting(
                args, model, device, task_type, loader_test,
                show_batch_pbar=args.show_batch_pbar
            )
            _, test_auc = compute_performance_classification(
                T=T, S=S, Y=Y, best_auc=best_val, file=csv_path, epoch="test"
            )
            print(f"Test AUC: {test_auc:.4f}")
        else:
            T, P, _ = predicting(
                args, model, device, task_type, loader_test,
                show_batch_pbar=args.show_batch_pbar
            )
            _, test_mse = compute_performance_regression(
                T, P, best_val, file=csv_path, epoch="test"
            )
            print(f"Test MSE: {test_mse:.4f}")

    if not args.only_fold:
        process_folds(args, out_dir, out_info)


def parse_args():
    p = argparse.ArgumentParser(description="FragC3 model training and evaluation")
    # Training / evaluation hyperparameters
    p.add_argument("--train_batch_size", type=int, default=512,
                   help="Training batch size")
    p.add_argument("--test_batch_size", type=int, default=512,
                   help="Test batch size")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate")
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs")
    p.add_argument("--early_stopping", type=int, default=20,
                   help="Early stopping patience")
    p.add_argument("--log_interval", type=int, default=20,
                   help="Logging interval (in number of batches)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed")
    p.add_argument("--folds", type=int, default=5,
                   help="Number of folds for cross-validation (default: 5)")
    p.add_argument("--only_fold", type=int, default=2,
                   help="Run a single specified fold only")
    p.add_argument("--groups", type=str, default="Drug",
                   choices=["Cell", "Drug", "none"],
                   help="Grouping strategy for data split")
    # Model configuration
    p.add_argument("--hidden", type=int, default=300,
                   help="Hidden dimension size")
    p.add_argument("--encoder", type=str, default="FragC3",
                   help="Encoder architecture")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate")
    p.add_argument("--frag_list", nargs="+", default=["brics", "fg", "murcko"],
                   help='Fragmentation views to use: "brics", "fg", "murcko"')
    p.add_argument("--frag_agg", type=str, default="cell_attn",
                   choices=["mlp", "gate", "cell_attn"],
                   help="Multi-view fragment aggregation mechanism")
    # C3Attention parameters
    p.add_argument("--use_C3Attn", type=bool, default=True,
                   help="Enable C3Attention-based bi-drug fragment encoding")
    p.add_argument("--tri_attn", type=bool, default=True,
                   help="Enable cell-conditioned tri-attention")
    p.add_argument("--tri_variant", type=str, default="scale_dot",
                   choices=["scale_dot", "add", "dot", "trilinear"],
                   help="Tri-attention interaction variant")
    p.add_argument("--cv_mode", type=str, default="bilinear",
                   choices=["mul", "add", "bilinear"],
                   help="Context–value interaction mode")
    p.add_argument("--tokenizer", type=str, default="conv",
                   choices=["conv", "linear"],
                   help="Cell-line tokenizer type")
    p.add_argument("--heads", type=int, default=2,
                   help="Number of attention heads")
    p.add_argument("--ffn_expansion", type=int, default=8,
                   help="Expansion ratio of the feed-forward network")
    p.add_argument("--cell_agg", type=int, default=256,
                   help="Hidden dimension for context-aware view aggregation")
    p.add_argument("--cell_pred", type=int, default=128,
                   help="Hidden dimension for prediction-specific cell embedding")
    p.add_argument("--Lc", type=int, default=32,
                   help="Number of cell-context tokens")
    # Dataset and device
    p.add_argument("--dataset", type=str, default="ONeil",
                   choices=["ALMANAC", "DrugComb", "ONeil"],
                   help="Dataset name (ONeil is treated as a binary classification task)")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="Computation device selection")
    # Optional: batch-level progress bar
    p.add_argument("--show_batch_pbar", type=bool, default=False,
                   help="Show batch-level tqdm progress bar")
    # Checkpoint saving
    p.add_argument("--ckpt_dir", type=str, default="saves",
                   help="Directory for saving model checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    main()
