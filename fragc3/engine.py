"""Training and validation-selected evaluation for FragC3."""

import copy, json, math, os, random, time
from pathlib import Path
import numpy as np, torch
from scipy import stats
from sklearn import metrics as skm
from .model import FragC3
from .stopping import EarlyStopping
from .data import loader, sha256


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def capture_rng(generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "loader_generator": generator.get_state(),
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    generator.set_state(state["loader_generator"])
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same number of visible CUDA devices")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def prediction_metrics(truth, scores, classification):
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != s.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(s).all():
        raise ValueError("Invalid targets or predictions")
    if classification:
        if set(np.unique(y)) != {0.0, 1.0}:
            raise ValueError("Classification metrics require both classes")
        hard = (s > 0.5).astype(int)
        precision, recall, _ = skm.precision_recall_curve(y, s)
        return {
            "AUROC": float(skm.roc_auc_score(y, s)),
            "AP": float(skm.average_precision_score(y, s)),
            "PR_AUC_trapezoidal": float(skm.auc(recall, precision)),
            "accuracy": float(skm.accuracy_score(y, hard)),
            "balanced_accuracy": float(skm.balanced_accuracy_score(y, hard)),
            "precision": float(skm.precision_score(y, hard, zero_division=0)),
            "recall": float(skm.recall_score(y, hard, zero_division=0)),
            "F1": float(skm.f1_score(y, hard, zero_division=0)),
        }
    mse = float(np.mean((y - s) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    variable = len(y) > 1 and np.ptp(y) > 0 and np.ptp(s) > 0
    return {
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "MAE": float(np.mean(np.abs(y - s))),
        "Pearson": float(stats.pearsonr(y, s).statistic) if variable else None,
        "Spearman": float(stats.spearmanr(y, s).statistic) if variable else None,
        "R2": float(1 - np.sum((y - s) ** 2) / total) if np.ptp(y) > 0 and total > 0 else None,
    }


def run_epoch(model, batches, expected_ids, device, classification, optimizer=None):
    model.train(optimizer is not None)
    loss_fn = torch.nn.CrossEntropyLoss() if classification else torch.nn.MSELoss()
    rows = []
    truth = []
    scores = []
    logits = []
    weighted_loss = 0.0
    steps = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for pack in batches:
        ids = pack["row_id"].reshape(-1).tolist()
        x = pack["input"].to(device)
        y = pack["y"].to(device).reshape(-1)
        target = y.long() if classification else y.float().reshape(-1, 1)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(x)
            if output.shape != (len(ids), 2 if classification else 1):
                raise ValueError("Unexpected prediction shape")
            loss = loss_fn(output, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            if optimizer is not None:
                loss.backward()
                parameters = [p for p in model.parameters() if p.grad is not None]
                if not parameters:
                    raise ValueError("No connected model gradients")
                torch.nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
                optimizer.step()
        weighted_loss += float(loss.detach().cpu()) * len(ids)
        score = (
            torch.softmax(output.detach(), dim=1)[:, 1]
            if classification
            else output.detach().reshape(-1)
        )
        rows.extend(ids)
        truth.extend(y.cpu().tolist())
        scores.extend(score.cpu().tolist())
        logits.extend(output.detach().cpu().tolist())
        steps += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if (
        len(rows) != len(expected_ids)
        or len(set(rows)) != len(rows)
        or set(rows) != set(int(i) for i in expected_ids)
    ):
        raise ValueError("Incomplete or repeated epoch rows")
    return {
        "rows": len(rows),
        "steps": steps,
        "seconds": time.perf_counter() - started,
        "loss": weighted_loss / len(rows),
        "metrics": prediction_metrics(truth, scores, classification),
    }, {
        "row_id": np.asarray(rows, dtype=np.int64),
        "y_true": np.asarray(truth),
        "y_score": np.asarray(scores),
        "logits": np.asarray(logits),
    }


def create_model(config, source, train_ids):
    model = FragC3(**config["model"])
    if config["initialize_output_bias_from_training_mean"]:
        if config["task"] != "regression":
            raise ValueError("Training-mean bias is for regression")
        mean = float(source.labels[np.asarray(train_ids, dtype=np.int64)].to(torch.float64).mean())
        with torch.no_grad():
            model.pred[-1].bias.fill_(mean)
    return model


def run_training(
    source,
    parts,
    config,
    output,
    setting,
    fold,
    seed=0,
    device="cuda:0",
    resume=False,
    pause_after=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; select --device cpu for a CPU run")
    if device.type == "cpu":
        torch.backends.mkldnn.enabled = False
    t = config["training"]
    torch.set_num_threads(t["cpu_threads"])
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    model = create_model(config, source, parts["train"]).to(device)
    inner = torch.optim.Adam(
        model.parameters(),
        lr=t["lr"],
        weight_decay=t["weight_decay"],
        betas=tuple(t["optimizer_betas"]),
        eps=t["optimizer_eps"],
        amsgrad=t["optimizer_amsgrad"],
    )
    classification = config["task"] == "classification"
    metric = "AUROC" if classification else "RMSE"
    mode = "max" if classification else "min"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        inner,
        mode=mode,
        factor=t["lr_plateau_factor"],
        patience=t["lr_plateau_patience"],
        threshold=t["early_stop_min_delta"],
        threshold_mode="abs",
        min_lr=t["min_lr"],
    )
    stopping = EarlyStopping(mode, t["patience"], t["early_stop_min_delta"], t["lr_grace_epochs"])
    partition_hash = {
        k: __import__("hashlib").sha256(np.asarray(v, dtype="<i8").tobytes()).hexdigest()
        for k, v in parts.items()
    }
    identity = {
        "dataset": source.name,
        "setting": setting,
        "fold": fold,
        "seed": seed,
        "config": config,
        "partitions": partition_hash,
        "data_checksums": source.manifest["sha256"],
    }
    last = output / "last.pt"
    history = []
    best_state = None
    start = 1
    if (output / "metrics.json").exists():
        raise FileExistsError("Completed run exists; choose a new output directory")
    if resume:
        if not last.exists():
            raise FileNotFoundError("No last.pt checkpoint to resume")
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck["identity"] != identity:
            raise ValueError("Resume data/configuration/partition identity changed")
        model.load_state_dict(ck["model_state"])
        inner.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        stopping.load_state_dict(ck["stopping_state"])
        history = ck["history"]
        best_state = ck["best_model_state"]
        start = ck["epoch"] + 1
        restore_rng(ck["rng"], generator)
    elif last.exists() or (output / "best.pt").exists():
        raise FileExistsError("Existing checkpoint; pass --resume or select a new output directory")
    atomic_json(output / "configuration.json", identity)
    train_batches = loader(source, parts["train"], t["batch_size"], generator, True)
    valid_batches = loader(source, parts["valid"], t["batch_size"], generator)
    for epoch in range(start, t["max_epochs"] + 1):
        if stopping.should_stop():
            break
        before = [g["lr"] for g in inner.param_groups]
        train, _ = run_epoch(model, train_batches, parts["train"], device, classification, inner)
        valid, _ = run_epoch(model, valid_batches, parts["valid"], device, classification)
        score = valid["metrics"][metric]
        scheduler.step(score)
        transition = stopping.observe(score, epoch, before, [g["lr"] for g in inner.param_groups])
        if transition["best_improved"]:
            best_state = cpu_state(model)
            atomic_torch(
                output / "best.pt",
                {
                    "model_state": best_state,
                    "config": config,
                    "epoch": stopping.best_epoch,
                    "validation_score": stopping.best_score,
                    "identity": identity,
                },
            )
        history.append(
            {
                "epoch": epoch,
                "train": train,
                "valid": valid,
                "best_epoch": stopping.best_epoch,
                "best_score": stopping.best_score,
                "learning_rate": before[0],
                "next_learning_rate": inner.param_groups[0]["lr"],
                "transition": transition,
            }
        )
        atomic_torch(
            last,
            {
                "identity": identity,
                "epoch": epoch,
                "model_state": cpu_state(model),
                "optimizer_state": inner.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "stopping_state": stopping.state_dict(),
                "best_model_state": best_state,
                "rng": capture_rng(generator),
                "history": history,
            },
        )
        atomic_json(output / "history.json", history)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "valid_" + metric: score,
                    "best_epoch": stopping.best_epoch,
                    "bad_epochs": stopping.bad_epochs,
                }
            ),
            flush=True,
        )
        if (
            pause_after is not None
            and epoch >= pause_after
            and not stopping.should_stop()
            and epoch < t["max_epochs"]
        ):
            atomic_json(output / "status.json", {"status": "paused", "completed_epoch": epoch})
            return {"status": "paused", "epoch": epoch}
    if best_state is None or not history:
        raise ValueError("No validation-selected checkpoint")
    model.load_state_dict(best_state)
    atomic_torch(
        output / "best.pt",
        {
            "model_state": best_state,
            "config": config,
            "epoch": stopping.best_epoch,
            "validation_score": stopping.best_score,
            "identity": identity,
        },
    )
    reports = {}
    for partition in ["train", "valid", "test"]:
        report, pred = run_epoch(
            model,
            loader(source, parts[partition], t["batch_size"], generator),
            parts[partition],
            device,
            classification,
        )
        reports[partition] = report
        np.savez_compressed(output / ("predictions_" + partition + ".npz"), **pred)
    result = {
        "dataset": source.name,
        "setting": setting,
        "fold": fold,
        "seed": seed,
        "best_epoch": stopping.best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "stop_reason": "patience" if stopping.should_stop() else "max_epochs",
        "best_validation_score": stopping.best_score,
        "test": reports["test"]["metrics"],
        "partitions": reports,
        "training_seconds": sum(r["train"]["seconds"] + r["valid"]["seconds"] for r in history),
    }
    atomic_json(output / "metrics.json", result)
    atomic_json(output / "status.json", {"status": "completed"})
    return result
