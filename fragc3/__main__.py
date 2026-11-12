"""Command-line entry points for FragC3 reproduction."""

import argparse, csv, json, hashlib
from pathlib import Path
import numpy as np, torch
from .data import BenchmarkDataset, load_split, loader
from .engine import prediction_metrics, run_epoch, run_training, atomic_json, seed_everything

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["ONeil", "ALMANAC", "DrugComb"]


def main():
    p = argparse.ArgumentParser(prog="python -m fragc3")
    from . import __version__

    p.add_argument("--version", action="version", version="FragC3 " + __version__)
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=DATASETS, required=True)
    common.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    common.add_argument("--split-root", type=Path, default=ROOT / "data/splits")
    common.add_argument("--setting", choices=["S0", "S1", "S2"], required=True)
    common.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    tr = sub.add_parser("train", parents=[common])
    tr.add_argument("--output", type=Path, required=True)
    tr.add_argument("--device", default="cuda:0")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--config", type=Path)
    tr.add_argument("--resume", action="store_true")
    tr.add_argument("--pause-after", type=int)
    ev = sub.add_parser("evaluate", parents=[common])
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--partition", choices=["train", "valid", "test"], default="test")
    ev.add_argument("--device", default="cuda:0")
    ev.add_argument("--output", type=Path, required=True)
    ve = sub.add_parser("verify-data")
    ve.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    ve.add_argument("--split-root", type=Path, default=ROOT / "data/splits")
    ve.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    su = sub.add_parser("summarize")
    su.add_argument("--runs", type=Path, required=True)
    su.add_argument("--output", type=Path, required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--dataset", choices=DATASETS, required=True)
    pr.add_argument("--samples", type=Path, required=True)
    pr.add_argument("--smiles", type=Path, required=True)
    pr.add_argument("--cells", type=Path, required=True)
    pr.add_argument("--cell-feature-order", type=Path)
    pr.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.command == "prepare":
        from .prepare import prepare_dataset

        print(
            json.dumps(
                prepare_dataset(
                    a.dataset, a.samples, a.smiles, a.cells, a.output, a.cell_feature_order
                ),
                indent=2,
            )
        )
        return
    if a.command == "summarize":
        groups = {}
        for path in sorted(a.runs.rglob("metrics.json")):
            r = json.loads(path.read_text())
            if not {"dataset", "setting", "seed", "fold", "test"} <= set(r):
                continue
            key = (r["dataset"], r["setting"], r["seed"])
            groups.setdefault(key, []).append(r)
        rows = []
        for (dataset, setting, seed), rs in sorted(groups.items()):
            if sorted(r["fold"] for r in rs) != [1, 2, 3, 4, 5]:
                raise ValueError(
                    "Exactly one result for each of five folds is required: "
                    + str((dataset, setting, seed))
                )
            for metric in rs[0]["test"]:
                values = [r["test"][metric] for r in rs]
                if any(v is None for v in values):
                    continue
                factor = 100 if metric in ["AUROC", "PR_AUC_trapezoidal"] else 1
                values = np.asarray(values, dtype=np.float64) * factor
                rows.append(
                    {
                        "dataset": dataset,
                        "setting": setting,
                        "seed": seed,
                        "metric": metric + ("_percent" if factor == 100 else ""),
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)),
                        "n": 5,
                    }
                )
        if not rows:
            raise ValueError("No completed five-fold results")
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(a.output)
        return
    names = DATASETS if getattr(a, "dataset", None) == "all" else [a.dataset]
    if a.command == "verify-data":
        for name in names:
            source = BenchmarkDataset(a.data_root, name)
            split_count = 0
            for setting in ["S0", "S1", "S2"] if name == "DrugComb" else ["S1", "S2"]:
                for fold in range(1, 6):
                    load_split(a.split_root, name, setting, fold, source)
                    split_count += 1
            for by_view in source.graphs.values():
                for g in by_view.values():
                    for nt in g.ntypes:
                        for _, value in g.nodes[nt].data.items():
                            if not torch.isfinite(value).all():
                                raise ValueError("Nonfinite graph features")
                    for et in g.canonical_etypes:
                        for _, value in g.edges[et].data.items():
                            if not torch.isfinite(value).all():
                                raise ValueError("Nonfinite edge features")
            print(
                json.dumps(
                    {
                        "dataset": name,
                        "rows": len(source),
                        "drugs": len(source.graphs),
                        "cell_dim": source.cell_dim,
                        "splits_checked": split_count,
                        "status": "passed",
                    }
                ),
                flush=True,
            )
        return
    source = BenchmarkDataset(a.data_root, a.dataset)
    parts = load_split(a.split_root, a.dataset, a.setting, a.fold, source)
    if a.command == "train":
        config = json.loads((a.config or ROOT / "configs" / f"{a.dataset}.json").read_text())
        if config["dataset"] != a.dataset or config["model"]["cell_dim"] != source.cell_dim:
            raise ValueError("Configuration/data mismatch")
        print(
            json.dumps(
                run_training(
                    source,
                    parts,
                    config,
                    a.output,
                    a.setting,
                    a.fold,
                    a.seed,
                    a.device,
                    a.resume,
                    a.pause_after,
                ),
                indent=2,
            )
        )
        return
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    identity = ck["identity"]
    if (identity["dataset"], identity["setting"], identity["fold"]) != (
        a.dataset,
        a.setting,
        a.fold,
    ):
        raise ValueError("Checkpoint dataset/setting/fold mismatch")
    if identity["data_checksums"] != source.manifest["sha256"]:
        raise ValueError("Checkpoint data checksums differ")
    for partition, indices in parts.items():
        digest = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
        if identity["partitions"][partition] != digest:
            raise ValueError("Checkpoint partition indices differ")
    torch.set_num_threads(ck["config"]["training"]["cpu_threads"])
    seed_everything(identity["seed"])
    if torch.device(a.device).type == "cpu":
        torch.backends.mkldnn.enabled = False
    from .model import FragC3

    model = FragC3(**ck["config"]["model"]).to(a.device)
    model.load_state_dict(ck["model_state"])
    generator = torch.Generator().manual_seed(identity["seed"])
    report, pred = run_epoch(
        model,
        loader(source, parts[a.partition], ck["config"]["training"]["batch_size"], generator),
        parts[a.partition],
        torch.device(a.device),
        ck["config"]["task"] == "classification",
    )
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output / "predictions.npz", **pred)
    atomic_json(a.output / "metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
