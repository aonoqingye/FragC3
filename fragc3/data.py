"""Compact ordered-pair inputs for the three benchmark datasets."""

import csv, hashlib, json
from pathlib import Path
from types import SimpleNamespace
import dgl, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from .batching import BalancedBatchSampler

VIEWS = ("brics", "fg", "murcko")
FIELDS = ("graph1", "graph2", "graph1_fg", "graph2_fg", "graph1_murcko", "graph2_murcko")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class PairBatch:
    def __init__(self, rows):
        self.cell1 = torch.cat([r.cell1 for r in rows], 0)
        for field in FIELDS:
            setattr(self, field, [getattr(r, field) for r in rows])

    def to(self, device):
        result = SimpleNamespace(cell1=self.cell1.to(device))
        for field in FIELDS:
            setattr(result, field, [g.to(device) for g in getattr(self, field)])
        return result


class BenchmarkDataset(Dataset):
    def __init__(self, root, name, verify=True):
        self.name = name
        self.directory = Path(root) / name
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["dataset"] != name:
            raise ValueError("Dataset identity mismatch")
        if verify:
            for file, h in self.manifest["sha256"].items():
                if sha256(self.directory / file) != h:
                    raise ValueError("Dataset checksum mismatch: " + file)
        with (self.directory / "samples.csv").open(newline="") as f:
            self.metadata = list(csv.DictReader(f))
        if any(int(r["row_id"]) != i for i, r in enumerate(self.metadata)):
            raise ValueError("Sample rows must be contiguous and ordered")
        self.labels = torch.tensor([float(r["y"]) for r in self.metadata], dtype=torch.float32)
        self.drug_names = json.loads((self.directory / "drugs.json").read_text())
        graphs, _ = dgl.load_graphs(str(self.directory / "graphs.bin"))
        if len(graphs) != 3 * len(self.drug_names):
            raise ValueError("Three graph views are required per drug")
        self.graphs = {
            name: {v: graphs[3 * i + j] for j, v in enumerate(VIEWS)}
            for i, name in enumerate(self.drug_names)
        }
        with np.load(self.directory / "cells.npz", allow_pickle=False) as f:
            ids = f["ids"].tolist()
            matrix = f["features"].copy()
        self.cells = {str(k): torch.from_numpy(matrix[i]) for i, k in enumerate(ids)}
        if matrix.ndim != 2 or matrix.dtype != np.float32:
            raise ValueError("Cell features must be a float32 matrix")
        self.cell_dim = matrix.shape[1]
        if (
            self.cell_dim != self.manifest["cell_dim"]
            or not torch.isfinite(self.labels).all()
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("Invalid target/cell features")
        for row in self.metadata:
            if (
                row["drug_a"] not in self.graphs
                or row["drug_b"] not in self.graphs
                or row["cell"] not in self.cells
            ):
                raise ValueError("Unknown sample identity")
        if name == "ONeil" and set(self.labels.tolist()) != {0.0, 1.0}:
            raise ValueError("O’Neil targets must be binary")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):
        i = int(i)
        row = self.metadata[i]
        a = self.graphs[row["drug_a"]]
        b = self.graphs[row["drug_b"]]
        return SimpleNamespace(
            graph1=a["brics"],
            graph1_fg=a["fg"],
            graph1_murcko=a["murcko"],
            graph2=b["brics"],
            graph2_fg=b["fg"],
            graph2_murcko=b["murcko"],
            cell1=self.cells[row["cell"]].unsqueeze(0),
            y=self.labels[i : i + 1],
        )


class RowIndices(Dataset):
    def __init__(self, ids):
        self.ids = np.asarray(ids, dtype=np.int64)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return int(self.ids[i])


class Collator:
    def __init__(self, source):
        self.source = source

    def __call__(self, ids):
        rows = [self.source[int(i)] for i in ids]
        return {
            "input": PairBatch(rows),
            "y": torch.cat([r.y for r in rows]),
            "row_id": torch.tensor(ids, dtype=torch.long),
        }


def loader(source, ids, batch_size, generator, training=False):
    rows = RowIndices(ids)
    common = dict(generator=generator, num_workers=0, collate_fn=Collator(source))
    if training:
        return DataLoader(
            rows,
            batch_sampler=BalancedBatchSampler(
                RandomSampler(rows, generator=generator), batch_size
            ),
            **common,
        )
    return DataLoader(rows, batch_size=batch_size, shuffle=False, drop_last=False, **common)


def load_split(split_root, dataset, setting, fold, source):
    root = Path(split_root)
    relative = f"{dataset}/{setting}/split_{fold}.npz"
    entry = json.loads((root / "manifest.json").read_text())["splits"][relative]
    path = root / relative
    if sha256(path) != entry["sha256"]:
        raise ValueError("Split checksum mismatch")
    with np.load(path, allow_pickle=False) as f:
        parts = {k: f[k].copy() for k in f.files}
    if set(parts) != {"train", "valid", "test", "quarantine"}:
        raise ValueError("Invalid split fields")
    for x in parts.values():
        if x.ndim != 1 or x.dtype != np.int64 or len(np.unique(x)) != len(x):
            raise ValueError("Invalid partition indices")
    if not np.array_equal(
        np.sort(np.concatenate(list(parts.values()))), np.arange(len(source))
    ) or len(parts["quarantine"]):
        raise ValueError("Partitions must be disjoint and exhaustive")
    return parts
