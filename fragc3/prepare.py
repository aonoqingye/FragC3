"""Prepare native three-view graph inputs from tabular benchmark inputs."""

import csv, json
from pathlib import Path
import dgl, numpy as np
from .chemistry import graphs_from_smiles
from .data import sha256


def prepare_dataset(name, samples, smiles, cells, output, feature_order=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    with Path(samples).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"row_id", "drug_a", "drug_b", "cell", "y"} <= set(rows[0]):
        raise ValueError("Expected sample columns: row_id,drug_a,drug_b,cell,y")
    if any(int(r["row_id"]) != i for i, r in enumerate(rows)):
        raise ValueError("Samples must retain contiguous original row order")
    with Path(smiles).open(newline="") as f:
        sr = list(csv.DictReader(f))
    molecules = {r["drug_id"]: r["smiles"] for r in sr}
    if len(molecules) != len(sr):
        raise ValueError("Drug IDs must have unique SMILES rows")
    drugs = list(molecules)
    graphs = []
    for drug in drugs:
        by_view = graphs_from_smiles(molecules[drug], partial_sanitization=name == "ALMANAC")
        graphs.extend(by_view[v] for v in ["brics", "fg", "murcko"])
    with np.load(cells, allow_pickle=False) as f:
        cell_ids = f["ids"].copy()
        matrix = f["features"].copy()
    if (
        matrix.ndim != 2
        or len(cell_ids) != len(matrix)
        or len(set(cell_ids.tolist())) != len(cell_ids)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Invalid cells.npz arrays")
    if matrix.dtype != np.float32:
        raise ValueError("Cell features must be float32")
    if any(
        r["drug_a"] not in molecules
        or r["drug_b"] not in molecules
        or r["cell"] not in set(cell_ids.tolist())
        for r in rows
    ):
        raise ValueError("Unknown sample drug or cell")
    dgl.save_graphs(str(output / "graphs.bin"), graphs)
    (output / "drugs.json").write_text(json.dumps(drugs, ensure_ascii=False))
    np.savez_compressed(output / "cells.npz", ids=cell_ids.astype(str), features=matrix)
    with (output / "samples.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["row_id", "drug_a", "drug_b", "cell", "y"])
        wr.writeheader()
        wr.writerows({k: r[k] for k in wr.fieldnames} for r in rows)
    with (output / "smiles.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["drug_id", "smiles"])
        wr.writeheader()
        wr.writerows(sr)
    if feature_order:
        import shutil

        shutil.copyfile(feature_order, output / "cell_feature_order.tsv")
    manifest = {
        "schema_version": 1,
        "dataset": name,
        "rows": len(rows),
        "drugs": len(drugs),
        "cells": len(matrix),
        "cell_dim": matrix.shape[1],
        "views": ["brics", "fg", "murcko"],
        "atom_features": 45,
        "fragment_features": 194,
        "bond_features": 14,
        "fragment_edge_features": 34,
        "sha256": {
            p.name: sha256(p) for p in sorted(output.iterdir()) if p.name != "manifest.json"
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
