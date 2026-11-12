"""Synthetic end-to-end fixture; creates no benchmark results."""

import csv, json, tempfile, unittest, subprocess, sys
from pathlib import Path
import numpy as np, torch, dgl
from fragc3.chemistry import graphs_from_smiles
from fragc3.data import BenchmarkDataset, sha256
from fragc3.engine import run_training


class Pipeline(unittest.TestCase):
    def test_data_training_evaluation_and_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="fragc3-test-") as folder:
            root = Path(folder)
            data = root / "data/ONeil"
            data.mkdir(parents=True)
            molecules = {"a": "CCO", "b": "CC(=O)O"}
            graphs = []
            for smi in molecules.values():
                views = graphs_from_smiles(smi)
                graphs.extend(views[v] for v in ["brics", "fg", "murcko"])
            dgl.save_graphs(str(data / "graphs.bin"), graphs)
            (data / "drugs.json").write_text(json.dumps(list(molecules)))
            np.savez_compressed(
                data / "cells.npz",
                ids=np.array(["x", "y"]),
                features=np.random.default_rng(0).normal(size=(2, 954)).astype(np.float32),
            )
            with (data / "samples.csv").open("w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["row_id", "drug_a", "drug_b", "cell", "y"])
                for i in range(8):
                    wr.writerow([i, "a", "b", "x" if i % 2 == 0 else "y", i % 2])
            manifest = {
                "dataset": "ONeil",
                "cell_dim": 954,
                "sha256": {p.name: sha256(p) for p in data.iterdir()},
            }
            (data / "manifest.json").write_text(json.dumps(manifest))
            source = BenchmarkDataset(root / "data", "ONeil")
            parts = {
                "train": np.arange(4, dtype=np.int64),
                "valid": np.arange(4, 6, dtype=np.int64),
                "test": np.arange(6, 8, dtype=np.int64),
                "quarantine": np.array([], dtype=np.int64),
            }
            config = json.loads(
                (Path(__file__).resolve().parents[1] / "configs/ONeil.json").read_text()
            )
            config["model"].update(hid_dim=48, ffn_expansion=2, cell_agg=16, cell_pred=16, Lc=8)
            config["training"].update(batch_size=4, max_epochs=2, cpu_threads=1)
            result = run_training(source, parts, config, root / "continuous", "S1", 1, 0, "cpu")
            pause = run_training(
                source, parts, config, root / "resumed", "S1", 1, 0, "cpu", pause_after=1
            )
            self.assertEqual(pause["status"], "paused")
            resumed = run_training(
                source, parts, config, root / "resumed", "S1", 1, 0, "cpu", resume=True
            )
            self.assertEqual(result["test"], resumed["test"])
            a = torch.load(root / "continuous/last.pt", map_location="cpu", weights_only=False)
            b = torch.load(root / "resumed/last.pt", map_location="cpu", weights_only=False)
            self.assertTrue(
                all(torch.equal(v, b["model_state"][k]) for k, v in a["model_state"].items())
            )
            for name in ["train", "valid", "test"]:
                with np.load(root / "continuous" / ("predictions_" + name + ".npz")) as x, np.load(
                    root / "resumed" / ("predictions_" + name + ".npz")
                ) as y:
                    self.assertTrue(all(np.array_equal(x[k], y[k]) for k in x.files))
            split = root / "splits/ONeil/S1/split_1.npz"
            split.parent.mkdir(parents=True)
            np.savez_compressed(split, **parts)
            (root / "splits/manifest.json").write_text(
                json.dumps({"splits": {"ONeil/S1/split_1.npz": {"sha256": sha256(split)}}})
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fragc3",
                    "evaluate",
                    "--dataset",
                    "ONeil",
                    "--setting",
                    "S1",
                    "--fold",
                    "1",
                    "--data-root",
                    str(root / "data"),
                    "--split-root",
                    str(root / "splits"),
                    "--checkpoint",
                    str(root / "continuous/best.pt"),
                    "--device",
                    "cpu",
                    "--output",
                    str(root / "evaluation"),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            with np.load(root / "continuous/predictions_test.npz") as x, np.load(
                root / "evaluation/predictions.npz"
            ) as y:
                self.assertTrue(np.array_equal(x["y_score"], y["y_score"]))
            with self.assertRaises(FileExistsError):
                run_training(source, parts, config, root / "continuous", "S1", 1, 0, "cpu")


if __name__ == "__main__":
    unittest.main()
