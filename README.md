# FragC3

Official implementation of **FragC3**:  
**Fragment-Centric Drug Synergy Prediction with Cell-Conditioned Cross-Fragment Attention**.

FragC3 is a fragment-centric framework for drug synergy prediction. It decomposes each drug into chemically meaningful fragment views and models **cell-conditioned cross-drug fragment interactions** through a tri-attention mechanism. The repository provides the model implementation, training and evaluation entry points, and the Fragment-aware Synergy Explanation (FSE) utilities used for post-hoc qualitative interpretation.

## Installation

### 1\. Clone the repository

```bash
git clone https://github.com/aonoqingye/FragC3.git
cd FragC3
```

### 2\. Create an environment

A CUDA-enabled environment is needed for full experiments.

```bash
conda create -n fragc3 python=3.10 -y
conda activate fragc3
```

### 3\. Install dependencies

```bash
pip install -r requirements.txt
```

## Data preparation

Place dataset files under `datas/` using the following structure:

```text
datas/
├── DrugComb/
├── ONeil/
└── NCI-ALMANAC/
```

## Evaluation protocols

There are three evaluation settings:

|Protocol|Grouping option|Description|
|-|-|-|
|S0 warm-start|`--groups none`|Sample-level held-out split. Test triplets are unseen, but drugs and cell lines may appear in training.|
|S1 leave-drug-out|`--groups Drug`|Repeated drug-disjoint hold-out split. Held-out drug identities are selected before training; any sample containing a held-out drug is assigned to the test partition and excluded from training and validation.|
|S2 leave-cell-line-out|`--groups Cell`|Repeated cell-line-disjoint hold-out split. All samples from held-out test cell lines are absent from training and validation.|

## Training

The main entry point is `train.py`.

### Common commands

S0 warm-start:

```bash
python train.py \\
  --dataset DrugComb \\
  --groups none \\
  --folds 5 \\
  --train\_batch\_size 512 \\
  --test\_batch\_size 512 \\
  --lr 2e-4 \\
  --epochs 100 \\
  --early\_stopping 20
```

S1 leave-drug-out:

```bash
python train.py \\
  --dataset DrugComb \\
  --groups Drug \\
  --folds 5 \\
  --train\_batch\_size 512 \\
  --test\_batch\_size 512 \\
  --lr 2e-4 \\
  --epochs 100 \\
  --early\_stopping 20
```

S2 leave-cell-line-out:

```bash
python train.py \\
  --dataset DrugComb \\
  --groups Cell \\
  --folds 5 \\
  --train\_batch\_size 512 \\
  --test\_batch\_size 512 \\
  --lr 2e-4 \\
  --epochs 100 \\
  --early\_stopping 20
```

Run O'Neil classification:

```bash
python train.py --dataset ONeil --groups Drug
```

Run NCI-ALMANAC regression:

```bash
python train.py --dataset ALMANAC --groups Drug
```

Run a single fold:

```bash
python train.py --dataset ONeil --groups Drug --only_fold 1
```

## License

This project is released under the MIT License.
