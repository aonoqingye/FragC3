# Benchmark inputs

## Sources

- **O'Neil:** O'Neil et al., *An Unbiased Oncology Compound Screen to Identify Novel Combination Strategies*, Molecular Cancer Therapeutics (2016). The binary benchmark and cell profiles follow the public [MultiSyn resources](https://github.com/HuazeLoong-Research/MultiSyn). The 954-coordinate CCLE expression source is also available from [DeepDDS](https://github.com/Sinwang404/DeepDDS). The NCIH23 cell vector uses the matching NCIH23_LUNG profile. The supplied feature-order file retains the coordinates used by the model.
- **NCI-ALMANAC:** [NCI ALMANAC](https://dtp.cancer.gov/ncialmanac/) provides the combination measurements. The processed ALMANAC–COSMIC benchmark uses ComboScore targets and 651 Cancer Gene Census expression coordinates. Feature identifiers are supplied in `cell_feature_order.tsv`.
- **DrugComb:** [DrugComb](https://drugcomb.org/) and the [Therapeutics Data Commons drug-combination benchmark](https://tdcommons.ai/multi_pred_tasks/drugsyn/) provide the benchmark context. The supplied CSS targets and cell profiles retain the processed benchmark coordinates: 23,808 RNA-seq, 3,171 protein and 627 microRNA values.

The supplied inputs are benchmark-derived research data. Their source terms remain applicable; the repository's software license does not replace the licenses or attribution requirements of the source datasets.

## Processed layout

Each dataset directory contains:

- `graphs.bin`: native DGL heterogeneous graphs; three consecutive graphs per drug in BRICS, EFG and Murcko order.
- `drugs.json`: drug identifiers in graph-file order.
- `cells.npz`: string array `ids` and float32 matrix `features`; one row per cell.
- `samples.csv`: `row_id,drug_a,drug_b,cell,y` in the benchmark's original row order.
- `smiles.csv`: `drug_id,smiles` used to construct the molecular graphs.
- `cell_feature_order.tsv`: cell-profile coordinate identifiers and provenance.
- `manifest.json`: dimensions, counts and SHA-256 checksums.

Graph node types are `a` (atoms) and `p` (fragments). Atom and fragment features have widths 45 and 194; chemical bond and fragment-relation features have widths 14 and 34. Membership edges connect each fragment to its constituent atoms. Node junction features concatenate the atom and fragment feature spaces.

The packaged graphs preserve source atom order, directed bond order, fragment membership and descriptor values. NCI-ALMANAC parses source molecules without assigning the additional stereochemistry introduced by full SMILES sanitization. O'Neil and DrugComb use full SMILES parsing. The same conventions are used by `python -m fragc3 prepare`.

The model applies L2 normalization to each cell vector during its forward pass. It does not treat neighboring feature coordinates as genomic adjacency. Labels are not standardized or clipped. Reverse-order observations remain in their recorded slots.

## Partitions

The code includes 35 NPZ files: ten each for O'Neil and ALMANAC, and fifteen for DrugComb. Each file contains int64 arrays `train`, `valid`, `test` and an empty `quarantine` array. The arrays are disjoint and collectively cover the sample rows. `data/splits/manifest.json` binds their checksums and counts.

## Preparing a dataset

Prepare the sample, SMILES and cell files in the formats above. Drug IDs and cell IDs must match exactly. Preserve row and feature order for benchmark reproduction. Pass the dataset name to select its molecular parsing convention, and write to an empty output directory. The three supplied configurations assume the cell dimensions and prediction endpoints listed in the README.
