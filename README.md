# FragC3

Official implementation of **FragC3**:  
*Fragment-Centric Drug Synergy Prediction with Cell-Conditioned Cross-Fragment Attention.*

## Overview
FragC3 models cell-conditioned cross-drug fragment interactions via a tri-attention mechanism,
providing accurate and interpretable drug synergy predictions.

## Requirements
See `requirements.txt`.

## Data
Place the dataset files in their respective directories under the datas/ folder (DrugComb, ONeil, NCI-ALMANAC).

## Training
```bash
python train.py --dataset DrugComb --groups Drug
