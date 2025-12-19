# ChemScreener

Code for the manuscript "ChemScreener: an Active Learning Enabled Hit Discovery Workflow with WDR5 Inhibitor Case Study" by Lingling Shen, Jian Fang, Lulu Liu, Rena Wang, Jeremy L. Jenkins, He Wang

ChemScreener is an active learning pipeline for small-molecule screening. In each cycle, it trains an ensemble of multi-task Chemprop models, scores an unlabeled library, and selects compounds to acquire based on different acquisition strategies.

## Acknowledgement
This project is based on
Chemprop https://github.com/chemprop/chemprop and
traversing_chem_space https://github.com/molML/traversing_chem_space

## Installation
### Create and activate a Conda environment
```bash
conda create -n ChemScreener python=3.11
conda activate ChemScreener
```

### Install the pytorch, lightning and chemprop v2
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install lightning==2.4.0
pip install chemprop==2.1.0
```
## Usage
```bash
python main.py \
  --assay_csv assay_labeled.csv \
  --test_csv library_unlabeled.csv \
  --smiles_column smiles \
  --target_columns Target1 Target2 \
  --output_dir results \
  --n_ensemble 10 \
  --batch_size 256 \
  --max_epochs 100 \
  --mode Balanced_Ranking \
  --n_acquire 300 \
  --accelerator auto \
  --devices 1 \
  --seed 42 \
  --run_id 0
```
You can use generate_random_toy_data.py to generate random test data for the code.

### Notes:
Input Data:
- Assay CSV (labeled): must contain a SMILES column and one or more binary target columns. 
- Test CSV (unlabeled library): must contain at least the SMILES column. If target columns exist, they are ignored for acquisition.
- smiles_column: name of the SMILES column (default: smiles)
- target_columns: one or more binary target columns (e.g., Active, T1, T2)

Optional split mapping (in the assay CSV):
- split_col: name of the split column (e.g., split)
- split_train: value used to mark training rows (e.g., train)
- split_val: value used to mark validation rows (e.g., val)


## Citation

ChemScreener: an Active Learning Enabled Hit Discovery Workflow with WDR5 Inhibitor Case Study. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-0c4mm


## Contact
For questions or support, please open an issue or contact the project maintainer.