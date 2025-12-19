import random
import csv
from rdkit import Chem
from rdkit import RDLogger

# Silence RDKit warnings/errors
RDLogger.DisableLog("rdApp.*")

CORE_FRAGMENTS = [
    "c1ccccc1",          # benzene
    "c1ccncc1",          # pyridine
    "c1ccccc1C(=O)",     # benzoyl
    "C1CCNCC1",          # piperazine
    "C1CCNCC1C",         # N-methylpiperazine
    "C1CCC(CC1)N",       # cyclohexylamine
]

SIDE_FRAGMENTS = [
    "C",                 # methyl
    "CC",                # ethyl
    "CO",                # methoxy
    "OC",                # hydroxy-methyl
    "N",                 # amino
    "CN",                # methylamino
    "C(=O)O",            # carboxylic acid
    "C(=O)N",            # amide
    "C(=O)NC",           # N-methylamide
    "OC(=O)C",           # ester
]

MAX_VALENCE = {
    6: 4,   # C
    7: 3,   # N
    8: 2,   # O
    9: 1,   # F
    16: 2,  # S
    17: 1,  # Cl
}

def can_add_single_bond(mol, atom_idx):
    mol.UpdatePropertyCache(strict=False)
    atom = mol.GetAtomWithIdx(atom_idx)
    Z = atom.GetAtomicNum()
    max_val = MAX_VALENCE.get(Z, 4)
    v = atom.GetExplicitValence()
    return (v + 1) <= max_val

def attach_side(mol, side_smiles, core_idx):
    side = Chem.MolFromSmiles(side_smiles)
    if side is None:
        return None

    if not can_add_single_bond(mol, core_idx):
        return None

    combo = Chem.CombineMols(mol, side)
    em = Chem.EditableMol(combo)

    offset = mol.GetNumAtoms()
    side_idx = 0
    em.AddBond(core_idx, offset + side_idx, order=Chem.BondType.SINGLE)

    new_mol = em.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
        return new_mol
    except Exception:
        return None

def generate_one(n_side_chains=1, max_attempts_per_chain=10):
    core_smiles = random.choice(CORE_FRAGMENTS)
    mol = Chem.MolFromSmiles(core_smiles)
    if mol is None:
        return None

    for _ in range(n_side_chains):
        attached = False
        for _ in range(max_attempts_per_chain):
            core_idx = random.randrange(mol.GetNumAtoms())
            new_mol = attach_side(mol, random.choice(SIDE_FRAGMENTS), core_idx)
            if new_mol is not None:
                mol = new_mol
                attached = True
                break
        if not attached:
            break

    try:
        smi = Chem.MolToSmiles(mol, canonical=True)
        if Chem.MolFromSmiles(smi) is None:
            return None
        return smi
    except Exception:
        return None

def random_smiles(n=20, n_side_chains_range=(1, 4)):
    out = []
    attempts = 0
    max_attempts = 50 * n
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        n_sc = random.randint(*n_side_chains_range)
        smi = generate_one(n_side_chains=n_sc)
        if smi is not None:
            out.append(smi)
    return out

def random_label(p_one=0.05):
    # 1 with probability p_one, else 0
    return 1 if random.random() < p_one else 0

def main(
    n_assay=1000,
    n_library=10000,
    assay_csv_path="assay_labeled.csv",
    library_csv_path="library_unlabeled.csv",
):
    assay_smiles = random_smiles(n=n_assay)
    library_smiles = random_smiles(n=n_library)

    # write assay_labeled.csv
    with open(assay_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "Target1", "Target2"])
        for smi in assay_smiles:
            t1 = random_label(0.05)
            t2 = random_label(0.05)
            writer.writerow([smi, t1, t2])

    # write library_unlabeled.csv
    with open(library_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        for smi in library_smiles:
            writer.writerow([smi])

    print(f"Wrote {len(assay_smiles)} rows to {assay_csv_path}")
    print(f"Wrote {len(library_smiles)} rows to {library_csv_path}")

if __name__ == "__main__":
    main(
        n_assay=1000,
        n_library=10000,
        assay_csv_path="assay_labeled.csv",
        library_csv_path="library_unlabeled.csv",
    )