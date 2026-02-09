import os, re
from typing import List, Tuple, Dict, Iterable, Optional
from rdkit import Chem
from rdkit.Chem import RDConfig

def _read_tsv_lines(path: str) -> List[List[str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("//"):
                continue
            parts = ln.split("\t")
            parts = [p.strip() for p in parts if p.strip()!=""]
            if parts:
                rows.append(parts)
    return rows

def load_functional_group_hierarchy(root: Optional[str]=None) -> List[Tuple[str,str]]:
    if root is None:
        root = RDConfig.RDDataDir
    path = os.path.join(root, "Functional_Group_Hierarchy.txt")
    out = []
    rows = _read_tsv_lines(path)
    for cols in rows:
        if len(cols) >= 2:
            name, smarts = cols[0], cols[1]
            out.append((name, smarts))
    return out

def load_functional_groups_flat(root: Optional[str]=None) -> List[Tuple[str,str]]:
    if root is None:
        root = RDConfig.RDDataDir
    path = os.path.join(root, "FunctionalGroups.txt")
    out = []
    rows = _read_tsv_lines(path)
    for cols in rows:
        if len(cols) >= 2:
            label, smarts = cols[0], cols[1]
            out.append((label, smarts))
    return out

def build_fg_smart_db(root: Optional[str]=None) -> List[Tuple[str, Chem.Mol]]:
    h = load_functional_group_hierarchy(root)
    f = load_functional_groups_flat(root)
    merged = h + f

    seen = set()
    uniq: List[Tuple[str,str]] = []
    for name, smarts in merged:
        key = (name, smarts)
        if key not in seen:
            seen.add(key); uniq.append(key)

    compiled: List[Tuple[str, Chem.Mol]] = []
    for name, smarts in uniq:
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            compiled.append((name, patt))
    compiled.sort(key=lambda x: -len(Chem.MolToSmarts(x[1]) or ""))
    return compiled
