"""BRICS, extended functional-group and Murcko graph construction."""

import os, logging
import numpy as np
import torch, dgl
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import MACCSkeys, AllChem, ChemicalFeatures
from rdkit.Chem.BRICS import FindBRICSBonds
from rdkit.Chem.Scaffolds import MurckoScaffold
from .efgs import get_dec_fgs

logger = logging.getLogger(__name__)
factory = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)
PHARM_TYPES = [i.split(".")[1] for i in factory.GetFeatureDefs().keys()]
PHARM_TYPES_LEN = len(PHARM_TYPES)
PHARM_FDIM = 167 + PHARM_TYPES_LEN
ELEMENTS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]

ATOM_FEATURES = {
    "atomic_num": ELEMENTS,
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],  # Formal charge of the atom
    "chiral_tag": [0, 1, 2, 3],  # Stereo labels of atoms
    "num_Hs": [0, 1, 2, 3, 4],  # Number of hydrogen atoms attached to the atom
    "hybridization": [  # Hybridization of atoms
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}


def _clean_mol(mol: Chem.Mol) -> Chem.Mol:
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol, sanitize=False)
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
    except Exception as e:
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            smi = "<MolToSmiles failed>"
        logger.warning(f"Clean mol failed, fallback continue. SMILES={smi} ; err={e}")
    return mol


def _safe_fragment_mol_from_atoms(mol: Chem.Mol, atom_idx_tuple) -> Chem.Mol:
    try:
        frag_smi = Chem.MolFragmentToSmiles(
            mol, atom_idx_tuple, isomericSmiles=True, kekuleSmiles=False
        )
        if not frag_smi:
            return None
        sub = Chem.MolFromSmiles(frag_smi, sanitize=False)
        if sub is None:
            return None
        Chem.SanitizeMol(
            sub, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
        sub = Chem.RemoveHs(sub, sanitize=False)
        return sub
    except Exception:
        return None


def bond_features(bond: Chem.rdchem.Bond):
    """Extract the feature vector of chemical bonds in RDKit."""
    if bond is None:
        raise ValueError(f"bond is None : {bond}")
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,  # single bond
            bt == Chem.rdchem.BondType.DOUBLE,  # double bond
            bt == Chem.rdchem.BondType.TRIPLE,  # triple bond
            bt == Chem.rdchem.BondType.AROMATIC,  # aromatic bond
            (bond.GetIsConjugated() if bt is not None else 0),  # conjugated bond
            (bond.IsInRing() if bt is not None else 0),  # ring
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def pharm_property_types_feats(mol, factory=factory):
    """Generate a binary vector indicating the presence of different pharmacophore property types in a given molecule using the RDKit feature factory."""
    types = [i.split(".")[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)] if mol is not None else []
    result = [0] * len(types)
    sfeats = set(feats)
    for i in range(len(types)):
        if types[i] in sfeats:
            result[i] = 1
    return result


def GetBricsBonds(mol):
    """
    Identify BRICS bonds in a molecule and assign corresponding BRICS rule-based features:
    1. A list of directed BRICS bond pairs (both directions),
    2. A list of BRICS bond rule-based features for each bond direction.
    """
    brics_bonds_feats = list()
    bonds_tmp = FindBRICSBonds(mol)
    bonds = [b for b in bonds_tmp]

    # Convert each bond to bidirectional mapping with BRICS features
    for item in bonds:  # item[0] is a key, item[1] is a BRICS type
        brics_bonds_feats.append(
            [
                [int(item[0][0]), int(item[0][1])],
                GetBricsBondFeature([item[1][0], item[1][1]]),
            ]
        )
        brics_bonds_feats.append(
            [
                [int(item[0][1]), int(item[0][0])],
                GetBricsBondFeature([item[1][1], item[1][0]]),
            ]
        )
    return brics_bonds_feats


def GetBricsBondFeature(action):
    """Generate a one-hot feature vector for a BRICS bond."""
    start_action_bond = int(action[0]) if (action[0] != "7a" and action[0] != "7b") else 7
    end_action_bond = int(action[1]) if (action[1] != "7a" and action[1] != "7b") else 7
    emb_0 = [0 for _ in range(17)]
    emb_1 = [0 for _ in range(17)]
    emb_0[start_action_bond] = 1
    emb_1[end_action_bond] = 1
    return emb_0 + emb_1


def maccskeys_emb(mol):
    """MACCS features after explicit ring-information initialization."""
    if mol is None:
        raise ValueError("Missing fragment molecule")
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    return list(MACCSkeys.GenMACCSKeys(mol))


def mol_with_atom_index(mol):
    """Set the mapping number of the atom according to the index of the atom"""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return mol


def GetFragmentFeats(mol):
    """Break bonds and split molecules based on BRICS rules, extract fragment features and atom-fragment mapping"""
    # i = ((20, 19), ('1', '3')) item[0] is a key, item[1] is a BRICS type
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0], i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    if not break_bonds:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol, break_bonds, addDummies=False)  # Cut into segments

    frags_idx_lst = Chem.GetMolFrags(tmp)  # Extract fragments

    # Initialize dictionary to store (mappings of atoms to fragments) and (fragment attributes)
    atom2frag = {}
    frag_feat = {}
    frag_id = 0

    # Iterate through the fragments
    for frag_idx in frags_idx_lst:
        for atom_id in frag_idx:
            atom2frag[atom_id] = frag_id
        try:
            mol_pharm = _safe_fragment_mol_from_atoms(mol, frag_idx)
            emb_0 = maccskeys_emb(mol_pharm)
            try:
                emb_1 = pharm_property_types_feats(mol_pharm if mol_pharm is not None else mol)
            except Exception:
                emb_1 = [0] * PHARM_TYPES_LEN
        except Exception as e:
            try:
                parent = Chem.MolToSmiles(mol, isomericSmiles=True)
                frag_smi = Chem.MolFragmentToSmiles(mol, frag_idx)
            except Exception:
                parent, frag_smi = "<MolToSmiles failed>", "<FragSMI failed>"
            logger.error(f"BRICS Feature gen failed; parent={parent} frag={frag_smi} ; err={e}")
            emb_0 = [0 for _ in range(167)]
            emb_1 = [0 for _ in range(PHARM_TYPES_LEN)]

        v = emb_0 + emb_1
        # ensure consistent dim
        if len(v) != PHARM_FDIM:
            v = (v + [0] * PHARM_FDIM)[:PHARM_FDIM]
        frag_feat[frag_id] = v
        frag_id += 1

    # Returns the mapping between atoms and fragments, fragment features
    return atom2frag, frag_feat


def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def atom_features(atom: Chem.rdchem.Atom):
    features = (
        onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
        + onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
        + onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES["chiral_tag"])
        + onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES["num_Hs"])
        + onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES["hybridization"])
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )  # scaled to about the same range as other features
    return features


def Mol2HeteroGraph(mol):
    """
    Converts an RDKit molecule object to a heterogeneous graph.
    The graph contains four edge types: interatomic bonds, interfragment connections, and cross-connections between atoms and fragments.
    Returns the constructed DGL heterogeneous graph object.
    """
    edge_types = [("a", "b", "a"), ("p", "r", "p"), ("a", "c", "p"), ("p", "c", "a")]
    edges = {k: [] for k in edge_types}

    atom2frag, frag_feat = GetFragmentFeats(mol)
    brics_bonds_feats = GetBricsBonds(mol)
    # BRICS bond pairs and their related feature information.
    # brics_bonds_feats = [[start atom index, end atom index], BRICS bond pair feature information].

    # atom-level
    for bond in mol.GetBonds():
        edges[("a", "b", "a")].append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges[("a", "b", "a")].append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

    # pharm-level
    for (s, t), _ in brics_bonds_feats:
        if s in atom2frag and t in atom2frag and atom2frag[s] != atom2frag[t]:
            edges[("p", "r", "p")].append([atom2frag[s], atom2frag[t]])
            edges[("p", "r", "p")].append([atom2frag[t], atom2frag[s]])

    # junction-level
    for k, v in atom2frag.items():  # atom_id,pharm_id
        edges[("a", "c", "p")].append([k, v])
        edges[("p", "c", "a")].append([v, k])

    g = dgl.heterograph(edges)

    # add atomic node features
    f_atom = []
    for idx in g.nodes("a"):
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(atom_features(atom))
    f_atom = torch.FloatTensor(f_atom)
    g.nodes["a"].data["f"] = f_atom
    dim_atom = len(f_atom[0]) if f_atom is not None else 0

    # add fragment node features
    f_frag = []
    for _, f in frag_feat.items():
        f_frag.append(f)
    g.nodes["p"].data["f"] = (
        torch.FloatTensor(f_frag) if f_frag else torch.zeros((0, PHARM_FDIM), dtype=torch.float32)
    )
    dim_frag = len(f_frag[0]) if f_frag else PHARM_FDIM

    num_atom = g.nodes["a"].data["f"].size()[0]
    num_frag = g.nodes["p"].data["f"].size()[0]

    # junction
    fa = g.nodes["a"].data["f"]
    fp = g.nodes["p"].data["f"]
    na = fa.shape[0]
    np_ = fp.shape[0]
    g.nodes["a"].data["f_junc"] = (
        torch.cat([fa, torch.zeros((na, fp.shape[1]), dtype=fa.dtype)], dim=1)
        if fp.numel() > 0
        else torch.cat([fa, torch.zeros((na, 0), dtype=fa.dtype)], dim=1)
    )
    g.nodes["p"].data["f_junc"] = (
        torch.cat([torch.zeros((np_, fa.shape[1]), dtype=fa.dtype), fp], dim=1)
        if np_ > 0
        else torch.zeros((0, fa.shape[1]), dtype=torch.float32)
    )

    # add atomic level edge features (type of bond)
    f_bond = []
    src, dst = g.edges(etype=("a", "b", "a"))  # beginnode, endnode
    for i in range(g.num_edges(etype=("a", "b", "a"))):
        f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(), dst[i].item())))
    g.edges[("a", "b", "a")].data["x"] = (
        torch.FloatTensor(f_bond) if len(f_bond) > 0 else torch.zeros((0, 13), dtype=torch.float32)
    )

    # add segment-level edge features (BRICS reaction pair information)
    f_reac = []
    src, dst = g.edges(etype=("p", "r", "p"))
    # build lookup for directed fragment-pair -> feature
    feat_map = {}
    for reac in brics_bonds_feats:  # [[begin_atom, end_atom], feature(34)]
        s, t = reac[0]
        if s in atom2frag and t in atom2frag:
            p0 = atom2frag[s]
            p1 = atom2frag[t]
            if p0 != p1:
                feat_map[(p0, p1)] = reac[1]
    zero_pad = [0] * 34
    for idx in range(g.num_edges(etype=("p", "r", "p"))):
        key = (int(src[idx].item()), int(dst[idx].item()))
        f_reac.append(feat_map.get(key, zero_pad))
    g.edges[("p", "r", "p")].data["x"] = (
        torch.FloatTensor(f_reac) if len(f_reac) > 0 else torch.zeros((0, 34), dtype=torch.float32)
    )

    return g


def _build_frag_bonds_from_sets(mol, mol_fragment_sets):
    frag_edges = []
    atom2frag = {}
    for fi, aset in enumerate(mol_fragment_sets):
        for a in aset:
            atom2frag[a] = fi
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        fu = atom2frag.get(u, None)
        fv = atom2frag.get(v, None)
        if fu is not None and fv is not None and fu != fv:
            frag_edges.append((fu, fv))
            frag_edges.append((fv, fu))
    return frag_edges


def _fragment_node_features_from_sets(mol, fragment_atom_sets, fragment_mol_sets=None):
    f_list = []
    if fragment_mol_sets is not None:
        for mol in fragment_mol_sets:
            # amap = list(aset)
            # mol = _safe_fragment_mol_from_atoms(mol, amap)
            maccs = maccskeys_emb(mol)
            try:
                pharm = pharm_property_types_feats(mol)
            except Exception:
                pharm = [0] * PHARM_TYPES_LEN
            f_list.append(maccs + pharm)
    else:
        for aset in fragment_atom_sets:
            amap = list(aset)
            mol_pharm = _safe_fragment_mol_from_atoms(mol, amap)
            maccs = maccskeys_emb(mol_pharm)
            try:
                pharm = pharm_property_types_feats(mol_pharm if mol_pharm is not None else mol)
            except Exception:
                pharm = [0] * PHARM_TYPES_LEN
            f_list.append(maccs + pharm)
    return f_list


def _mol2hetero_from_sets(
    mol, fragment_atom_sets, frag_edges, edge_feat_dim=None, fragment_mol_sets=None
):
    num_atoms = mol.GetNumAtoms()
    num_frags = len(fragment_atom_sets)

    # a-b-a
    a_src, a_dst = [], []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        a_src += [u, v]
        a_dst += [v, u]

    # a-c-p
    p_src, p_dst = [], []
    for fi, aset in enumerate(fragment_atom_sets):
        for a in aset:
            p_src.append(a)
            p_dst.append(fi)

    edges = {
        ("a", "b", "a"): (
            torch.tensor(a_src, dtype=torch.int64),
            torch.tensor(a_dst, dtype=torch.int64),
        ),
        ("a", "c", "p"): (
            torch.tensor(p_src, dtype=torch.int64),
            torch.tensor(p_dst, dtype=torch.int64),
        ),
        ("p", "c", "a"): (
            torch.tensor(p_dst, dtype=torch.int64),
            torch.tensor(p_src, dtype=torch.int64),
        ),
        ("p", "r", "p"): (
            torch.tensor([s for s, _ in frag_edges], dtype=torch.int64),
            torch.tensor([t for _, t in frag_edges], dtype=torch.int64),
        ),
    }
    g = dgl.heterograph(edges, num_nodes_dict={"a": num_atoms, "p": num_frags})

    f_atom = []
    for idx in g.nodes("a"):
        atom = mol.GetAtomWithIdx(int(idx.item()))
        f_atom.append(atom_features(atom))
    f_atom = torch.FloatTensor(f_atom)
    g.nodes["a"].data["f"] = f_atom

    f_list = (
        _fragment_node_features_from_sets(mol, fragment_atom_sets, fragment_mol_sets)
        if num_frags > 0
        else []
    )
    if len(f_list) > 0:
        g.nodes["p"].data["f"] = torch.FloatTensor(f_list)
    else:
        g.nodes["p"].data["f"] = torch.zeros((0, 167 + PHARM_TYPES_LEN), dtype=torch.float32)

    fa = g.nodes["a"].data["f"]
    fp = g.nodes["p"].data["f"]
    na = fa.shape[0]
    np_ = fp.shape[0]
    g.nodes["a"].data["f_junc"] = (
        torch.cat([fa, torch.zeros((na, fp.shape[1]), dtype=fa.dtype)], dim=1)
        if fp.numel() > 0
        else torch.cat([fa, torch.zeros((na, 0), dtype=fa.dtype)], dim=1)
    )
    g.nodes["p"].data["f_junc"] = (
        torch.cat([torch.zeros((np_, fa.shape[1]), dtype=fa.dtype), fp], dim=1)
        if np_ > 0
        else torch.zeros((0, fa.shape[1]), dtype=torch.float32)
    )

    f_bond = []
    src, dst = g.edges(etype=("a", "b", "a"))
    for i in range(g.num_edges(etype=("a", "b", "a"))):
        f_bond.append(
            bond_features(mol.GetBondBetweenAtoms(int(src[i].item()), int(dst[i].item())))
        )
    g.edges[("a", "b", "a")].data["x"] = (
        torch.FloatTensor(f_bond) if len(f_bond) > 0 else torch.zeros((0, 13), dtype=torch.float32)
    )

    if edge_feat_dim is not None and g.num_edges(etype=("p", "r", "p")) > 0:
        g.edges[("p", "r", "p")].data["x"] = torch.zeros(
            (g.num_edges(etype=("p", "r", "p")), edge_feat_dim), dtype=torch.float32
        )
    else:
        g.edges[("p", "r", "p")].data["x"] = torch.zeros(
            (g.num_edges(etype=("p", "r", "p")), 0), dtype=torch.float32
        )

    return g


def _get_mol_fragment_sets_murcko(mol):
    num_atoms = mol.GetNumAtoms()
    core = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_index = set(mol.GetSubstructMatch(core))
    frag_bond_list = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        link = (u in scaffold_index) + (v in scaffold_index)
        if link == 1:
            frag_bond_list.append(bond.GetIdx())
    if len(frag_bond_list) == 0:
        mol_fragment_sets = [tuple(range(num_atoms))]
    else:
        mfrag = Chem.FragmentOnBonds(mol, frag_bond_list)
        tmp_sets = []
        Chem.GetMolFrags(mfrag, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=tmp_sets)
        mol_fragment_sets = [tuple(sorted([a for a in s if a < num_atoms])) for s in tmp_sets]
    frag_edges = _build_frag_bonds_from_sets(mol, mol_fragment_sets)
    return mol_fragment_sets, frag_edges


def _return_fg_without_ca_i_wash(fg_with_ca_i, fg_without_ca_i):
    washed = []
    for with_ca in fg_with_ca_i:
        for without_ca in fg_without_ca_i:
            if set(without_ca).issubset(set(with_ca)):
                washed.append(list(without_ca))
    return washed


def _merge_hit_fg_atoms(sorted_all_hit_fg_atoms):
    merged = []
    for atoms in sorted_all_hit_fg_atoms:
        if atoms not in merged:
            if len(merged) == 0:
                merged.append(atoms)
                continue
            aset = set(atoms)
            keep = True
            for m in merged:
                if aset.issubset(set(m)):
                    keep = False
                    break
            if keep:
                merged.append(atoms)
    return merged


def _get_mol_fragment_sets_fg(mol):
    num_atoms = mol.GetNumAtoms()

    hit_fg_atoms_list, fg_smiles, fg_submols = get_dec_fgs(mol)
    fg_atoms_flat = []
    for group in hit_fg_atoms_list:
        for frag in group:
            fg_atoms_flat.append(frag)

    not_fg_atoms = [i for i in range(num_atoms) if i not in fg_atoms_flat]
    background = set()
    if len(not_fg_atoms) > 0:
        background = set(not_fg_atoms)
        hit_fg_atoms_list.append(background)

    frag_edges = _build_frag_bonds_from_sets(mol, hit_fg_atoms_list)
    return hit_fg_atoms_list, background, frag_edges


def Mol2HeteroGraph_multi(mol):
    # 1) BRICS
    g_brics = Mol2HeteroGraph(mol)

    # 2) Functional Groups
    num_atoms = mol.GetNumAtoms()
    fg_sets, fg_smiles, fg_submols = get_dec_fgs(mol)
    fg_atoms_flat = []
    for group in fg_sets:
        for frag in group:
            fg_atoms_flat.append(frag)

    not_fg_atoms = [i for i in range(num_atoms) if i not in fg_atoms_flat]
    background = set()
    if len(not_fg_atoms) > 0:
        background = set(not_fg_atoms)
        fg_sets.append(background)
        bg_mol = _safe_fragment_mol_from_atoms(mol, background)
        fg_submols.append(bg_mol)

    fg_edges = _build_frag_bonds_from_sets(mol, fg_sets)
    g_fg = _mol2hetero_from_sets(
        mol, fg_sets, fg_edges, edge_feat_dim=34, fragment_mol_sets=fg_submols
    )

    # 3) Murcko scaffold
    murcko_sets, murcko_edges = _get_mol_fragment_sets_murcko(mol)
    g_murcko = _mol2hetero_from_sets(mol, murcko_sets, murcko_edges, edge_feat_dim=34)

    return {"brics": g_brics, "fg": g_fg, "murcko": g_murcko}
