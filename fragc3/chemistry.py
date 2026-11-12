"""Native fragment descriptors with initialized RDKit ring information."""

import collections, json
import torch
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from . import graph_builder as builder


def _validate_view_atom_order(mol, graph, builder, view):
    expected = torch.tensor(
        [builder.atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32
    )
    if graph.num_nodes("a") != mol.GetNumAtoms() or not torch.allclose(
        graph.nodes["a"].data["f"], expected, atol=1e-6, rtol=0
    ):
        raise ValueError(f"{view}: source atom features/order differ")
    bonds = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.extend(((u, v), (v, u)))
    src, dst = graph.edges(etype=("a", "b", "a"))
    actual = list(zip(src.tolist(), dst.tolist()))
    if actual != bonds:
        raise ValueError(f"{view}: ordered source bond indices differ")
    if actual:
        features = torch.tensor(
            [builder.bond_features(mol.GetBondBetweenAtoms(u, v)) for u, v in actual],
            dtype=torch.float32,
        )
        if not torch.equal(features, graph.edges[("a", "b", "a")].data["x"]):
            raise ValueError(f"{view}: source bond features/stereo differ")


def _membership_sets(graph):
    src, dst = graph.edges(etype=("a", "c", "p"))
    members = collections.defaultdict(set)
    for atom, fragment in zip(src.tolist(), dst.tolist()):
        members[fragment].add(atom)
    return [members[index] for index in range(graph.num_nodes("p"))]


def _parent_atom_validation(smiles, views, builder):
    errors = []
    for recipe in (
        "full_SMILES_sanitize_then_RemoveHs",
        "released_partial_sanitize_after_RemoveHs",
    ):
        try:
            if recipe == "full_SMILES_sanitize_then_RemoveHs":
                parent = Chem.MolFromSmiles(smiles)
                if parent is None:
                    raise ValueError("MolFromSmiles failed")
                parent = Chem.RemoveHs(parent)
            else:
                parent = Chem.MolFromSmiles(smiles, sanitize=False)
                if parent is None:
                    raise ValueError("MolFromSmiles failed")
                parent = Chem.RemoveHs(parent, sanitize=False)
                Chem.SanitizeMol(
                    parent,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                )
                Chem.GetSymmSSSR(parent)
            # BRICS source constructs atom/bond features before Ertl mutates
            # the parent molecule's stereo flags. Match this source stage.
            _validate_view_atom_order(parent, views["brics"], builder, "brics")
            brics_parent = Chem.Mol(parent)
            breaks = [
                parent.GetBondBetweenAtoms(x[0][0], x[0][1]).GetIdx()
                for x in builder.FindBRICSBonds(parent)
            ]
            split_parent = (
                Chem.FragmentOnBonds(parent, breaks, addDummies=False) if breaks else parent
            )
            brics_sets = [set(atoms) for atoms in Chem.GetMolFrags(split_parent)]
            if brics_sets != _membership_sets(views["brics"]):
                raise ValueError("BRICS source partition IDs/order differ from cached membership")
            post_efg_parent = Chem.Mol(parent)
            fg_sets, fg_smiles, fg_mols = builder.get_dec_fgs(post_efg_parent)
            fg_sets = [set(atoms) for atoms in fg_sets]
            background = (
                set(range(parent.GetNumAtoms())) - set().union(*fg_sets)
                if fg_sets
                else set(range(parent.GetNumAtoms()))
            )
            if background:
                fg_sets.append(background)
                bg_mol = builder._safe_fragment_mol_from_atoms(post_efg_parent, sorted(background))
                if bg_mol is None:
                    raise ValueError("Native FG background subset construction failed")
                fg_mols.append(bg_mol)
            if len(fg_mols) != len(fg_sets) or fg_sets != _membership_sets(views["fg"]):
                raise ValueError("Native EFG sets/IDs/order do not exactly match cached membership")
            _validate_view_atom_order(post_efg_parent, views["fg"], builder, "fg")
            _validate_view_atom_order(post_efg_parent, views["murcko"], builder, "murcko")
            murcko_sets, _ = builder._get_mol_fragment_sets_murcko(post_efg_parent)
            if [set(atoms) for atoms in murcko_sets] != _membership_sets(views["murcko"]):
                raise ValueError("Murcko source partition IDs/order differ from cached membership")
            validation = {
                "source_atom_order_validation": "all views match source-stage 45D atom features and ordered 14D bond features/indices",
                "source_parent_recipe": recipe,
                "atom_feature_comparison_atol": 1e-6,
                "source_canonical_isomeric_smiles_before_efg": Chem.MolToSmiles(
                    brics_parent, isomericSmiles=True
                ),
                "native_EFG_membership_and_fragment_ID_order_exact": True,
                "source_EFG_stereo_mutation": "get_dec_fgs clears parent atom chirality and bond stereo; FG and Murcko atom/bond arrays are validated after this original mutation",
                "native_EFG_decorated_molecule_count": len(fg_mols),
                "native_EFG_background_fragment_present": bool(background),
            }
            return (
                {"brics": brics_parent, "fg": post_efg_parent, "murcko": post_efg_parent},
                fg_mols,
                validation,
            )
        except Exception as exc:
            errors.append({"recipe": recipe, "error": str(exc)})
    raise ValueError(json.dumps(errors, ensure_ascii=False))


def _repair_one_fragment(parent, atom_indices, factory, type_order):
    if not atom_indices or len(atom_indices) != len(set(atom_indices)):
        raise ValueError("Empty or duplicate fragment atom set")
    if min(atom_indices) < 0 or max(atom_indices) >= parent.GetNumAtoms():
        raise ValueError("Fragment atom index outside parent")
    text = Chem.MolFragmentToSmiles(
        parent, atomsToUse=list(atom_indices), isomericSmiles=True, kekuleSmiles=False
    )
    sub = Chem.MolFromSmiles(text, sanitize=False)
    if sub is None:
        raise ValueError("Fragment SMILES could not be reparsed")
    sub = Chem.RemoveHs(sub, sanitize=False)
    # RemoveHs(sanitize=False) clears RingInfo. Sanitize/find rings AFTER it.
    flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    Chem.SanitizeMol(sub, sanitizeOps=flags)
    Chem.GetSymmSSSR(sub)
    sub.GetRingInfo().NumRings()  # Explicit initialized-ring assertion.
    maccs = list(MACCSkeys.GenMACCSKeys(sub))  # No zero fallback on failure.
    features = factory.GetFeaturesForMol(sub)
    present = {feature.GetType() for feature in features}
    pharmacophore = [int(name in present) for name in type_order]
    if len(maccs) != 167 or len(pharmacophore) != 27:
        raise ValueError("Repair no longer has the frozen 167+27 feature definition")
    tensor = torch.tensor(maccs + pharmacophore, dtype=torch.float32)
    if tensor.shape != (194,) or not torch.isfinite(tensor).all():
        raise ValueError("Invalid repaired fragment feature vector")
    aromatic_nonring = [
        atom.GetIdx() for atom in sub.GetAtoms() if atom.GetIsAromatic() and not atom.IsInRing()
    ]
    return tensor, {
        "fragment_smiles": text,
        "fragment_atoms_after_H_removal": sub.GetNumAtoms(),
        "ring_count": sub.GetRingInfo().NumRings(),
        "MACCS_nonzero_bits": sum(maccs),
        "pharmacophore_nonzero_types": sum(pharmacophore),
        "retained_aromatic_nonring_atom_indices": aromatic_nonring,
        "all_zero_vector_from_successful_computation": not bool(tensor.any()),
        "sanitization": "RemoveHs(False), SanitizeMol(ALL ^ KEKULIZE), GetSymmSSSR; no fingerprint/feature exception fallback",
    }


def _repair_native_efg_fragment(native_mol, factory, type_order):
    if native_mol is None:
        raise ValueError("Missing native decorated EFG molecule")
    # Preserve the original get_dec_fgs decoration, atom attributes and caps.
    # Only computed cache/ring information is initialized; no repartitioning.
    sub = Chem.Mol(native_mol)
    sub.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(sub)
    sub.GetRingInfo().NumRings()
    maccs = list(MACCSkeys.GenMACCSKeys(sub))
    present = {feature.GetType() for feature in factory.GetFeaturesForMol(sub)}
    pharmacophore = [int(name in present) for name in type_order]
    if len(maccs) != 167 or len(pharmacophore) != 27:
        raise ValueError("Native EFG feature dimension changed")
    tensor = torch.tensor(maccs + pharmacophore, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError("Native EFG has non-finite features")
    return tensor, {
        "fragment_smiles": Chem.MolToSmiles(sub, isomericSmiles=True),
        "native_decorated_atom_count": sub.GetNumAtoms(),
        "ring_count": sub.GetRingInfo().NumRings(),
        "MACCS_nonzero_bits": sum(maccs),
        "pharmacophore_nonzero_types": sum(pharmacophore),
        "retained_aromatic_nonring_atom_indices": [
            atom.GetIdx() for atom in sub.GetAtoms() if atom.GetIsAromatic() and not atom.IsInRing()
        ],
        "all_zero_vector_from_successful_computation": not bool(tensor.any()),
        "native_EFG_decoration_retained": True,
        "sanitization": "Original get_dec_fgs decorated subMol copied; UpdatePropertyCache(strict=False), GetSymmSSSR; no fingerprint/feature exception fallback",
    }


def graphs_from_smiles(smiles, partial_sanitization=False):
    if partial_sanitization:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError("Invalid molecular SMILES")
        mol = Chem.RemoveHs(mol, sanitize=False)
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
        Chem.GetSymmSSSR(mol)
    else:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError("Invalid molecular SMILES")
        mol = Chem.RemoveHs(mol, sanitize=False)
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
        Chem.GetSymmSSSR(mol)
    elif not partial_sanitization:
        mol = Chem.RemoveHs(mol)
    graphs = builder.Mol2HeteroGraph_multi(mol)
    parents, fg_mols, _ = _parent_atom_validation(smiles, graphs, builder)
    for view, graph in graphs.items():
        members = _membership_sets(graph)
        features = []
        for i, atoms in enumerate(members):
            if view == "fg":
                f, _ = _repair_native_efg_fragment(fg_mols[i], builder.factory, builder.PHARM_TYPES)
            else:
                f, _ = _repair_one_fragment(
                    parents[view], sorted(atoms), builder.factory, builder.PHARM_TYPES
                )
            features.append(f)
        f = torch.stack(features)
        graph.nodes["p"].data["f"] = f
        graph.nodes["p"].data["f_junc"] = torch.cat((torch.zeros((len(f), 45)), f), dim=1)
    return graphs
