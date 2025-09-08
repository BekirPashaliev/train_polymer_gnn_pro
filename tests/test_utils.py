import sys, types
import pytest

# minimal pandas stub for import
def _ensure_pandas_stub():
    if 'pandas' not in sys.modules:
        sys.modules['pandas'] = types.ModuleType('pandas')


_ensure_pandas_stub()


def test_compile_smarts_list_ignore_invalid():
    tpg = pytest.importorskip('train_polymer_gnn_pro')
    pytest.importorskip('rdkit.Chem')
    compiled = tpg._compile_smarts_list(['[C]', 'invalid'])
    assert len(compiled) == 1


def test_match_anchor_atoms_hbd():
    tpg = pytest.importorskip('train_polymer_gnn_pro')
    Chem = pytest.importorskip('rdkit.Chem')
    mol = Chem.MolFromSmiles('CC(=O)O')
    anchors = tpg._match_anchor_atoms(mol, tpg._HBD_P)
    symbols = {mol.GetAtomWithIdx(i).GetSymbol() for i in anchors}
    assert 'O' in symbols


def test_counts_within_r1_r2():
    tpg = pytest.importorskip('train_polymer_gnn_pro')
    r1, r2 = tpg._counts_within_r1_r2([[1], [0, 2], [1]], {0})
    assert r1[1] == pytest.approx(1/3)
    assert r2[2] == pytest.approx(1/3)


def test_smiles_to_graph_removes_stars():
    tpg = pytest.importorskip('train_polymer_gnn_pro')
    pytest.importorskip('rdkit.Chem')
    x, edge_index, edge_attr, gdesc = tpg.smiles_to_graph('C[*]C')
    assert x.shape[0] == 2
    assert edge_index.shape[1] == 2
    assert edge_attr.shape[0] == 2
