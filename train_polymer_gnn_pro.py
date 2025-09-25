import os, time, random, argparse, json, io
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import re

import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import rdchem, Descriptors, Crippen, rdMolDescriptors, rdMolDescriptors as rdMD, rdmolops
from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')   # скрыть предупреждения
RDLogger.DisableLog('rdApp.error')   # скрыть и ошибки парсинга

# progress bars (fallback, если tqdm не установлен)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):  # тихий заглушка-прогресс
        return x

import signal, platform
from datetime import datetime
import torch.nn.functional as F

from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# === Unified AMP dtype (bf16 if supported, else fp16) ===
try:
    AMP_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
except AttributeError:
    # старые версии torch без is_bf16_supported
    AMP_DTYPE = torch.float16


import gc, torch
import hashlib  # для хеш-бакинга n-грамм


import platform
from rdkit import DataStructs

import platform, os
from joblib import Parallel, delayed, parallel_config

try:
    from sklearn.isotonic import IsotonicRegression
    _HAS_SK_ISO = True
except Exception:
    _HAS_SK_ISO = False

# =============================
# Targets / Constants
# =============================
TARGETS = ["Tg","FFV","Tc","Density","Rg"]

GLOBAL_DESCS = [
    # all scaled crudely; standardization done later
    lambda m: Descriptors.MolWt(m),
    lambda m: Descriptors.HeavyAtomCount(m),
    lambda m: Descriptors.NumAromaticRings(m),
    lambda m: Descriptors.NumAliphaticRings(m),
    lambda m: Descriptors.RingCount(m),
    lambda m: Descriptors.TPSA(m),
    lambda m: Descriptors.NumHAcceptors(m),
    lambda m: Descriptors.NumHDonors(m),
    lambda m: Descriptors.NumRotatableBonds(m),
    lambda m: Crippen.MolLogP(m),
    lambda m: Crippen.MolMR(m),
    lambda m: rdMD.CalcChi0n(m),
    lambda m: rdMD.CalcChi1n(m),
    lambda m: rdMD.CalcKappa1(m),
]

# сколько новых фич добавляем к узлам/глобальным дескрипторам
# [is_backbone, dist2bb, is_star_nb, dist2star, pos_sin, pos_cos,
#  conj_seg_len_norm, conj_seg_frac_of_bb,
#  ewg_r1, ewg_r2, edg_r1, edg_r2,
#  is_hbd, is_hba, hbd_nb1, hba_nb1,
#  randic_r1, chi1_local]
EXTRA_NODE_FEATS = 18

# сколько новых фич добавляем к КАЖДОМУ ребру (до финального is_bb_bond)
EXTRA_EDGE_FEATS = 4   # [is_khop_edge, khop_norm, is_rotatable, is_on_bb_pair]
KHOP_MAX = 3           # k-hop по бэкбону до 3
KHOP_MIN = 2           # начиная со 2-хопа


# [bb_len, bb_arom_frac, bb_sp3_frac, bb_rot_per_len, bb_branch_avg, bb_branch_max]

# ========= Полимерные глобальные признаки =========
# базовые 6 уже есть (_poly_backbone_descriptors):
#   [bb_len, bb_arom_frac, bb_sp3_frac, bb_rot_per_len, bb_branch_avg, bb_branch_max]
# добавляем:
# B1: 4 фракции функционалов (карбонил/имид/сульфон/арил)
# B2: 2 "bulk" метрики боковых цепей (avg, max)
# B3: 4 зарядовых статистики (|q| на bb/side: mean,std)
# F1: 1 мол. масса закрытого CRU (M0)
# F2: 1 доля гетероатомов (N,O,S) на бэкбоне
# F3: 1 средняя геодезическая длина боковых цепей
# B4: K биграммных n-грамм (хешированные) вдоль primary_path
# B5: 1 доля вращательных связей среди всех связей бэкбона
BB_NGRAM_TOPK = 64


# --- размеры глобальных дескрипторов ---
BASE_BB_DIM = 7  # _poly_backbone_descriptors возвращает РОВНО 7

EXTRA_PLUS_DIM = 17  # cseg(2) + tpsa(2) + hba/hbd(4) + side_geom(2) + ortho(2) + ewg/edg(2) + spec(3)


EXTRA_EXT_DIM = (4 + 2 + 4 + 1 + 1 + 1 + BB_NGRAM_TOPK) + EXTRA_PLUS_DIM

TOTAL_GDESC_DIM = len(GLOBAL_DESCS) + BASE_BB_DIM + EXTRA_EXT_DIM

# === Универсальная упаковка признаков ребра (один источник правды) ===========

def edge_attr_build(
    bond: Optional[rdchem.Bond],
    *,
    is_poly: bool = False,
    is_khop: float = 0.0,
    khop_norm: float = 0.0,
    is_rot: float = 0.0,
    is_bb_pair: float = 0.0,
    is_bb_bond: float = 0.0
) -> list:
    """
    Возвращает ПОЛНЫЙ список признаков ребра длиной ровно EDGE_FEAT_DIM.
    Никаких «магических» конкатенаций по месту.
    """
    base = bond_features(bond, is_poly_edge=is_poly)  # длина = BOND_BASE_DIM
    ext  = [float(is_khop), float(khop_norm), float(is_rot), float(is_bb_pair), float(is_bb_bond)]
    v = base + ext
    # Страховка от несовпадений — пад/обрезка, чтобы НИКОГДА не падать на cat():
    if len(v) < EDGE_FEAT_DIM:
        v += [0.0] * (EDGE_FEAT_DIM - len(v))
    elif len(v) > EDGE_FEAT_DIM:
        v = v[:EDGE_FEAT_DIM]
    return v

def ensure_edge_attr_dim_tensor(ea: torch.Tensor) -> torch.Tensor:
    """Гарантирует shape [E, EDGE_FEAT_DIM] (пад/обрезка, если надо)."""
    if ea.numel() == 0:
        return ea
    if ea.size(1) == EDGE_FEAT_DIM:
        return ea
    if ea.size(1) < EDGE_FEAT_DIM:
        pad = torch.zeros((ea.size(0), EDGE_FEAT_DIM - ea.size(1)), dtype=ea.dtype, device=ea.device)
        return torch.cat([ea, pad], dim=1)
    return ea[:, :EDGE_FEAT_DIM]

# =============================
# Utils
# =============================

# Небольшой фолбэк: PAV для изотонической регрессии (кусочно-постоянная),
# а применяем как кусочно-линейную интерполяцию по узлам.
def _pav_fit(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    w = np.ones_like(y, dtype=np.float64)
    # блоки
    yhat = y.copy()
    n = len(yhat)
    i = 0
    while i < n-1:
        if yhat[i] <= yhat[i+1]:
            i += 1
            continue
        j = i
        s, ww = yhat[i]*w[i], w[i]
        while j >= 0 and yhat[j] > yhat[j+1]:
            s += yhat[j]*w[j]; ww += w[j]
            val = s/ww
            yhat[j] = yhat[j+1] = val
            j -= 1
        i += 1
    # сжимаем в узлы
    knots_x, knots_y = [x[0]], [yhat[0]]
    for t in range(1, n):
        if yhat[t] != yhat[t-1]:
            knots_x.append(x[t]); knots_y.append(yhat[t])
    if knots_x[-1] != x[-1]:
        knots_x.append(x[-1]); knots_y.append(yhat[-1])
    return np.asarray(knots_x, float), np.asarray(knots_y, float)

def _iso_fit_1d(pred, true):
    if _HAS_SK_ISO:
        ir = IsotonicRegression(y_min=None, y_max=None, increasing=True, out_of_bounds="clip")
        ir.fit(pred, true)
        # Узлы: sklearn хранит как thresholds_/y_thresholds_
        return np.asarray(ir.X_thresholds_, float), np.asarray(ir.y_thresholds_, float)
    return _pav_fit(pred, true)

def _iso_apply_1d(x, knots_x, knots_y):
    # Кусочно-линейная интерполяция по монотонным узлам
    return np.interp(x, knots_x, knots_y, left=knots_y[0], right=knots_y[-1]).astype(np.float32)


def apply_head_constraints(mu: torch.Tensor, target_names=None) -> torch.Tensor:
    """
    Возвращает преобразованное mu с ограничениями:
      FFV -> sigmoid ∈ (0,1)
      Rg  -> softplus > 0
      Density -> softplus > 0
    Остальные цели без изменений.
    """
    if target_names is None:
        target_names = TARGETS
    out = mu
    # делаем копию только если нужные столбцы есть
    if "FFV" in target_names:
        i = target_names.index("FFV")
        v = torch.sigmoid(out[:, i])
        out = out.clone()
        out[:, i] = v
    if "Rg" in target_names:
        i = target_names.index("Rg")
        v = F.softplus(out[:, i])
        out = out.clone()
        out[:, i] = v
    if "Density" in target_names:
        i = target_names.index("Density")
        v = F.softplus(out[:, i])
        out = out.clone()
        out[:, i] = v
    return out

def _encode_dataset(model, loader, device, use_amp=True):
    """Возвращает эмбеддинги (N,H), y (N,T), mask (N,T)."""
    G, Ys, Ms = [], [], []
    model.eval(); _enable_mc_dropout(model, False)
    with torch.no_grad():
        for x, ei, ea, bvec, gdesc, y, ymask in loader:
            x, ei, ea, bvec, gdesc = x.to(device), ei.to(device), ea.to(device), bvec.to(device), gdesc.to(device)
            with torch.autocast('cuda', enabled=use_amp, dtype=AMP_DTYPE):
                g = model.encode(x, ei, ea, bvec, gdesc, use_gdesc=True, use_poly_hints=True)
            G.append(g.detach().to(torch.float32).cpu().numpy())
            Ys.append(y.detach().cpu().numpy()); Ms.append(ymask.detach().cpu().numpy())
    G = np.vstack(G)
    Y = np.vstack(Ys) if len(Ys) else None
    M = np.vstack(Ms) if len(Ms) else None
    return G, Y, M

def _knn_regress(test_G, train_G, train_Y, train_M, k=32, tau=0.2):
    """
    Косинусная близость + softmax(weights/tau); per-target маскирование отсутствующих лейблов.
    """
    t0 = time.perf_counter()

    # нормализуем
    eps = 1e-8
    test_N = test_G / np.maximum(np.linalg.norm(test_G, axis=1, keepdims=True), eps)
    train_N = train_G / np.maximum(np.linalg.norm(train_G, axis=1, keepdims=True), eps)

    # инфо о размерах
    print(f"[KNN] test_G={test_G.shape}, train_G={train_G.shape}, "
          f"targets={train_Y.shape[1]}, k={k}, tau={tau}")

    # батчим по 4096 чтобы не съесть память
    B = 4096
    out = np.zeros((test_G.shape[0], train_Y.shape[1]), dtype=np.float32)

    # прогресс по блокам
    steps = (test_G.shape[0] + B - 1) // B
    it = range(0, test_G.shape[0], B)
    try:
        it = tqdm(it, total=steps, desc="[KNN] blocks")  # tqdm уже есть в файле
    except Exception:
        pass

    for s in it:
        e = min(s + B, test_G.shape[0])
        # косинусные сходства (через скалярные произведения нормированных)
        S = test_N[s:e] @ train_N.T  # [b, Ntrain]

        # для каждого таргета отдельно учитываем маски отсутствующих лейблов
        # (веса на таких строках = -inf → softmax=0)
        for t in range(train_Y.shape[1]):
            m = train_M[:, t].astype(bool)            # [Ntrain]
            if not m.any():
                continue
            sims = S[:, m]                             # [b, Navail]
            # top-k по доступным лейблам
            kk = min(k, sims.shape[1])
            if kk <= 0:
                continue
            idx = np.argpartition(-sims, kk-1, axis=1)[:, :kk]  # без полной сортировки
            rows = np.arange(sims.shape[0])[:, None]
            top = sims[rows, idx]                     # [b, kk]
            w = np.exp(top / max(tau, 1e-6))
            w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-8)

            yk = train_Y[m, t][idx]                   # [b, kk]
            out[s:e, t] = (w * yk).sum(axis=1)

    dt = time.perf_counter() - t0
    print(f"[KNN] done in {dt:.2f}s")
    return out

def _enable_mc_dropout(model: nn.Module, enable: bool = True):
    """
    Включает train() только у Dropout-слоёв, оставляя остальное в eval().
    Удобно для TTA без дергания LayerNorm/BatchNorm.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train(enable)

class Tic:
    def __init__(self, tag): self.tag, self.t0 = tag, time.perf_counter()
    def done(self, extra=""):
        dt = time.perf_counter() - self.t0
        print(f"[UNL-TIMER] {self.tag} {extra} | {dt:,.2f}s")


def _add_khop_backbone_edges(adj, keep_mask, backbone_set, old2new, src, dst, eattr,
                             kmin=2, kmax=3):
    if not backbone_set:
        return
    bb = set(int(i) for i in backbone_set)
    # множество уже существующих реальных ребер (в новых индексах)
    existing = set(zip(src, dst))  # ок, т.к. мы вызываем после добавления хим. ребер

    # BFS по бэкбону от каждого узла
    from collections import deque
    for a in bb:
        # старт — только из бэкбон узлов
        q = deque([(a, 0)])
        seen = {a: 0}
        while q:
            u, d = q.popleft()
            if d >= kmax:
                continue
            for v in adj[u]:
                if (v not in bb):        # ходим только по бэкбону
                    continue
                nd = d + 1
                if v not in seen or nd < seen[v]:
                    seen[v] = nd
                    q.append((v, nd))
        # добавляем пары на расстояниях [kmin..kmax]
        for v, dist in seen.items():
            if v == a or dist < kmin or dist > kmax:
                continue
            ua = int(old2new[a]); va = int(old2new[v])
            if ua < 0 or va < 0:
                continue
            # пропускаем, если такое ребро уже есть
            if (ua, va) in existing:
                continue
            bf = bond_features(None, is_poly_edge=False)  # виртуальное, но не poly
            khop_norm = 1.0 / float(dist)
            ea_khop = edge_attr_build(
                None, is_poly=False,
                is_khop=1.0, khop_norm=khop_norm, is_rot=0.0,
                is_bb_pair=1.0, is_bb_bond=0.0
            )
            src += [ua, va]; dst += [va, ua]; eattr += [ea_khop, ea_khop]; existing.add((ua, va)); existing.add((va, ua))


# ---- 2D физичные локальные признаки ----------------------------------------
# Якоря EWG/EDG — берём минимальный устойчивый набор
# ---------- баланс EWG/EDG на бэкбоне ----------
_EWG_SMARTS = [
    "[N;X3,+](=O)[O-]",    # Нитро-группа
    "[CX2]#N",             # Нитрил (циано, –C≡N)
    "[C;!$(CO),$(C(=O)OC(=O)[#6])](=O)",       # Ацильный карбонил (сильные: альдегид, кетон, ацил-галогенид, ангидридный фрагмент)
    "C(=O)[OX2][#6]",  # Сложный эфир (–C(=O)–O–R)
    "C(=O)[NX3]",  # Амид (–C(=O)–N–)
    "C(=O)[O;X1,-]",  # Карбоксилат (–C(=O)O⁻)
    "[SX4;!$(SO)](=O)(=O)",   # Сульфон(ил) (S(VI)=O,=O)
    "[SX4](=O)(=O)O",  # Сульфонаты/сульфаты
    "[PX4v5](=[O,S])[O,S]",  # Фосфаты/фосфонаты (P(V) с =O)
    "[CX4](F)(F)F"     # CF3 (грубый якорь)
]
_EDG_SMARTS = [
    "[OX2H;!$(OC=O);!$(OS=O);!$(OP=O)]",          # спиртовый O–H (донор/EDG)
    "[OX2;!$(OC=O);!$(OS=O);!$(OP=O);!$(OO)]",   # Алкокси/эфирный O (–O–R, –OR на ариле)
    "[NX3v3;H0,H1,H2;!$(NC=O);!$(N=O);!$(NN);!$(NS=O);!$(NP=O);!$(N[c])]",  # Нейтральные амины (aliph/aryl, перв/втор/третичные)
    "[SX2;!$(SS)]",            # Тиоэфиры (R–S–R′)
    "[c][NX3v3;H0,H1,H2;!$(NC=O);!$(N=O);!$(NN);!$(NS=O);!$(NP=O)]"  # Анилиноподобные доноры (–NH₂/–NR₂ на ариле)
    # "[CX4H3]"  # Алкильные доноры (прокси): терминальный метил/этил
]

# H-bond доноры/акцепторы (узел-локально)
# ---------- TPSA и HBA/HBD раздельно для bb/side ----------
_HBD_SMARTS = ["[OX2H]",
               "[NX3;!+;H1,H2]",
               "[N+;H1,H2,H3]",
               "[SX2H]",
               "[nH]"
               ]  # O–H, N–H
_HBA_SMARTS = ["[O;$(O=C);!$(O=C[OH]);!$(O=COO);!$(O=C[O+])]",
               "[OX2;!$(OO);!$([O+]);!$(OC=O);!$(OS=O)]",
               "[O-]",
               "[n;!$([nH]);!$([n+])]",
               "[N;$([NX1]#[CX2])]",
               "[O;$(ON=O),$(O=NO)]",
               "[S;$([SX2]);!$([SH]);!$([S-]);!$([S+])]",
               "[S;$(S=[CX3])]",
               "[O;$([OX1]=S)]",
               "[O;$(OP=O),$([O-]P),$(O(C)P),$(O=P)]",
               ]

def _compile_smarts_list(patts):
    out = []
    for s in patts:
        m = Chem.MolFromSmarts(s)
        if m is not None: out.append(m)
    return out

_EWG_P = _compile_smarts_list(_EWG_SMARTS)
_EDG_P = _compile_smarts_list(_EDG_SMARTS)
_HBD_P = _compile_smarts_list(_HBD_SMARTS)
_HBA_P = _compile_smarts_list(_HBA_SMARTS)

def _match_anchor_atoms(mol, patt_list):
    anchors = set()
    for p in patt_list:
        for match in mol.GetSubstructMatches(p):
            if match:
                anchors.add(int(match[0]))   # первый атом матча как якорь
    return anchors

def _counts_within_r1_r2(adj, anchors):
    # возвращаем два массива длиной N: counts в r=1 и r<=2
    N = len(adj)
    r1 = [0]*N; r2 = [0]*N
    if not anchors:
        return r1, r2
    from collections import deque
    for a in anchors:
        # BFS до глубины 2
        dist = [-1]*N
        q = deque([a]); dist[a] = 0
        while q:
            u = q.popleft()
            if dist[u] >= 2: continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for i, d in enumerate(dist):
            if d == 1: r1[i] += 1
            if 1 <= d <= 2: r2[i] += 1
    # клип до 3 и нормируем
    def _norm(v): return [min(3, x)/3.0 for x in v]
    return _norm(r1), _norm(r2)

def _hb_flags_per_atom(mol):
    N = mol.GetNumAtoms()
    is_hbd = [0.0]*N; is_hba = [0.0]*N
    def mark(patts, arr):
        for p in patts:
            for match in mol.GetSubstructMatches(p):
                for idx in match:
                    arr[int(idx)] = 1.0
    mark(_HBD_P, is_hbd)
    mark(_HBA_P, is_hba)
    return is_hbd, is_hba

def _neighbor_counts(adj, flag_arr, *, clip=3):
    # сколько "помеченных" соседей на расстоянии 1
    out = []
    for i, nbrs in enumerate(adj):
        cnt = sum(flag_arr[j] > 0.5 for j in nbrs)
        out.append(min(clip, cnt)/float(clip))
    return out

def _conjugated_backbone_segments(mol, backbone_set, keep_mask):
    # граф только по бэкбону и только по конъюгированным/ароматическим связям
    N = mol.GetNumAtoms()
    adj = [[] for _ in range(N)]
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if not (keep_mask[u] and keep_mask[v]): continue
        if (u in backbone_set) and (v in backbone_set):
            if b.GetIsConjugated() or b.GetBondType() == rdchem.BondType.AROMATIC:
                adj[u].append(v); adj[v].append(u)
    # компоненты
    comp_id = [-1]*N; comp_sizes = {}
    cid = 0
    from collections import deque
    for i in backbone_set:
        if comp_id[i] != -1: continue
        q = deque([i]); comp_id[i] = cid; size = 0
        while q:
            u = q.popleft(); size += 1
            for v in adj[u]:
                if comp_id[v] == -1:
                    comp_id[v] = cid; q.append(v)
        comp_sizes[cid] = size
        cid += 1
    # длины компонентов по атомам
    seg_len = [0]*N
    for i in backbone_set:
        cid = comp_id[i]
        seg_len[i] = comp_sizes.get(cid, 1)
    max_seg = max([seg_len[i] for i in backbone_set], default=1)
    bb_len = max(1, len(backbone_set))
    # нормировки
    conj_norm = [ (seg_len[i]/max_seg) if i in backbone_set else 0.0 for i in range(N) ]
    conj_frac = [ (seg_len[i]/bb_len)  if i in backbone_set else 0.0 for i in range(N) ]
    return conj_norm, conj_frac

def _local_topo_indices(adj):
    # randic_r1 и chi1_local на основе степеней
    deg = [len(n) for n in adj]
    def safe_sqrt(x): return math.sqrt(max(1.0, x))
    randic_r1 = []
    chi1 = []
    for i, nbrs in enumerate(adj):
        if not nbrs:
            randic_r1.append(0.0); chi1.append(0.0); continue
        s1 = 0.0
        for j in nbrs:
            s1 += 1.0 / (safe_sqrt(deg[i]) * safe_sqrt(deg[j]))
        # клип/норма для устойчивости
        s1 = min(3.0, s1) / 3.0
        randic_r1.append(s1)
        s2 = 0.0
        for j in nbrs:
            dj = max(1, deg[j]-1)
            s2 += 1.0 / math.sqrt(dj)
        s2 = min(3.0, s2) / 3.0
        chi1.append(s2)
    return randic_r1, chi1



# === Poly-edge helpers (мультирежим) ========================================
_VALID_POLY_MODES = ("pair", "cycle", "clique")

def get_poly_modes(args):
    """Возвращает список режимов poly_edge_mode, уникальный и отфильтрованный."""
    raw = (args.poly_edge_modes or "").strip()
    if not raw:
        modes = [args.poly_edge_mode]
    else:
        modes = [m.strip() for m in raw.split(",") if m.strip()]
    modes = [m for m in modes if m in _VALID_POLY_MODES]
    if not modes:  # страховка на случай опечаток
        modes = [args.poly_edge_mode if args.poly_edge_mode in _VALID_POLY_MODES else "cycle"]
    # убрать дубликаты, сохранив порядок
    modes = list(dict.fromkeys(modes))
    return modes

def canonical_poly_mode(args, modes):
    """Какой режим использовать на валидации/инфере."""
    ev = (args.poly_edge_eval_mode or "first").strip().lower()
    if ev == "first":
        return modes[0]
    if ev in _VALID_POLY_MODES and ev in modes:
        return ev
    return modes[0]  # дефолт


def print_loader_info(name, batch_size, num_workers, pin_memory, persistent_workers, prefetch_factor=None):
    pf = f", prefetch={prefetch_factor}" if prefetch_factor is not None else ""
    print(f"[LOADER] {name}: bs={batch_size}, workers={num_workers}, pin={pin_memory}, persistent={persistent_workers}{pf}")

@dataclass
class BatchStage:
    bs: int
    patience: int
    max_epochs: int  # 0 = без потолка на стадию (ограничит только patience/глобальный лимит)

def parse_batch_growth_3(spec: str) -> List[BatchStage]:
    spec = (spec or "").strip()
    if not spec:
        return []
    out: List[BatchStage] = []
    for tok in spec.split(","):
        a = tok.strip().split(":")
        if len(a) != 3:
            raise ValueError(f"Ожидается 'bs:patience:max_epochs', получено: '{tok}'")
        bs, pat, me = map(int, a)
        if bs <= 0 or pat <= 0 or me < 0:
            raise ValueError(f"Невалидные числа в '{tok}' (bs>0, patience>0, max_epochs>=0)")
        out.append(BatchStage(bs, pat, me))
    for i in range(1, len(out)):
        if out[i].bs < out[i-1].bs:
            raise ValueError("batch sizes должны не убывать")
    return out

class BatchGrowthController:
    """Хранит стадию, считает эпохи/патенс и говорит, когда перейти на следующий bs."""
    def __init__(self, stages: List[BatchStage], *, global_max_epochs: Optional[int] = None,
                 mode: str = "min", min_delta: float = 1e-4):
        self.stages = stages
        self.global_max_epochs = (None if (not global_max_epochs or global_max_epochs <= 0)
                                  else int(global_max_epochs))
        self.mode = mode
        self.min_delta = float(min_delta)
        self.stage_idx = 0
        self.best = None
        self.no_imp = 0
        self.ep_in_stage = 0
        self.ep_global = 0

    @property
    def done(self) -> bool:
        return (self.stage_idx >= len(self.stages)) or (
            self.global_max_epochs is not None and self.ep_global >= self.global_max_epochs
        )

    @property
    def bs(self) -> int:
        return self.stages[self.stage_idx].bs

    def _improved(self, v: float) -> bool:
        if self.best is None: return True
        return (self.best - v) > self.min_delta if self.mode == "min" else (v - self.best) > self.min_delta

    def on_epoch_end(self, metric_value: float) -> bool:
        """Вернёт True, если пора завершать стадию (патенс/лимит)."""
        self.ep_global += 1
        self.ep_in_stage += 1
        if self._improved(metric_value):
            self.best = metric_value; self.no_imp = 0
        else:
            self.no_imp += 1
        st = self.stages[self.stage_idx]
        stop_pat = (self.no_imp >= st.patience)
        stop_max = (st.max_epochs > 0 and self.ep_in_stage >= st.max_epochs)
        stop_glob = (self.global_max_epochs is not None and self.ep_global >= self.global_max_epochs)
        return stop_pat or stop_max or stop_glob

    def advance(self) -> None:
        """Переход к след. стадии или завершение, если это последняя."""
        if self.stage_idx + 1 < len(self.stages):
            self.stage_idx += 1
            self.best = None; self.no_imp = 0; self.ep_in_stage = 0
        else:
            self.stage_idx = len(self.stages)  # помечаем как done

def parse_seed_list(args) -> List[int]:
    if getattr(args, "seed_list", ""):
        try:
            seeds = [int(s.strip()) for s in args.seed_list.split(",") if s.strip()]
            return seeds
        except Exception:
            pass
    n = max(1, int(getattr(args, "n_seeds", 1)))
    base = int(getattr(args, "seed_base", 42))
    return [base + i for i in range(n)]

def make_split_id(args: argparse.Namespace) -> str:
    # для кешей данных: от фолда зависит, от сида — нет
    return f"f{args.fold}" if getattr(args, "n_folds", 0) >= 2 else "split"

def make_run_tag(args: argparse.Namespace) -> str:
    # для весов/логов/сабмитов: зависит и от фолда, и от сида
    if getattr(args, "run_tag", ""):
        return args.run_tag
    parts = []
    if getattr(args, "n_folds", 0) >= 2:
        parts.append(f"f{args.fold}")
    parts.append(f"s{args.seed}")
    return "__".join(parts)

def seeds_from_args(args: argparse.Namespace) -> List[int]:
    # реюз твоей логики, просто короткое имя
    return parse_seed_list(args)


def blend_submissions(csv_paths: List[str], out_path: str):
    # усредняем по столбцам TARGETS, id берём из первого файла
    dfs = []
    for p in csv_paths:
        if os.path.isfile(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        print("[CV] Нет файлов для бленда сабмита.")
        return
    base = dfs[0][["id"]].copy()
    for t in TARGETS:
        vals = [df[t].values for df in dfs if t in df.columns]
        if not vals:
            continue
        base[t] = np.mean(np.vstack(vals), axis=0)
    base.to_csv(out_path, index=False)
    print("[CV] Blended submission saved:", out_path)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def one_hot(x, choices):
    v = [0]*len(choices)
    try:
        idx = choices.index(x)
        v[idx] = 1
    except ValueError:
        v[-1] = 1
    return v

def _mol_adj_no_stars(mol, keep_mask):
    """Adjacency только по не-звёздам."""
    N = mol.GetNumAtoms()
    adj = [[] for _ in range(N)]
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if keep_mask[u] and keep_mask[v]:
            adj[u].append(v); adj[v].append(u)
    return adj

def _components_no_stars(mol, keep_mask):
    """Компоненты связности по не-звёздам: comp[i] = id или -1 для звёзд."""
    N = mol.GetNumAtoms()
    adj = _mol_adj_no_stars(mol, keep_mask)
    comp = [-1]*N; cid = 0
    from collections import deque
    for i in range(N):
        if keep_mask[i] and comp[i] == -1:
            q = deque([i]); comp[i] = cid
            while q:
                u = q.popleft()
                for w in adj[u]:
                    if comp[w] == -1:
                        comp[w] = cid; q.append(w)
            cid += 1
    return comp, adj

def _shortest_path_atoms(mol, a, b, keep_mask):
    """Кратчайший путь между a,b (индексы в mol), отфильтрованный от звёзд."""
    try:
        path = list(rdmolops.GetShortestPath(mol, a, b))
        return [p for p in path if keep_mask[p]]
    except Exception:
        return []

def _backbone_atoms_by_cycle_paths(mol, comp, keep_mask, star_neighbors):
    """
    Бэкбон = объединение кратчайших путей между соседями звёзд
    внутри каждого компонента, соединённых 'кольцом' (cycle).
    """
    from collections import defaultdict
    by_comp = defaultdict(list)
    for n in star_neighbors:
        cid = comp[n] if 0 <= n < len(comp) else -1
        if cid >= 0:
            by_comp[cid].append(n)

    backbone = set()
    primary_path = []  # самый длинный из путей (для позиционных фич)
    for cid, nbrs in by_comp.items():
        if len(nbrs) < 2:
            continue
        # cycle: (n0,n1), (n1,n2), ... , (nK,n0)
        for i in range(len(nbrs)):
            a, b = nbrs[i], nbrs[(i+1)%len(nbrs)]
            path = _shortest_path_atoms(mol, a, b, keep_mask)
            if len(path) >= 2:
                backbone.update(path)
                if len(path) > len(primary_path):
                    primary_path = path
    return backbone, primary_path

def _dist_to_set(adj, keep_mask, target_set):
    """Минимальное расстояние от каждого узла до множества target_set (по графу без звёзд)."""
    from collections import deque
    N = len(adj)
    INF = 10**9
    dist = [INF]*N
    q = deque()
    for t in target_set:
        dist[t] = 0; q.append(t)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    # для звёзд вернём INF, но потом их не используем
    return dist


# ---------- функциональные группы на бэкбоне ----------
_POLY_SMARTS = {
    # Карбонильные/амидные/эстерные узлы
    "bb_carbonyl":      Chem.MolFromSmarts("[CX3](=O)"),                 # любой ацильный центр
    "bb_amide":         Chem.MolFromSmarts("[CX3](=O)[NX3]"),            # амидная связь
    "bb_ester":         Chem.MolFromSmarts("[CX3](=O)O"),                # сложный эфир (вкл. лактон)
    "bb_carbamate":     Chem.MolFromSmarts("[NX3]C(=O)O"),               # –NH–C(=O)–O–
    "bb_urea":          Chem.MolFromSmarts("[NX3]C(=O)[NX3]"),           # –NH–C(=O)–NH–
    "bb_imide":         Chem.MolFromSmarts("O=C[NX3]C=O"),             # обобщённый имид (тот же N к двум C=O)

    # Сера и оксо-сера
    "bb_thioether":     Chem.MolFromSmarts("[SX2]"),                     # R–S–R′
    "bb_thione":        Chem.MolFromSmarts("[CX3](=S)"),                 # C=S
    "bb_sulfoxide":     Chem.MolFromSmarts("[#16X3](=O)"),               # S(=O)
    "bb_sulfone":       Chem.MolFromSmarts("[#16X4](=O)(=O)"),           # S(=O)2
    "bb_sulfonamide":   Chem.MolFromSmarts("[#16X4](=O)(=O)N"),          # –SO2–NH–

    # Фосфор
    "bb_phosphoryl":    Chem.MolFromSmarts("[PX4](=O)"),                 # P(V)=O (фосфаты/фосфонаты)
    "bb_phosphate_ester": Chem.MolFromSmarts("P(=O)O"),                  # P(=O)–O–

    # Ароматика и C=C
    "bb_arylC":         Chem.MolFromSmarts("[c]"),                       # арильный C (исключает гетероатомы)
    "bb_heteroaryl":    Chem.MolFromSmarts("[a;!c]"),                    # ароматические гетероатомы (пиридин и др.)
    "bb_vinyl":         Chem.MolFromSmarts("[CX3]=[CX3]"),               # винильная C=C

    # EWG-«якоря» на бэкбоне
    "bb_cyano":         Chem.MolFromSmarts("[CX2]#N"),                   # –C≡N
    "bb_nitro":         Chem.MolFromSmarts("[N+](=O)[O-]"),              # –NO2

}
def _bb_functional_fractions(mol, backbone_set):
    if not backbone_set:
        return [0.0, 0.0, 0.0, 0.0]
    bb = set(backbone_set); total = float(len(bb))
    out = []
    for patt in _POLY_SMARTS.values():
        matches = mol.GetSubstructMatches(patt) or []
        c = 0
        for mt in matches:
            if any((idx in bb) for idx in mt):
                c += 1
        out.append(c / total)
    return out  # 4 числа

# ---------- «bulk» и глубина боковых цепей ----------
def _sidechain_bulk_and_depth(mol, keep_mask, backbone_set):
    """Возвращает [avg_bulk, max_bulk, avg_depth] по бэкбону.
    bulk = число атомов в ветке; depth = макс. расстояние от узла bb в сторону ветки."""
    if not backbone_set:
        return [0.0, 0.0, 0.0]
    bb = set(backbone_set)
    bulks, depths = [], []
    for i in bb:
        a = mol.GetAtomWithIdx(i)
        # собираем стартовые соседи вне бэкбона
        starts = [nb.GetIdx() for nb in a.GetNeighbors()
                  if keep_mask[nb.GetIdx()] and (nb.GetIdx() not in bb)]
        if not starts:
            bulks.append(0); depths.append(0); continue
        # суммарный размер и глубина веток от каждого старта
        total_bulk_i = 0; max_depth_i = 0
        for s in starts:
            # DFS до возвращения на бэкбон; считаем размер и максимальную глубину
            stack = [(s, 1)]
            seen = {i}
            local_nodes = set()
            local_depth = 0
            while stack:
                u, d = stack.pop()
                if u in seen:
                    continue
                seen.add(u); local_nodes.add(u)
                local_depth = max(local_depth, d)
                for w in mol.GetAtomWithIdx(u).GetNeighbors():
                    k = w.GetIdx()
                    if keep_mask[k] and (k not in bb):
                        stack.append((k, d+1))
            total_bulk_i += len(local_nodes)
            max_depth_i = max(max_depth_i, local_depth)
        bulks.append(total_bulk_i)
        depths.append(max_depth_i)
    if not bulks:
        return [0.0, 0.0, 0.0]
    return [float(np.mean(bulks)), float(np.max(bulks)), float(np.mean(depths))]

# ---------- зарядовые статистики (Gasteiger) ----------
def _charge_stats_bb_vs_side(mol, keep_mask, backbone_set):
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]
    bb_absq, side_absq = [], []
    bb = set(backbone_set)
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if not keep_mask[i]:
            continue
        q = 0.0
        if a.HasProp('_GasteigerCharge'):
            try:
                q = 0.0
                if a.HasProp('_GasteigerCharge'):
                    try:
                        q = float(a.GetProp('_GasteigerCharge'))
                    except Exception:
                        q = 0.0
                # отбрасываем NaN/Inf
                if not math.isfinite(q):
                    q = 0.0
            except Exception:
                q = 0.0
        (bb_absq if i in bb else side_absq).append(abs(q))

    def _ms(v):
        if not v:
            return [0.0, 0.0]
        a = np.asarray(v, dtype=np.float32)
        a = a[np.isfinite(a)]  # выбросить NaN/±Inf
        if a.size == 0:
            return [0.0, 0.0]
        return [float(a.mean()), float(a.std(ddof=0))]
    return _ms(bb_absq) + _ms(side_absq)  # 4 числа


# ---------- конъюгированные сегменты на бэкбоне ----------
def _bb_conjugation_stats(mol, keep_mask, backbone_set):
    """Средняя и максимальная доля длины конъюгированных отрезков на бэкбоне."""
    bb = [i for i in backbone_set] if backbone_set else []
    n = len(bb)
    if n < 1:
        return [0.0, 0.0]
    bb_set = set(bb)

    # построим граф только по бондам бэкбона, оставляя ТОЛЬКО конъюгированные связи
    adj = {i: [] for i in bb}
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if keep_mask[u] and keep_mask[v] and (u in bb_set) and (v in bb_set) and b.GetIsConjugated():
            adj[u].append(v); adj[v].append(u)

    # компоненты связности по конъюгированным бондам
    seen = set()
    sizes = []
    from collections import deque
    for s in bb:
        if s in seen:
            continue
        q = deque([s]); seen.add(s); comp = 0
        while q:
            x = q.popleft(); comp += 1
            for w in adj[x]:
                if w not in seen:
                    seen.add(w); q.append(w)
        sizes.append(comp)

    if not sizes:
        return [0.0, 0.0]
    sizes = np.asarray(sizes, dtype=np.float32)
    mean_frac = float(sizes.mean() / max(1.0, n))
    max_frac  = float(sizes.max()  / max(1.0, n))
    return [mean_frac, max_frac]


def _tpsa_bb_side(mol, backbone_set):
    """
    Сумма атомных вкладов TPSA отдельно по бэкбону и «сайдам».
    Работает устойчиво к разным вариантам возвращаемого значения RDKit.
    """
    bb = set(backbone_set or [])
    N = mol.GetNumAtoms()

    # Унификация возврата rdMolDescriptors._CalcTPSAContribs:
    # - может вернуть list[float]
    # - или tuple(...), где один из элементов — list[float] длины N
    try:
        res = Chem.rdMolDescriptors._CalcTPSAContribs(mol)
        contribs = None
        if isinstance(res, (list, tuple)):
            # случай: вернули уже список вкладов по атомам
            if len(res) == N and all(isinstance(x, (int, float)) for x in res):
                contribs = [float(x) for x in res]
            # случай: вернули кортеж, где 1-й элемент — список вкладов
            elif len(res) > 0 and isinstance(res[0], (list, tuple)):
                lst = list(res[0])
                if len(lst) == N and all(isinstance(x, (int, float)) for x in lst):
                    contribs = [float(x) for x in lst]
            # общий поиск подходящего элемента внутри кортежа
            if contribs is None and isinstance(res, tuple):
                for el in res:
                    if isinstance(el, (list, tuple)) and len(el) == N and all(isinstance(x, (int, float)) for x in el):
                        contribs = [float(x) for x in el]
                        break
        if contribs is None:
            contribs = [0.0] * N
    except Exception:
        contribs = [0.0] * N

    # страховка на несовпадение длины
    if len(contribs) != N:
        if len(contribs) < N:
            contribs = list(contribs) + [0.0] * (N - len(contribs))
        else:
            contribs = list(contribs[:N])

    t_bb = 0.0
    t_side = 0.0
    for i, c in enumerate(contribs):
        # dummy-атомы ('*') пропускаем
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 0:
            continue
        if i in bb:
            t_bb += float(c)
        else:
            t_side += float(c)
    return [t_bb, t_side]


def _hba_hbd_bb_side(mol, keep_mask, backbone_set):
    """
    Возвращает [hba_bb, hbd_bb, hba_side, hbd_side] — числа доноров/акцепторов,
    посчитанные по атомам, разделённые на бэкбон и боковые цепи.
    Использует заранее скомпилированные паттерны _HBA_P/_HBD_P.
    """
    bb = set(backbone_set or [])
    N = mol.GetNumAtoms()

    # Собираем множества совпадающих атомных индексов
    hba_atoms = set()
    for p in _HBA_P:
        for mt in mol.GetSubstructMatches(p) or []:
            hba_atoms.update(int(i) for i in mt)

    hbd_atoms = set()
    for p in _HBD_P:
        for mt in mol.GetSubstructMatches(p) or []:
            hbd_atoms.update(int(i) for i in mt)

    hba_bb = hbd_bb = hba_side = hbd_side = 0
    for i in range(N):
        if not keep_mask[i]:  # пропускаем звёзды
            continue
        if i in hba_atoms:
            if i in bb: hba_bb += 1
            else:       hba_side += 1
        if i in hbd_atoms:
            if i in bb: hbd_bb += 1
            else:       hbd_side += 1

    return [float(hba_bb), float(hbd_bb), float(hba_side), float(hbd_side)]


# ---------- «геометрия» side: ротируемость и кольцеватость на атом ----------
def _side_geom_per_atom(mol, keep_mask, backbone_set):
    bb = set(backbone_set or [])
    side_atoms = [i for i in range(mol.GetNumAtoms()) if keep_mask[i] and (i not in bb)]
    denom = max(1.0, float(len(side_atoms)))

    # ротируемые бонды на side: оба конца вне bb, single, не в кольце, не амид
    rot = 0
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if not (keep_mask[u] and keep_mask[v]):
            continue
        if (u not in bb) and (v not in bb):
            if (b.GetBondType() == rdchem.BondType.SINGLE) and (not b.IsInRing()):
                # простая амидная маска
                if not (mol.GetAtomWithIdx(u).GetAtomicNum()==7 and mol.GetAtomWithIdx(v).GetAtomicNum()==6 and any(nb.GetAtomicNum()==8 and mol.GetBondBetweenAtoms(v, nb.GetIdx()).GetBondType()==rdchem.BondType.DOUBLE for nb in mol.GetAtomWithIdx(v).GetNeighbors())) and \
                   not (mol.GetAtomWithIdx(v).GetAtomicNum()==7 and mol.GetAtomWithIdx(u).GetAtomicNum()==6 and any(nb.GetAtomicNum()==8 and mol.GetBondBetweenAtoms(u, nb.GetIdx()).GetBondType()==rdchem.BondType.DOUBLE for nb in mol.GetAtomWithIdx(u).GetNeighbors())):
                    rot += 1
    # кольцевые атомы на side
    rings = sum(1 for i in side_atoms if mol.GetAtomWithIdx(i).IsInRing())
    return [float(rot)/denom, float(rings)/denom]


# ---------- ароматические орто-замещения на бэкбоне ----------
def _bb_aromatic_ortho_stats(mol, backbone_set):
    """Считаем для каждого ароматического атома бэкбона число «орто-заместителей»
    (внешние соседи у его соседей по кольцу). Возвращаем среднее и долю >0."""
    bb = set(backbone_set or [])
    vals = []
    for i in bb:
        a = mol.GetAtomWithIdx(i)
        if not (a.GetIsAromatic() and a.IsInRing()):
            continue
        # соседи по тому же кольцу
        ring_nbrs = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetIsAromatic() and nb.IsInRing()]
        c = 0
        for j in ring_nbrs:
            # внешние (не в том же кольце) соседи j
            for nb in mol.GetAtomWithIdx(j).GetNeighbors():
                k = nb.GetIdx()
                if (not nb.IsInRing()) or (not nb.GetIsAromatic()):
                    if k != i:
                        c += 1
        vals.append(float(c))
    if not vals:
        return [0.0, 0.0]
    mean_c = float(np.mean(vals))
    frac_pos = float(np.mean([v > 0.0 for v in vals]))
    return [mean_c, frac_pos]


def _ewg_edg_balance_bb(mol, backbone_set):
    bb = set(backbone_set or [])
    if not bb:
        return [0.0, 0.0]
    def _hits(patts):
        hit_atoms = set()
        for p in patts:
            for mt in mol.GetSubstructMatches(p) or []:
                hit_atoms.update(mt)
        return hit_atoms

    ewg = _hits(_EWG_P); edg = _hits(_EDG_P)
    # считаем попавшие именно на бэкбон
    ewg_bb = len([i for i in ewg if i in bb])
    edg_bb = len([i for i in edg if i in bb])
    balance = float(ewg_bb - edg_bb) / max(1.0, float(len(bb)))
    # доля bb-атомов, имеющих EWG/EDG в радиусе 1 (сам атом или его сосед)
    tagged = set(i for i in bb if (i in ewg) or (i in edg))
    for i in bb:
        if i in tagged:
            continue
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nb.GetIdx()
            if (j in ewg) or (j in edg):
                tagged.add(i); break
    frac_tagged = float(len(tagged)) / max(1.0, float(len(bb)))
    return [balance, frac_tagged]


# ---------- спектральные признаки лапласианы бэкбона ----------
def _bb_laplacian_eigs(mol, keep_mask, backbone_set, k=3):
    bb = [i for i in backbone_set] if backbone_set else []
    n = len(bb)
    if n < 2:
        return [0.0]*k
    idx2pos = {i: p for p, i in enumerate(bb)}
    A = np.zeros((n, n), dtype=np.float64)
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if keep_mask[u] and keep_mask[v] and (u in idx2pos) and (v in idx2pos):
            iu, iv = idx2pos[u], idx2pos[v]
            A[iu, iv] = 1.0; A[iv, iu] = 1.0
    d = A.sum(axis=1)
    d[d < 1e-12] = 1.0
    Dm12 = np.diag(1.0/np.sqrt(d))
    L = np.eye(n) - Dm12 @ A @ Dm12
    try:
        w = np.linalg.eigvalsh(L)  # симметричная матрица
    except Exception:
        w = np.linalg.eigvals(L).real
    w = np.clip(np.sort(w), 0.0, 2.0)
    # берём первые k НЕ нулевых
    nz = w[w > 1e-8]
    out = (nz[:k] if nz.size >= k else np.pad(nz, (0, k - nz.size), constant_values=0.0)).astype(np.float64)
    return [float(x) for x in out]


# ---------- масса «закрытого» повторяющегося звена ----------
def _closed_cru_molwt(mol):
    m2 = strip_stars_and_add_hs(mol)
    if m2 is None:
        return 0.0
    try:
        return float(Descriptors.MolWt(m2))
    except Exception:
        return 0.0

# ---------- доля гетероатомов на бэкбоне ----------
def _bb_hetero_frac(mol, backbone_set):
    if not backbone_set:
        return 0.0
    bb = list(backbone_set)
    hetero = 0
    for i in bb:
        z = mol.GetAtomWithIdx(i).GetAtomicNum()
        if z in (7, 8, 16):  # N, O, S
            hetero += 1
    return float(hetero) / float(len(bb))

# ---------- биграммы вдоль primary_path (хеш-бакинг) ----------
def _bb_node_token(atom: rdchem.Atom) -> str:
    sym = atom.GetSymbol()
    hyb = atom.GetHybridization()
    hyb_s = {rdchem.HybridizationType.SP: "sp",
             rdchem.HybridizationType.SP2: "sp2",
             rdchem.HybridizationType.SP3: "sp3"}.get(hyb, "oth")
    het = ("H" if atom.GetAtomicNum() in (7,8,16,9,17,35,53) else "C")
    aro = ("a" if atom.GetIsAromatic() else "al")
    return f"{sym}|{hyb_s}|{het}|{aro}"

def _bb_ngram_vector(mol, primary_path, K=BB_NGRAM_TOPK):
    # считаем биграммы по последовательности токенов вдоль primary_path
    if len(primary_path) < 2:
        return [0.0]*K
    toks = [_bb_node_token(mol.GetAtomWithIdx(i)) for i in primary_path]
    counts = np.zeros(K, dtype=np.float32)
    for a, b in zip(toks[:-1], toks[1:]):
        s = f"{a}->{b}"
        h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % K
        counts[h] += 1.0
    # нормируем на число биграмм
    norm = max(1.0, len(toks)-1)
    return (counts / norm).tolist()

# ---------- Компоновщик всех новых глобальных фич ----------
def _poly_extra_descriptors(mol, keep_mask, backbone_set, primary_path):
    # B1 (4) + B2 (2) + B3 (4) + F1 (1) + F2 (1) + F3 (1) + B4 (K)
    b1 = _bb_functional_fractions(mol, backbone_set)                   # 4
    b2_bulk_avg, b2_bulk_max, f3_avg_depth = _sidechain_bulk_and_depth(mol, keep_mask, backbone_set)  # 2 + 1
    b3 = _charge_stats_bb_vs_side(mol, keep_mask, backbone_set)        # 4
    f1 = [_closed_cru_molwt(mol)]                                      # 1
    f2 = [_bb_hetero_frac(mol, backbone_set)]                          # 1
    f3 = [f3_avg_depth]                                                # 1
    b4 = _bb_ngram_vector(mol, primary_path, K=BB_NGRAM_TOPK)          # K

    cseg = _bb_conjugation_stats(mol, keep_mask, backbone_set)  # 2
    tpsa = _tpsa_bb_side(mol, backbone_set)  # 2
    hba_hbd = _hba_hbd_bb_side(mol, keep_mask, backbone_set)  # 4
    side_geom = _side_geom_per_atom(mol, keep_mask, backbone_set)  # 2
    ortho = _bb_aromatic_ortho_stats(mol, backbone_set)  # 2
    ewg_edg = _ewg_edg_balance_bb(mol, backbone_set)  # 2
    spec = _bb_laplacian_eigs(mol, keep_mask, backbone_set, k=3)  # 3

    return (list(b1) + [b2_bulk_avg, b2_bulk_max] + list(b3) + f1 + f2 + f3 + list(b4) +
            list(cseg) + list(tpsa) + list(hba_hbd) + list(side_geom) +
            list(ortho) + list(ewg_edg) + list(spec))


def _poly_backbone_descriptors(mol, keep_mask, backbone_set):
    """
    D-11: дескрипторы по бэкбону (возвращает список из BASE_BB_DIM чисел).
    """
    if not backbone_set:
        return [0.0]*BASE_BB_DIM
    bb = set(backbone_set)
    # длина (число атомов)
    bb_len = float(len(bb))

    # доли arom и sp3 на бэкбоне
    arom = 0; sp3 = 0
    for i in bb:
        a = mol.GetAtomWithIdx(i)
        if a.GetIsAromatic(): arom += 1
        if a.GetHybridization() == rdchem.HybridizationType.SP3: sp3 += 1
    bb_arom_frac = arom / bb_len
    bb_sp3_frac  = sp3 / bb_len

    # "вращательные" связи на бэкбоне (грубая эвристика)
    rot = 0; bonds_on_bb = 0
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u in bb and v in bb:
            bonds_on_bb += 1
            if (b.GetBondType() == rdchem.BondType.SINGLE) and (not b.IsInRing()):
                rot += 1
    bb_rot_per_len = (rot / max(bb_len - 1.0, 1.0))
    bb_rot_frac = float(rot) / max(float(bonds_on_bb), 1.0)

    # ветвистость: степени "наружу" (соседи не на бэкбоне)
    branch_counts = []
    for i in bb:
        a = mol.GetAtomWithIdx(i)
        c = 0
        for nb in a.GetNeighbors():
            if keep_mask[nb.GetIdx()] and (nb.GetIdx() not in bb):
                c += 1
        branch_counts.append(c)
    bb_branch_avg = float(np.mean(branch_counts)) if branch_counts else 0.0
    bb_branch_max = float(np.max(branch_counts)) if branch_counts else 0.0

    return [bb_len, bb_arom_frac, bb_sp3_frac, bb_rot_per_len, bb_rot_frac, bb_branch_avg, bb_branch_max]


# def sanitize_polymer_smiles(s: str) -> str:
#     if not isinstance(s, str):
#         return s
#     # [R], [R1], [R'], [R2] → *
#     s = re.sub(r"\[R[0-9']*\]", "*", s)
#     # иногда встречаются «косые» апострофы
#     s = s.replace("’", "'")
#     # # RDKit терпит '*', но валентности могут ломаться → заменим на углерод
#     # s = s.replace("*", "C")
#     return s

def canonical_poly_smiles(smiles: str) -> str:
    m = safe_mol_from_smiles(smiles)
    if m is None:
        return smiles
    # нормализуем полимер: убираем '*' и добавляем H, чтобы канонизация была стабильной
    m2 = strip_stars_and_add_hs(m) or m
    try:
        return Chem.MolToSmiles(m2, canonical=True)
    except Exception:
        return smiles

def smiles_dedup_key(smiles: str, mode: str = "canon") -> str:
    if mode == "smiles":
        return smiles
    if mode == "inchi":
        try:
            return Chem.MolToInchiKey(strip_stars_and_add_hs(safe_mol_from_smiles(smiles)))
        except Exception:
            return smiles
    # default: canonical
    return canonical_poly_smiles(smiles)


def safe_mol_from_smiles(smiles: str):
    try:
        # NEW: если есть плейсхолдеры в квадратных скобках — выкидываем молекулу
        if isinstance(smiles, str) and re.search(r"\[R[0-9']*\]", smiles):
            return None

        mol = Chem.MolFromSmiles(smiles)
        # if mol is None:
        #     # пробуем санитайзинг полимерных плейсхолдеров
        #     fixed = sanitize_polymer_smiles(smiles)
        #     if fixed != smiles:
        #         mol = Chem.MolFromSmiles(fixed)
        if mol is None:
            return None
        try:
            Chem.Kekulize(mol, clearAromaticFlags=False)
        except Exception:
            pass  # иногда не критично
        return mol
    except Exception:
        return None


def ssl_encoder_state_dict(model: nn.Module) -> dict:
    # сохраняем всё, кроме головы и rho
    sd = {}
    for k, v in model.state_dict().items():
        if k.startswith("head."):
            continue
        if k == "rho":
            continue
        sd[k] = v
    return sd


# =============================
# Featurization
# =============================
def atom_features(atom: rdchem.Atom):
    common = [0,1,5,6,7,8,9,14,15,16,17,35,53]
    num = atom.GetAtomicNum()
    degree = atom.GetTotalDegree()
    valence = atom.GetTotalValence()
    formal = atom.GetFormalCharge()
    aromatic = atom.GetIsAromatic()
    in_ring = atom.IsInRing()
    hyb = atom.GetHybridization()
    chiral = atom.GetChiralTag()

    pt = Chem.GetPeriodicTable()
    mass = pt.GetAtomicWeight(num) if num>0 else 0.0
    vdw = pt.GetRvdw(num) if num>0 else 0.0
    # electronegativity proxy: Pauling if available else 0
    try:
        en = pt.GetRcovalent(num)  # not EN but correlated; avoid extra deps
    except Exception:
        en = 0.0

    feats = []
    feats += one_hot(num if num in common else -1, common + [-1])
    feats += one_hot(min(degree, 5), list(range(6)))
    feats += one_hot(min(valence, 6), list(range(7)))
    feats += one_hot(int(max(-2, min(2, formal)))+2, list(range(5)))
    feats += [1.0 if aromatic else 0.0, 1.0 if in_ring else 0.0]
    feats += one_hot(hyb, [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3,
                           rdchem.HybridizationType.SP3D, rdchem.HybridizationType.SP3D2, None])
    feats += one_hot(chiral, [rdchem.ChiralType.CHI_UNSPECIFIED,
                              rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                              rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                              rdchem.ChiralType.CHI_OTHER])
    feats += one_hot(min(atom.GetTotalNumHs(),4), list(range(5)))
    # real-valued
    feats += [mass/200.0, vdw/2.5, en/2.0, float(atom.GetImplicitValence())/6.0, float(atom.GetNumRadicalElectrons())/2.0]
    return feats

def one_hot_enum(x, choices):
    # всегда добавляем "OTHER" в конец
    v = [0]* (len(choices) + 1)
    try:
        idx = choices.index(x)
    except ValueError:
        idx = len(choices)  # OTHER
    v[idx] = 1
    return v

def bond_features(bond: Optional[rdchem.Bond], is_poly_edge: bool = False):
    """Фичи ребра; для виртуального ребра bond=None и is_poly_edge=True."""
    if bond is None:
        # 'OTHER' / нули для атрибутов и флаг виртуальности
        feats = one_hot_enum(None,
                             [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
                              rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC])
        feats += [0.0, 0.0]  # conjugated, in_ring
        feats += one_hot_enum(None,
                              [rdchem.BondStereo.STEREONONE, rdchem.BondStereo.STEREOZ, rdchem.BondStereo.STEREOE])
        feats += one_hot_enum(None,
                              [rdchem.BondDir.NONE, rdchem.BondDir.BEGINWEDGE, rdchem.BondDir.BEGINDASH,
                               rdchem.BondDir.ENDDOWNRIGHT, rdchem.BondDir.ENDUPRIGHT])
    else:
        feats = one_hot_enum(bond.GetBondType(),
                             [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
                              rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC])
        feats += [1.0 if bond.GetIsConjugated() else 0.0, 1.0 if bond.IsInRing() else 0.0]
        feats += one_hot_enum(bond.GetStereo(),
                              [rdchem.BondStereo.STEREONONE, rdchem.BondStereo.STEREOZ, rdchem.BondStereo.STEREOE])
        feats += one_hot_enum(bond.GetBondDir(),
                              [rdchem.BondDir.NONE, rdchem.BondDir.BEGINWEDGE, rdchem.BondDir.BEGINDASH,
                               rdchem.BondDir.ENDDOWNRIGHT, rdchem.BondDir.ENDUPRIGHT])

    # новый последний признак
    feats += [1.0 if is_poly_edge else 0.0]
    return feats

# Базовая длина, возвращаемая bond_features(...). ВАЖНО: включает последний бит is_poly_edge.
BOND_BASE_DIM = len(bond_features(None, True))

# Полная длина edge_attr: base + EXTRA_EDGE_FEATS ([is_khop, khop_norm, is_rot, is_on_bb_pair]) + is_bb_bond
EDGE_FEAT_DIM = BOND_BASE_DIM + EXTRA_EDGE_FEATS + 1

# Удобные индексы (чтобы не резать «на глаз» внутри моделей):
IDX_IS_POLY     = BOND_BASE_DIM - 1
IDX_KHOP        = BOND_BASE_DIM + 0
IDX_KHOP_NORM   = BOND_BASE_DIM + 1
IDX_IS_ROT      = BOND_BASE_DIM + 2
IDX_ON_BB_PAIR  = BOND_BASE_DIM + 3
IDX_IS_BB_BOND  = BOND_BASE_DIM + 4

# def mol_global_features(mol: Optional[Chem.Mol]):
#     if mol is None:
#         return [0.0]*len(GLOBAL_DESCS)
#     try:
#         m = mol
#         # если есть '*', считаем дескрипторы на "закрытой" версии
#         if has_star(m):
#             m_cap = cap_polymer_stars(m, cap="[H]") or cap_polymer_stars(m, cap="[CH3]")
#             if m_cap is not None:
#                 m = m_cap
#             else:
#                 # как fallback — не рушим пайплайн
#                 return [0.0]*len(GLOBAL_DESCS)
#         return [float(fn(m)) for fn in GLOBAL_DESCS]
#     except Exception:
#         return [0.0]*len(GLOBAL_DESCS)

def mol_global_features(mol: Optional[Chem.Mol]):
    if mol is None:
        return [0.0]*len(GLOBAL_DESCS)
    try:
        m = mol
        if has_star(m):
            m2 = strip_stars_and_add_hs(m)
            if m2 is None:
                return [0.0]*len(GLOBAL_DESCS)
            m = m2
        vals = [float(fn(m)) for fn in GLOBAL_DESCS]
        # фикс: заменяем nan/inf на 0
        vals = [v if math.isfinite(v) else 0.0 for v in vals]
        return vals
    except Exception:
        return [0.0]*len(GLOBAL_DESCS)

def _infer_feat_dims() -> Tuple[int, int]:
    try:
        g = smiles_to_graph("CC")  # всегда валидный граф без звёзд
        if g is None:
            raise RuntimeError("smiles_to_graph returned None")
        x, _, ea, _ = g
        node_dim = int(x.size(1))
        edge_dim = int(ea.size(1)) if ea.numel() > 0 else len(bond_features(None, is_poly_edge=True)) + 1
        return node_dim, edge_dim
    except Exception:
        return 54 + EXTRA_NODE_FEATS, len(bond_features(None, True)) + EXTRA_EDGE_FEATS + 1


NODE_FEAT_DIM, EDGE_FEAT_DIM = _infer_feat_dims()

def _connected_components_without_stars(mol, keep_mask):
    # строим списки смежности только по не-звёздам
    N = mol.GetNumAtoms()
    adj = [[] for _ in range(N)]
    for b in mol.GetBonds():
        u = b.GetBeginAtomIdx(); v = b.GetEndAtomIdx()
        if keep_mask[u] and keep_mask[v]:
            adj[u].append(v); adj[v].append(u)
    comp = [-1]*N
    cid = 0
    from collections import deque
    for i in range(N):
        if keep_mask[i] and comp[i] == -1:
            q = deque([i]); comp[i] = cid
            while q:
                u = q.popleft()
                for w in adj[u]:
                    if comp[w] == -1:
                        comp[w] = cid; q.append(w)
            cid += 1
    return comp  # длина N, -1 для звёзд

def _add_poly_edges_for_group(idx_list, old2new, src, dst, eattr, mode="cycle"):
    idx_list = [int(old2new[i]) for i in idx_list if int(old2new[i]) >= 0]
    if len(idx_list) < 2:
        return
    ea_poly = edge_attr_build(None, is_poly=True, is_bb_bond=0.0)

    if mode == "pair":
        for i in range(0, len(idx_list)-1, 2):
            u, v = idx_list[i], idx_list[i+1]
            src += [u, v]; dst += [v, u]; # eattr += [bf, bf]
            eattr += [ea_poly, ea_poly]
    elif mode == "cycle":
        for i in range(len(idx_list)):
            u, v = idx_list[i], idx_list[(i+1) % len(idx_list)]
            src += [u, v]; dst += [v, u]; # eattr += [bf, bf]
            eattr += [ea_poly, ea_poly]
    elif mode == "clique":
        for i in range(len(idx_list)):
            for j in range(i+1, len(idx_list)):
                u, v = idx_list[i], idx_list[j]
                src += [u, v]; dst += [v, u]; # eattr += [bf, bf]
                eattr += [ea_poly, ea_poly]

def augment_smiles_polymer_local(smiles: str, p: float = 0.2, seed: int = 0) -> str:
    """
    С вероятностью p применяет ОДНУ случайную локальную замену вне бэкбона:
      - Me <-> Et на терминале сайд-чейна
      - F <-> Cl (галоалкан на сайд-чейн-углероде)
      - OMe <-> OEt (алкоксигруппа в боковой цепи)
    Возвращает валидный SMILES (или исходный при неуспехе).
    """
    if p <= 1e-6:
        return smiles
    rng = np.random.default_rng(seed)
    m = safe_mol_from_smiles(smiles)
    if m is None:
        return smiles

    try:
        # распознаём «звёзды» и бэкбон
        N = m.GetNumAtoms()
        star_idxs = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
        keep_mask = np.ones(N, dtype=bool);
        for i in star_idxs: keep_mask[i] = False

        comp, adj = _components_no_stars(m, keep_mask)
        # соседи звёзд (как в smiles_to_graph)
        star_neighbors = []
        for si in star_idxs:
            a = m.GetAtomWithIdx(si)
            nbrs = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetAtomicNum() != 0]
            if len(nbrs) >= 1:
                star_neighbors.append(nbrs[0])
        backbone_set, _ = _backbone_atoms_by_cycle_paths(m, comp, keep_mask, star_neighbors)
        bb = set(backbone_set)

        # кандидаты под замену = атомы НЕ на бэкбоне
        side_mask = np.array([keep_mask[i] and (i not in bb) for i in range(N)], dtype=bool)

        # --- подстановки по SMARTS/реакциям ---
        rxns = []

        # F <-> Cl (только если атом не на бэкбоне)
        # Реализуем как прямую модификацию атома:
        def _swap_halogen(mm, z_from, z_to):
            em = Chem.EditableMol(mm)
            changed = False
            for a in mm.GetAtoms():
                i = a.GetIdx()
                if side_mask[i] and a.GetAtomicNum() == z_from:
                    # проверим валентность (одинарная связь допустима)
                    changed = True
                    em.ReplaceAtom(i, Chem.Atom(z_to))
            return em.GetMol() if changed else None

        # OMe <-> OEt через реакции
        rxn_OMe_to_OEt = AllChem.ReactionFromSmarts("[O:1]-[CH3:2]>>[O:1]-[CH2:2]-C")
        rxn_OEt_to_OMe = AllChem.ReactionFromSmarts("[O:1]-[CH2:2]-[CH3:3]>>[O:1]-[CH3:2]")

        # Me <-> Et на терминале сайд-чейна (грубая эвристика)
        rxn_Me_to_Et = AllChem.ReactionFromSmarts("[CH3:1]-[*:2]>>[CH2:1]-[CH3]-[*:2]")
        rxn_Et_to_Me = AllChem.ReactionFromSmarts("[CH2:1]-[CH3:2]-[*:3]>>[CH3:1]-[*:3]")

        # список действий (каждое — callable(m) -> m' или None)
        actions = []

        def act_halogen_swap(mm, a, b):
            m2 = _swap_halogen(mm, a, b)
            if m2: Chem.SanitizeMol(m2, catchErrors=True)
            return m2

        actions.append(lambda mm: act_halogen_swap(mm, 9, 17))   # F->Cl
        actions.append(lambda mm: act_halogen_swap(mm, 17, 9))   # Cl->F

        def _apply_rxn_on_side(mm, rxn):
            # фильтруем продукты: оставляем те, у которых изменённые атомы вне бэкбона
            prods = rxn.RunReactants((mm,))
            for tpl in prods:
                m2 = tpl[0]
                try:
                    Chem.SanitizeMol(m2, catchErrors=True)
                except Exception:
                    continue
                # простая проверка связности сохраняется автоматически
                return m2
            return None

        actions.append(lambda mm: _apply_rxn_on_side(mm, rxn_OMe_to_OEt))
        actions.append(lambda mm: _apply_rxn_on_side(mm, rxn_OEt_to_OMe))
        actions.append(lambda mm: _apply_rxn_on_side(mm, rxn_Me_to_Et))
        actions.append(lambda mm: _apply_rxn_on_side(mm, rxn_Et_to_Me))

        if rng.random() < p:
            order = rng.permutation(len(actions))
            for k in order:
                m_try = actions[k](m)
                if m_try is not None:
                    m = m_try
                    break

        out = Chem.MolToSmiles(m, canonical=True)
        return out or smiles
    except Exception:
        return smiles


def smiles_to_graph(smiles: str, poly_edge_mode: Optional[str] = None):
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return None

    # --- Находим звёзды и их соседей ---
    N_atoms = mol.GetNumAtoms()
    star_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    keep_mask = np.ones(N_atoms, dtype=bool); keep_mask[star_idxs] = False

    # соседи звёзд, которые НЕ звёзды (обычно ровно по одному на звезду)
    star_neighbors = []
    for si in star_idxs:
        a = mol.GetAtomWithIdx(si)
        nbrs = [nb.GetIdx() for nb in a.GetNeighbors() if nb.GetAtomicNum() != 0]
        # берём первый валидный (в норме он один)
        if len(nbrs) >= 1:
            star_neighbors.append(nbrs[0])

    # --- Формируем отображение старых индексов -> новые (без звёзд) ---
    old2new = -np.ones(N_atoms, dtype=int)
    new_idx = 0
    for i in range(N_atoms):
        if keep_mask[i]:
            old2new[i] = new_idx
            new_idx += 1

    # Если вдруг все узлы оказались звёздами — защитимся
    if new_idx == 0:
        # фоллбек: пустой граф
        x = torch.zeros((1, NODE_FEAT_DIM), dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)
        # gdesc длиной base+extra
        gdesc = torch.zeros((TOTAL_GDESC_DIM,), dtype=torch.float32)
        return x, edge_index, edge_attr, gdesc

    # --- Узловые признаки (без звёзд) ---
    # базовые узловые фичи
    base_feats = []
    for i in range(N_atoms):
        if keep_mask[i]:
            base_feats.append(atom_features(mol.GetAtomWithIdx(i)))

    # x = torch.tensor(x, dtype=torch.float32)

    # --- компоненты и бэкбон ---
    comp, adj = _components_no_stars(mol, keep_mask)
    backbone_set, primary_path = _backbone_atoms_by_cycle_paths(mol, comp, keep_mask, star_neighbors)

    # расстояния до бэкбона (для всех не-звёзд)
    dist2bb_full = _dist_to_set(adj, keep_mask, backbone_set if backbone_set else set())
    # расстояния до ближайшей звезды среди не-звёзд: считаем от соседей звёзд
    dist2star_full = _dist_to_set(adj, keep_mask, set(star_neighbors) if len(star_neighbors)>0 else set())

    # --- Обычные рёбра между «не-звёздами» + флаг is_bb_bond ---
    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        u0 = b.GetBeginAtomIdx()
        v0 = b.GetEndAtomIdx()
        if keep_mask[u0] and keep_mask[v0]:
            u = int(old2new[u0]);
            v = int(old2new[v0])
            is_bb = 1.0 if ((u0 in backbone_set) and (v0 in backbone_set)) else 0.0
            is_rot = 1.0 if (b.GetBondType() == rdchem.BondType.SINGLE and (not b.IsInRing()) and (
                not b.GetIsConjugated())) else 0.0
            ea_full = edge_attr_build(
                b,
                is_poly=False,
                is_khop=0.0,
                khop_norm=0.0,
                is_rot=is_rot,
                is_bb_pair=is_bb,
                is_bb_bond=is_bb,
            )
            src += [u, v]
            dst += [v, u]
            eattr += [ea_full, ea_full]
        # связи, где участвует звезда, намеренно пропускаем

    # режим берём из args, по умолчанию 'cycle'
    mode = poly_edge_mode or "cycle"
    # сгруппируем соседей звёзд по компонентам
    groups = {}
    for n in star_neighbors:
        cid = comp[n] if n < len(comp) else -1
        if cid >= 0:
            groups.setdefault(cid, []).append(n)
    for _, idx_list in groups.items():
        _add_poly_edges_for_group(idx_list, old2new, src, dst, eattr, mode=mode)

    _add_khop_backbone_edges(adj, keep_mask, backbone_set, old2new, src, dst, eattr,
                             kmin=KHOP_MIN, kmax=KHOP_MAX)

    # --- позиционные фичи вдоль основного пути ---
    pos_index = {i: 0 for i in range(N_atoms) if keep_mask[i]}  # по умолчанию 0
    L = max(1, len(primary_path))
    if L >= 2:
        for k, atom_idx in enumerate(primary_path):
            pos_index[atom_idx] = k
    # sin/cos
    def _pos_pair(i):
        k = pos_index.get(i, 0)
        return (math.sin(2*math.pi*k/L), math.cos(2*math.pi*k/L))

    # --- Подготовка новых локальных 2D-признаков ---

    # 1) Конъюгированные сегменты на бэкбоне
    conj_norm_full, conj_frac_full = _conjugated_backbone_segments(mol, backbone_set, keep_mask)

    # 2) EWG/EDG окружение
    adj_no_stars = adj  # уже построенный без звёзд
    ewg_anchors = _match_anchor_atoms(mol, _EWG_P)
    edg_anchors = _match_anchor_atoms(mol, _EDG_P)
    ewg_r1_full, ewg_r2_full = _counts_within_r1_r2(adj_no_stars, ewg_anchors)
    edg_r1_full, edg_r2_full = _counts_within_r1_r2(adj_no_stars, edg_anchors)

    # 3) H-bond роли
    is_hbd_full, is_hba_full = _hb_flags_per_atom(mol)
    hbd_nb1_full = _neighbor_counts(adj_no_stars, is_hbd_full, clip=3)
    hba_nb1_full = _neighbor_counts(adj_no_stars, is_hba_full, clip=3)

    # 4) Локальные топо-индексы
    randic_r1_full, chi1_full = _local_topo_indices(adj_no_stars)

    # --- собираем расширенные узловые фичи ---
    x = []
    for i in range(N_atoms):
        if keep_mask[i]:
            bb_flag = 1.0 if (i in backbone_set) else 0.0
            # clamp до 3 и нормируем в [0,1]
            d_bb = dist2bb_full[i];
            d_bb = 3 if d_bb > 3 else d_bb
            d_bb = 0 if d_bb >= 10**8 else d_bb
            d_bb = float(d_bb)/3.0

            star_nb = 1.0 if (i in star_neighbors) else 0.0

            d_st = dist2star_full[i]
            d_st = 3 if d_st > 3 else d_st
            d_st = 0 if d_st >= 10**8 else d_st
            d_st = float(d_st)/3.0

            ps, pc = _pos_pair(i)

            x.append(
                base_feats[old2new[i]] + [
                    bb_flag, d_bb, star_nb, d_st, ps, pc,
                    conj_norm_full[i], conj_frac_full[i],
                    ewg_r1_full[i], ewg_r2_full[i], edg_r1_full[i], edg_r2_full[i],
                    is_hbd_full[i], is_hba_full[i], hbd_nb1_full[i], hba_nb1_full[i],
                    randic_r1_full[i], chi1_full[i],
                ]
            )

    x = torch.tensor(x, dtype=torch.float32)

    edge_index = torch.tensor([src, dst], dtype=torch.long) if len(src) else torch.zeros((2,0), dtype=torch.long)
    edge_attr = torch.tensor(eattr, dtype=torch.float32) if len(eattr) else torch.zeros((0, EDGE_FEAT_DIM),
                                                                                        dtype=torch.float32)
    edge_attr = ensure_edge_attr_dim_tensor(edge_attr)  # ← страховка, чтобы никогда не падать на cat()

    # --- Глобальные дескрипторы как и раньше (со снятием звёзд) ---
    g_base = mol_global_features(mol)
    g_bb = _poly_backbone_descriptors(mol, keep_mask, backbone_set)

    g_ext = _poly_extra_descriptors(mol, keep_mask, backbone_set, primary_path)
    gdesc = torch.tensor(list(g_base) + list(g_bb) + list(g_ext), dtype=torch.float32)

    return x, edge_index, edge_attr, gdesc


def compute_scaffold(smiles: str) -> str:
    m = safe_mol_from_smiles(smiles)
    if m is None:
        return ""
    # для скелета: сперва пытаемся закрыть метилом, иначе – удалить звезды и добавить H
    m2 = cap_polymer_stars(m, cap="[CH3]") or strip_stars_and_add_hs(m) or m
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m2, includeChirality=False)
    except Exception:
        # запасной вариант – канонический SMILES после капа, чтобы группа не терялась
        try:
            return Chem.MolToSmiles(m2, canonical=True)
        except Exception:
            return ""


# =============================
# Dataset / Collate
# =============================
@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    gdesc: torch.Tensor
    y: torch.Tensor
    y_mask: torch.Tensor
    id: int


class TensorUnlabeledDataset(torch.utils.data.Dataset):
    """Хранит кортежи (x, edge_index, edge_attr, gdesc). Никакого RDKit."""
    def __init__(self, items):
        self.items = items  # list of tuples
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]

# === one-shot materialization for SSL ===
def materialize_unlabeled_dataset(smiles_list, poly_edge_mode="cycle", desc="unl", cache_path=None):
    """
    Строит тензоры один раз. Если cache_path задан и файл существует — читает его.
    Возвращает TensorUnlabeledDataset и путь до кеша (если был указан).
    """
    items = None
    if cache_path and os.path.isfile(cache_path):
        try:
            items = torch.load(cache_path, map_location="cpu", weights_only=True)
        except TypeError:
            items = torch.load(cache_path, map_location="cpu")

    if items is None:
        ds = UnlabeledDS(smiles_list, poly_edge_mode=poly_edge_mode)
        items = []
        it = range(len(ds))
        try:
            from tqdm import tqdm
            it = tqdm(it, desc=f"[BUILD] {desc}")
        except Exception:
            pass
        for i in it:
            # здесь RDKit вызывается ОДИН раз на SMILES
            x, ei, ea, gdesc = ds[i]
            items.append((x, ei, ea, gdesc))

        if cache_path:
            torch.save(items, cache_path)

    return TensorUnlabeledDataset(items), cache_path

# === helper for stable cache file name ===
def _ssl_cache_path(unl_df: pd.DataFrame, out_dir: str, poly_edge_mode: str, tag: str = "") -> str:
    smi = unl_df["SMILES"].astype(str).tolist()
    key = "||".join(smi) + f"|{poly_edge_mode}|{len(smi)}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
    suffix = f"_{tag}" if tag else ""
    return os.path.join(out_dir, f"cache_unlabeled_{poly_edge_mode}_{h}{suffix}.pt")


class TensorGraphDataset(torch.utils.data.Dataset):
    """Хранит уже собранные тензоры графов; никакого RDKit в __getitem__."""
    def __init__(self, items):
        # items: list of (x, ei, ea, gdesc, y, ymask)
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        x, ei, ea, g, y, m = self.items[i]
        return GraphData(x, ei, ea, g, y, m, i)

def materialize_dataset(ds, desc=""):
    """Один раз строим тензоры и складываем в список (можно потом сохранить на диск)."""
    mat = []
    it = range(len(ds))
    try:
        from tqdm import tqdm
        it = tqdm(it, desc=f"[BUILD] {desc}")
    except Exception:
        pass
    for i in it:
        g = ds[i]  # здесь RDKit вызовется ОДИН раз на элемент
        mat.append((g.x, g.edge_index, g.edge_attr, g.gdesc, g.y, g.y_mask))
    return TensorGraphDataset(mat)


class UnlabeledDS(Dataset):
    def __init__(self, smiles_list, poly_edge_mode="cycle",
                 aug_local_p: float = 0.0, aug_local_dynamic: bool = False, seed: int = 0):
        self.smiles = smiles_list
        self.poly_edge_mode = poly_edge_mode
        self.aug_local_p = aug_local_p
        self.aug_local_dynamic = aug_local_dynamic
        self.rng = np.random.default_rng(seed)

    def __len__(self): return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        if self.aug_local_dynamic and (self.aug_local_p > 0.0) and (self.rng.random() < self.aug_local_p):
            smi = augment_smiles_polymer_local(smi, p=self.aug_local_p, seed=int(self.rng.integers(0, 1<<31)))
        g = smiles_to_graph(smi, self.poly_edge_mode)
        if g is None:
            x = torch.zeros((1, NODE_FEAT_DIM), dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)
            gdesc = torch.zeros((TOTAL_GDESC_DIM,), dtype=torch.float32)
        else:
            x, edge_index, edge_attr, gdesc = g
        return x, edge_index, edge_attr, gdesc


def collate_unl(batch):
    xs, eis, eas, gdescs, bvecs = [], [], [], [], []
    node_offset = 0
    for i, (x, ei, ea, gdesc) in enumerate(batch):
        N = x.size(0)
        xs.append(x)
        bvecs.append(torch.full((N,), i, dtype=torch.long))
        if ei.numel() > 0:
            eis.append(ei + node_offset); eas.append(ea)
        node_offset += N
        gdescs.append(gdesc.unsqueeze(0))
    x = torch.cat(xs, dim=0)
    ei = torch.cat(eis, dim=1) if eis else torch.zeros((2,0), dtype=torch.long)
    ea = torch.cat(eas, dim=0) if eas else torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)
    bvec = torch.cat(bvecs, dim=0)

    # gdescs: список 1D или [1, D] тензоров разной длины
    _gdescs = [gd.reshape(-1).to(torch.float32) for gd in gdescs]
    lens = [int(t.numel()) for t in _gdescs]
    L_max = max(lens)

    # # (необязательно) Раз в батч печатать статистику – оставьте, если полезно
    # if len(set(lens)) > 1:
    #     print(f"[COLLATE_UNL] gdesc lengths in batch: {lens} | bad idx: "
    #           f"{[i for i, L in enumerate(lens) if L != L_max]}")

    # Паддинг справа нулями до L_max и укладка в [B, L_max]
    gdesc = torch.stack([F.pad(t, (0, L_max - t.numel())) for t in _gdescs], dim=0)

    return x, ei, ea, bvec, gdesc

class PolymerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_cols: List[str], poly_edge_mode="cycle"):
        self.rows = df.reset_index(drop=True)
        self.target_cols = target_cols
        self.poly_edge_mode = poly_edge_mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        smi = row["SMILES"]
        gid = int(row["id"]) if "id" in row else idx
        g = smiles_to_graph(smi, self.poly_edge_mode)
        if g is None:
            x = torch.zeros((1, NODE_FEAT_DIM), dtype=torch.float32)  # ← было 40
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)  # ← было 12
            gdesc = torch.zeros((TOTAL_GDESC_DIM,), dtype=torch.float32)

        else:
            x, edge_index, edge_attr, gdesc = g

        if all(c in row for c in self.target_cols):
            y = torch.tensor([float(row[c]) if pd.notna(row[c]) else float("nan")
                              for c in self.target_cols], dtype=torch.float32)
            y_mask = torch.tensor([pd.notna(row[c]) for c in self.target_cols], dtype=torch.bool)
        else:
            y = torch.full((len(self.target_cols),), float("nan"))
            y_mask = torch.zeros((len(self.target_cols),), dtype=torch.bool)

        return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr, gdesc=gdesc, y=y, y_mask=y_mask, id=gid)

def collate(batch: List[GraphData]):
    xs, eis, eas = [], [], []
    ys, ymasks = [], []
    batch_idx = []
    gdescs = []
    node_offset = 0
    for i, g in enumerate(batch):
        N = g.x.size(0)
        xs.append(g.x)
        batch_idx.append(torch.full((N,), i, dtype=torch.long))
        if g.edge_index.numel() > 0:
            eis.append(g.edge_index + node_offset)
            eas.append(g.edge_attr)
        else:
            eis.append(torch.zeros((2, 0), dtype=torch.long))
            eas.append(torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32))
        ys.append(g.y.unsqueeze(0))
        ymasks.append(g.y_mask.unsqueeze(0))
        gdescs.append(g.gdesc.unsqueeze(0))
        node_offset += N
    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if eis else torch.zeros((2,0), dtype=torch.long)
    edge_attr  = torch.cat(eas, dim=0) if eas else torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)
    y = torch.cat(ys, dim=0)
    y_mask = torch.cat(ymasks, dim=0)
    batch_vec = torch.cat(batch_idx, dim=0)

    # плоские вектора и длины
    _lens = [int(g.reshape(-1).numel()) for g in gdescs]

    # выберем «целевую» длину один раз и запомним в атрибуте функции
    #    — берём моду по первой встрече, чтобы минимально трогать данные
    if not hasattr(collate, "_gdesc_len") or collate._gdesc_len is None:
        from collections import Counter
        collate._gdesc_len = Counter(_lens).most_common(1)[0][0]  # например, 129
    L = collate._gdesc_len

    # мягко паддим/усекаем под L (на CPU, float32)
    _gfixed = []
    for i, g in enumerate(gdescs):
        gi = g.reshape(-1).to(torch.float32)
        Li = gi.numel()
        if Li < L:
            gi = F.pad(gi, (0, L - Li))
        elif Li > L:
            gi = gi[:L]
        _gfixed.append(gi)

    #  редкое логирование, чтобы не засорять вывод
    if not hasattr(collate, "_gdesc_warns"):
        collate._gdesc_warns = 0
    if len(set(_lens)) != 1 and collate._gdesc_warns < 3:
        bad = [idx for idx, Lx in enumerate(_lens) if Lx != L]
        # print(f"[COLLATE] gdesc lengths in batch: {_lens} | target={L} | bad idx: {bad[:8]}")
        collate._gdesc_warns += 1

    # теперь можно батчить в (B, L)
    gdesc = torch.stack(_gfixed, dim=0)

    return x, edge_index, edge_attr, batch_vec, gdesc, y, y_mask


# === GLOBAL MATERIALIZATION (supervised) =====================================

def materialize_graphs_for_df(df: pd.DataFrame,
                              target_cols: List[str],
                              poly_edge_mode: str = "cycle",
                              cache_path: Optional[str] = None):
    """
    ОДИН раз строит тензоры (x, ei, ea, gdesc, y, y_mask) для КАЖДОЙ строки df.
    Если cache_path задан и файл существует — читает его. Возвращает list[tuples].
    """
    items = None
    if cache_path and os.path.isfile(cache_path):
        try:
            items = torch.load(cache_path, map_location="cpu", weights_only=True)
        except TypeError:
            items = torch.load(cache_path, map_location="cpu")
        # лёгкая валидация: размер совпадает?
        if not isinstance(items, list) or len(items) != len(df):
            items = None  # перестроим

    if items is None:
        items = []
        it = range(len(df))
        try:
            from tqdm import tqdm
            it = tqdm(it, desc=f"[BUILD] all_graphs({len(df)})")
        except Exception:
            pass

        for i in it:
            row = df.iloc[i]
            smi = row["SMILES"]
            g = smiles_to_graph(smi, poly_edge_mode)
            if g is None:
                x = torch.zeros((1, NODE_FEAT_DIM), dtype=torch.float32)
                ei = torch.zeros((2, 0), dtype=torch.long)
                ea = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float32)
                gdesc = torch.zeros((TOTAL_GDESC_DIM,), dtype=torch.float32)
            else:
                x, ei, ea, gdesc = g

            if all(c in row for c in target_cols):
                y = torch.tensor([float(row[c]) if pd.notna(row[c]) else float("nan")
                                  for c in target_cols], dtype=torch.float32)
                y_mask = torch.tensor([pd.notna(row[c]) for c in target_cols], dtype=torch.bool)
            else:
                y = torch.full((len(target_cols),), float("nan"))
                y_mask = torch.zeros((len(target_cols),), dtype=torch.bool)

            items.append((x, ei, ea, gdesc, y, y_mask))

        if cache_path:
            torch.save(items, cache_path)

    return items


def _slice_tensor_ds(all_items: List[Tuple[torch.Tensor, ...]], idx_list: List[int]) -> TensorGraphDataset:
    """
    Берёт precomputed список items и возвращает TensorGraphDataset
    с нужной подвыборкой по индексам исходного df.
    """
    sliced = [all_items[i] for i in idx_list]
    return TensorGraphDataset(sliced)


# === SINGLE-SPLIT (main-only) по ИНДЕКСАМ (без копий df) =====================

def split_main_only_indices(sup_df: pd.DataFrame, val_size: float, seed: int):
    """
    Повторяет вашу логику split_main_only, но возвращает ИНДЕКСЫ sup_df:
      - train_idx = индексы train_main ∪ ext (id == -1)
      - val_idx   = индексы val_main
    """
    main_mask = (sup_df["id"] != -1)
    ext_mask  = ~main_mask

    main_df = sup_df[main_mask]            # ВАЖНО: не reset_index, чтобы индексы совпадали с sup_df
    ext_df  = sup_df[ext_mask]

    # используем ваш scaffold_split, он Сохраняет индексы входного df
    tr_main, va_main = scaffold_split(main_df.copy(), val_size, seed)

    # индексы относительно sup_df
    tr_idx = list(tr_main.index) + list(ext_df.index)
    va_idx = list(va_main.index)

    return tr_idx, va_idx

# =============================
# Model
# =============================
class EdgeMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    def forward(self, e):
        return self.net(e)

class GatedMP(nn.Module):
    """Edge-conditioned gated message passing (stable)"""
    def __init__(self, node_dim, edge_dim, dropout=0.1):
        super().__init__()
        self.in_norm = nn.LayerNorm(node_dim)             # NEW: pre-norm
        self.edge_mlp = EdgeMLP(edge_dim, node_dim)
        self.msg = nn.Sequential(
            nn.Linear(node_dim*2, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )
        self.gru = nn.GRUCell(node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return x
        # safety on inputs
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

        src, dst = edge_index
        h_in = self.in_norm(x)                             # PRE-NORM

        # bounded gate for stability
        e_emb = torch.tanh(self.edge_mlp(edge_attr))       # [-1, 1]

        # message with bb boost
        m = h_in[src] * e_emb

        BOND_BASE_DIM = len(bond_features(None, is_poly_edge=True))  # вычислить один раз выше по модулю
        bb_flag = edge_attr[:, -1:].to(m.dtype)  # is_bb_bond
        poly_flag = edge_attr[:, BOND_BASE_DIM - 1:BOND_BASE_DIM].to(m.dtype)  # is_poly_edge (из базового блока)
        khop_flag = edge_attr[:, BOND_BASE_DIM:BOND_BASE_DIM + 1].to(m.dtype)  # is_khop_edge
        khop_w = edge_attr[:, BOND_BASE_DIM + 1:BOND_BASE_DIM + 2].to(m.dtype)  # khop_norm \in {0.5, 0.333...}

        # аккуратные бусты: bb > poly > khop
        m = m * (1.0
                 + 0.30 * bb_flag
                 + 0.20 * poly_flag
                 + 0.12 * khop_flag * (2.0 * khop_w))  # 2hop≈0.12, 3hop≈0.08

        N, D = h_in.size()

        # FP32 accumulation + degree normalization
        agg = torch.zeros((N, D), device=h_in.device, dtype=torch.float32)
        agg.index_add_(0, dst, m.to(torch.float32))
        deg = torch.bincount(dst, minlength=N).unsqueeze(1).to(torch.float32).clamp_min_(1.0)
        agg = (agg / deg).to(h_in.dtype)

        # update
        upd_in = self.msg(torch.cat([h_in, agg], dim=1))
        h = self.gru(upd_in, h_in)
        h = self.norm(h)
        h = self.dropout(h)
        return x + h


class AttentivePool(nn.Module):
    def __init__(self, in_dim, att_dim=128, dropout=0.0):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(in_dim, att_dim),
            nn.SiLU(),
            nn.Linear(att_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_vec):
        a = self.att(x).squeeze(-1)                        # [N]
        a = torch.nan_to_num(a, nan=-1e6)                  # любую NaN -> «минус бесконечность»

        num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0
        pooled = torch.zeros((num_graphs, x.size(1)), device=x.device, dtype=x.dtype)
        for b in torch.unique(batch_vec):
            mask = (batch_vec == b)
            if mask.any():
                w = torch.softmax(a[mask], dim=0)          # СТАБИЛЬНО
                pooled[b] = (x[mask] * w.unsqueeze(-1)).sum(dim=0)
        return self.dropout(pooled)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)

    def store(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}

    def copy_to(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    v.copy_(self.shadow[k])

class PolymerGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, gdesc_dim, hidden=256, layers=8, targets=5, dropout=0.1, use_global_token=True, predict_sigma=False, constrain=False, per_target_last=False):

        super().__init__()
        self.extra_node_feats = EXTRA_NODE_FEATS  # важно знать, где искать флаги
        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.layers = nn.ModuleList([GatedMP(hidden, edge_in_dim, dropout=dropout) for _ in range(layers)])
        self.pool = AttentivePool(hidden, att_dim=128, dropout=dropout)
        self.use_global_token = use_global_token

        self.gdesc_norm = nn.LayerNorm(gdesc_dim)

        # вход fuse: hidden (all) + hidden (bb) + gdesc
        self.fuse = nn.Sequential(
            nn.Linear(hidden*2 + gdesc_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, targets)
        )

        h2 = hidden//2
        self.head_core = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.per_target_last = bool(per_target_last)
        if self.per_target_last:
            self.head_out_list = nn.ModuleList([nn.Linear(h2, 1) for _ in range(targets)])
        else:
            self.head_out = nn.Linear(h2, targets)

        self.predict_sigma = bool(predict_sigma)
        if self.predict_sigma:
            # sigma предсказываем из того же h2
            self.head_sigma_core = nn.Sequential(nn.LayerNorm(hidden),
                                                 nn.Linear(hidden, h2),
                                                 nn.ReLU(),
                                                 nn.Dropout(dropout))
            self.head_sigma_out = nn.Linear(h2, targets)

        self.constrain = bool(constrain)

        self.rho = nn.Parameter(torch.zeros(targets))

    @staticmethod
    def _masked_mean_pool(h, batch_vec, mask_bool):
        D = h.size(1)
        num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0
        sums  = torch.zeros((num_graphs, D), device=h.device, dtype=h.dtype)
        counts = torch.zeros((num_graphs, 1), device=h.device, dtype=h.dtype)
        if mask_bool.dtype != torch.bool:
            mask_bool = mask_bool > 0.5
        if mask_bool.any():
            idx = torch.nonzero(mask_bool, as_tuple=False).view(-1)
            # добавляем только выбранные узлы
            sums.index_add_(0, batch_vec[idx], h[idx])
            counts.index_add_(0, batch_vec[idx], torch.ones((idx.numel(),1), device=h.device, dtype=h.dtype))
        counts = counts.clamp_min_(1.0)
        return sums / counts

    def encode(self, x, edge_index, edge_attr, batch_vec, gdesc, *, use_gdesc: bool = True, use_poly_hints: bool = True):
        # маска бэкбона из сырых фич x: ПЕРВАЯ из 6 новых — is_backbone
        bb_col = x.size(1) - self.extra_node_feats  # начало «хвоста»
        backbone_mask = x[:, bb_col]  # float (0/1)
        # нормализация NAN
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        gdesc = torch.nan_to_num(gdesc, nan=0.0, posinf=0.0, neginf=0.0)

        # В SSL можно отрубить 6 «poly-hints» (хвост узловых фич)
        if not use_poly_hints:
            # не меняем размерность: просто зануляем хвост
            x = x.clone()
            x[:, bb_col:bb_col + self.extra_node_feats] = 0.0

        # Возвращает графовый эмбеддинг размера `hidden`
        h = self.node_embed(x)
        for mp in self.layers:
            h = mp(h, edge_index, edge_attr)

        g_all = self.pool(h, batch_vec)
        g_bb  = self._masked_mean_pool(h, batch_vec, backbone_mask)
        # gdesc можно отключить в SSL, подставив нули того же размера
        gd = self.gdesc_norm(gdesc) if use_gdesc else torch.zeros_like(gdesc)
        g = torch.cat([g_all, g_bb, gd], dim=1)
        g = self.fuse(g)
        return g

    def forward(self, x, edge_index, edge_attr, batch_vec, gdesc):
        # supervised-путь: всё включено
        g = self.encode(x, edge_index, edge_attr, batch_vec, gdesc,
                        use_gdesc=True, use_poly_hints=True)

        h = self.head_core(g)
        if self.per_target_last:
            mu = torch.cat([m(h) for m in self.head_out_list], dim=1)
        else:
            mu = self.head_out(h)

        if self.constrain:
            mu = apply_head_constraints(mu, TARGETS)

        if self.predict_sigma:
            hs = self.head_sigma_core(g)
            sig = self.head_sigma_out(hs)
            return mu, sig
        return mu


# =============================
# Metric / Loss
# =============================
def compute_norm_and_weights(df: pd.DataFrame, target_cols: List[str]):
    # страховка: убеждаемся, что колонки числовые
    _df = df.copy()
    for c in target_cols:
        if c in _df.columns and not np.issubdtype(_df[c].dtype, np.number):
            _df[c] = pd.to_numeric(_df[c], errors='coerce')

    ranges, counts = {}, {}
    for c in target_cols:
        vals = _df[c].dropna().values
        if len(vals) == 0:
            ranges[c] = 1.0
            counts[c] = 1.0
        else:
            r = float(np.nanmax(vals) - np.nanmin(vals))
            ranges[c] = r if r > 0 else 1.0
            counts[c] = float(len(vals))
    arr = np.array([1.0/np.sqrt(counts[c]) for c in target_cols], dtype=np.float32)
    arr = arr / max(arr.sum(), 1e-9)
    return np.array([ranges[c] for c in target_cols], dtype=np.float32), arr

# ===== Kaggle-like validation metric (r_i по предсказаниям, w_i нормируются к K) =====
def _kaggle_weights_from_mask(val_mask: np.ndarray, K: int) -> np.ndarray:
    # val_mask: (N, T) bool
    n_i = val_mask.sum(axis=0).astype(np.float32)
    n_i[n_i <= 0] = 1.0
    w = 1.0 / np.sqrt(n_i)
    w = (K * w) / max(w.sum(), 1e-9)
    return w

def kaggle_wmae_from_preds(pred: np.ndarray, true: np.ndarray, mask: np.ndarray, target_cols=TARGETS):
    """
    pred/true: (N, T) float32, mask: (N, T) bool
    r_i = max(pred_i[mask]) - min(pred_i[mask]);  w_i = K * sqrt(1/n_i) / sum_j sqrt(1/n_j)
    """
    assert pred.shape == true.shape == mask.shape
    K = pred.shape[1]
    # r_i по предсказаниям текущего чекпоинта
    r = []
    for t in range(K):
        m = mask[:, t]
        if m.sum() == 0:
            r.append(1.0)
            continue
        p = pred[m, t]
        ri = float(np.nanmax(p) - np.nanmin(p))
        r.append(ri if ri > 0 else 1.0)
    r = np.asarray(r, dtype=np.float32)
    # веса как на Kaggle
    w = _kaggle_weights_from_mask(mask, K)
    # MAE по маске, делённый на r_i
    abs_err = np.abs((pred - true))
    abs_err[~mask] = 0.0
    cnt = mask.sum(axis=0).clip(min=1)
    per_t = abs_err.sum(axis=0) / cnt
    per_t = per_t / r
    score = float((per_t * w).sum())
    return score, per_t.tolist(), r.tolist(), w.tolist()

# === replace your masked losses ===
def masked_wmae(pred, true, mask, ranges, weights):
    # если прилетел (mu, sigma) — берём mu
    if isinstance(pred, (tuple, list)):
        pred = pred[0]

    # работаем ТОЛЬКО по маске с самого начала
    denom = torch.as_tensor(ranges, device=pred.device, dtype=pred.dtype).view(1, -1).clamp_min_(1e-8)
    w = torch.as_tensor(weights, device=pred.device, dtype=pred.dtype).view(1, -1)

    # обнуляем вне маски ещё до арифметики, чтобы не порождать NaN
    diff = torch.where(mask, pred - true, torch.zeros_like(pred))
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

    err = (diff.abs() / denom)

    per_t_sum = err.sum(dim=0)
    per_t_count = mask.sum(dim=0).clamp(min=1)

    per_t_mae = per_t_sum / per_t_count


    score = (per_t_mae * w.squeeze(0)).sum()
    return score, per_t_mae

def masked_wmae_with_uncertainty(pred, true, mask, ranges, weights, rho):
    # если прилетел (mu, sigma) — берём mu
    if isinstance(pred, (tuple, list)):
        pred = pred[0]

    # константы без градиента
    denom = torch.as_tensor(ranges, device=pred.device, dtype=pred.dtype).view(1, -1)
    denom = denom.clamp_min(1e-8)
    w = torch.as_tensor(weights, device=pred.device, dtype=pred.dtype).view(1, -1)

    # работаем строго по маске (не порождаем NaN/inf)
    diff = torch.where(mask, pred - true, torch.zeros_like(pred))
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

    # # параметризация σ: всегда положительна и гладкая
    # sigma = F.softplus(rho).view(1, -1) + 1e-8
    # стабильная параметризация: sigma >= 0, клампы от экстремальных значений
    sigma = (F.softplus(rho).clamp(1e-3, 100.0)).view(1, -1)

    base = (diff.abs() / denom)
    term = base / sigma + torch.log(sigma)
    err = torch.where(mask, term, torch.zeros_like(term))

    per_t_sum = err.sum(dim=0)
    per_t_count = mask.sum(dim=0).clamp(min=1)
    per_t_mae = per_t_sum / per_t_count
    score = (per_t_mae * w.squeeze(0)).sum()
    return score, per_t_mae


def masked_wmae_hetero(pred_mu, true, mask, ranges, weights, sigma_pred):
    """Пер-сэмпловая (гетероскедастическая) версия: σ предсказывается сетью."""
    denom = torch.as_tensor(ranges, device=pred_mu.device, dtype=pred_mu.dtype).view(1, -1)
    denom = denom.clamp_min(1e-8)
    w = torch.as_tensor(weights, device=pred_mu.device, dtype=pred_mu.dtype).view(1, -1)

    diff = torch.where(mask, pred_mu - true, torch.zeros_like(pred_mu))
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

    sigma = F.softplus(sigma_pred).clamp(1e-3, 100.0)          # σ>0
    sigma = torch.where(mask, sigma, torch.ones_like(sigma))   # там где нет таргета — нейтрально

    base = (diff.abs() / denom)
    term = base / sigma + torch.log(sigma)
    err  = torch.where(mask, term, torch.zeros_like(term))

    per_t_sum   = err.sum(dim=0)
    per_t_count = mask.sum(dim=0).clamp(min=1)
    per_t_mae   = per_t_sum / per_t_count
    score = (per_t_mae * w.squeeze(0)).sum()
    return score, per_t_mae

# =============================
# Data Loading / Merging
# =============================
def load_main_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str,pd.DataFrame]]:
    # support plain folder OR a single zip file path that contains train.csv/test.csv
    if data_dir.endswith('.zip') and os.path.isfile(data_dir):
        import zipfile
        z = zipfile.ZipFile(data_dir, 'r')
        train_df = pd.read_csv(z.open('train.csv'))
        test_df  = pd.read_csv(z.open('test.csv'))
        return train_df, test_df, {}
    elif os.path.isdir(data_dir):
        train_csv = os.path.join(data_dir, "train.csv")
        test_csv  = os.path.join(data_dir, "test.csv")
        train_df = pd.read_csv(train_csv)
        test_df  = pd.read_csv(test_csv)
        return train_df, test_df, {}
    else:
        raise FileNotFoundError(f"No zip or folder found at {data_dir}")

def load_supplemental(data_dir: str) -> Dict[str, pd.DataFrame]:
    d: Dict[str,pd.DataFrame] = {}
    if data_dir.endswith('.zip') and os.path.isfile(data_dir):
        import zipfile
        z = zipfile.ZipFile(data_dir, 'r')
        for name in z.namelist():
            if name.startswith('train_supplement/') and name.endswith('.csv'):
                try:
                    d[os.path.basename(name)] = pd.read_csv(z.open(name))
                except Exception:
                    pass
        return d
    supp_dir = os.path.join(data_dir, "train_supplement")
    if os.path.isdir(supp_dir):
        for fn in os.listdir(supp_dir):
            if fn.endswith(".csv"):
                try:
                    d[fn] = pd.read_csv(os.path.join(supp_dir, fn))
                except Exception:
                    pass
    return d

def ingest_external_sources(args) -> Dict[str, pd.DataFrame]:
    out = {}

    def read_csv_or_zip(folder_or_zip, inner_file):
        zip_path = f"{folder_or_zip}.zip"
        folder_path = folder_or_zip
        if os.path.isfile(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as z:
                return pd.read_csv(z.open(inner_file))
        elif os.path.isfile(os.path.join(folder_path, inner_file)):
            return pd.read_csv(os.path.join(folder_path, inner_file))
        return None

    def read_excel_or_zip(folder_or_zip, inner_file):
        zip_path = f"{folder_or_zip}.zip"
        folder_path = folder_or_zip
        if os.path.isfile(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as z:
                return pd.read_excel(z.open(inner_file))
        elif os.path.isfile(os.path.join(folder_path, inner_file)):
            return pd.read_excel(os.path.join(folder_path, inner_file))
        return None

    # Tc
    df = read_csv_or_zip(os.path.join(args.external_dir, "Tc_SMILES"), "Tc_SMILES.csv")
    if df is not None:
        out["Tc"] = df.rename(columns={"TC_mean": "Tc"}) if "TC_mean" in df.columns else df

    # Tg
    df = read_csv_or_zip(os.path.join(args.external_dir, "Tg_SMILES_PID_Polymer_Class"), "TgSS_enriched_cleaned.csv")
    if df is not None:
        out["Tg"] = df.rename(columns={"Tg": "Tg"}) if "Tg" in df.columns else df

    # Extra
    jcim = read_csv_or_zip(os.path.join(args.external_dir, "SMILES_Extra_Data"), "JCIM_sup_bigsmiles.csv")
    if jcim is not None:
        if "Tg (C)" in jcim.columns:
            jcim = jcim.rename(columns={"Tg (C)": "Tg"})
        out["JCIM_Tg"] = jcim[["SMILES", "Tg"]]

    dnst = read_excel_or_zip(os.path.join(args.external_dir, "SMILES_Extra_Data"), "data_dnst1.xlsx")
    if dnst is not None:
        dnst = dnst.rename(columns={"density(g/cm3)": "Density"})
        out["Density"] = dnst[["SMILES", "Density"]].dropna()

    tg3 = read_excel_or_zip(os.path.join(args.external_dir, "SMILES_Extra_Data"), "data_tg3.xlsx")
    if tg3 is not None:
        tg3 = tg3.rename(columns={"Tg [K]": "Tg"})
        tg3["Tg"] = tg3["Tg"].astype(float) - 273.15
        out["Tg2"] = tg3[["SMILES", "Tg"]]

    extra = read_csv_or_zip(os.path.join(args.external_dir, "SMILE_DATA"), "SMILES_EXTRA_DATA (1).csv")
    if extra is not None:
        out["EXTRA_DATA"] = extra[["SMILES", "Tg"]]


    return out

def has_star(m: Chem.Mol) -> bool:
    return any(a.GetAtomicNum() == 0 for a in m.GetAtoms())

def cap_polymer_stars(m: Chem.Mol, cap: str = "[H]") -> Optional[Chem.Mol]:
    """
    Заменяет все '*' на выбранную заглушку (по умолчанию явный водород),
    затем санация. Возвращает новую молекулу или None при неудаче.
    """
    try:
        patt = Chem.MolFromSmarts("[*]")
        repl = Chem.MolFromSmiles(cap)
        res = Chem.ReplaceSubstructs(m, patt, repl, replaceAll=True)
        mc = res[0] if isinstance(res, (list, tuple)) else res
        Chem.SanitizeMol(mc)
        return mc
    except Exception:
        return None

def strip_stars_and_add_hs(m: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Удаляет все dummy-атомы ('*'), корректно разрывая связи.
    Затем санитизация и добавление ЯВНЫХ H для корректной валентности/массы.
    Возвращает новую молекулу или None при неудаче.
    """
    try:
        em = Chem.EditableMol(m)
        # Собираем индексы звёзд и их связи
        star_idxs = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
        # Удаляем связи со звёздами
        for si in star_idxs:
            a = m.GetAtomWithIdx(si)
            for nb in list(a.GetNeighbors()):
                bi = m.GetBondBetweenAtoms(si, nb.GetIdx()).GetIdx()
                em.RemoveBond(si, nb.GetIdx())
        # Удаляем сами звёзды (идти в обратном порядке индексов)
        for si in sorted(star_idxs, reverse=True):
            em.RemoveAtom(si)
        m2 = em.GetMol()
        Chem.SanitizeMol(m2)
        # Добавляем явные H, чтобы дескрипторы (MolWt/LogP и т.п.) были корректными
        m2h = Chem.AddHs(m2)
        Chem.SanitizeMol(m2h)
        return m2h
    except Exception:
        return None

def jsonl_append(path: str, record: dict):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass

def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_event(events_path: str, event: str, msg: str = "", **kv):
    rec = {"time": _ts(), "event": event, "msg": msg}
    rec.update(kv)
    # короткий принт
    tail = (" " + " ".join(f"{k}={v}" for k,v in kv.items())) if kv else ""
    print(f"[{rec['time']}] [{event}] {msg}{tail}")
    # и в jsonl-файл
    jsonl_append(events_path, rec)


def current_lr(optimizer):
    try:
        return optimizer.param_groups[0]["lr"]
    except Exception:
        return float("nan")


# === Dataset fingerprint & cache paths ======================================

def _dataset_fingerprint_from_smiles(smiles_list: List[str]) -> str:
    """Короткий md5-отпечаток набора SMILES в заданном порядке."""
    s = "||".join(smiles_list) + f"|N={len(smiles_list)}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]

def _scaffold_cache_paths(out_dir: str, ds_fp: str):
    """Возвращает кандидаты путей для карты скаффолдов (parquet и csv)."""
    p_parq = os.path.join(out_dir, f"scaffold_map_{ds_fp}.parquet")
    p_csv  = os.path.join(out_dir, f"scaffold_map_{ds_fp}.csv")
    return p_parq, p_csv

def _kfold_cache_path(out_dir: str, ds_fp: str, n_folds: int, seed: int) -> str:
    """Путь для кеша индексов K-fold — json."""
    return os.path.join(out_dir, f"kfold_{n_folds}_{seed}_{ds_fp}.json")

# === Scaffold map cache (SMILES -> scaffold) ================================

def _load_scaffold_map(p_parq: str, p_csv: str) -> Optional[pd.DataFrame]:
    if os.path.isfile(p_parq):
        try:
            return pd.read_parquet(p_parq)
        except Exception:
            pass
    if os.path.isfile(p_csv):
        try:
            return pd.read_csv(p_csv)
        except Exception:
            pass
    return None

def _save_scaffold_map(df_map: pd.DataFrame, p_parq: str, p_csv: str):
    # Пытаемся Parquet (быстро и компактно), фоллбэк — CSV
    try:
        df_map.to_parquet(p_parq, index=False)
    except Exception:
        df_map.to_csv(p_csv, index=False)


def ensure_scaffold_map_cached(smiles: List[str], out_dir: str) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками ['SMILES','scaffold'] для ВСЕХ переданных SMILES.
    Если часть SMILES уже в кеше — дозаполняет только недостающие и обновляет файл.
    """
    ds_fp = _dataset_fingerprint_from_smiles(smiles)
    p_parq, p_csv = _scaffold_cache_paths(out_dir, ds_fp)

    # читаем, если есть
    df_map = _load_scaffold_map(p_parq, p_csv)
    need_compute = []
    if df_map is None:
        df_map = pd.DataFrame({"SMILES": np.unique(np.asarray(smiles, dtype=object))})
        df_map["scaffold"] = None
        need_compute = df_map["SMILES"].tolist()
    else:
        # удостоверимся, что покрываем все SMILES (на случай изменений датасета)
        known = set(df_map["SMILES"].astype(str).tolist())
        allu = list(np.unique(np.asarray(smiles, dtype=str)))
        need_compute = [s for s in allu if s not in known]
        if need_compute:
            df_map = pd.concat([df_map, pd.DataFrame({"SMILES": need_compute, "scaffold": [None]*len(need_compute)})],
                               ignore_index=True)

    # посчитаем только недостающие
    if need_compute:
        it = need_compute
        try:
            it = tqdm(need_compute, desc="[SCAFFOLD] compute missing")
        except Exception:
            pass
        map_new = []
        for s in it:
            try:
                scaf = compute_scaffold(s)
            except Exception:
                scaf = ""
            map_new.append((s, scaf))
        upd = pd.DataFrame(map_new, columns=["SMILES","scaffold"])
        # обновляем значения в df_map
        df_map = (df_map.drop_duplicates(subset=["SMILES"], keep="first")
                         .set_index("SMILES"))
        df_map.update(upd.set_index("SMILES"))
        df_map = df_map.reset_index()

        # сохраняем
        _save_scaffold_map(df_map, p_parq, p_csv)

    # теперь у нас есть карта для всех уникальных SMILES
    return df_map


def cuda_mem_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0

def print_env_banner():
    dev = "cpu"
    if torch.cuda.is_available():
        try:
            dev = f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})"
        except Exception:
            dev = "cuda"

    amp = "bf16" if AMP_DTYPE == torch.bfloat16 else ("fp16" if AMP_DTYPE == torch.float16 else str(AMP_DTYPE))

    try:
        import rdkit
        rdkit_ver = getattr(rdkit, "__version__", "unknown")
    except Exception:
        rdkit_ver = "unknown"
    try:
        git_hash = os.popen("git rev-parse --short HEAD").read().strip()
    except Exception:
        git_hash = ""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ENV] {ts} | torch {torch.__version__} | rdkit {rdkit_ver} | device {dev}" + (f" | git {git_hash}" if git_hash else "") + f" | amp {amp}")


def recommend_num_workers(requested: Optional[int] = None) -> int:
    if platform.system() == "Windows":
        if requested and requested > 0:
            print("[INFO] Windows detected: overriding num_workers to 0 for stability.")
        return 0
    if requested is not None:
        return max(0, int(requested))
    cpu = os.cpu_count() or 2
    return min(8, max(1, cpu - 1))


def make_loader_kwargs(num_workers: int):
    kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0 and platform.system() != "Windows"),
    )
    if num_workers > 0:
        kw["prefetch_factor"] = 2
    return kw


def seed_worker(worker_id):
    # сделаем воспроизводимость при num_workers>0
    # (используем глобальный SEED, см. parse_args)
    import random as _random
    _seed = torch.initial_seed() % 2**32
    _random.seed(_seed + worker_id)
    np.random.seed((_seed + worker_id) % (2**32 - 1))



def coerce_targets_numeric(df: pd.DataFrame, target_cols):
    df = df.copy()
    for c in target_cols:
        if c in df.columns:
            # убираем лишние символы, если вдруг есть единицы измерения
            df[c] = (df[c]
                     .astype(str)
                     .str.replace(',', '.', regex=False)    # десятичные запятые → точки
                     .str.extract(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')[0])  # вытягиваем число
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def build_supervised_dataframe(main_train: pd.DataFrame,
                               supp_from_zip: Dict[str,pd.DataFrame],
                               external: Dict[str,pd.DataFrame]) -> pd.DataFrame:
    df_list = []
    # main
    df_list.append(main_train[["SMILES","id"] + TARGETS].copy())

    # supplement inside NeurIPS zip, if available
    if "dataset1.csv" in "".join(supp_from_zip.keys()):
        for k, v in supp_from_zip.items():
            if "dataset1" in k and "TC_mean" in v.columns:
                df = v.rename(columns={"TC_mean":"Tc"})
                df["id"] = -1
                for t in TARGETS:
                    if t not in df.columns:
                        df[t] = np.nan
                df_list.append(df[["SMILES","id"]+TARGETS])

            if "dataset3" in k and "Tg" in v.columns:
                df = v.copy()
                df["id"] = -1
                for t in TARGETS:
                    if t not in df.columns:
                        df[t] = np.nan
                df_list.append(df[["SMILES","id"]+TARGETS])

            if "dataset4" in k and "FFV" in v.columns:
                df = v.copy()
                df["id"] = -1
                for t in TARGETS:
                    if t not in df.columns:
                        df[t] = np.nan
                df_list.append(df[["SMILES","id"]+TARGETS])

    # external provided alongside
    for k, v in external.items():
        df = v.copy()
        df["id"] = -1
        for t in TARGETS:
            if t not in df.columns:
                df[t] = np.nan
        # handle Tg columns potentially in Kelvin already handled earlier
        df_list.append(df[["SMILES","id"]+TARGETS])

    sup = pd.concat(df_list, axis=0, ignore_index=True).drop_duplicates(
        subset=["SMILES", "Tg","FFV","Tc","Density","Rg"], keep="first"
    )

    sup = coerce_targets_numeric(sup, TARGETS)

    sup = sup[sup["SMILES"].apply(lambda s: isinstance(s, str) and s.count("*") >= 2)].reset_index(drop=True)

    print(f"[DATA][SUP] total={len(sup)} | "
          f"with_targets={int(sup[TARGETS].notna().any(axis=1).sum())} | "
          f"stars>=2={int(sup['SMILES'].apply(lambda s: isinstance(s, str) and s.count('*') >= 2).sum())}")

    return sup

def build_unlabeled_dataframe_smart(main_train: pd.DataFrame,
                                    main_test: pd.DataFrame,
                                    supp_from_zip: Dict[str,pd.DataFrame],
                                    args,
                                    sup_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    tic = Tic("unlabeled_build")

    # 0) echo настроек
    print(f"[UNL-ARGS] pretrain_samples={getattr(args, 'pretrain_samples', None)} | "
          f"near_frac={getattr(args, 'unl_near_frac', None)} | "
          f"far_frac={getattr(args, 'unl_far_frac', None)} | "
          f"near_tmin={getattr(args, 'unl_near_tanimoto_min', 0.5)} | "
          f"far_tmax={getattr(args, 'unl_far_tanimoto_max', 0.25)} | "
          f"max_ref={getattr(args, 'unl_max_ref', None)} | "
          f"pi1m_cap={getattr(args, 'unl_pi1m_cap', 0)} | "
          f"dedup_key={getattr(args, 'unl_dedup_key', 'canon')}")

    # Собираем исходный пул
    unl = []
    # dataset2 inside NeurIPS zip if present
    for k, v in supp_from_zip.items():
        if "dataset2" in k and "SMILES" in v.columns:
            unl.append(v[["SMILES"]].copy())

    # PI1M.csv (huge ~1M)
    pi1m_path = os.path.join(args.external_dir, "PI1M.csv")
    if os.path.isfile(pi1m_path):
        try:
            pi = pd.read_csv(pi1m_path, usecols=["SMILES"])
            cap = int(getattr(args, "unl_pi1m_cap", 0) or 0)
            if cap > 0 and len(pi) > cap:
                # дешёвая первичная очистка перед сэмплом
                pi = pi[pi["SMILES"].apply(lambda s: isinstance(s, str) and s.count("*") >= 2)]
                pi = pi.drop_duplicates(subset=["SMILES"])
                pi = pi.sample(n=cap, random_state=args.seed)
            unl.append(pi[["SMILES"]])
        except Exception:
            pass

    # all main SMILES (train+test)
    unl.append(main_train[["SMILES"]])
    if args.unl_use_main_test:
        unl.append(main_test[["SMILES"]])


    # (опц.) добавляем ВСЕ SMILES из supervised пула как безлейбловые
    if args.unl_use_sup_df and (sup_df is not None) and ("SMILES" in sup_df.columns):
        unl.append(sup_df[["SMILES"]])

    print(f"[UNL] raw_concat={sum(len(x) for x in unl)}")

    pool = pd.concat(unl, axis=0, ignore_index=True)
    before = len(pool)
    # базовая полимер-фильтрация
    pool = pool[pool["SMILES"].apply(lambda s: isinstance(s, str) and s.count("*") >= 2)].reset_index(drop=True)
    print(f"[UNL] star>=2: {before} -> {len(pool)}");
    tic.done("star_filter")

    # сразу после конкатенации pool:
    before = len(pool)
    pool = pool.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    print(f"[UNL] raw_dedup: {before} -> {len(pool)}");
    tic.done("raw_dedup")

    # 2) Дедупликация по выбранному ключу
    key_mode = getattr(args, "unl_dedup_key", "canon")
    pool["__key__"] = pool["SMILES"].astype(str).apply(lambda s: smiles_dedup_key(s, key_mode))
    before = len(pool)
    pool = pool.drop_duplicates(subset=["__key__"]).drop(columns=["__key__"]).reset_index(drop=True)
    print(f"[UNL] smart_dedup({key_mode}): {before} -> {len(pool)}");
    tic.done("smart_dedup")

    # Если итог меньше квоты — возвращаем сразу
    target_N = int(args.pretrain_samples) if getattr(args, "pretrain_samples", 0) > 0 else len(pool)
    if (target_N <= 0) or (len(pool) <= target_N):
        print(f"[UNL-SELECT] pool={len(pool)} <= target={target_N} → use all");
        tic.done("early_return")
        return pool[["SMILES"]].copy()

    # 3) Доменные референсы (для near/far)
    ref_df = None
    if sup_df is not None:
        ref_df = sup_df[["SMILES"]].copy()
        ref_df = ref_df[ref_df["SMILES"].apply(lambda s: isinstance(s, str) and s.count("*") >= 2)]
        ref_df = ref_df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    else:
        # если sup_df нет — используем train+test как минимум
        ref_df = pd.concat([main_train[["SMILES"]], main_test[["SMILES"]]], axis=0).drop_duplicates().reset_index(drop=True)

    # 3a) Scaffold-сеты (быстрое "near" по совпадению скелета)
    print(f"[UNL] ref_df={len(ref_df)} (after star>=2 & dedup)")
    tic2 = Tic("scaffold_maps")
    ref_scaf = ensure_scaffold_map_cached(ref_df["SMILES"].astype(str).tolist(), args.out_dir)
    pool_scaf = ensure_scaffold_map_cached(pool["SMILES"].astype(str).tolist(), args.out_dir)
    tic2.done()
    print(f"[UNL] unique_ref_scaffolds={ref_scaf['scaffold'].nunique()} | pool_scaffolds={pool_scaf['scaffold'].nunique()}")


    ref_scaf_set = set(ref_scaf["scaffold"].tolist())
    pool = pool.merge(pool_scaf, on="SMILES", how="left")
    pool["__is_near_scaf__"] = pool["scaffold"].isin(ref_scaf_set)
    print(f"[UNL] near_by_scaffold={int(pool['__is_near_scaf__'].sum())} / {len(pool)}")

    # tanimoto
    near_thr = float(getattr(args, "unl_near_tanimoto_min", 0.5))
    far_thr = float(getattr(args, "unl_far_tanimoto_max", 0.25))
    if near_thr <= far_thr:
        print(f"[WARN] near_tmin({near_thr}) <= far_tmax({far_thr}) — границы пересекаются; проверьте параметры.")

    # 3b) Оценка max-Tanimoto до ближайшего ref (ограничим референсы)
    _fp_cache = {}

    def fp_cached(smiles):
        v = _fp_cache.get(smiles)
        if v is not None: return v
        v = _morgan_fp_from_smiles(smiles, nBits=args.ssl_fp_bits)
        _fp_cache[smiles] = v
        return v

    ref_sample = ref_df["SMILES"].sample(
        n=min(len(ref_df), max(1, int(args.unl_max_ref))), random_state=args.seed
    ).tolist()
    ref_fps = [fp_cached(s) for s in ref_sample]

    # фильтрация невалидных отпечатков
    ref_fps = [fp for fp in ref_fps if fp is not None]
    if len(ref_fps) < len(ref_sample):
        print(f"[UNL] filtered_invalid_ref_fps={len(ref_sample) - len(ref_fps)} | kept={len(ref_fps)}")

    need = (~pool["__is_near_scaf__"]).sum()
    print(f"[UNL] need_tanimoto_for={need} | ref_fps={len(ref_fps)}")
    tic3 = Tic("tanimoto")

    # для скорости считаем только там, где нужно (кандидаты вне прямого scaffold-матча)
    pool_fps = pool["SMILES"].astype(str).apply(fp_cached)
    pool["__max_t__"] = 0.0

    mask_need = ~pool["__is_near_scaf__"]
    idxs = pool.index[mask_need].tolist()

    def _max_t_for_idx(i):
        fp = pool_fps[i]
        if fp is None or not ref_fps:
            return 0.0
        # безопасный вызов: на случай редких странностей внутри RDKit
        try:
            sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
            return max(sims) if sims else 0.0
        except Exception:
            best = 0.0
            for r in ref_fps:
                if r is not None:
                    s = DataStructs.TanimotoSimilarity(fp, r)
                    if s > best: best = s
            return best

    n_jobs = int(getattr(args, "unl_n_jobs", 0))
    # чтобы не душить CPU внутри процессов (иначе over-subscription)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    use_threads = (platform.system() == "Windows")  # на Windows — threads, на *nix — процессы

    if n_jobs and len(idxs) > 1000:
        if use_threads:
            # ВНИМАНИЕ: для threading НЕЛЬЗЯ задавать inner_max_num_threads
            print(f"[UNL][JBL] backend=threading prefer=threads n_jobs={n_jobs} | tasks={len(idxs)}")
            with parallel_config(backend="threading"):
                vals = Parallel(
                    n_jobs=n_jobs,
                    prefer="threads",
                    batch_size=512,
                )(delayed(_max_t_for_idx)(i) for i in idxs)
        else:
            print(f"[UNL][JBL] backend=loky prefer=processes n_jobs={n_jobs} | tasks={len(idxs)}")
            # Для процессов можно и НУЖНО ограничить внутренний трединг, чтобы не было оверсабскрипшена
            with parallel_config(backend="loky", inner_max_num_threads=1):
                vals = Parallel(
                    n_jobs=n_jobs,
                    prefer="processes",
                    batch_size=512,
                )(delayed(_max_t_for_idx)(i) for i in idxs)

        pool.loc[idxs, "__max_t__"] = vals
    else:
        pool.loc[idxs, "__max_t__"] = [_max_t_for_idx(i) for i in idxs]

    tic3.done()

    # краткая статистика по max_t
    if need > 0:
        q = pool.loc[~pool["__is_near_scaf__"], "__max_t__"].quantile([0.0, 0.5, 0.9, 0.99]).to_dict()
        print(f"[UNL] __max_t__ quantiles (non-scaffold): {q}")

    # квоты и выбор
    print(f"[UNL] target={target_N} | near_frac={getattr(args, 'unl_near_frac', 0.6)} "
          f"| far_frac={getattr(args, 'unl_far_frac', 0.2)} | scaffold_cap={getattr(args, 'unl_scaffold_cap', 0)}")

    # 4) Квоты near/far
    near_frac = float(getattr(args, "unl_near_frac", 0.6))
    far_frac  = float(getattr(args, "unl_far_frac", 0.2))
    near_N = int(round(near_frac * target_N))
    far_N  = int(round(far_frac  * target_N))
    rest_N = max(0, target_N - near_N - far_N)

    # near кандидаты: совпавшие по scaffold ИЛИ с высоким max-T (расположим по убыванию)
    near_candidates = pool[(pool["__is_near_scaf__"]) | (pool["__max_t__"] >= near_thr)].copy()

    # scaffold-баланс (cap)
    cap = int(getattr(args, "unl_scaffold_cap", 0))
    if cap > 0 and "scaffold" in near_candidates.columns:
        near_selected = []
        for scaf, grp in near_candidates.groupby("scaffold"):
            k = min(len(grp), cap)
            if k > 0:
                grp2 = grp.sort_values(
                    by=["__is_near_scaf__", "__max_t__"],
                    ascending=[False, False]
                ).head(k)
                near_selected.append(grp2)
        near_candidates = pd.concat(near_selected, axis=0, ignore_index=True) if near_selected else near_candidates

    # если переизбыток — обрежем по приоритету: сначала scaffold, потом Tanimoto
    if len(near_candidates) > near_N:
        near_candidates = near_candidates.sort_values(
            by=["__is_near_scaf__", "__max_t__"],
            ascending=[False, False]
        ).head(near_N)

    # иначе — возможно, доберём ещё «высоким Tanimoto» из оставшегося пула
    if len(near_candidates) < near_N:
        rest_pool = pool.drop(index=near_candidates.index)
        extra = rest_pool.sort_values("__max_t__", ascending=False).head(near_N - len(near_candidates))
        near_candidates = pd.concat([near_candidates, extra], axis=0)

    # far кандидаты: низкий max-Tanimoto
    far_pool = pool.drop(index=near_candidates.index)
    far_candidates = far_pool[far_pool["__max_t__"] <= far_thr]
    if len(far_candidates) > far_N:
        far_candidates = far_candidates.sample(n=far_N, random_state=args.seed)
    # остаток — случайно из оставшегося пула
    rest_pool = pool.drop(index=pd.concat([near_candidates, far_candidates]).index)
    if rest_N > 0 and len(rest_pool) > 0:
        rest_candidates = rest_pool.sample(n=min(rest_N, len(rest_pool)), random_state=args.seed)
    else:
        rest_candidates = rest_pool.iloc[0:0]

    rest_candidates = rest_pool.sample(n=min(rest_N, len(rest_pool)), random_state=args.seed) if rest_N > 0 else rest_pool.iloc[0:0]

    out = pd.concat([near_candidates, far_candidates, rest_candidates], axis=0).drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    out = out[["SMILES"]]

    print(f"[UNL-SELECT] pool={len(pool)} → near={len(near_candidates)} | "
          f"far={len(far_candidates)} | rest={len(rest_candidates)} | target={target_N}")
    print(f"[UNL] unique_scaffolds_selected={pd.concat([near_candidates, far_candidates, rest_candidates])['scaffold'].nunique()}")
    tic.done("done")

    return out

# =============================
# Pretraining (GraphCL-style)
# =============================
def random_node_drop(x, edge_index, edge_attr, batch_vec, drop_prob=0.1):
    """
    Дропает узлы и СИНХРОННО обновляет batch_vec.
    Гарантирует, что у каждого графа останется >=1 узел.
    """
    assert torch.is_tensor(batch_vec) and batch_vec.dim() == 1 and batch_vec.numel() == x.size(0)

    if x.size(0) <= 2 or drop_prob <= 1e-6:
        return x, edge_index, edge_attr, batch_vec

    N = x.size(0)
    device = x.device
    keep = (torch.rand(N, device=device) > drop_prob)

    # гарантируем минимум 1 узел на граф
    for g in torch.unique(batch_vec):
        mask_g = (batch_vec == g)
        if (keep & mask_g).sum() == 0:
            idxs = torch.nonzero(mask_g, as_tuple=False).view(-1)
            keep[idxs[torch.randint(0, len(idxs), (1,), device=device)]] = True

    idx = torch.nonzero(keep, as_tuple=False).view(-1)
    # перенумеровка узлов
    new_index = -torch.ones(N, dtype=torch.long, device=device)
    new_index[idx] = torch.arange(idx.numel(), device=device)

    # фильтруем рёбра
    if edge_index.numel() > 0:
        mask_e = keep[edge_index[0]] & keep[edge_index[1]]
        ei = edge_index[:, mask_e]
        ea = edge_attr[mask_e]
        ei = new_index[ei]
    else:
        ei = edge_index
        ea = edge_attr

    x_out = x[idx]
    bvec_out = batch_vec[idx]
    return x_out, ei, ea, bvec_out


def random_edge_drop(x, edge_index, edge_attr, drop_prob=0.1):
    if edge_index.numel() == 0 or drop_prob <= 1e-6:
        return x, edge_index, edge_attr
    E = edge_index.size(1)
    keep = torch.rand(E, device=x.device) > drop_prob
    if keep.sum() < 1:
        keep[torch.randint(0, E, (1,), device=x.device)] = True
    return x, edge_index[:, keep], edge_attr[keep]

def random_supercell_2d(x, edge_index, edge_attr, batch_vec, p=0.0, repeats=3, wrap=False):
    """
    На части графов батча строит 'суперячейки' из repeats копий:
      - дублирует узлы/рёбра с соответствующим смещением индексов
      - соединяет между копиями узлы со флагом is_star_nb POLY-рёбрами
    Требует: в x хвост из EXTRA_NODE_FEATS, где [0]=is_backbone, [2]=is_star_nb.
    """
    if p <= 1e-6 or repeats <= 1:
        return x, edge_index, edge_attr, batch_vec

    device = x.device
    N, D = x.size()
    E = edge_index.size(1)

    extra = EXTRA_NODE_FEATS  # уже есть в твоём модуле
    star_col = D - extra + 2  # [bb, dist2bb, star_nb, dist2star, pos_sin, pos_cos]
    is_star_nb = (x[:, star_col] > 0.5)

    bf_poly_list = edge_attr_build(None, is_poly=True, is_bb_bond=0.0)
    bf_poly = torch.tensor(bf_poly_list, dtype=edge_attr.dtype, device=edge_attr.device).unsqueeze(0)

    # будем собирать новые тензоры порционно по каждому графу в батче
    new_x, new_ei, new_ea, new_b = [], [], [], []
    node_offset_global = 0

    num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0
    for g in range(num_graphs):
        # --- вырежем подграф g ---
        mask_g = (batch_vec == g)
        idx_g = torch.nonzero(mask_g, as_tuple=False).view(-1)
        if idx_g.numel() == 0:
            continue
        # локальная перенумерация
        new_index = -torch.ones(N, dtype=torch.long, device=device)
        new_index[idx_g] = torch.arange(idx_g.numel(), device=device)
        # рёбра внутри g
        if E > 0:
            mask_e = mask_g[edge_index[0]] & mask_g[edge_index[1]]
            ei_g = new_index[edge_index[:, mask_e]]
            ea_g = edge_attr[mask_e]
        else:
            ei_g = torch.zeros((2,0), dtype=torch.long, device=device)
            ea_g = torch.zeros((0, edge_attr.size(1)), dtype=edge_attr.dtype, device=device)

        x_g = x[idx_g]
        star_g = is_star_nb[idx_g]
        # решение: с вероятностью p применяем суперячейку
        if torch.rand(1, device=device).item() < p and star_g.any():
            # повторяем узлы/рёбра repeats раз
            R = int(max(2, repeats))
            copies_x = [x_g] + [x_g.clone() for _ in range(R-1)]
            copies_b = [torch.full((x_g.size(0),), len(new_b), dtype=torch.long, device=device) for _ in range(R)]
            # рёбра каждой копии
            copies_ei, copies_ea = [], []
            for r in range(R):
                shift = r * x_g.size(0)
                if ei_g.numel() > 0:
                    copies_ei.append(ei_g + shift)
                    copies_ea.append(ea_g)
            # POLY-линки между копиями по порядку 'один-к-одному' среди star_nb
            star_idx_local = torch.nonzero(star_g, as_tuple=False).view(-1)
            # защитимся: если нечётное число, лишний игнорируем
            S = star_idx_local.numel() - (star_idx_local.numel() % 2)
            star_idx_local = star_idx_local[:S]
            # строим пары (0,1), (2,3), ... или просто 0..S-1 «по месту»
            # здесь предпочтём «один-к-одному» по индексу
            poly_src, poly_dst = [], []
            if S >= 2:
                for r in range(R - (0 if wrap else 1)):
                    a = r
                    b = (r+1) % R
                    # соединим одинаковые индексы star_nb в копиях a и b
                    for k in range(S):
                        u = int(star_idx_local[k]) + a * x_g.size(0)
                        v = int(star_idx_local[k]) + b * x_g.size(0)
                        poly_src += [u, v]; poly_dst += [v, u]
            if poly_src:
                poly_ei = torch.tensor([poly_src, poly_dst], dtype=torch.long, device=device)
                poly_ea = bf_poly.repeat(len(poly_src), 1)
                copies_ei.append(poly_ei)
                copies_ea.append(poly_ea)

            # склеиваем копии
            x_gc = torch.cat(copies_x, dim=0)
            b_gc = torch.cat(copies_b, dim=0)
            if copies_ei:
                ei_gc = torch.cat(copies_ei, dim=1)
                ea_gc = torch.cat(copies_ea, dim=0)
            else:
                ei_gc = torch.zeros((2,0), dtype=torch.long, device=device)
                ea_gc = torch.zeros((0, edge_attr.size(1)), dtype=edge_attr.dtype, device=device)
        else:
            # без изменений
            x_gc, ei_gc, ea_gc = x_g, ei_g, ea_g
            b_gc = torch.full((x_gc.size(0),), len(new_b), dtype=torch.long, device=device)

        # глобальный сдвиг индексов при склейке батча
        if ei_gc.numel() > 0:
            ei_gc = ei_gc + node_offset_global
        new_x.append(x_gc)
        new_b.append(b_gc)
        new_ei.append(ei_gc)
        new_ea.append(ea_gc)
        node_offset_global += x_gc.size(0)

    x_out = torch.cat(new_x, dim=0) if new_x else x
    b_out = torch.cat(new_b, dim=0) if new_b else batch_vec
    if new_ei:
        ei_out = torch.cat(new_ei, dim=1)
        ea_out = torch.cat(new_ea, dim=0)
    else:
        ei_out = edge_index
        ea_out = edge_attr
    return x_out, ei_out, ea_out, b_out


class ContrastiveHead(nn.Module):
    def __init__(self, in_dim, proj_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )
    def forward(self, z):
        z = self.net(z)
        z = nn.functional.normalize(z, dim=-1)
        return z

def info_nce_loss(z1, z2, temperature=0.2):
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.t()  # cosine since z normalized
    n = z.size(0)
    labels = torch.arange(n, device=z.device)
    labels = (labels + (n//2)) % n  # positives offset by half
    mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim / temperature
    sim = sim.masked_fill(mask, float('-inf'))
    loss = nn.functional.cross_entropy(sim, labels)
    return loss

# =============================
# SSL Evaluation (A-F)
# =============================
from rdkit.Chem import AllChem
from rdkit import DataStructs

@torch.no_grad()
def _encode_loader(enc, loader, device, *, use_gdesc: bool, use_poly_hints: bool):
    enc.eval()
    outs = []
    for x, ei, ea, bvec, gdesc in loader:
        x, ei, ea, bvec, gdesc = x.to(device), ei.to(device), ea.to(device), bvec.to(device), gdesc.to(device)
        z = enc.encode(x, ei, ea, bvec, gdesc, use_gdesc=use_gdesc, use_poly_hints=use_poly_hints)  # [B, D]
        outs.append(z.detach().to(torch.float32).cpu().numpy())
    import numpy as _np
    return _np.vstack(outs) if outs else _np.zeros((0, getattr(enc, "hidden", 256)), dtype=_np.float32)

def _morgan_fp_from_smiles(smiles, nBits=2048, radius=2):
    m = safe_mol_from_smiles(smiles)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits)

def _tanimoto(fp1, fp2):
    if fp1 is None or fp2 is None: return 0.0
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def _triplet_accuracy(Z, fps, pos_thr=0.7, neg_thr=0.2, K=50000, seed=42):
    import numpy as np
    rng = np.random.default_rng(seed)
    N = len(fps)
    ok = 0; tot = 0
    for _ in range(K):
        i = int(rng.integers(0, N))
        # ищем positive и negative
        # быстрый рандомный поиск: до лимита попыток
        tries = 0
        p = n = None
        while tries < 50 and (p is None or n is None):
            j = int(rng.integers(0, N))
            if j == i:
                tries += 1; continue
            s = _tanimoto(fps[i], fps[j])
            if s >= pos_thr and p is None: p = j
            if s <= neg_thr and n is None: n = j
            tries += 1
        if p is None or n is None:
            continue
        da = np.linalg.norm(Z[i] - Z[p])
        dn = np.linalg.norm(Z[i] - Z[n])
        ok += (da < dn); tot += 1
    return (ok / max(1, tot), int(tot))

def _corr_embed_vs_fp(Z, fps, pairs=50000, seed=123):
    import numpy as np
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    N = len(fps)
    if N < 3: return np.nan
    idx1 = rng.integers(0, N, size=pairs)
    idx2 = rng.integers(0, N, size=pairs)
    d_emb = np.linalg.norm(Z[idx1] - Z[idx2], axis=1)
    d_fp = np.array([1.0 - _tanimoto(fps[i], fps[j]) for i,j in zip(idx1, idx2)], dtype=np.float64)
    rho = spearmanr(d_emb, d_fp).correlation
    return float(rho)

def _retrieval_at_k(Z, labels, ks=(1,5,10), sample_q=2000, seed=7):
    """
    labels: либо scaffold-строки, либо булевские множества «похожие» — здесь используем scaffold.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    N, D = Z.shape
    # косинусной близостью удобнее: нормируем
    Zn = Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9, None)
    # выбираем запросы
    Q = min(sample_q, N)
    q_idx = rng.choice(N, size=Q, replace=False)
    # скалярные произведения к остальным
    sims = Zn[q_idx] @ Zn.T  # [Q, N]
    # себя исключаем
    for qi, i in enumerate(q_idx):
        sims[qi, i] = -1e9
    recalls = {k:0 for k in ks}
    valid = 0
    for qi, i in enumerate(q_idx):
        lab = labels[i]
        # если у этого скелета уникальный экземпляр, «правильных» соседей может не быть
        has_pos = any((labels[j]==lab and j!=i) for j in range(N))
        if not has_pos:
            continue
        valid += 1
        order = np.argpartition(-sims[qi], kth=max(ks))[:max(ks)]
        top_sorted = order[np.argsort(-sims[qi, order])]
        for k in ks:
            topk = top_sorted[:k]
            hit = any(labels[j]==lab for j in topk)
            recalls[k] += int(hit)
    if valid == 0:
        return {k: np.nan for k in ks}, 0
    return {k: recalls[k]/valid for k in ks}, valid

def _alignment_uniformity(
    enc, ds, device, *,
    batches=4,
    aug_node_drop=0.1,
    aug_edge_drop=0.1,
    loader_bs=256,
    nw=0,
    aug_supercell_p=0.0,
    aug_supercell_repeats=3,
    aug_supercell_wrap=False,
):
    """
    Alignment = mean ||z(x) - z(aug(x))||^2
    Uniformity = log E exp(-2 ||z_i - z_j||^2)
    """
    import numpy as np
    kw = dict(
        batch_size=loader_bs, shuffle=True, collate_fn=collate_unl,
        num_workers=nw, pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
    )
    if nw > 0:
        kw["prefetch_factor"] = 2
    loader = DataLoader(ds, **kw)
    enc.eval()
    A, Z_all = [], []
    it = 0
    for x, ei, ea, bvec, gdesc in loader:
        it += 1
        # supercell перед view-аугациями
        if aug_supercell_p > 0:
            x_sc, ei_sc, ea_sc, b_sc = random_supercell_2d(
                x, ei, ea, bvec,
                p=aug_supercell_p,
                repeats=aug_supercell_repeats,
                wrap=aug_supercell_wrap
            )
        else:
            x_sc, ei_sc, ea_sc, b_sc = x, ei, ea, bvec

        x1, ei1, ea1, bvec1 = random_node_drop(x_sc, ei_sc, ea_sc, b_sc, aug_node_drop)
        x1, ei1, ea1 = random_edge_drop(x1, ei1, ea1, aug_edge_drop)
        x2, ei2, ea2, bvec2 = random_node_drop(x_sc, ei_sc, ea_sc, b_sc, aug_node_drop)
        x2, ei2, ea2 = random_edge_drop(x2, ei2, ea2, aug_edge_drop)
        with torch.no_grad():
            z1 = enc.encode(x1.to(device), ei1.to(device), ea1.to(device), bvec1.to(device), gdesc.to(device),
                            use_gdesc=False, use_poly_hints=False)
            z2 = enc.encode(x2.to(device), ei2.to(device), ea2.to(device), bvec2.to(device), gdesc.to(device),
                            use_gdesc=False, use_poly_hints=False)
        d2 = (z1 - z2).pow(2).sum(dim=1).detach().cpu().numpy()
        A.append(d2.mean())
        Z_all.append(z1.detach().cpu().numpy())
        if it >= batches:
            break
    A = float(np.mean(A)) if A else np.nan
    if not Z_all:
        return A, np.nan
    Z = np.vstack(Z_all)
    Z = Z / np.clip(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9, None)
    M = min(20000, Z.shape[0] * (Z.shape[0] - 1) // 2)
    rng = np.random.default_rng(123)
    idx1 = rng.integers(0, Z.shape[0], size=M)
    idx2 = rng.integers(0, Z.shape[0], size=M)
    d2 = ((Z[idx1] - Z[idx2]) ** 2).sum(axis=1)
    U = float(np.log(np.exp(-2.0 * d2).mean()))
    return A, U

def _collapse_check(Z, eps=1e-6):
    import numpy as np
    X = Z - Z.mean(axis=0, keepdims=True)
    # SVD на ковариации
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    s1 = float(S[0]) if S.size>0 else 0.0
    ratio = s1 / max(1e-9, S.sum())
    num_small = int((S < eps).sum())
    return ratio, num_small

def _linear_knn_probe(Z, smiles, props=("MolLogP","TPSA","NumAromaticRings","NumRotatableBonds"), train_frac=0.8, seed=101):
    """
    Псевдо-лейблы RDKit на том же holdout-е; обучаем линейку и kNN на эмбеддингах.
    Возвращает {prop: {"R2":.., "Spearman":.., "MAE":..} ...}
    """
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from scipy.stats import spearmanr
    # считаем свойства
    vals = []
    for s in smiles:
        m = safe_mol_from_smiles(s)
        if m is None:
            vals.append({p: np.nan for p in props}); continue
        d = {
            "MolLogP": Crippen.MolLogP(m),
            "TPSA": Descriptors.TPSA(m),
            "NumAromaticRings": Descriptors.NumAromaticRings(m),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(m),
        }
        vals.append({p: d[p] for p in props})
    vals = {p: np.array([v[p] for v in vals], dtype=np.float64) for p in props}
    rng = np.random.default_rng(seed)
    N = Z.shape[0]; idx = rng.permutation(N); cut = int(train_frac*N)
    tr, te = idx[:cut], idx[cut:]
    out = {}
    for p in props:
        y = vals[p]
        mask = np.isfinite(y)
        trm = np.intersect1d(tr, np.where(mask)[0], assume_unique=False)
        tem = np.intersect1d(te, np.where(mask)[0], assume_unique=False)
        if trm.size<10 or tem.size<10:
            out[p] = {"R2": np.nan, "Spearman": np.nan, "MAE": np.nan}; continue
        # линейный пробинг
        ridge = Ridge(alpha=1.0, random_state=seed).fit(Z[trm], y[trm])
        yp = ridge.predict(Z[tem])
        r2 = 1 - np.sum((y[tem]-yp)**2)/max(1e-12, np.sum((y[tem]-y[tem].mean())**2))
        rho = spearmanr(y[tem], yp).correlation
        mae = float(np.mean(np.abs(y[tem]-yp)))
        # kNN (5-NN) для справки
        knn = KNeighborsRegressor(n_neighbors=5).fit(Z[trm], y[trm])
        yp2 = knn.predict(Z[tem])
        r2_knn = 1 - np.sum((y[tem]-yp2)**2)/max(1e-12, np.sum((y[tem]-y[tem].mean())**2))
        rho_knn = spearmanr(y[tem], yp2).correlation
        mae_knn = float(np.mean(np.abs(y[tem]-yp2)))
        out[p] = {"R2": float(r2), "Spearman": float(rho), "MAE": float(mae),
                  "R2_kNN": float(r2_knn), "Spearman_kNN": float(rho_knn), "MAE_kNN": float(mae_knn)}
    return out

def run_ssl_evaluation(enc, val_ds, val_smiles, device, args, out_dir, tag=""):
    """
    Считает метрики A–F, пишет JSON и печатает сводку с универсальными оценками
    (независимо от прошлых запусков).
    """
    import json, os, numpy as np
    from math import isfinite, log10
    # === helpers ===
    LABELS = [(95,"идеально"), (85,"отлично"), (70,"хорошо"),
              (55,"удовлетворительно"), (40,"плохо"), (0,"ужасно")]
    def _label(score):
        for th, name in LABELS:
            if score >= th: return name
        return LABELS[-1][1]

    def _linpiece(x, xs, ys):
        """Линейная интерполяция по точкам xs (возр.) → баллы ys (0..100)."""
        if x is None or not isfinite(x): return 0.0
        if x <= xs[0]: return ys[0]
        if x >= xs[-1]: return ys[-1]
        for i in range(len(xs)-1):
            if xs[i] <= x <= xs[i+1]:
                t = (x - xs[i]) / (xs[i+1] - xs[i] + 1e-12)
                return ys[i] + t * (ys[i+1] - ys[i])
        return ys[-1]

    # === универсальные пороги (по смыслу метрик) ===
    # Retrieval:
    #   R@1:   <0.15—слабо; ~0.35—хорошо; ≥0.50—отлично (очень строгая задача)
    #   R@10:  <0.50—слабо; ~0.70—хорошо; ≥0.80—отлично
    # Spearman ρ (дист. эмбеддингов vs 1−Tanimoto):
    #   <0.25—слабо; 0.35—удовл.; 0.45—хорошо; ≥0.55—отлично
    # TripletAcc:
    #   <0.75—слабо; 0.85—удовл.; 0.92—хорошо; ≥0.97—отлично (но метрика легко насыщается)
    # Alignment — ниже лучше; нормализуем log10(align):
    #   log10 ~ 2.3–3.6 — норма; >4.5 — уже плохо; >5.5 — очень плохо
    # Uniformity — более отрицательно лучше; берём -Unif:
    #   ~1.6 — норма; ~1.8 — хорошо; ≥1.95 — отлично
    # Collapse ratio — доля малых синг. значений (ниже лучше):
    #   ≤0.10 — отлично; 0.12–0.16 — ок; ≥0.22 — тревожно (коллапс/перекластеризация)
    THRESHOLDS = {
        "r1":   ( [0.05, 0.15, 0.25, 0.35, 0.50, 0.60], [0, 25, 45, 70, 90, 100] ),
        "r10":  ( [0.30, 0.50, 0.60, 0.70, 0.80, 0.90], [0, 25, 45, 70, 90, 100] ),
        "rho":  ( [0.10, 0.25, 0.35, 0.45, 0.55, 0.65], [0, 25, 45, 70, 90, 100] ),
        "trip": ( [0.60, 0.75, 0.85, 0.92, 0.97, 0.995],[0, 25, 45, 70, 90, 100] ),
        # ниже — лучше (оценка падает с ростом):
        "align_log10": ( [2.3, 2.8, 3.2, 3.6, 4.5, 5.5], [100, 90, 70, 45, 25, 0] ),
        # выше (по модулю) — лучше:
        "unif_abs": ( [1.20, 1.40, 1.60, 1.80, 1.95, 2.10],[0, 25, 45, 70, 90, 100] ),
        # ниже — лучше:
        "col":  ( [0.08, 0.10, 0.12, 0.16, 0.22, 0.30],   [100, 90, 70, 45, 25, 0] ),
    }

    def _grade(res, ks_sorted):
        # берём R@1 (если есть) и ближайший к k=10 (если есть)
        k_small = ks_sorted[0] if ks_sorted else None
        k10 = next((k for k in ks_sorted if k >= 10), (ks_sorted[-1] if ks_sorted else None))
        r_small = res["retrieval"].get(f"R@{k_small}") if k_small else None
        r10 = res["retrieval"].get(f"R@{k10}") if k10 else None

        # оценка retrieval
        s_r_small = _linpiece(r_small, *THRESHOLDS["r1"]) if (k_small == 1) else (
                    _linpiece(r_small, *THRESHOLDS["r10"]) if r_small is not None else 0.0)
        s_r10     = _linpiece(r10, *THRESHOLDS["r10"]) if r10 is not None else 0.0

        # прочие
        s_rho   = _linpiece(res["spearman_embed_vs_fp"], *THRESHOLDS["rho"])
        s_trip  = _linpiece(res["triplet_acc"], *THRESHOLDS["trip"])
        s_unif  = _linpiece(abs(res["uniformity"]), *THRESHOLDS["unif_abs"])
        a = res["alignment"]
        s_align = _linpiece(log10(a+1e-12), *THRESHOLDS["align_log10"]) if isfinite(a) and a>0 else 0.0
        s_col   = _linpiece(res["collapse_ratio_sigma1_over_sum"], *THRESHOLDS["col"])

        # итог: акцент на Retrieval и ρ; Align/Unif контролируют «геометрию», col — анти-коллапс
        weights = {
            "retr_small": 0.30,  # R@1
            "rho":        0.25,
            "retr_10":    0.15,
            "unif":       0.10,
            "align":      0.10,
            "col":        0.07,
            "trip":       0.03,
        }
        overall = (s_r_small*weights["retr_small"] + s_rho*weights["rho"] +
                   s_r10*weights["retr_10"] + s_unif*weights["unif"] +
                   s_align*weights["align"] + s_col*weights["col"] +
                   s_trip*weights["trip"])
        per_metric = {
            f"R@{k_small}": {"score": round(s_r_small,1), "label": _label(s_r_small)} if k_small else None,
            f"R@{k10}":     {"score": round(s_r10,1),     "label": _label(s_r10)}     if k10 else None,
            "rho":          {"score": round(s_rho,1),     "label": _label(s_rho)},
            "triplet_acc":  {"score": round(s_trip,1),    "label": _label(s_trip)},
            "alignment":    {"score": round(s_align,1),   "label": _label(s_align)},
            "uniformity":   {"score": round(s_unif,1),    "label": _label(s_unif)},
            "col_ratio":    {"score": round(s_col,1),     "label": _label(s_col)},
        }
        per_metric = {k:v for k,v in per_metric.items() if v is not None}
        return float(overall), _label(overall), per_metric

    # ===== 1) эмбеддинги валидации
    val_loader = DataLoader(
        val_ds, batch_size=args.ssl_eval_batch, shuffle=False, collate_fn=collate_unl,
        **make_loader_kwargs(args.pre_num_workers)
    )

    Z = _encode_loader(enc, val_loader, device, use_gdesc = args.ssl_use_gdesc, use_poly_hints = args.ssl_use_poly_hints)

    # ===== 2) fingerprints & scaffolds
    fps = [_morgan_fp_from_smiles(s, nBits=args.ssl_fp_bits) for s in val_smiles]

    scaff = [compute_scaffold(s) for s in val_smiles]

    # === A. Triplet Acc
    trip_acc, trip_used = _triplet_accuracy(Z, fps, args.ssl_triplet_pos, args.ssl_triplet_neg, args.ssl_triplet_K)
    # === B. Corr(embed, 1-Tanimoto)
    rho = _corr_embed_vs_fp(Z, fps, pairs=min(100000, len(val_smiles)*10))
    # === C. Retrieval@k по scaffold'ам
    ks = tuple(int(x.strip()) for x in args.ssl_retrieval_k.split(",") if x.strip())
    retk, ret_valid = _retrieval_at_k(Z, scaff, ks=ks)
    # === D. Alignment/Uniformity
    align, unif = _alignment_uniformity(
        enc, val_ds, device,
        batches=args.ssl_align_batches,
        aug_node_drop=args.aug_node_drop,
        aug_edge_drop=args.aug_edge_drop,
        loader_bs=args.ssl_eval_batch,
        nw=(args.pre_num_workers or 0),
        aug_supercell_p=args.aug_supercell_p,
        aug_supercell_repeats=args.aug_supercell_repeats,
        aug_supercell_wrap=args.aug_supercell_wrap,
    )

    # === E. Collapse-чек
    col_ratio, col_small = _collapse_check(Z)
    # === F. Probing
    probe = _linear_knn_probe(Z, val_smiles)

    # raw results
    res = {
        "N_val": len(val_smiles),
        "triplet_acc": float(trip_acc), "triplet_used": int(trip_used),
        "spearman_embed_vs_fp": float(rho),
        "retrieval": {f"R@{k}": float(retk.get(k, float('nan'))) for k in ks},
        "retrieval_valid": int(ret_valid),
        "alignment": float(align),
        "uniformity": float(unif),
        "collapse_ratio_sigma1_over_sum": float(col_ratio),
        "collapse_num_small_sv": int(col_small),
        "probe": probe,
    }

    # scoring
    ks_sorted = sorted(ks)
    overall_score, overall_label, per_metric = _grade(res, ks_sorted)
    res["score"] = {
        "overall": round(overall_score, 1),
        "label": overall_label,
        "per_metric": per_metric
    }

    # write JSON
    out_json = os.path.join(out_dir, f"ssl_eval{('_'+tag) if tag else ''}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("[SSL-EVAL] →", out_json)

    # raw print
    print(
        "[SSL-EVAL] "
        f"TripAcc={trip_acc:.3f} (n={trip_used}) | ρ={rho:.3f} | "
        + " ".join([f"R@{k}={retk.get(k, float('nan')):.3f}" for k in ks_sorted])
        + f" | Align={align:.3f} | Unif={unif:.3f} | col_ratio={col_ratio:.3f}"
    )

    # scored print
    def _pm(name):
        d = per_metric.get(name)
        return f"{name}: {d['score']:.0f} ({d['label']})" if d else None
    pieces = [
        f"ИТОГ={res['score']['overall']:.1f}/100 ({res['score']['label']})",
        _pm(f"R@{ks_sorted[0]}") if ks_sorted else None,
        next((_pm(f"R@{k}") for k in ks_sorted if k>=10 and f"R@{k}" in per_metric), None),
        _pm("rho"), _pm("uniformity"), _pm("alignment"), _pm("col_ratio"), _pm("triplet_acc"),
    ]
    print("\n\n[SSL-EVAL][SCORE] " + " | ".join([p for p in pieces if p]) + "\n\n")

    # предупреждения по здравому смыслу
    notes = []
    if res["retrieval_valid"] < max(100, 0.01*res["N_val"]):
        notes.append("мало валидных пар для Retrieval — оценка может быть шумной")
    if res["collapse_ratio_sigma1_over_sum"] >= 0.22:
        notes.append("возможен коллапс/узкие кластеры — проверьте аугментации/температуру")
    if len(notes):
        print("[SSL-EVAL][NOTE] " + " | ".join(notes))

    return res



# =============================
# Training / Eval
# =============================
def make_model(sample: GraphData, args):
    node_in = sample.x.size(1)
    edge_in = sample.edge_attr.size(1) if sample.edge_attr.numel() > 0 else EDGE_FEAT_DIM
    gdesc_dim = sample.gdesc.numel()
    model = PolymerGNN(node_in, edge_in, gdesc_dim,
                       hidden=args.hidden, layers=args.layers, targets=len(TARGETS),
                       dropout=args.dropout,
                       predict_sigma=getattr(args, "hetero_sigma_head", False),
                       constrain=getattr(args, "constrain_head", False),
                       per_target_last=getattr(args, "per_target_last", False))


    return model, node_in, edge_in, gdesc_dim

def scaffold_kfold_indices(df: pd.DataFrame, n_folds: int = 5, seed: int = 12345) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Разбивает датасет на K фолдов по Murcko scaffolds без утечки.
    Балансируем фолды по числу образцов (жадная раскладка групп).
    Возвращает список кортежей (train_idx, val_idx).
    """
    assert n_folds >= 2
    _df = df.copy()
    _df["scaffold"] = _df["SMILES"].apply(compute_scaffold)

    # группы: scaffold -> индексы строк
    groups = _df.groupby("scaffold").indices
    scafs = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(scafs)

    fold_bins = [[] for _ in range(n_folds)]
    fold_sizes = [0 for _ in range(n_folds)]

    for s in scafs:
        idxs = list(groups[s])
        # кладём текущую группу в наименее заполненный фолд
        k = min(range(n_folds), key=lambda i: fold_sizes[i])
        fold_bins[k].extend(idxs)
        fold_sizes[k] += len(idxs)

    folds = []
    all_idx = np.arange(len(_df), dtype=int)
    for k in range(n_folds):
        val_idx = np.array(sorted(fold_bins[k]), dtype=int)
        tr_mask = np.ones(len(_df), dtype=bool)
        tr_mask[val_idx] = False
        tr_idx = all_idx[tr_mask]
        folds.append((tr_idx, val_idx))

    return folds


def scaffold_split(df: pd.DataFrame, frac_val: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group by Murcko scaffold to reduce leakage; fallback to random if needed"""
    df = df.copy()
    try:
        df["scaffold"] = df["SMILES"].apply(compute_scaffold)
        groups = df.groupby("scaffold")
        scafs = list(groups.groups.keys())
        random.Random(seed).shuffle(scafs)
        val_size = int(len(df) * frac_val)
        val_scafs = []
        cur = 0
        for s in scafs:
            gsize = len(groups.get_group(s))
            if cur + gsize <= val_size:
                val_scafs.append(s); cur += gsize
        val_mask = df["scaffold"].isin(val_scafs)
        va_df = df[val_mask].drop(columns=["scaffold"])
        tr_df = df[~val_mask].drop(columns=["scaffold"])
        if len(va_df)==0 or len(tr_df)==0:
            raise RuntimeError("Empty split")
        return tr_df, va_df
    except Exception:
        from sklearn.model_selection import train_test_split
        avail_count = df[TARGETS].notna().sum(axis=1)
        tr_df, va_df = train_test_split(df, test_size=frac_val, random_state=seed, stratify=avail_count)
        return tr_df, va_df

def split_main_only(sup_df: pd.DataFrame, val_size: float, seed: int):
    main = sup_df[sup_df["id"] != -1].copy()    # ровно то, что в train.csv
    ext  = sup_df[sup_df["id"] == -1].copy()    # всё внешнее

    tr_main, va_main = scaffold_split(main, val_size, seed)  # с новым compute_scaffold
    train = pd.concat([tr_main, ext], ignore_index=True)     # «внешку» только в train
    val   = va_main
    return train, val

def train_supervised(args):
    set_seed(args.seed)
    # Load main & supplemental
    train_df, test_df, _ = load_main_data(args.data_dir)
    supp_zip = load_supplemental(args.data_dir)
    external = ingest_external_sources(args)
    sup_df = build_supervised_dataframe(train_df, supp_zip, external)

    # === Split ID / Run tag ===
    split_id = make_split_id(args)
    run_tag = make_run_tag(args)

    modes = get_poly_modes(args)
    eval_mode = canonical_poly_mode(args, modes)

    # --- Глобальные кэши графов для КАЖДОГО режима (один раз)
    all_items_by_mode = {}
    for m in modes:
        all_cache_m = os.path.join(args.out_dir, f"cache_all_graphs_{m}.pt")
        print(f"[CACHE] {'Load' if os.path.isfile(all_cache_m) else 'Build'} supervised graph cache: {all_cache_m}")
        items_m = materialize_graphs_for_df(
            sup_df,
            TARGETS,
            poly_edge_mode=m,
            cache_path=all_cache_m
        )
        all_items_by_mode[m] = items_m


    # --- Получаем индексы train/val относительно sup_df ---
    if args.n_folds > 1:
        folds = scaffold_kfold_indices(sup_df, n_folds=args.n_folds, seed=args.split_seed)
        if not (0 <= args.fold < args.n_folds):
            raise ValueError(f"--fold должен быть в [0..{args.n_folds - 1}], получено {args.fold}")
        tr_idx, va_idx = folds[args.fold]
        tr_idx = tr_idx.tolist() if isinstance(tr_idx, np.ndarray) else list(tr_idx)
        va_idx = va_idx.tolist() if isinstance(va_idx, np.ndarray) else list(va_idx)
        print(f"[CV] Using fold {args.fold}/{args.n_folds - 1} | "
              f"train={len(tr_idx)} val={len(va_idx)} (split_seed={args.split_seed})")
    else:
        tr_idx, va_idx = split_main_only_indices(sup_df, args.val_size, args.split_seed)

    train_ds_by_mode = {m: _slice_tensor_ds(all_items_by_mode[m], tr_idx) for m in modes}
    val_ds_by_mode   = {m: _slice_tensor_ds(all_items_by_mode[m], va_idx) for m in modes}

    def make_sup_loaders(bs: int, train_mode: str, val_mode: str):
        g = torch.Generator();
        g.manual_seed(args.seed)
        train_ds = train_ds_by_mode[train_mode]
        sampler = None
        if getattr(args, 'balanced_sampler', False):
            # собираем маски наличия таргетов у каждого образца
            masks = []
            for it in getattr(train_ds, "items", []):  # (x, ei, ea, gdesc, y, ymask)
                m = it[5].reshape(-1)
                masks.append(m.numpy() if isinstance(m, torch.Tensor) else np.asarray(m))
            if len(masks):
                M = np.vstack(masks).astype(np.float32)  # (N, T)
                freq = M.mean(axis=0)  # доля наличия по каждому таргету
                eps = 1e-6
                beta = float(getattr(args, "sampler_beta", 0.5))
                strength = float(getattr(args, "sampler_strength", 1.0))
                bonus = np.power(np.maximum(freq, eps), -beta)  # ~ (1/freq)^beta
                bonus = bonus / max(bonus.mean(), eps)  # норм к среднему 1
                w = 1.0 + strength * (M @ bonus)  # (N,)
                w = w / max(w.mean(), eps)
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=torch.as_tensor(w, dtype=torch.double),
                    num_samples=len(train_ds),
                    replacement=True
                )
        tr_loader = DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate,
            generator=g,
            worker_init_fn=seed_worker,
            **make_loader_kwargs(args.train_num_workers)
        )
        va_loader = DataLoader(
            val_ds_by_mode[val_mode],
            batch_size=bs,
            shuffle=False,
            collate_fn=collate,
            generator=g,
            worker_init_fn=seed_worker,
            **make_loader_kwargs(args.train_num_workers)
        )
        return tr_loader, va_loader

    stages_ft = parse_batch_growth_3(args.ft_batch_growth)
    if stages_ft:
        plan = " → ".join(f"bs={s.bs}/pat={s.patience}/max={s.max_epochs}" for s in stages_ft)
        print(f"[SUP][BATCH-GROWTH] {plan}")

        controller_ft = BatchGrowthController(stages_ft, global_max_epochs=args.epochs, mode="min", min_delta=1e-4)

        active_mode = modes[0]                         # на старте тренируемся в первом режиме
        train_loader, val_loader = make_sup_loaders(controller_ft.bs, active_mode, eval_mode)

        print_loader_info(f"SUP/train[{active_mode}]", controller_ft.bs, args.train_num_workers,
                          torch.cuda.is_available(), (args.train_num_workers > 0))
        print_loader_info(f"SUP/val[{eval_mode}]", controller_ft.bs, args.train_num_workers,
                          torch.cuda.is_available(), (args.train_num_workers > 0))

    else:
        # Loaders
        g = torch.Generator()
        g.manual_seed(args.seed)

        active_mode = modes[0]
        train_loader, val_loader = make_sup_loaders(args.train_batch_size, active_mode, eval_mode)

        print_loader_info(f"SUP/train[{active_mode}]", args.train_batch_size, args.train_num_workers,
                          torch.cuda.is_available(), (args.train_num_workers > 0))
        print_loader_info(f"SUP/val[{eval_mode}]", args.train_batch_size, args.train_num_workers,
                          torch.cuda.is_available(), (args.train_num_workers > 0))


    # ranges, weights = compute_norm_and_weights(tr_df, TARGETS)
    ranges, weights = compute_norm_and_weights(sup_df.iloc[tr_idx], TARGETS)

    # --- Диагностика покрытия таргетов на train
    try:
        # Диагностика покрытия таргетов на train/val
        tr_cov = sup_df.iloc[tr_idx][TARGETS].notna().mean().round(3).to_dict()
        va_cov = sup_df.iloc[va_idx][TARGETS].notna().mean().round(3).to_dict()
        print(f"[SPLIT] train={len(tr_idx)} val={len(va_idx)} | target coverage train={tr_cov} val={va_cov}")

        # Сколько «внешних» в train (val по определению только main)
        tr_main = int((sup_df.iloc[tr_idx]["id"] != -1).sum())
        tr_ext = int((sup_df.iloc[tr_idx]["id"] == -1).sum())
        print(f"[SPLIT] train main={tr_main} ext={tr_ext}")

    except Exception:
        pass

    # Model (build from the active_mode dataset)
    # если активный режим пуст (случайно), берём первый непустой
    if len(train_ds_by_mode[active_mode]) == 0:
        for m in modes:
            if len(train_ds_by_mode[m]) > 0:
                active_mode = m
                break
    assert len(train_ds_by_mode[active_mode]) > 0, "Empty training dataset"

    sample = train_ds_by_mode[active_mode][0]
    model, node_in, edge_in, gdesc_dim = make_model(sample, args)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device)
    if args.debug_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # безопасный фолбэк: если нет Triton/Inductor, работаем без compile
    use_compile = (not args.cpu) and (not getattr(args, "no_compile", False))
    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")  # Inductor + Triton
        except Exception as e:
            print("[compile] TorchInductor/Triton недоступны, откатываюсь к eager:", e)
            # либо вообще без компиляции:
            # просто ничего не делаем
            # либо мягкий бэкенд без Inductor:
            # model = torch.compile(model, backend="eager")

    use_amp = (torch.cuda.is_available() and not args.cpu)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Warm-start from SSL encoder if available
    ssl_path = os.path.join(args.out_dir, "ssl_encoder.pt")


    if os.path.isfile(ssl_path):
        try:
            try:
                sd = torch.load(ssl_path, map_location=device, weights_only=True)
            except TypeError:
                sd = torch.load(ssl_path, map_location=device)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"Loaded SSL weights with {len(missing)} missing and {len(unexpected)} unexpected keys."),

            # --- свежая голова и rho ---
            def _reset_linear(m):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            def _apply_reset(m):
                if hasattr(model, "head_core"): model.head_core.apply(_reset_linear)
                if hasattr(model, "head_out"):  model.head_out.apply(_reset_linear)
                if hasattr(model, "head_out_list"):
                    for lin in model.head_out_list: lin.apply(_reset_linear)
                if hasattr(model, "head_sigma_core"): model.head_sigma_core.apply(_reset_linear)
                if hasattr(model, "head_sigma_out"):  model.head_sigma_out.apply(_reset_linear)

            _apply_reset(model)

            with torch.no_grad():
                model.rho.zero_()

        except Exception as e:
            print("Could not load SSL weights:", e)

    # Optim / schedulers
    base_lr = args.lr
    head_lr = args.head_lr if args.head_lr is not None else base_lr
    head_wd = args.head_weight_decay if args.head_weight_decay is not None else args.weight_decay
    sigma_lr = args.sigma_head_lr if args.sigma_head_lr is not None else head_lr

    # собираем параметры без дублей
    def _params_of(m):
        return list(m.parameters()) if m is not None else []
    backbone = set(p for p in model.parameters())
    head_params = []
    if hasattr(model, "head_core"): head_params += _params_of(model.head_core)
    if hasattr(model, "head_out"):  head_params += _params_of(model.head_out)
    if hasattr(model, "head_out_list"):
        for lin in model.head_out_list: head_params += _params_of(lin)
    if hasattr(model, "head_sigma_core"): head_params += _params_of(model.head_sigma_core)
    if hasattr(model, "head_sigma_out"):  head_params += _params_of(model.head_sigma_out)
    for p in head_params:
        if p in backbone: backbone.remove(p)
    # rho — отдельная группа без weight decay
    if model.rho in backbone: backbone.remove(model.rho)

    # группы
    param_groups = [
        {"params": list(backbone), "lr": base_lr, "weight_decay": args.weight_decay, "role": "backbone"},
    ]
    # head_core/out — единый LR/WD
    core_and_shared = []
    if hasattr(model, "head_core"): core_and_shared += _params_of(model.head_core)
    if hasattr(model, "head_out"):  core_and_shared += _params_of(model.head_out)
    if len(core_and_shared):
        param_groups.append({"params": core_and_shared, "lr": head_lr, "weight_decay": head_wd, "role": "head_core"})

    # per-target last
    if hasattr(model, "head_out_list"):
        mults = [float(x) for x in (args.head_lr_mults.split(",") if isinstance(args.head_lr_mults, str) else args.head_lr_mults)]
        if len(mults) != len(TARGETS):
            mults = (mults + [1.0]*len(TARGETS))[:len(TARGETS)]
        for i, lin in enumerate(model.head_out_list):
            param_groups.append({"params": list(lin.parameters()), "lr": head_lr*mults[i], "weight_decay": head_wd, "role": f"head_t{i}"})

    # sigma-head
    if hasattr(model, "head_sigma_core") or hasattr(model, "head_sigma_out"):
        sig_params = _params_of(model.head_sigma_core) + _params_of(model.head_sigma_out)
        if len(sig_params):
            param_groups.append({"params": sig_params, "lr": sigma_lr, "weight_decay": head_wd, "role": "head_sigma"})

    # rho
    param_groups.append({"params": [model.rho], "lr": head_lr, "weight_decay": 0.0, "role": "rho"})


    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.99))
    # диагностика
    print(f"[OPT] groups: backbone={len(param_groups[0]['params'])}, head_core/shared={len(core_and_shared)}, "
          f"per-target={(hasattr(model,'head_out_list') and len(model.head_out_list)) or 0}, sigma_head="
          f"{(hasattr(model,'head_sigma_out') and model.head_sigma_out.out_features) if hasattr(model,'head_sigma_out') else 0}")


    # --- Проверка, что голова и rho в оптимизаторе (п.4)
    opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    def _in_opt(mod):
        return any(id(p) in opt_param_ids for p in getattr(mod, "parameters")())

    print(f"[CHECK] head in optimizer: {_in_opt(model.head)} | "
          f"head_sigma in optimizer: {getattr(model, 'head_sigma', None) is not None and _in_opt(model.head_sigma)} | "
          f"rho in optimizer: {id(model.rho) in opt_param_ids}")

    # === Stage-aware scheduler: warmup_frac применяется ВНУТРИ каждой стадии ===

    def _estimate_stage_epochs_for_current(controller, default_ep: int) -> int:
        if controller is None:
            return max(1, default_ep)
        st = controller.stages[controller.stage_idx]
        # у тебя в планe поле называется max_epochs
        return max(1, int(getattr(st, "max_epochs", default_ep)))

    def _build_stage_scheduler(opt, steps_in_stage: int, warmup_frac: float, warmup_steps_override: int = 0,
                               bb_frac=None, bb_steps=0, bb_floor=0.0,
                               head_frac=None, head_steps=0, head_floor=0.0,
                               dyn_mults=None):
        """
        По-групповой LambdaLR:
          - для групп с role in {backbone} применяются (bb_frac|bb_steps, bb_floor)
          - для групп с role in {head_* , rho} применяются (head_frac|head_steps, head_floor)
          - dyn_mults[i] — внешний динамический множитель (для авто-LR per-target)
        """
        steps_in_stage = max(1, int(steps_in_stage))
        # дефолты
        if bb_frac is None: bb_frac = warmup_frac
        if head_frac is None: head_frac = warmup_frac
        dyn_mults = dyn_mults if dyn_mults is not None else [1.0] * len(opt.param_groups)

        def _mk_lambda(warm_frac, steps_override, floor):
            warm = int(steps_override) if steps_override and steps_override > 0 else max(1,
                                                                                         int(warm_frac * steps_in_stage))
            floor = float(max(0.0, min(1.0, floor)))

            def _lam(step):
                if step < warm:
                    return float(step + 1) / float(max(1, warm))
                remain = max(1, steps_in_stage - warm)
                progress = (step - warm) / float(remain)
                # косина от 1 к floor
                return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

            return _lam, warm

        # под каждую группу — своя лямбда
        lams = []
        warms = []
        for gi, g in enumerate(opt.param_groups):
            role = str(g.get("role", "backbone"))
            if role.startswith("head_") or role == "rho":
                lam, warm = _mk_lambda(head_frac, head_steps, head_floor)
            else:
                lam, warm = _mk_lambda(bb_frac, bb_steps, bb_floor)
            # заворачиваем в мультипликатор
            lams.append(lambda step, lam=lam, gi=gi: float(dyn_mults[gi]) * lam(step))
            warms.append(warm)

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lams)
        # warmup_steps наименьший для логов (не критично)
        return scheduler, int(np.median(warms)), steps_in_stage

    # первичная инициализация шедулера под текущий bs
    steps_per_epoch = len(train_loader)
    est_ep = _estimate_stage_epochs_for_current(controller_ft if stages_ft else None, args.epochs)
    _stage_total_steps = max(1, steps_per_epoch * est_ep)

    # динамические множители lr по группам (для авто-LR per-target)
    dyn_mults = [1.0] * len(opt.param_groups)

    sched, _stage_warmup_steps, _stage_total_steps = _build_stage_scheduler(
        opt,
        steps_in_stage=_stage_total_steps,
        warmup_frac=args.warmup_frac,
        warmup_steps_override=getattr(args, "warmup_steps", 0),
        bb_frac=(args.bb_warmup_frac if args.bb_warmup_frac is not None else args.warmup_frac),
        bb_steps=args.bb_warmup_steps, bb_floor=args.bb_cosine_floor,
        head_frac=(args.head_warmup_frac if args.head_warmup_frac is not None else args.warmup_frac),
        head_steps=args.head_warmup_steps, head_floor=args.head_cosine_floor,
        dyn_mults=dyn_mults
    )

    # --- Авто-LR per-target: подготовка
    head_group_indices = []  # индексы param_groups, соответствующие head_out_list[i]
    if getattr(model, "per_target_last", False):
        for gi, g in enumerate(opt.param_groups):
            r = str(g.get("role", ""))
            if r.startswith("head_t"):
                head_group_indices.append(gi)  # порядок совпадает с индексом таргета
    # буферы норм градиента
    from collections import deque
    grad_hist = [deque(maxlen=max(1, int(args.auto_lr_window))) for _ in range(len(head_group_indices))]

    def _module_grad_norm(mod):
        s = 0.0
        for p in mod.parameters():
            if p.grad is not None:
                v = p.grad.detach()
                s += float((v * v).sum().item())
        return math.sqrt(max(s, 0.0))


    _stage_step = 0  # локальный шаг в текущей стадии
    last_lr = None  # для диагностики
    print(f"[SUP][BATCH-GROWTH] stage scheduler init | steps/ep={steps_per_epoch} "
          f"| est_ep={est_ep} | steps_in_stage={_stage_total_steps} "
          f"| warmup_frac={args.warmup_frac} -> warmup_steps={_stage_warmup_steps}")

    # SWA
    swa_model = None
    swa_sched = None
    swa_started = False
    if args.swa:
        swa_model = AveragedModel(model).to(device)
        swa_sched = SWALR(
            opt,
            swa_lr=args.swa_lr,
            anneal_epochs=args.swa_anneal_epochs,
            anneal_strategy=args.swa_anneal_strategy  # 'cos' | 'linear'
        )

    # EMA
    ema = EMA(model, decay=args.ema_decay)

    # Train loop
    best = float("inf")
    history = []
    global_step = 0
    periodic_ckpts = []
    patience = args.early_stop
    bad_epochs = 0
    best_ep = 0

    for ep in range(1, args.epochs+1):
        try:
            t0 = time.time()
            model.train()
            tot = 0.0; nb = 0
            train_wmae_sum = 0.0
            per_t_sum_train = torch.zeros(len(TARGETS), device=device)
            per_t_batches = 0

            train_log_path = os.path.join(args.out_dir, f"supervised_train_log_{run_tag}.jsonl")
            val_log_path = os.path.join(args.out_dir, f"val_log_{run_tag}.jsonl")

            # --- в начале каждой эпохи переключаем train-режим (round-robin)
            wanted_mode = modes[(ep - 1) % len(modes)]
            if wanted_mode != active_mode:
                active_mode = wanted_mode
                train_loader, val_loader = make_sup_loaders(
                    controller_ft.bs if stages_ft else args.train_batch_size,
                    active_mode, eval_mode
                )
                print(f"[SUP] Epoch {ep:02d}: switch train poly_edge_mode -> {active_mode}")


            # старт SWA до батчей эпохи
            if args.swa and (not swa_started) and (ep >= args.swa_start):
                swa_started = True
                print(
                    f"[SWA] start at epoch {ep} "
                    f"| lr={args.swa_lr} "
                    f"| anneal={args.swa_anneal_epochs} "
                    f"| freq={args.swa_freq} "
                    f"| strategy={args.swa_anneal_strategy}")

            phase = "SWA" if (args.swa and swa_started) else "SUP"
            desc = f"[{phase}] Epoch {ep:02d}"
            iterator = train_loader
            batches = len(train_loader)
            it = 0
            step0 = time.time()
            phys_announced = False
            swa_updates = 0

            if not args.no_tqdm:
                iterator = tqdm(train_loader, total=batches, desc=desc, leave=False)

            for x, ei, ea, bvec, gdesc, y, ymask in iterator:
                it += 1
                x, ei, ea, bvec, gdesc, y, ymask = x.to(device), ei.to(device), ea.to(device), bvec.to(device), gdesc.to(
                    device), y.to(device), ymask.to(device)

                with torch.autocast('cuda', enabled=use_amp, dtype=AMP_DTYPE):

                    out = model(x, ei, ea, bvec, gdesc)
                    if isinstance(out, tuple):  # hetero head → (mu, sigma)
                        pred, sig = out
                        if getattr(args, "hetero_sigma_head", False):
                            loss, _ = masked_wmae_hetero(pred, y, ymask, ranges, weights, sig)
                        else:
                            loss, _ = masked_wmae_with_uncertainty(pred, y, ymask, ranges, weights, model.rho)
                    else:
                        pred = out
                        loss, _ = masked_wmae_with_uncertainty(pred, y, ymask, ranges, weights, model.rho)

                    # --- Physics-inspired reg: Density ~ (1 - FFV) ---
                    lam = getattr(args, "physics_reg_lambda", 0.01)
                    if (lam > 0.0) and (not phys_announced):
                        # print(f"[PHYS] Density ≈ (1-FFV) reg enabled | λ={lam}")
                        phys_announced = True

                    if lam > 0.0 and pred.size(1) >= len(TARGETS):
                        idxD = TARGETS.index("Density");
                        idxF = TARGETS.index("FFV")
                        # берём только те объекты, где обе цели присутствуют (по маскам)
                        both_mask = (ymask[:, idxD] & ymask[:, idxF])
                        both_cnt = int(both_mask.sum().item())
                        if both_cnt >= 2 and (global_step % 1000 == 0):
                            print(f"[PHYS] pairs this step: {both_cnt}")

                        if both_mask.sum() >= 2:
                            d = pred[both_mask, idxD]
                            f = pred[both_mask, idxF]
                            # # центрирование и нормировка, чтобы штраф был масштабоинвариантным
                            # d_n = (d - d.mean()) / (d.std().clamp_min(1e-6))
                            # g = (1.0 - f)
                            # g_n = (g - g.mean()) / (g.std().clamp_min(1e-6))

                            # std без смещения + страховка eps
                            d_std = d.std(correction=0).clamp_min(1e-6)
                            g = 1.0 - f
                            g_std = g.std(correction=0).clamp_min(1e-6)
                            d_n = (d - d.mean()) / d_std
                            g_n = (g - g.mean()) / g_std

                            reg = F.mse_loss(d_n, g_n)
                            loss = loss + lam * reg

                # «чистый» wMAE для логов
                with torch.no_grad():
                    wmae_batch, per_t_batch = masked_wmae(pred, y, ymask, ranges, weights)
                train_wmae_sum += wmae_batch.item()

                per_t_sum_train += per_t_batch.to(device)
                per_t_batches += 1

                opt.zero_grad(set_to_none=True)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                for p in model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # === Авто-LR per-target: собрать нормы градиента у последнего слоя каждого таргета
                if args.auto_head_lr_mults and getattr(model, "per_target_last", False) and len(
                        head_group_indices) == len(TARGETS):
                    try:
                        for ti, gi in enumerate(head_group_indices):
                            lin = model.head_out_list[ti]
                            gnorm = _module_grad_norm(lin)
                            grad_hist[ti].append(gnorm)
                    except Exception:
                        pass

                if torch.isfinite(grad_norm):
                    scaler.step(opt)
                else:
                    print(
                        f"[WARN] Non-finite grad_norm={grad_norm} at epoch {ep} iter {it}. Skipping optimizer.step().")
                    opt.zero_grad(set_to_none=True)

                scaler.update()  # вызывать ВСЕГДА ровно один раз на итерацию

                # EMA
                ema.update(model)

                # Шаг LR: по батчам — либо базовый, либо SWALR
                if args.swa and swa_started:
                    swa_sched.step() # step СРАЗУ ПОСЛЕ optimizer.step()
                else:
                    sched.step()

                # === Авто-LR per-target: периодическое обновление множителей dyn_mults
                if args.auto_head_lr_mults and getattr(model, "per_target_last", False) and len(
                        head_group_indices) == len(TARGETS):
                    do_update = (global_step % max(1, int(args.auto_lr_every)) == 0)
                    # при желании — стартовать только после warmup текущей стадии
                    if args.auto_lr_start_after_warmup:
                        do_update = do_update and (_stage_step > _stage_warmup_steps)
                    if do_update:
                        means = np.array([(sum(h) / max(1, len(h))) if len(h) > 0 else 0.0 for h in grad_hist],
                                         dtype=np.float64)
                        # опорная величина — медиана по ненулевым
                        nz = means[means > 0]
                        if nz.size > 0:
                            ref = float(np.median(nz))
                            gamma = float(args.auto_lr_gamma)
                            lo, hi = float(args.auto_lr_min_mult), float(args.auto_lr_max_mult)
                            for ti, gi in enumerate(head_group_indices):
                                gv = float(means[ti])
                                if gv <= 0:
                                    mult = 1.0
                                else:
                                    mult = (ref / gv) ** gamma
                                mult = float(max(lo, min(hi, mult)))
                                dyn_mults[gi] = mult
                            # для читаемого лога — редко
                            if (it % max(50, args.auto_lr_every) == 0) or it == batches:
                                txt = ", ".join([f"{TARGETS[ti]}×{dyn_mults[gi]:.2f}" for ti, gi in
                                                 enumerate(head_group_indices)])
                                print(f"[AUTO-LR] step {global_step} | per-target mults: {txt}")

                # локальный шаг текущей стадии
                _stage_step += 1



                # --- Warmup/LR лог
                cur_lr = current_lr(opt)
                lr_name = "swa_lr" if (args.swa and swa_started) else "lr"

                if args.lr_debug_steps > 0 and global_step <= args.lr_debug_steps:
                    in_warmup = (_stage_step <= _stage_warmup_steps)
                    print(f"[LRDBG] gstep {global_step} | stage_step {_stage_step}/{_stage_total_steps} "
                          f"| lr={cur_lr:.6e} | warmup={in_warmup}")
                    if in_warmup and (last_lr is not None) and (cur_lr + 1e-12 < last_lr):
                        print(f"[LRDBG][WARN] LR decreased during warmup: prev={last_lr:.6e} -> cur={cur_lr:.6e}")
                last_lr = cur_lr

                # SWA усреднение по частоте
                if args.swa and swa_started and ((global_step % max(1, args.swa_freq) == 0) or (it == batches)):
                    swa_model.update_parameters(model)
                    swa_updates += 1

                    # print(f"[SWA] epoch {ep} updates={swa_updates} last_lr={cur_lr:.2e}")

                tot += loss.item()
                nb += 1
                global_step += 1

                if (it % args.log_every == 0) or (it == 1) or (it == batches):
                    elapsed = time.time() - step0

                    remaining = max(0, batches - it)
                    ips = (it * max(1, args.train_batch_size)) / max(1e-9, elapsed)
                    eta_s = (remaining * max(1, args.train_batch_size)) / max(ips, 1e-9)

                    rec = {
                        "phase": "train",
                        "epoch": ep,
                        "iter": it,
                        "iters_per_epoch": batches,
                        "loss_unc_avg": tot / max(1, nb),
                        "wmae_avg": train_wmae_sum / max(1, nb),
                        "lr": cur_lr,
                        "ips": ips,
                        "cuda_gb": cuda_mem_gb(),
                        "grad_norm": float(grad_norm),
                        "eta_min": eta_s / 60.0,

                    }
                    jsonl_append(train_log_path, rec)
                    if not args.no_tqdm:
                        bs_eff = y.size(0)
                        mask_fill = ymask.float().mean().item()  # доля заполненности таргетов в батче

                        post = {
                            "loss_unc": f"{rec['loss_unc_avg']:.4f}",
                            "wmae": f"{rec['wmae_avg']:.4f}",
                            "ips": f"{rec['ips']:.1f}/s",
                            "mem": f"{rec['cuda_gb']:.2f}GB",
                            "eta": f"{rec['eta_min']:.1f}m",
                            "gn": f"{grad_norm:.2f}",
                            "bs": bs_eff,
                            "fill": f"{mask_fill * 100:.0f}%"
                        }
                        # один-единственный ключ для learning-rate
                        post[lr_name] = f"{cur_lr:.2e}"

                        iterator.set_postfix(**post)


                # Quick mid-epoch validation (optional)
                if args.eval_every > 0 and (it % args.eval_every == 0):
                    model.eval()
                    ckpt_state_mid = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    ema.copy_to(model)
                    with torch.no_grad():
                        vtot_q = 0.0;
                        vb_q = 0
                        for j, (xq, eiq, eaq, bq, gq, yq, ymaskq) in enumerate(val_loader):
                            if j >= 3:  # быстрый срез на несколько батчей
                                break
                            xq, eiq, eaq, bq, gq, yq, ymaskq = xq.to(device), eiq.to(device), eaq.to(device), bq.to(
                                device), gq.to(device), yq.to(device), ymaskq.to(device)
                            with torch.autocast('cuda', enabled=use_amp, dtype=AMP_DTYPE):
                                predq = model(xq, eiq, eaq, bq, gq)
                                wmae_q, _ = masked_wmae(predq, yq, ymaskq, ranges, weights)
                            vtot_q += wmae_q.item();
                            vb_q += 1
                        quick_val = vtot_q / max(1, vb_q)
                    model.load_state_dict(ckpt_state_mid)
                    # Вынесем в stdout и в jsonl
                    print(f"[QUICK VAL] ep {ep:02d} it {it}/{batches} | wMAE~{quick_val:.4f} | lr={current_lr(opt):.2e} | select={args.select_model}")

                    jsonl_append(val_log_path, {
                        "phase": "val_quick", "epoch": ep, "iter": it, "wMAE": quick_val
                    })
                    model.train()

            # лог по эпохе
            train_loss = tot / max(1, nb)
            train_wmae = train_wmae_sum / max(1, nb)
            train_per_t = (per_t_sum_train / max(1, per_t_batches)).detach().cpu().tolist()

            # # >>> после окончания обучающих батчей эпохи:
            # if args.swa and swa_started and swa_model is not None:
            #     # гарантируем, что хотя бы раз за эпоху обновили среднее
            #     swa_model.update_parameters(model)
            # if args.swa and swa_started and (swa_sched is not None):
            #     # один шаг SWALR на ЭПОХУ
            #     swa_sched.step()


            # # val with EMA params
            # model.eval()
            # # store and load EMA weights
            # ckpt_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # ema.copy_to(model)
            # with torch.no_grad():
            #     vtot = 0.0
            #     vb = 0
            #     per_t_sum = torch.zeros(len(TARGETS))
            #     iterator_val = val_loader
            #     if not args.no_tqdm:
            #         iterator_val = tqdm(val_loader, total=len(val_loader), desc=f"[VAL] Epoch {ep:02d}", leave=False)
            #
            #     for x, ei, ea, bvec, gdesc, y, ymask in iterator_val:
            #         x, ei, ea, bvec, gdesc, y, ymask = x.to(device), ei.to(device), ea.to(device), bvec.to(
            #             device), gdesc.to(device), y.to(device), ymask.to(device)
            #         with torch.autocast('cuda', enabled=use_amp, dtype=AMP_DTYPE):
            #             out = model(x, ei, ea, bvec, gdesc)
            #             pred = out[0] if isinstance(out, tuple) else out
            #             vloss, per_t = masked_wmae(pred, y, ymask, ranges, weights)
            #
            #         vtot += vloss.item()
            #         vb += 1
            #         per_t_sum += per_t.cpu()
            #         if not args.no_tqdm:
            #             iterator_val.set_postfix(wMAE=f"{(vtot / max(1, vb)):.4f}")
            #     val_loss = vtot / max(1, vb)
            #     per_t_avg = (per_t_sum / max(1, vb)).tolist()


            # --- собираем предсказания/таргеты для Kaggle-like метрики ---
            val_pred_agg = None  # будем суммировать по eval-модам, а потом усреднять
            val_true_all, val_mask_all = None, None
            val_true_chunks, val_mask_chunks = [], []
            eval_modes_to_use = []
            if (args.poly_edge_eval_mode or "first").strip().lower() == "avg":
                eval_modes_to_use = modes
            else:
                eval_modes_to_use = [eval_mode]

            # → валидация на EMA-параметрах
            model.eval()
            ckpt_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            vtot = 0.0;
            vb = 0;
            per_t_sum = torch.zeros(len(TARGETS))

            with torch.no_grad():
                for m_eval in eval_modes_to_use:
                    # если текущий loader не из нужного режима — создаём временный
                    loader = val_loader if m_eval == eval_mode else DataLoader(
                        val_ds_by_mode[m_eval],
                        batch_size=controller_ft.bs if stages_ft else args.train_batch_size,
                        shuffle=False, collate_fn=collate, generator=torch.Generator().manual_seed(args.seed),
                        worker_init_fn=seed_worker, **make_loader_kwargs(args.train_num_workers)
                    )
                    it = loader if args.no_tqdm else tqdm(loader, total=len(loader),
                                                          desc=f"[VAL:{m_eval}] Epoch {ep:02d}", leave=False)
                    preds_here = []
                    collect_truth = (m_eval == eval_modes_to_use[0])  # GT/маски собираем только в первом режиме

                    for x, ei, ea, bvec, gdesc, y, ymask in it:
                        x, ei, ea, bvec, gdesc = x.to(device), ei.to(device), ea.to(device), bvec.to(device), gdesc.to(device)
                        y, ymask = y.to(device), ymask.to(device)
                        with torch.autocast('cuda', enabled=use_amp, dtype=AMP_DTYPE):
                            out = model(x, ei, ea, bvec, gdesc)
                            pred = out[0] if isinstance(out, tuple) else out
                            vloss, per_t = masked_wmae(pred, y, ymask, ranges, weights)
                        vtot += vloss.item(); vb += 1; per_t_sum += per_t.cpu()
                        preds_here.append(pred.detach().to(torch.float32).cpu().numpy())
                        if val_true_all is None:
                            val_true_all = [y.detach().cpu().numpy()]
                            val_mask_all = [ymask.detach().cpu().numpy()]
                        else:
                            # только первые проходы сохраняют y/ymask
                            pass
                        if collect_truth:
                            # накапливаем GT/маски для всех батчей ПЕРВОГО режима
                            val_true_chunks.append(y.detach().cpu().numpy())
                            val_mask_chunks.append(ymask.detach().cpu().numpy())
                    preds_here = np.vstack(preds_here)
                    val_pred_agg = preds_here if val_pred_agg is None else (val_pred_agg + preds_here)

                # старая внутренняя wMAE (по train ranges/weights) — оставим для мониторинга
                val_loss = vtot / max(1, vb)
                per_t_avg = (per_t_sum / max(1, vb)).tolist()

                # Kaggle-like wMAE по усреднённым предсказаниям
                val_pred_all = (val_pred_agg / float(len(eval_modes_to_use))).astype(np.float32, copy=False)
                # GT/маски — конкатенация всех батчей первого режима
                val_true_all = np.vstack(val_true_chunks).astype(np.float32, copy=False)
                val_mask_all = np.vstack(val_mask_chunks).astype(bool, copy=False)
                # строгая проверка совпадения форм
                if not (val_pred_all.shape[0] == val_true_all.shape[0] == val_mask_all.shape[0]):
                    raise RuntimeError(f"[VAL][BUG] shape mismatch after fix: "
                                       f"pred={val_pred_all.shape} "
                                       f"true={val_true_all.shape} "
                                       f"mask={val_mask_all.shape}")
                val_wmae_kaggle, per_t_kaggle, r_used, w_used = kaggle_wmae_from_preds(val_pred_all, val_true_all, val_mask_all)

                # ---- Сохраняем OOF и (опц.) обучаем изотоническую калибровку ----
                oof_pack = {
                    "targets": TARGETS,
                    "pred": val_pred_all.astype(np.float32).tolist(),
                    "true": val_true_all.astype(np.float32).tolist(),
                    "mask": val_mask_all.astype(bool).tolist(),
                }
                oof_path = os.path.join(args.out_dir, f"oof_{run_tag}.json")
                with open(oof_path, "w") as f:
                    json.dump(oof_pack, f)
                # print(f"[VAL] saved OOF → {oof_path}")

                if getattr(args, "fit_isotonic", False):
                    iso = {}
                    for i, t in enumerate(TARGETS):
                        m = val_mask_all[:, i].astype(bool)
                        if m.sum() < 10:
                            continue
                        px = val_pred_all[m, i].astype(np.float64)
                        py = val_true_all[m, i].astype(np.float64)
                        kx, ky = _iso_fit_1d(px, py)
                        iso[t] = {"x": kx.tolist(), "y": ky.tolist()}
                    iso_path = os.path.join(args.out_dir, f"isotonic_{run_tag}.json")
                    with open(iso_path, "w") as f:
                        json.dump({"targets": TARGETS, "maps": iso}, f)
                    # print(f"[VAL] fitted isotonic calibrators → {iso_path}")

                # ---- Сохраняем вал-статистики предсказаний для пост-обработки ----
                if getattr(args, "save_val_range_stats", False):
                    stats = {}
                    for i, t in enumerate(TARGETS):
                        m = val_mask_all[:, i].astype(bool)
                        if m.sum() == 0:
                            stats[t] = {"q_low": None, "q_high": None, "min": None, "max": None}
                            continue
                        v = val_pred_all[m, i]
                        ql = float(np.percentile(v, args.winsor_q_low)) if 0.0 < args.winsor_q_low < 50.0 else None
                        qh = float(np.percentile(v, args.winsor_q_high)) if 50.0 < args.winsor_q_high < 100.0 else None
                        stats[t] = {"q_low": ql, "q_high": qh, "min": float(v.min()), "max": float(v.max())}
                    stats_path = os.path.join(args.out_dir, f"calib_stats_{run_tag}.json")
                    with open(stats_path, "w") as f:
                        json.dump({"targets": TARGETS, "stats": stats}, f, indent=2)
                    # print(f"[VAL] saved calib stats → {stats_path}")

                sigma = F.softplus(model.rho).detach().cpu().tolist()
                val_rec = {
                    "phase": "val",
                    "epoch": ep,
                    "val_wMAE": val_loss,
                    "per_target": per_t_avg,
                    "sigma": sigma
                }
                jsonl_append(val_log_path, val_rec)

            # restore non-EMA weights
            model.load_state_dict(ckpt_state)

            # ===== Валидация: SWA (если есть) =====
            val_loss_swa = None
            if args.swa and swa_model is not None and swa_started:
                # оценим SWA-модель отдельно
                model.eval()
                with torch.no_grad():
                    vtot_swa = 0.0
                    vb_swa = 0
                    per_t_sum_swa = torch.zeros(len(TARGETS))
                    iterator_val = val_loader
                    if not args.no_tqdm:
                        iterator_val = tqdm(val_loader, total=len(val_loader), desc=f"[VAL:SWA] Epoch {ep:02d}", leave=False)
                    # подменяем веса на SWA-параметры (без трогания ema.shadow)
                    orig_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    model.load_state_dict(swa_model.module.state_dict(), strict=True)
                    for x, ei, ea, bvec, gdesc, y, ymask in iterator_val:
                        x, ei, ea, bvec, gdesc, y, ymask = x.to(device), ei.to(device), ea.to(device), bvec.to(device), gdesc.to(device), y.to(device), ymask.to(device)
                        with torch.autocast('cuda', enabled=use_amp, dtype=AMP_DTYPE):
                            out = model(x, ei, ea, bvec, gdesc)
                            pred = out[0] if isinstance(out, tuple) else out
                            vloss_swa, per_t_swa = masked_wmae(pred, y, ymask, ranges, weights)
                        vtot_swa += vloss_swa.item()
                        vb_swa += 1
                        per_t_sum_swa += per_t_swa.cpu()
                    val_loss_swa = vtot_swa / max(1, vb_swa)
                    # возвращаем исходное состояние
                    model.load_state_dict(orig_state, strict=True)


            # История/печать
            history.append({
                "epoch": ep,
                "train_loss_unc": train_loss,  # оптимизационная
                "train_wMAE": train_wmae,      # интерпретируемая
                "val_wMAE": val_loss,
                "val_wMAE_internal": val_loss,
                "val_wMAE_kaggle": val_wmae_kaggle,
                "per_target": per_t_avg,
                "sec": time.time()-t0
            })

            # печать с понятными именами
            print(f"Epoch {ep:02d} | train_loss_unc {train_loss:.4f} | train_wMAE {train_wmae:.4f} | "
                  f"val_wMAE(int) {val_loss:.4f} | val_wMAE(K) {val_wmae_kaggle:.4f} | "
                  f"train per-target {np.round(np.array(train_per_t), 4)} | "
                  f"val per-target {np.round(np.array(per_t_avg), 4)} | "
                  f"{history[-1]['sec']:.1f}s")


            # --- выбор «лучшей» модели по настроенному критерию
            # candidate_metric = val_loss  # это EMA-валидация
            candidate_metric = val_wmae_kaggle  # выбираем чекпоинт по Kaggle-like
            candidate_weights = {k: v.detach().cpu() for k, v in ema.shadow.items()}
            candidate_kind = "ema"

            if args.swa and swa_started and (val_loss_swa is not None):
                if args.select_model == "swa":
                    candidate_metric = val_loss_swa
                    candidate_weights = {k: v.detach().cpu() for k, v in swa_model.module.state_dict().items()}
                    candidate_kind = "swa"
                else:
                    # select_model == "ema": оставляем как есть
                    pass

            prev_best = best
            if candidate_metric < best:
                best_path = os.path.join(args.out_dir, f"best_polymer_gnn_{run_tag}.pt")
                print(f"[BEST] epoch {ep}: {candidate_kind} improved {prev_best:.4f} → {candidate_metric:.4f} | saved {best_path}")

                best = candidate_metric

                best_ep = ep
                bad_epochs = 0
                to_save = {
                    "model": candidate_weights,
                    "node_in": node_in,
                    "edge_in": edge_in,
                    "gdesc_dim": gdesc_dim,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "targets": list(TARGETS),
                    "ranges": torch.as_tensor(ranges, dtype=torch.float32),
                    "weights": torch.as_tensor(weights, dtype=torch.float32),
                    "kind": candidate_kind,
                }
                torch.save(to_save, best_path)

                with open(os.path.join(args.out_dir, f"history_{run_tag}.json"), "w") as f:
                    json.dump(history, f, indent=2)
            else:
                if patience > 0:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        print(
                            f"[EARLY STOP] no improvement for {patience} epochs (best {best:.4f} at epoch {best_ep}).")
                        break

            # Periodic EMA checkpoint
            if args.ckpt_every > 0 and (ep % args.ckpt_every == 0):
                path = os.path.join(args.out_dir, f"ckpt_{run_tag}_epoch_{ep:03d}.pt")
                torch.save({
                    "model": {k: v.detach().cpu() for k, v in ema.shadow.items()},
                    "node_in": node_in,
                    "edge_in": edge_in,
                    "gdesc_dim": gdesc_dim,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "targets": TARGETS,
                    "ranges": ranges,
                    "weights": weights,
                    "epoch": ep
                }, path)
                periodic_ckpts.append(path)
                # Keep only last K
                if len(periodic_ckpts) > max(0, args.keep_last_k):
                    old = periodic_ckpts.pop(0)
                    try:
                        os.remove(old)
                    except Exception:
                        pass

            # --- batch-growth: критерий = val wMAE (или SWA по твоему select_model), минимизируем
            if stages_ft:
                if controller_ft.on_epoch_end(candidate_metric):
                    print(
                        f"[SUP][BATCH-GROWTH] stage end: ep={ep}, "
                        f"metric={candidate_metric:.4f}, "
                        f"next_bs={'done' if controller_ft.stage_idx + 1 >= len(controller_ft.stages) else controller_ft.stages[controller_ft.stage_idx + 1].bs}")

                    controller_ft.advance()
                    if controller_ft.done:
                        break  # все стадии пройдены или достигнут глобальный лимит
                    train_loader, val_loader = make_sup_loaders(controller_ft.bs, active_mode, eval_mode)  # НОВОЕ
                    # только пересоздаём лоадеры

                    print(f"[SUP][BATCH-GROWTH] now bs={controller_ft.bs}")

                    # === Пересборка stage-scheduler под новый bs ===
                    steps_per_epoch = len(train_loader)
                    est_ep = _estimate_stage_epochs_for_current(controller_ft, args.epochs)
                    _stage_total_steps = max(1, steps_per_epoch * est_ep)
                    sched, _stage_warmup_steps, _stage_total_steps = _build_stage_scheduler(
                        opt,
                        steps_in_stage=_stage_total_steps,
                        warmup_frac=args.warmup_frac,
                        warmup_steps_override=getattr(args, "warmup_steps", 0)
                    )
                    _stage_step = 0
                    print(f"[SUP][BATCH-GROWTH] stage scheduler reset | bs={controller_ft.bs} "
                          f"| steps/ep={steps_per_epoch} | est_ep={est_ep} "
                          f"| steps_in_stage={_stage_total_steps} | warmup_steps={_stage_warmup_steps}")




        except KeyboardInterrupt:
            print("\n[INTERRUPT] Saving last EMA checkpoint and exiting...")
            torch.save({"model": {k: v.detach().cpu() for k, v in ema.shadow.items()},
                        "node_in": node_in, "edge_in": edge_in, "gdesc_dim": gdesc_dim,
                        "hidden": args.hidden, "layers": args.layers, "targets": TARGETS,
                        "ranges": ranges, "weights": weights, "epoch": ep},
                       os.path.join(args.out_dir, f"ckpt_interrupted_{run_tag}.pt"))
            raise

    # =============================
    # DOT-STAGE (полирующий этап)
    # =============================
    if args.dot_stage:
        print(f"[DOT] Starting dot-stage for {args.dot_epochs} epochs with LR={args.dot_lr:g} "
              f"({'head-only' if args.dot_head_only else 'full model'})")

        # Загружаем лучший чекпоинт перед доточкой
        best_ckpt = safe_load_ckpt(os.path.join(args.out_dir, f"best_polymer_gnn_{run_tag}.pt"), device)

        model = PolymerGNN(best_ckpt["node_in"], best_ckpt["edge_in"], best_ckpt["gdesc_dim"],
                           hidden=best_ckpt["hidden"], layers=best_ckpt["layers"],
                           targets=len(best_ckpt["targets"])).to(device)
        sd = model.state_dict()
        for k in sd.keys():
            if k in best_ckpt["model"]:
                sd[k] = best_ckpt["model"][k].to(device)
        model.load_state_dict(sd, strict=True)

        # Переинициализируем EMA/SWA под новые стартовые веса
        ema = EMA(model, decay=args.ema_decay)
        if args.swa:
            swa_model = AveragedModel(model).to(device)

        # разморозка/заморозка
        if args.dot_head_only:
            for p in model.parameters(): p.requires_grad = False
            if hasattr(model, "head_core"):
                for p in model.head_core.parameters(): p.requires_grad = True
            if hasattr(model, "head_out"):
                for p in model.head_out.parameters(): p.requires_grad = True
            if hasattr(model, "head_out_list"):
                for lin in model.head_out_list:
                    for p in lin.parameters(): p.requires_grad = True
            if hasattr(model, "head_sigma_core"):
                for p in model.head_sigma_core.parameters(): p.requires_grad = True
            if hasattr(model, "head_sigma_out"):
                for p in model.head_sigma_out.parameters(): p.requires_grad = True
            model.rho.requires_grad_(True)
        else:
            for p in model.parameters():
                p.requires_grad = True

        dot_params = [p for p in model.parameters() if p.requires_grad]
        dot_opt = torch.optim.AdamW(dot_params, lr=args.dot_lr, weight_decay=0.0)
        dot_total_steps = args.dot_epochs * max(1, len(train_loader))
        dot_warmup = max(1, int(0.1 * dot_total_steps))
        def dot_lr_lambda(step):
            if step < dot_warmup:
                return float(step)/max(1, dot_warmup)
            progress = (step - dot_warmup) / max(1, dot_total_steps - dot_warmup)
            return max(0.0, 0.5*(1 + math.cos(math.pi * progress)))
        dot_sched = torch.optim.lr_scheduler.LambdaLR(dot_opt, lr_lambda=dot_lr_lambda)

        for dep in range(1, args.dot_epochs+1):
            model.train()
            tot = 0.0; nb = 0
            iterator = train_loader
            if not args.no_tqdm:
                iterator = tqdm(train_loader, total=len(train_loader), desc=f"[DOT] Epoch {dep:02d}", leave=False)

            it = 0
            for x, ei, ea, bvec, gdesc, y, ymask in iterator:
                it += 1
                x, ei, ea, bvec, gdesc, y, ymask = x.to(device), ei.to(device), ea.to(device), bvec.to(device), gdesc.to(device), y.to(device), ymask.to(device)
                with torch.autocast('cuda', enabled=use_amp, dtype=AMP_DTYPE):
                    out = model(x, ei, ea, bvec, gdesc)
                    if isinstance(out, tuple):  # hetero head → (mu, sigma)
                        pred, sig = out
                        if getattr(args, "hetero_sigma_head", False):
                            # доточкa тем же «uncertainty»-лоссом, чтобы rho было осмысленным
                            loss_dot, _ = masked_wmae_hetero(pred, y, ymask, ranges, weights, sig)
                        else:
                            loss_dot, _ = masked_wmae_with_uncertainty(pred, y, ymask, ranges, weights, model.rho)
                    else:
                        pred = out
                        loss_dot, _ = masked_wmae_with_uncertainty(pred, y, ymask, ranges, weights, model.rho)

                dot_opt.zero_grad(set_to_none=True)
                scaler.scale(loss_dot).backward()
                scaler.unscale_(dot_opt)
                torch.nn.utils.clip_grad_norm_(dot_params, 1.0)
                scaler.step(dot_opt)
                scaler.update()
                dot_sched.step()

                # EMA/SWA поверх доточкu
                ema.update(model)
                if args.swa and swa_model is not None:
                    if (it % max(1, args.swa_freq) == 0) or (it == len(train_loader)):
                        swa_model.update_parameters(model)

                tot += loss_dot.item(); nb += 1

            # валидация на EMA
            model.eval()
            ckpt_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            with torch.no_grad():
                vtot = 0.0; vb = 0
                for x, ei, ea, bvec, gdesc, y, ymask in val_loader:
                    x, ei, ea, bvec, gdesc, y, ymask = x.to(device), ei.to(device), ea.to(device), bvec.to(device), gdesc.to(device), y.to(device), ymask.to(device)
                    with torch.autocast('cuda', enabled=use_amp, dtype=AMP_DTYPE):
                        out = model(x, ei, ea, bvec, gdesc)
                        pred = out[0] if isinstance(out, tuple) else out
                        vloss, _ = masked_wmae(pred, y, ymask, ranges, weights)
                    vtot += vloss.item(); vb += 1
                val_dot = vtot / max(1, vb)
            model.load_state_dict(ckpt_state)

            # при select_model==swa посчитаем и SWA-валидацию
            val_dot_swa = None
            if args.swa and swa_model is not None and (args.select_model=="swa"):
                with torch.no_grad():
                    vtot_swa = 0.0; vb_swa = 0
                    orig_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    model.load_state_dict(swa_model.module.state_dict(), strict=True)
                    for x, ei, ea, bvec, gdesc, y, ymask in val_loader:
                        x, ei, ea, bvec, gdesc, y, ymask = x.to(device), ei.to(device), ea.to(device), bvec.to(device), gdesc.to(device), y.to(device), ymask.to(device)
                        with torch.autocast('cuda', enabled=use_amp, dtype=AMP_DTYPE):
                            out = model(x, ei, ea, bvec, gdesc)
                            pred = out[0] if isinstance(out, tuple) else out
                            vloss_swa, _ = masked_wmae(pred, y, ymask, ranges, weights)
                        vtot_swa += vloss_swa.item(); vb_swa += 1
                    val_dot_swa = vtot_swa / max(1, vb_swa)
                    model.load_state_dict(orig_state, strict=True)

            # сохраняем лучший прогресс доточкu
            candidate_metric = val_dot
            candidate_weights = {k: v.detach().cpu() for k, v in ema.shadow.items()}
            candidate_kind = "ema"
            if args.select_model == "swa" and (val_dot_swa is not None):
                candidate_metric = val_dot_swa
                candidate_weights = {k: v.detach().cpu() for k, v in swa_model.module.state_dict().items()}
                candidate_kind = "swa"

            prev_best = best
            if candidate_metric < best:
                best_path = os.path.join(args.out_dir, f"best_polymer_gnn_{run_tag}.pt")
                print(f"[BEST][DOT] epoch {dep}: {candidate_kind} improved {prev_best:.4f} → {candidate_metric:.4f} | saved {best_path}")

                best = candidate_metric
                best_ep = f"dot:{dep}"
                to_save = {
                    "model": candidate_weights,
                    "node_in": node_in,
                    "edge_in": edge_in,
                    "gdesc_dim": gdesc_dim,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "targets": list(TARGETS),
                    "ranges": torch.as_tensor(ranges, dtype=torch.float32),
                    "weights": torch.as_tensor(weights, dtype=torch.float32),
                    "kind": candidate_kind,
                }
                torch.save(to_save, best_path)

                print(f"[DOT] Improved best to {best:.4f} ({candidate_kind})")

    # >>> В КОНЦЕ train_supervised, перед return
    if args.swa and args.swa_update_bn and swa_model is not None:
        try:
            print("[SWA] Updating BatchNorm statistics on training loader...")
            update_bn(train_loader, swa_model, device=device)
            print("[SWA] BN stats updated")

        except Exception as e:
            print("[SWA] update_bn skipped:", e)

    return best


def build_unlabeled_loader(unl_df: pd.DataFrame, batch_size: int, num_workers: int, poly_edge_mode: str):
    ds = UnlabeledDS(unl_df["SMILES"].tolist(), poly_edge_mode=poly_edge_mode)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_unl,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def pretrain(args):
    set_seed(args.seed)
    train_df, test_df, _ = load_main_data(args.data_dir)
    supp_zip = load_supplemental(args.data_dir)

    # соберём sup_df, чтобы знать «домен»
    external = ingest_external_sources(args)
    sup_df   = build_supervised_dataframe(train_df, supp_zip, external)

    unl_df = build_unlabeled_dataframe_smart(train_df, test_df, supp_zip, args, sup_df=sup_df)

    if len(unl_df) == 0:
        print("[SSL] Unlabeled set is empty after filtering. Skipping SSL.")
        return os.path.join(args.out_dir, "ssl_encoder.pt")

    # (опц.) ограничение числа сэмплов для скорости — применяем ДО кеша,
    # чтобы не материализовать лишнее
    if args.pretrain_samples > 0 and len(unl_df) > args.pretrain_samples:
        unl_df = unl_df.sample(n=args.pretrain_samples, random_state=args.seed).reset_index(drop=True)

    # === ВАЖНО: сперва делаем scaffold-holdout для SSL-валидации ===
    # (иначе утечка, если сперва прособираем кэш под train)
    tr_unl, va_unl = scaffold_split(unl_df[["SMILES"]].copy(), frac_val=args.ssl_val_frac, seed=args.split_seed)
    if args.ssl_val_max > 0 and len(va_unl) > args.ssl_val_max:
        va_unl = va_unl.sample(n=args.ssl_val_max, random_state=args.seed).reset_index(drop=True)

    # === режимы poly-edge
    modes = get_poly_modes(args)
    eval_mode = canonical_poly_mode(args, modes)

    # === Кэши SSL (train) для всех режимов
    ssl_train_by_mode = {}
    if args.aug_local_dynamic and args.aug_local_subst_p > 0:
        # динамический путь: без кэша, т.к. SMILES меняем на лету
        for m in modes:
            ds_m = UnlabeledDS(
                tr_unl["SMILES"].tolist(),
                poly_edge_mode=m,
                aug_local_p=args.aug_local_subst_p,
                aug_local_dynamic=True,
                seed=args.seed
            )
            ssl_train_by_mode[m] = ds_m
        print("[SSL] dynamic local SMILES augmentations enabled (no train cache).")
    else:
        # быстрый путь с кэшем (train)
        for m in modes:
            cache_tr_m = _ssl_cache_path(tr_unl, args.out_dir, m, tag=f"train_{m}")
            print(f"[CACHE][SSL] train[{m}]: {'Load' if os.path.isfile(cache_tr_m) else 'Build'} {cache_tr_m} | rows={len(tr_unl)}")
            ds_m, _ = materialize_unlabeled_dataset(tr_unl["SMILES"].tolist(),
                                                    poly_edge_mode=m,
                                                    desc=f"SSL-TRAIN[{m}]",
                                                    cache_path=cache_tr_m)
            ssl_train_by_mode[m] = ds_m

    # === Валидационный кэш делаем 1 (канонический) — стабильные метрики
    cache_va = _ssl_cache_path(va_unl, args.out_dir, eval_mode, tag=f"val_{eval_mode}")
    print(f"[CACHE][SSL]  val[{eval_mode}]: {'Load' if os.path.isfile(cache_va) else 'Build'} {cache_va} | rows={len(va_unl)}")
    ssl_val_ds, _ = materialize_unlabeled_dataset(va_unl["SMILES"].tolist(),
                                                  poly_edge_mode=eval_mode,
                                                  desc=f"SSL-VAL[{eval_mode}]",
                                                  cache_path=cache_va)



    def make_ssl_loader(ds, bs: int):
        kw = dict(
            batch_size=bs,
            shuffle=True,
            collate_fn=collate_unl,
            num_workers=args.pre_num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(args.pre_num_workers > 0),
        )
        if args.pre_num_workers > 0:
            kw["prefetch_factor"] = 2
        return DataLoader(ds, **kw)

    # batch-growth остаётся как есть, но лоадер теперь зависит от режима
    stages_ssl = parse_batch_growth_3(args.ssl_batch_growth)
    if stages_ssl:
        controller_ssl = BatchGrowthController(stages_ssl, global_max_epochs=args.pretrain_epochs, mode="min",
                                               min_delta=1e-4)
        cur_bs = controller_ssl.bs
    else:
        controller_ssl = None
        cur_bs = args.pre_batch_size

    # активный режим на старте
    active_mode = modes[0]
    loader = make_ssl_loader(ssl_train_by_mode[active_mode], cur_bs)
    print_loader_info(f"SSL[{active_mode}]", cur_bs, args.pre_num_workers, torch.cuda.is_available(),
                      (args.pre_num_workers > 0), prefetch_factor=(2 if args.pre_num_workers > 0 else None))

    # small dummy to infer dims — берём из уже собранных ds
    sample = None
    for m in modes:
        ds_m = ssl_train_by_mode[m]
        if len(ds_m) > 0:
            x, ei, ea, gdesc = ds_m[0]
            sample = (x, ei, ea, gdesc)
            break
    if sample is None and len(ssl_val_ds) > 0:
        x, ei, ea, gdesc = ssl_val_ds[0]
        sample = (x, ei, ea, gdesc)

    if sample is None:
        dummy = GraphData(
            torch.zeros((1, NODE_FEAT_DIM)),
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, EDGE_FEAT_DIM)),
            torch.zeros((TOTAL_GDESC_DIM,)),
            torch.zeros(5),
            torch.zeros(5).bool(),
            0
        )
    else:
        x, ei, ea, gdesc = sample
        dummy = GraphData(x, ei, ea, gdesc, torch.zeros(5), torch.zeros(5).bool(), 0)

    enc, _, _, _ = make_model(dummy, args)
    # proj = ContrastiveHead(enc.head[0].normalized_shape[0] if isinstance(enc.head[0], nn.LayerNorm) else args.hidden)
    proj = ContrastiveHead(args.hidden)  # ← было с LayerNorm проверкой

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    enc = enc.to(device)
    proj = proj.to(device)

    if (not args.no_compile) and torch.cuda.is_available() and not args.cpu:
        try:
            enc = torch.compile(enc, mode="reduce-overhead")
            proj = torch.compile(proj, mode="reduce-overhead")
            print("[SSL] torch.compile enabled for encoder+proj")
        except Exception as e:
            print("[SSL] compile skipped:", e)

    # AMP: включено на CUDA; GradScaler нужен только для fp16 (для bf16 не нужен)
    use_amp = (torch.cuda.is_available() and not args.cpu)
    scaler_ssl = torch.amp.GradScaler('cuda', enabled=(use_amp and (AMP_DTYPE == torch.float16)))

    params = list(enc.parameters()) + list(proj.parameters())
    # AdamW с fused-ядрами (если доступно) — меньше CPU-overhead, больше работы на GPU
    try:
        opt = torch.optim.AdamW(params, lr=args.pre_lr, weight_decay=1e-4,
                                fused=(torch.cuda.is_available() and not args.cpu))
    except TypeError:
        # на старых версиях PyTorch параметра fused может не быть
        opt = torch.optim.AdamW(params, lr=args.pre_lr, weight_decay=1e-4)

    # === Stage-aware cosine LR с тёплым стартом на каждую стадию SSL ===

    def _make_ssl_sched_for_stage(batches_per_epoch: int, max_epochs_stage: int):
        """Создаёт новый LambdaLR под текущую стадию (bs), с собственным warmup.
        Возвращает (scheduler, warmup_steps, total_steps_stage)."""
        total_steps_stage = max(1, batches_per_epoch) * max(1, max_epochs_stage)
        # warmup: либо явные steps, либо доля от этапа
        if getattr(args, "warmup_steps", 0) and args.warmup_steps > 0:
            warmup_steps = int(args.warmup_steps)
        else:
            warmup_steps = max(1, int(float(args.warmup_frac) * total_steps_stage))

        def _lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps_stage - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
        return sch, warmup_steps, total_steps_stage

    # Инициализация шедулера для текущего bs/стадии
    if stages_ssl:
        cur_stage = controller_ssl.stages[controller_ssl.stage_idx]
        batches_now = max(1, len(loader))
        # ВАЖНО: обнуляем LR к базовому при старте этапа
        for g in opt.param_groups:
            g["lr"] = args.pre_lr
        sched, ssl_warmup_steps, ssl_total_steps_stage = _make_ssl_sched_for_stage(
            batches_now, cur_stage.max_epochs
        )
    else:
        # Режим без batch-growth: warmup на всю SSL-фазу
        batches_now = max(1, len(loader))
        for g in opt.param_groups:
            g["lr"] = args.pre_lr
        sched, ssl_warmup_steps, ssl_total_steps_stage = _make_ssl_sched_for_stage(
            batches_now, args.pretrain_epochs
        )

    enc.train(); proj.train()
    ssl_log_path = os.path.join(args.out_dir, "ssl_train_log.jsonl")

    ssl_periodic = []

    # === SSL early stop по train loss: только если НЕТ batch-growth ===
    use_ssl_early_stop = (controller_ssl is None) and (getattr(args, "ssl_early_stop", 0) > 0)
    best_ssl_loss = float("inf")
    ssl_no_improve = 0

    for ep in range(1, args.pretrain_epochs + 1):
        try:
            t0 = time.time()
            tot = 0.0
            nb = 0
            batches = len(loader)
            step0 = time.time()
            it = 0

            # --- переключение режима по эпохам: round-robin
            wanted_mode = modes[(ep - 1) % len(modes)]
            if wanted_mode != active_mode:
                active_mode = wanted_mode
                loader = make_ssl_loader(ssl_train_by_mode[active_mode], cur_bs)
                print(f"[SSL] Epoch {ep:02d}: switch poly_edge_mode -> {active_mode}")

            iterator = loader
            if not args.no_tqdm:
                iterator = tqdm(loader, total=batches, desc=f"[SSL] Epoch {ep:02d}", leave=False)

            for batch in iterator:
                it += 1
                x, ei, ea, bvec, gdesc = batch

                x = x.to(device, non_blocking=True)
                ei = ei.to(device, non_blocking=True)
                ea = ea.to(device, non_blocking=True)
                bvec = bvec.to(device, non_blocking=True)
                gdesc = gdesc.to(device, non_blocking=True)

                # supercell перед view-аугациями
                if args.aug_supercell_p > 0:
                    x_sc, ei_sc, ea_sc, b_sc = random_supercell_2d(
                        x, ei, ea, bvec,
                        p=args.aug_supercell_p,
                        repeats=args.aug_supercell_repeats,
                        wrap=args.aug_supercell_wrap
                    )
                else:
                    x_sc, ei_sc, ea_sc, b_sc = x, ei, ea, bvec

                # две аугментации (важно: синхронный bvec)
                x1, ei1, ea1, bvec1 = random_node_drop(x_sc, ei_sc, ea_sc, b_sc, args.aug_node_drop)
                x1, ei1, ea1 = random_edge_drop(x1, ei1, ea1, args.aug_edge_drop)
                x2, ei2, ea2, bvec2 = random_node_drop(x_sc, ei_sc, ea_sc, b_sc, args.aug_node_drop)
                x2, ei2, ea2 = random_edge_drop(x2, ei2, ea2, args.aug_edge_drop)

                z1 = enc.encode(x1, ei1, ea1, bvec1, gdesc, use_gdesc = args.ssl_use_gdesc, use_poly_hints = args.ssl_use_poly_hints)
                z2 = enc.encode(x2, ei2, ea2, bvec2, gdesc, use_gdesc = args.ssl_use_gdesc, use_poly_hints = args.ssl_use_poly_hints)

                with torch.autocast('cuda', enabled=use_amp, dtype=AMP_DTYPE):
                    p1, p2 = proj(z1), proj(z2)
                    loss = info_nce_loss(p1, p2, temperature=args.nce_temp)

                accum = max(1, int(getattr(args, "pre_accum_steps", 1)))

                # нормируем лосс под аккумуляцию и делаем backward
                if scaler_ssl.is_enabled():
                    scaler_ssl.scale(loss / accum).backward()
                else:
                    (loss / accum).backward()

                # шаг делаем только раз в 'accum' микрошагов
                if it % accum == 0:
                    # gradient clipping после unscale (если fp16) либо сразу (если bf16)
                    if scaler_ssl.is_enabled():
                        scaler_ssl.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)

                    if torch.isfinite(gn):
                        if scaler_ssl.is_enabled():
                            scaler_ssl.step(opt)
                            scaler_ssl.update()
                        else:
                            opt.step()
                    else:
                        print(f"[SSL][WARN] Non-finite grad_norm={gn} at epoch {ep} iter {it}. Skip step.")

                    opt.zero_grad(set_to_none=True)

                    # Шедулер двигаем ТОЛЬКО когда был фактический шаг оптимизатора
                    sched.step()

                if args.lr_debug_steps > 0 and sched.last_epoch <= args.lr_debug_steps:
                    print(f"[SSL][LRDBG] stage={controller_ssl.stage_idx if stages_ssl else 0} "
                          f"step={sched.last_epoch} | lr={current_lr(opt):.6e} "
                          f"| warmup={(sched.last_epoch < ssl_warmup_steps)}")

                tot += loss.item()
                train_loss = tot / max(1, nb)
                nb += 1

                # периодический лог
                if (it % args.log_every == 0) or (it == 1) or (it == batches):
                    elapsed = time.time() - step0
                    eff_bs = max(1, args.pre_batch_size) * max(1, accum)
                    ips = (it * eff_bs) / max(1e-9, elapsed)

                    remaining = max(0, batches - it)
                    eta_s = (remaining * max(1, args.pre_batch_size)) / max(ips, 1e-9)

                    rec = {
                        "phase": "ssl",
                        "epoch": ep,
                        "iter": it,
                        "iters_per_epoch": batches,
                        "loss_avg": train_loss,
                        "lr": sched.get_last_lr()[0],
                        "ips": ips,
                        "cuda_gb": cuda_mem_gb(),
                        "elapsed_s": elapsed,
                        "eta_min": eta_s / 60.0,

                    }
                    jsonl_append(ssl_log_path, rec)
                    if not args.no_tqdm:
                        iterator.set_postfix(
                            loss=f"{rec['loss_avg']:.4f}",
                            lr=f"{rec['lr']:.2e}",
                            ips=f"{rec['ips']:.1f}/s",
                            mem=f"{rec['cuda_gb']:.2f}GB",
                            eta=f"{eta_s / 60:.1f}m",
                        )

            # --- batch-growth: критерий = train InfoNCE (минимизируем)
            if stages_ssl:
                if controller_ssl.on_epoch_end(train_loss):
                    print(f"[SSL][BATCH-GROWTH] stage end: ep={ep}, loss={train_loss:.4f}")

                    controller_ssl.advance()
                    if controller_ssl.done:
                        break  # завершаем SSL

                    # пересобираем лоадер под ТЕКУЩИЙ режим и новый bs
                    cur_bs = controller_ssl.bs
                    loader = make_ssl_loader(ssl_train_by_mode[active_mode], cur_bs)

                    print(f"[SSL][BATCH-GROWTH] now bs={controller_ssl.bs}")

                    # >>> обновляем LR-шедулер под новую стадию
                    batches_now = max(1, len(loader))
                    cur_stage = controller_ssl.stages[controller_ssl.stage_idx]
                    # Сбрасываем фактический LR к базовому, чтобы warmup начинался «с нуля»
                    for g in opt.param_groups:
                        g["lr"] = args.pre_lr
                    sched, ssl_warmup_steps, ssl_total_steps_stage = _make_ssl_sched_for_stage(
                        batches_now, cur_stage.max_epochs
                    )
                    print(f"[SSL][LR] stage#{controller_ssl.stage_idx}: steps={ssl_total_steps_stage}, "
                          f"warmup={ssl_warmup_steps} (frac={float(args.warmup_frac):.3f})")



        except KeyboardInterrupt:
            print("\n[INTERRUPT][SSL] Saving encoder snapshot and exiting...")
            torch.save(ssl_encoder_state_dict(enc), os.path.join(args.out_dir, "ssl_encoder_interrupted.pt"))

            raise

        print(f"[SSL] Epoch {ep:02d} | loss {tot / max(1, nb):.4f} | {time.time() - t0:.1f}s")
        if args.ckpt_every > 0 and (ep % args.ckpt_every == 0):
            path = os.path.join(args.out_dir, f"ssl_encoder_ep{ep:03d}.pt")
            torch.save(ssl_encoder_state_dict(enc), path)
            ssl_periodic.append(path)
            if len(ssl_periodic) > max(0, args.keep_last_k):
                old = ssl_periodic.pop(0)
                try:
                    os.remove(old)
                except Exception:
                    pass

        # === SSL EARLY STOP: только если НЕ используется ssl_batch_growth ===
        if use_ssl_early_stop:
            epoch_loss = tot / max(1, nb)
            # улучшение считается, если упало хотя бы на min_delta
            if (best_ssl_loss - epoch_loss) >= float(getattr(args, "ssl_early_min_delta", 1e-4)):
                best_ssl_loss = epoch_loss
                ssl_no_improve = 0
            else:
                ssl_no_improve += 1
                print(f"[SSL][EARLY-STOP] no-improve={ssl_no_improve}/{args.ssl_early_stop} "
                      f"(best={best_ssl_loss:.6f} | cur={epoch_loss:.6f})")
                if ssl_no_improve >= int(args.ssl_early_stop):
                    print(f"[SSL][EARLY-STOP] stop at epoch {ep} "
                          f"(best loss {best_ssl_loss:.6f}, patience={args.ssl_early_stop})")
                    break

    # save encoder weights for fine-tuning
    torch.save(ssl_encoder_state_dict(enc), os.path.join(args.out_dir, "ssl_encoder.pt"))

    # запуск SSL-оценки A–F на holdout-е ===
    try:
        print("[SSL-EVAL] Running A–F on scaffold-holdout...")
        _ = run_ssl_evaluation(
            enc=enc,
            val_ds=ssl_val_ds,
            val_smiles=va_unl["SMILES"].tolist(),
            device=device,
            args=args,
            out_dir=args.out_dir,
            tag=f"s{args.seed}"
        )
    except Exception as e:
        print("[SSL-EVAL] skipped due to error:", e)

    return os.path.join(args.out_dir, "ssl_encoder.pt")

def safe_load_ckpt(path, device):
    import torch, numpy as _np
    from torch.serialization import add_safe_globals

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # Разрешим старые numpy-классы (если уверены в источнике файла)
        try:
            add_safe_globals([_np._core.multiarray._reconstruct])  # PyTorch 2.5+
            return torch.load(path, map_location=device, weights_only=True)
        except Exception:
            # последний шанс: обычный load (допускает код в pickle — использовать только для СВОИХ файлов)
            return torch.load(path, map_location=device, weights_only=False)


def run_inference_with_tag(args, run_tag: str) -> str:

    modes = get_poly_modes(args)
    eval_mode = canonical_poly_mode(args, modes)
    use_avg = (args.poly_edge_eval_mode or "first").strip().lower() == "avg"
    eval_modes = modes if use_avg else [eval_mode]
    print(f"[INFER] poly_edge_mode={'avg(' + ','.join(eval_modes) + ')' if use_avg else eval_mode}")


    # фикс: генератор для детерминированной раздачи батчей
    g = torch.Generator()
    g.manual_seed(args.seed)

    # # load test
    # test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    # test_ds = PolymerDataset(test_df, TARGETS, poly_edge_mode=eval_mode)
    #
    # test_loader = DataLoader(
    #     test_ds,
    #     batch_size=args.train_batch_size,
    #     shuffle=False,
    #     collate_fn=collate,
    #     generator=g,
    #     worker_init_fn=seed_worker,
    #     **make_loader_kwargs(args.train_num_workers)
    # )

    # load test once
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    print(f"[INFER] test_df rows: {len(test_df)}")
    if len(test_df) == 0:
        print("[INFER][WARN] test.csv пуст → сабмит будет без строк.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = (torch.cuda.is_available() and not args.cpu)

    # ckpt этого прогона
    ckpt_path = os.path.join(args.out_dir, f"best_polymer_gnn_{run_tag}.pt")
    if not os.path.isfile(ckpt_path):
        # fallback на последний периодический/прерванный, если есть
        for alt in [f"ckpt_interrupted_{run_tag}.pt"]:
            altp = os.path.join(args.out_dir, alt)
            if os.path.isfile(altp):
                ckpt_path = altp
                break
    ckpt = safe_load_ckpt(ckpt_path, device)

    print(f"[INFER] using checkpoint: {ckpt_path} ({ckpt.get('kind', 'ema')})")

    model = PolymerGNN(ckpt["node_in"], ckpt["edge_in"], ckpt["gdesc_dim"],
                       hidden=ckpt["hidden"], layers=ckpt["layers"], targets=len(ckpt["targets"])).to(device)
    # load EMA weights from shadow dict
    sd = model.state_dict()
    for k in sd.keys():
        if k in ckpt["model"]:
            sd[k] = ckpt["model"][k].to(device)
    model.load_state_dict(sd)
    model.eval()

    # preds = []
    # with torch.no_grad():
    #     for x, ei, ea, bvec, gdesc, y, ymask in test_loader:
    #         x, ei, ea, bvec, gdesc = x.to(device), ei.to(device), ea.to(device), bvec.to(device), gdesc.to(device)
    #         with torch.autocast('cuda', enabled=use_amp, dtype=AMP_DTYPE):
    #             o = model(x, ei, ea, bvec, gdesc).detach().to(torch.float32).cpu().numpy()
    #
    #
    #         preds.append(o)
    # preds = np.vstack(preds)

    def _make_loader(ds):
        return DataLoader(
            ds, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate,
            generator=g, worker_init_fn=seed_worker, **make_loader_kwargs(args.train_num_workers)
        )

    preds_sum = None
    with torch.no_grad():
        for m_eval in eval_modes:
            test_ds = PolymerDataset(test_df, TARGETS, poly_edge_mode=m_eval)
            test_loader = _make_loader(test_ds)

            # прогресс-бар без лишних хелперов
            if not getattr(args, "no_tqdm", False):
                test_loader = tqdm(test_loader, total=len(test_loader), desc=f"[INFER] {m_eval}")

            # TTA: N прогонов с активным Dropout
            tta_reps = max(1, int(getattr(args, "tta", 0)) or 1)
            mode_sum = None
            for rep in range(tta_reps):
                model.eval()
                _enable_mc_dropout(model, enable=(tta_reps > 1))
                cur = []
                for x, ei, ea, bvec, gdesc, y, ymask in test_loader:
                    x, ei, ea, bvec, gdesc = x.to(device), ei.to(device), ea.to(device), bvec.to(device), gdesc.to(
                        device)
                    with torch.autocast('cuda', enabled=use_amp, dtype=AMP_DTYPE):
                        out = model(x, ei, ea, bvec, gdesc)
                    cur.append(out.detach().to(torch.float32).cpu().numpy())
                cur = np.vstack(cur)
                mode_sum = cur if mode_sum is None else (mode_sum + cur)
            preds_mode = mode_sum / float(tta_reps)
            preds_sum = preds_mode if preds_sum is None else (preds_sum + preds_mode)
    preds = preds_sum / float(len(eval_modes))
    print(f"[INFER] preds shape: {preds.shape}  (rows must equal len(test_df)={len(test_df)})")

    # ===== Изотоническая калибровка (если есть и запросили) =====
    iso_path = os.path.join(args.out_dir, f"isotonic_{run_tag}.json")
    if getattr(args, "apply_isotonic", False) and os.path.exists(iso_path):
        with open(iso_path, "r") as f:
            iso = json.load(f).get("maps", {})
        for i, t in enumerate(TARGETS):
            m = iso.get(t)
            if m and m.get("x") and m.get("y"):
                preds[:, i] = _iso_apply_1d(preds[:, i].astype(np.float64),
                                            np.asarray(m["x"], float), np.asarray(m["y"], float))
        print("[INFER] isotonic calibration applied")
    else:
        if getattr(args, "apply_isotonic", False):
            print(f"[INFER] isotonic maps not found for tag {run_tag} → skip")

    # ===== Пост-обработка: winsorize и range-match по вал-статистике =====
    stats_path = os.path.join(args.out_dir, f"calib_stats_{run_tag}.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            cinfo = json.load(f)
        st = cinfo.get("stats", {})
        # 1) winsorize
        if getattr(args, "apply_winsor", False):
            for i, t in enumerate(TARGETS):
                s = st.get(t, {})
                ql, qh = s.get("q_low"), s.get("q_high")
                if ql is not None:
                    preds[:, i] = np.maximum(preds[:, i], ql)
                if qh is not None:
                    preds[:, i] = np.minimum(preds[:, i], qh)
            print("[INFER] winsorize applied from validation percentiles")
        # 2) range-match (аффинное преобразование min/max тест-предсказаний к вал-диапазону)
        if getattr(args, "range_match", False):
            for i, t in enumerate(TARGETS):
                s = st.get(t, {})
                vmin, vmax = s.get("min"), s.get("max")
                if vmin is None or vmax is None or vmax <= vmin:
                    continue
                pmin, pmax = float(preds[:, i].min()), float(preds[:, i].max())
                if pmax <= pmin:
                    continue
                scale = (vmax - vmin) / (pmax - pmin)
                preds[:, i] = (preds[:, i] - pmin) * scale + vmin
            print("[INFER] range-match applied to align test range to validation range")
    else:
        print(f"[INFER] calib stats not found: {stats_path} (skip post-processing)")

    # ====== kNN-blend на SSL-эмбеддинге (опционально) ======
    alpha = float(getattr(args, "knn_alpha", 0.0) or 0.0)
    if alpha > 0.0:
        print(f"[INFER] kNN-blend enabled: alpha={alpha}, k={args.knn_k}, tau={args.knn_tau}")

        # соберём supervised DF так же, как в трене
        tr_df, te_df, _ = load_main_data(args.data_dir)
        supp_zip = load_supplemental(args.data_dir)
        external = ingest_external_sources(args)
        sup_df = build_supervised_dataframe(tr_df, supp_zip, external)

        # только примеры с хотя бы 1 лейблом
        sup_df = sup_df[~sup_df[TARGETS].isna().all(axis=1)].reset_index(drop=True)
        print(f"[KNN] labeled sup_df: {len(sup_df)} rows")

        train_ds = PolymerDataset(sup_df, TARGETS, poly_edge_mode=eval_modes[0])
        train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=False,
                                  collate_fn=collate, generator=g, worker_init_fn=seed_worker,
                                  **make_loader_kwargs(args.train_num_workers))

        # подготовим test_loader, чтобы видеть прогресс кодирования
        test_ds = PolymerDataset(test_df, TARGETS, poly_edge_mode=eval_modes[0])
        test_loader = _make_loader(test_ds)

        # оборачиваем оба лоадера в tqdm, если пользователь не отключил бары
        enc_test_loader = test_loader
        enc_train_loader = train_loader
        if not getattr(args, "no_tqdm", False):
            enc_test_loader = tqdm(test_loader, total=len(test_loader), desc="[ENC] test")
            enc_train_loader = tqdm(train_loader, total=len(train_loader), desc="[ENC] train")

        # эмбеддинги (ничего в _encode_dataset не трогаем)
        G_test, _, _ = _encode_dataset(model, enc_test_loader, device, use_amp)
        G_train, Y_train, M_train = _encode_dataset(model, enc_train_loader, device, use_amp)
        print(f"[ENC] G_test={G_test.shape} | G_train={G_train.shape}")

        Y_knn = _knn_regress(
            G_test, G_train, Y_train.astype(np.float32), M_train.astype(bool),
            k=int(getattr(args, "knn_k", 32)), tau=float(getattr(args, "knn_tau", 0.2))
        )
        print(f"[KNN] Y_knn shape: {Y_knn.shape}")
        preds = (1.0 - alpha) * preds + alpha * Y_knn
        print(f"[KNN] blended preds shape: {preds.shape}")

    out_csv = os.path.join(args.out_dir, f"submission_{run_tag}.csv")

    sub = pd.DataFrame({"id": test_df["id"].values})
    for i, t in enumerate(TARGETS):
        sub[t] = preds[:,i]
    sub.to_csv(out_csv, index=False)
    print(f"[INFER] predictions saved → {out_csv} | rows={len(sub)}")

    return out_csv

def run_many_and_blend(args):
    # 1) при необходимости — SSL один раз (никаких подпапок; файл лежит в out_dir/ssl_encoder.pt)
    if args.pretrain:
        print("=== Pretraining (SSL) ===")
        ssl_path = pretrain(args)
        print("Saved SSL encoder:", ssl_path)

    # 2) сформируем список сидов
    seed_list = seeds_from_args(args)
    print(f"[CV] seeds: {seed_list}")

    submission_paths = []

    if args.n_folds >= 2:
        print(f"[CV] {args.n_folds}-fold K-split")
        for f in range(args.n_folds):
            args.fold = f
            for s in seed_list:
                args.seed = s
                args.run_tag = ""  # пусть генерится автоматически: f{f}__s{s}
                print(f"\n=== Supervised: fold={f} seed={s} ===")
                best = train_supervised(args)
                print("Best val wMAE:", best)
                print("=== Inference ===")
                run_tag = make_run_tag(args)
                csv_path = run_inference_with_tag(args, run_tag)
                submission_paths.append(csv_path)
    else:
        # single split, много сидов
        args.n_folds = 1  # чтобы внутри train_supervised пошёл обычный scaffold_split
        for s in seed_list:
            args.seed = s
            args.run_tag = ""  # генерится: s{s}
            print(f"\n=== Supervised: seed={s} ===")
            best = train_supervised(args)
            print("Best val wMAE:", best)
            print("=== Inference ===")
            run_tag = make_run_tag(args)
            csv_path = run_inference_with_tag(args, run_tag)
            submission_paths.append(csv_path)

    # 3) бленд всех сабмитов
    blend_out = os.path.join(args.out_dir, "submission.csv")
    blend_submissions(submission_paths, blend_out)
    print("[CV] Blended submission:", blend_out)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================
# CLI
# =============================
def parse_args():
    p = argparse.ArgumentParser(
        description="Polymer GNN: SSL pretrain + supervised + CV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ---- Paths & IO ---------------------------------------------------------
    g_io = p.add_argument_group("Paths & IO")
    g_io.add_argument("--data_dir", type=str,
                      default="C:/Users/pasha/My_SMARTS_fragmentation/NeurIPS_polymer/GPT5/polymer_comp",
                      help="Папка ИЛИ zip с train.csv/test.csv и (опц.) train_supplement/")
    g_io.add_argument("--external_dir", type=str,
                      default="C:/Users/pasha/My_SMARTS_fragmentation/NeurIPS_polymer/GPT5/external_dir",
                      help="Внешние датасеты (PI1M.csv, Tg/Tc/Density и др.) — папка или zip-структуры")
    g_io.add_argument("--out_dir", type=str,
                      default="C:/Users/pasha/My_SMARTS_fragmentation/NeurIPS_polymer/GPT5/polymer_gnn_pro_ssl_train_test_50",
                      help="Куда сохранять логи/веса/сабмиты и т.п.")

    # ---- Modes (что запускать) ---------------------------------------------
    g_mode = p.add_argument_group("Run modes")
    g_mode.add_argument("--pretrain", action="store_true",
                        help="Сначала сделать SSL, затем supervised + инференс")
    g_mode.add_argument("--pretrain_only", action="store_true",
                        help="Только SSL (сохранит ssl_encoder.pt) и выйти")
    g_mode.add_argument("--finetune_only", action="store_true",
                        help="Только supervised + инференс (без SSL)")

    # ---- Supervised training ------------------------------------------------
    g_sup = p.add_argument_group("Supervised training")
    g_sup.add_argument("--epochs", type=int, default=80, help="Число эпох supervised")
    g_sup.add_argument("--train_batch_size", type=int, default=None,
                       help="Batch size для supervised (если None, см. --batch_size)")
    g_sup.add_argument("--lr", type=float, default=2e-3, help="Начальный LR для supervised")
    g_sup.add_argument("--weight_decay", type=float, default=3e-4, help="Weight decay (AdamW)")
    g_sup.add_argument("--hidden", type=int, default=256, help="Размер скрытого эмбеддинга")
    g_sup.add_argument("--layers", type=int, default=8, help="Число слоёв message passing")
    g_sup.add_argument("--dropout", type=float, default=0.1, help="Dropout в энкодере/голове")
    g_sup.add_argument("--val_size", type=float, default=0.15,
                       help="Если n_folds=1 — доля валидации при scaffold-split")
    g_sup.add_argument("--early_stop", type=int, default=0,
                       help="Патенс в эпохах (0 — выключено)")
    g_sup.add_argument("--cpu", action="store_true", help="Принудительно обучать на CPU")
    # deprecated общий batch_size
    g_sup.add_argument("--batch_size", type=int, default=64,
                       help="(deprecated) Если указан, используется для обеих фаз, "
                            "если не задано pre/train_batch_size")
    g_sup.add_argument("--poly_edge_mode", type=str, default="cycle", choices=["pair", "cycle", "clique"])

    # CLI
    g_sup.add_argument("--warmup_frac", type=float, default=0.03)
    g_sup.add_argument("--warmup_steps", type=int, default=0)

    g_sup.add_argument("--physics_reg_lambda", type=float, default=0.01,
                       help="Вес мягкого регуляризатора: Density ~= 1-FFV (0 — выключить)")

    g_sup.add_argument("--no_compile", action="store_true",
                       help="Отключить torch.compile (индуктор/тритон)")

    # ---- SSL pretraining ----------------------------------------------------
    g_ssl = p.add_argument_group("SSL pretraining")
    g_ssl.add_argument("--pretrain_epochs", type=int, default=10, help="Эпохи SSL")
    g_ssl.add_argument("--pretrain_samples", type=int, default=250000,
                       help="Сэмплов из unlabeled набора для SSL (0 — все)")
    g_ssl.add_argument("--pre_batch_size", type=int, default=None,
                       help="Batch size для SSL (если None, см. --batch_size)")
    g_ssl.add_argument("--pre_lr", type=float, default=1e-3, help="LR в SSL")
    g_ssl.add_argument("--aug_node_drop", type=float, default=0.1, help="Вероятность дропа узлов")
    g_ssl.add_argument("--aug_edge_drop", type=float, default=0.1, help="Вероятность дропа рёбер")
    g_ssl.add_argument("--nce_temp", type=float, default=0.2, help="Температура в InfoNCE")

    # ---- Checkpoints & logging ---------------------------------------------
    g_log = p.add_argument_group("Logging & checkpoints")
    g_log.add_argument("--ckpt_every", type=int, default=0,
                       help="Сохранять чекпоинт каждые N эпох (0 — только лучший)")
    g_log.add_argument("--keep_last_k", type=int, default=3,
                       help="Сколько последних периодических чекпоинтов хранить")
    g_log.add_argument("--eval_every", type=int, default=0,
                       help="Быстрая валидация раз в N батчей (0 — только по эпохам)")
    g_log.add_argument("--log_every", type=int, default=5,
                       help="Логировать метрики каждые N батчей")
    g_log.add_argument("--no_tqdm", action="store_true", help="Отключить прогресс-бары")

    g_log.add_argument("--lr_debug_steps", type=int, default=0,
                       help="Сколько первых шагов печатать подробный LR/warmup-лог (0 — выключено)")
    g_log.add_argument("--debug_anomaly", action="store_true",
                       help="Включить torch.autograd.set_detect_anomaly (True) и проверки finiteness градиента")

    # ---- DataLoader / performance ------------------------------------------
    g_dl = p.add_argument_group("DataLoader / performance")
    g_dl.add_argument("--pre_num_workers", type=int, default=None,
                      help="num_workers для SSL (None — авто; на Windows → 0)")
    g_dl.add_argument("--train_num_workers", type=int, default=None,
                      help="num_workers для supervised/инференса (None — авто; Windows → 0)")

    # ---- EMA ---------------------------------------------------------------
    g_ema = p.add_argument_group("EMA")
    g_ema.add_argument("--ema_decay", type=float, default=0.999,
                       help="EMA decay (Polyak) для слежения за ‘средними’ весами")

    # ---- SWA ---------------------------------------------------------------
    g_swa = p.add_argument_group("SWA")
    g_swa.add_argument("--swa", action="store_true", help="Включить Stochastic Weight Averaging")
    g_swa.add_argument("--swa_start", type=int, default=40,
                       help="С какой эпохи начинать усреднение SWA (1-based)")
    g_swa.add_argument("--swa_lr", type=float, default=5e-4, help="LR в фазе SWA")
    g_swa.add_argument("--swa_anneal_epochs", type=int, default=1,
                       help="Сколько эпох делать anneal LR для SWA")
    g_swa.add_argument("--swa_freq", type=int, default=1,
                       help="Как часто обновлять средние SWA (в батчах)")
    g_swa.add_argument("--swa_update_bn", action="store_true",
                       help="В конце пересчитать BN-статистики для SWA-модели (если есть BN)")
    g_swa.add_argument("--swa_anneal_strategy", type=str, default="cos",
                       choices=["cos", "linear"], help="Стратегия anneal LR для SWA")
    g_swa.add_argument("--select_model", type=str, default="ema",
                       choices=["ema", "swa"],
                       help="Какие веса считать ‘лучшими’ для сохранения")

    # ---- Dot-stage (полирующий этап) --------------------------------------
    g_dot = p.add_argument_group("Dot-stage (polishing)")
    g_dot.add_argument("--dot_stage", action="store_true",
                       help="Сделать короткий полирующий этап после основного тренинга")
    g_dot.add_argument("--dot_epochs", type=int, default=5, help="Эпохи доточкu")
    g_dot.add_argument("--dot_lr", type=float, default=5e-4, help="LR для доточкu")
    g_dot.add_argument("--dot_head_only", action="store_true",
                       help="Учить только голову (+rho), энкодер заморожен")

    # --- Cross-validation / seeds -----------------------------------------
    g_cv = p.add_argument_group("Cross-validation / seeds")
    g_cv.add_argument("--n_folds", type=int, default=0,
                      help="Сколько фолдов. 0/1 — обычный single split; >=2 — K-fold.")
    g_cv.add_argument("--fold", type=int, default=0,
                      help="Индекс фолда (нужен только для единичного запуска конкретного фолда).")
    g_cv.add_argument("--seed_list", type=str, default="",
                      help="Явный список сидов через запятую. Пример: '11,22,33'.")
    g_cv.add_argument("--n_seeds", type=int, default=1,
                      help="Сколько сидов, если seed_list не задан.")
    g_cv.add_argument("--seed_base", type=int, default=42,
                      help="База для генерации сидов при n_seeds (seed_base + [0..n_seeds-1]).")
    g_cv.add_argument("--split_seed", type=int, default=12345,
                      help="Сид для построения K-fold сплитов (независим от обучающего сидa).")

    # небольшая утилита-метка запуска (не трогать руками — генерится автоматически)
    g_cv.add_argument("--run_tag", type=str, default="",
                      help="Суффикс файлов текущего прогона. Если пусто — формируется автоматически, например 'f2__s123'.")

    # ---- Misc / reproducibility -------------------------------------------
    g_misc = p.add_argument_group("Misc / reproducibility")
    g_misc.add_argument("--deterministic", action="store_true",
                        help="Детерминизм (медленнее, но воспроизводимо)")

    # ---- SSL pretraining ----------------------------------------------------
    g_ssl.add_argument("--ssl_batch_growth", type=str, default="",
                       help='Стадии "bs:patience:max_epochs", напр. "4:2:3,8:2:3" (пусто = выключено)')

    # ---- Supervised training ------------------------------------------------
    g_sup.add_argument("--ft_batch_growth", type=str, default="",
                       help='Стадии "bs:patience:max_epochs", напр. "8:2:2,16:3:5" (пусто = выключено)')


    # ---- SSL evaluation (A-F) ----------------------------------------------
    g_ssl.add_argument("--ssl_val_frac", type=float, default=0.10,
                       help="Доля holdout для SSL-оценки (scaffold-holdout).")
    g_ssl.add_argument("--ssl_val_max", type=int, default=20000,
                       help="Максимум молекул в SSL-валидации (для скорости). 0 = без лимита.")
    g_ssl.add_argument("--ssl_eval_batch", type=int, default=256,
                       help="Batch size на инференсе SSL-оценки.")
    g_ssl.add_argument("--ssl_fp_bits", type=int, default=2048,
                       help="Размер Morgan-битов для Tanimoto.")
    g_ssl.add_argument("--ssl_triplet_pos", type=float, default=0.70,
                       help="Порог Tanimoto для позитивной пары в Triplet Acc.")
    g_ssl.add_argument("--ssl_triplet_neg", type=float, default=0.20,
                       help="Порог Tanimoto для негативной пары в Triplet Acc.")
    g_ssl.add_argument("--ssl_triplet_K", type=int, default=50000,
                       help="Сколько триплетов сэмплировать для Triplet Acc (по возможности).")
    g_ssl.add_argument("--ssl_retrieval_k", type=str, default="1,5,10",
                       help="K для Retrieval@k (через запятую).")
    g_ssl.add_argument("--ssl_align_batches", type=int, default=4,
                       help="Сколько батчей валид.набора использовать для Alignment/Uniformity.")

    # === Управление шорткатами в SSL ===
    g_ssl.add_argument("--ssl_use_gdesc", action="store_true",
                       help = "Использовать глобальные дескрипторы gdesc в SSL (по умолчанию ВЫКЛ.)")
    g_ssl.add_argument("--ssl_use_poly_hints", action="store_true",
                       help = "Использовать последние 6 узловых poly-подсказок в SSL (по умолчанию ВЫКЛ.)")

    # --- в parse_args(), после g_sup.add_argument("--poly_edge_mode", ...)
    g_sup.add_argument("--poly_edge_modes", type=str, default="cycle,pair",
                       help="Список режимов poly_edge_mode через запятую (напр., 'cycle,pair'). Если пусто — используется --poly_edge_mode.")
    g_sup.add_argument("--poly_edge_eval_mode", type=str, default="first",
                       help="Какой режим использовать на валидации/инференсе: 'first' (первый из --poly_edge_modes) или явное значение ('cycle'|'pair'|'clique|avg').")

    g_ssl.add_argument("--aug_supercell_p", type=float, default=0.0,
                       help="Вероятность применить 2D суперячейку (0 — выкл.)")
    g_ssl.add_argument("--aug_supercell_repeats", type=int, default=3,
                       help="Сколько копий CRU склеить (2–4 разумно)")
    g_ssl.add_argument("--aug_supercell_wrap", action="store_true",
                       help="Замкнуть последнюю копию на первую (периодический wrap)")

    g_ssl.add_argument("--aug_local_subst_p", type=float, default=0.0,
                       help="Вероятность сделать локальную замену Me/Et, F/Cl, OMe/OEt вне бэкбона")
    g_ssl.add_argument("--aug_local_dynamic", action="store_true",
                       help="Делать замены на лету в UnlabeledDS (иначе только при материализации)")

    # --- SSL early stop (по train loss) ---
    g_ssl.add_argument("--ssl_early_stop", type=int, default=5,
                       help="Ранняя остановка в SSL по train loss (кол-во эпох без улучшений). 0 = выключено.")
    g_ssl.add_argument("--ssl_early_min_delta", type=float, default=1e-4,
                       help="Минимальное улучшение train loss между эпохами для SSL-ранней остановки.")

    # ---- SSL unlabeled selection ------------------------------------------------
    g_ssl.add_argument("--unl_use_sup_df", action="store_true",
                       help="Включать ВСЕ SMILES из sup_df в пул SSL (без лейблов). Рекомендуется.")
    g_ssl.add_argument("--unl_use_main_test", action="store_true", default=True,
                       help="Включать SMILES из main_test в пул SSL. По умолчанию включено.")
    g_ssl.add_argument("--unl_near_frac", type=float, default=0.6,
                       help="Доля 'near-domain' молекул в итоговом unlabeled (scaffold совпадает с sup_df и/или высокий Tanimoto).")
    g_ssl.add_argument("--unl_far_frac", type=float, default=0.2,
                       help="Доля 'far-domain' молекул с низким Tanimoto для разнообразия.")
    g_ssl.add_argument("--unl_max_ref", type=int, default=5000,
                       help="Сколько референсных sup_df молекул использовать для оценки max-Tanimoto (для скорости).")
    g_ssl.add_argument("--unl_far_tanimoto_max", type=float, default=0.25,
                       help="Порог max-Tanimoto для 'far-domain'.")
    g_ssl.add_argument("--unl_near_tanimoto_min", type=float, default=0.5,
                       help="Порог max-Tanimoto для near-кандидатов")
    g_ssl.add_argument("--unl_scaffold_cap", type=int, default=200,
                       help="Мягкий лимит выборок на один scaffold при отборе (баланс разнообразия). 0 = без лимита.")
    g_ssl.add_argument("--unl_dedup_key", type=str, choices=["canon", "smiles", "inchi"], default="canon",
                       help="Ключ для дедупликации unlabeled: канонический SMILES после нормализации ('canon'), исходный ('smiles') или InChIKey ('inchi').")
    g_ssl.add_argument("--unl_pi1m_cap", type=int, default=300000,
                       help="Сколько строк из PI1M.csv максимум брать в пул до тяжёлых операций (scaffold/FP). 0 = всё.")
    g_ssl.add_argument("--unl_n_jobs", type=int, default=0,
                       help="Параллель при расчёте Tanimoto (0=выкл, -1=все ядра, >0=число процессов)")


    g_ssl.add_argument("--pre_accum_steps", type=int, default=1,
                       help="Градиентная аккумуляция: сколько микробатчей по pre_batch_size сливать перед optimizer.step(). 1 = выкл.")

    # ---- Inference ----------------------------------------------------------
    g_inf = p.add_argument_group("Inference")
    g_inf.add_argument("--tta", type=int, default=0, help="MC-Dropout TTA: число прогонов на инференсе (0 — выкл.)")
    g_inf.add_argument("--knn_alpha", type=float, default=0.0, help="Доля kNN-бленда (0 — выкл.)")
    g_inf.add_argument("--knn_k", type=int, default=32, help="Число соседей в kNN")
    g_inf.add_argument("--knn_tau", type=float, default=0.2, help="Температура в softmax весах kNN")

    g_sup.add_argument("--hetero_sigma_head", action="store_true",
                       help = "Добавить вторую голову sigma для гетероскедастической регрессии (per-sample σ)")

    g_sup.add_argument("--balanced_sampler", action="store_true",
                       help="Взвешенный семплинг батчей по наличию редких таргетов")
    g_sup.add_argument("--sampler_strength", type=float, default=1.0,
                       help="Сила бонуса для объектов с редкими таргетами")
    g_sup.add_argument("--sampler_beta", type=float, default=0.5,
                       help="Степень в (1/freq)^beta при расчёте бонуса")

    g_sup.add_argument("--constrain_head", action="store_true",
                       help = "Включить физически осмысленные ограничения в голове: FFV∈[0,1], Rg>0, Density>0")

    g_sup.add_argument("--winsor_q_low", type=float, default=0.0,
                       help="Нижний перцентиль предсказаний на валидации для winsorize (0=выкл.)")
    g_sup.add_argument("--winsor_q_high", type=float, default=100.0,
                       help="Верхний перцентиль предсказаний на валидации для winsorize (100=выкл.)")
    g_sup.add_argument("--save_val_range_stats", action="store_true",
                       help="Сохранять статистики предсказаний на валидации (перцентили и min/max) для пост-обработки")

    g_inf.add_argument("--apply_winsor", action="store_true",
                       help="Применить winsorize к тест-предсказаниям по сохранённым вал-перцентилям")
    g_inf.add_argument("--range_match", action="store_true",
                       help="Аффинно привести диапазон тест-предсказаний к валидационному (по сохранённым min/max)")

    g_sup.add_argument("--fit_isotonic", action="store_true",
                       help="Фит изотонической калибровки на вал-предсказаниях и сохранение узлов")
    g_inf.add_argument("--apply_isotonic", action="store_true",
                       help="Применить изотоническую калибровку (если сохранена для этого run_tag)")

    g_sup.add_argument("--per_target_last", action="store_true",
                       help="Сделать выход головы per-target (отдельный Linear на каждый таргет)")
    g_sup.add_argument("--head_lr", type=float, default=None,
                       help="LR для головы (если None — как --lr)")
    g_sup.add_argument("--head_weight_decay", type=float, default=None,
                       help="WD для головы (если None — как --weight_decay)")
    g_sup.add_argument("--head_lr_mults", type=str, default="1,1,1,1,1",
                       help="Мультипликаторы LR для последнего слоя головы по таргетам (используется при --per_target_last)")
    g_sup.add_argument("--sigma_head_lr", type=float, default=None,
                       help="LR для sigma-головы (если None — как head_lr)")

    # Раздельный warmup/косина для разных групп параметров
    g_sup.add_argument("--bb_warmup_frac", type=float, default=None,
                       help="Доля шагов warmup для бэкбона (если None — как --warmup_frac)")
    g_sup.add_argument("--bb_warmup_steps", type=int, default=0,
                       help="Жёсткий оверрайд шагов warmup для бэкбона (0 — выкл.)")
    g_sup.add_argument("--bb_cosine_floor", type=float, default=0.0,
                       help="Пол косинусного расписания для бэкбона (0..1)")

    g_sup.add_argument("--head_warmup_frac", type=float, default=None,
                       help="Доля шагов warmup для головы (если None — как --warmup_frac)")
    g_sup.add_argument("--head_warmup_steps", type=int, default=0,
                       help="Жёсткий оверрайд шагов warmup для головы (0 — выкл.)")
    g_sup.add_argument("--head_cosine_floor", type=float, default=0.0,
                       help="Пол косинусного расписания для головы (0..1)")

    g_sup.add_argument("--auto_head_lr_mults", action="store_true",
                       help="Авто-подбор LR-мультипликаторов per-target по градиентным нормам (требует --per_target_last)")
    g_sup.add_argument("--auto_lr_window", type=int, default=200,
                       help="Окно усреднения градиентных норм в шагах")
    g_sup.add_argument("--auto_lr_every", type=int, default=100,
                       help="Как часто обновлять множители (в шагах)")
    g_sup.add_argument("--auto_lr_gamma", type=float, default=0.5,
                       help="Степень в (median/gnorm)^gamma")
    g_sup.add_argument("--auto_lr_min_mult", type=float, default=0.5,
                       help="Нижняя граница множителя")
    g_sup.add_argument("--auto_lr_max_mult", type=float, default=2.0,
                       help="Верхняя граница множителя")
    g_sup.add_argument("--auto_lr_start_after_warmup", action="store_true",
                       help="Стартовать авто-подбор только после warmup-фазы текущей стадии")

    # ----- parse & postprocess ----------------------------------------------
    args = p.parse_args()

    # if len(args.seed_list) == 1:
    args.seed = int(seeds_from_args(args)[0])

    # back-compat: общий batch_size прокинуть, если спец. не заданы
    if args.pre_batch_size is None and args.train_batch_size is None:
        if args.batch_size is not None:
            args.pre_batch_size = args.batch_size
            args.train_batch_size = args.batch_size
        else:
            args.pre_batch_size = 64
            args.train_batch_size = 64
    if args.pre_batch_size is None:   args.pre_batch_size = 64
    if args.train_batch_size is None: args.train_batch_size = 64

    # разумные дефолты num_workers (Windows → 0)
    args.pre_num_workers = recommend_num_workers(args.pre_num_workers)
    args.train_num_workers = recommend_num_workers(args.train_num_workers)

    return args

def main():
    args = parse_args()

    if platform.system() == "Windows":
        torch.multiprocessing.set_start_method("spawn", force=True)

    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.out_dir, exist_ok=True)

    # Save args snapshot
    with open(os.path.join(args.out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print_env_banner()

    if args.pretrain_only:
        print("=== Pretraining (SSL) ===")
        ssl_path = pretrain(args)
        print("Saved SSL encoder:", ssl_path)
        return

    # Если больше одного запуска (несколько сидов и/или фолдов) — идём через оркестратор
    multiple_runs = (args.n_folds >= 2) or (len(seeds_from_args(args)) > 1)
    if args.finetune_only:
        if multiple_runs:
            run_many_and_blend(args)
        else:
            print("=== Supervised fine-tuning ===")
            best = train_supervised(args)
            print("Best val wMAE:", best)
            print("=== Inference on test ===")
            run_tag = make_run_tag(args)
            run_inference_with_tag(args, run_tag)
        return

    # поведение по умолчанию: если указан --pretrain, делаем SSL, затем fine-tune
    if multiple_runs:
        run_many_and_blend(args)
    else:
        if args.pretrain:
            print("=== Pretraining (SSL) ===")
            ssl_path = pretrain(args)
            print("Saved SSL encoder:", ssl_path)
        print("=== Supervised fine-tuning ===")
        best = train_supervised(args)
        print("Best val wMAE:", best)
        print("=== Inference on test ===")
        run_tag = make_run_tag(args)
        run_inference_with_tag(args, run_tag)


if __name__ == "__main__":
    main()
