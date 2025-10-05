import numpy as np
import pandas as pd
from typing import Optional, Literal, Tuple
import torch
from torch.utils.data import Dataset, DataLoader


# -----------------------
# 1) TIME PREPROCESS
# -----------------------
def _build_time_projection(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    year: Optional[int] = None,
    t_size: int = 5,
    binning: Literal["uniform","quantile"] = "uniform",
):
    d = df.copy()
    d[datetime_col] = pd.to_datetime(d[datetime_col], errors="coerce")
    d["date"] = d[datetime_col].dt.normalize()
    if year is not None:
        d = d[d["date"].dt.year == year].copy()

    if d["date"].notna().sum() == 0:
        empty_day = pd.DataFrame(columns=["date","date_number"])
        d["date_number"] = pd.Series(dtype="Int64")
        d["time_proj"] = pd.Series(dtype="Int64")
        return d, empty_day, {"message":"No valid dates after parsing/filter."}

    # build continuous daily table
    full_range = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
    day_table = pd.DataFrame({"date": full_range})
    day_table["date_number"] = np.arange(1, len(day_table)+1, dtype=int)

    # left-join back
    d = d.merge(day_table, on="date", how="left")

    # bin to time_proj in {0..t_size-1}
    if binning == "uniform":
        edges = np.linspace(1, len(day_table), t_size + 1)
        d["time_proj"] = pd.cut(d["date_number"], bins=edges, labels=False, include_lowest=True)
    else:
        # quantile bins
        try:
            bins_per_day = pd.qcut(day_table["date_number"], q=t_size, labels=False, duplicates="drop")
            time_map = day_table[["date","date_number"]].assign(time_proj=bins_per_day)
            d = d.merge(time_map[["date","time_proj"]], on="date", how="left", suffixes=("", "_q"))
        except ValueError:
            edges = np.linspace(1, len(day_table), t_size + 1)
            d["time_proj"] = pd.cut(d["date_number"], bins=edges, labels=False, include_lowest=True)

    d["time_proj"] = d["time_proj"].astype("Int64")
    return d, day_table[["date","date_number"]], {
        "n_unique_days": int(day_table.shape[0]),
        "t_size": t_size,
        "binning": binning
    }

# -----------------------
# 2) SPACE PREPROCESS
# -----------------------
def _build_spatial_projection(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    x_size: int = 50,
    y_size: int = 50,
    bbox: Optional[Tuple[float,float,float,float]] = None,  # (lat_min, lat_max, lon_min, lon_max)
):
    d = df.copy()

    # ensure numeric
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")
    d = d.dropna(subset=[lat_col, lon_col]).copy()

    if bbox is None:
        lat_min, lat_max = d[lat_col].min(), d[lat_col].max()
        lon_min, lon_max = d[lon_col].min(), d[lon_col].max()
    else:
        lat_min, lat_max, lon_min, lon_max = bbox

    # clamp to bbox
    d[lat_col] = d[lat_col].clip(lat_min, lat_max)
    d[lon_col] = d[lon_col].clip(lon_min, lon_max)

    # build edges (IMPORTANT: size+1)
    lon_edges = np.linspace(lon_min, lon_max, x_size + 1)
    lat_edges = np.linspace(lat_min, lat_max, y_size + 1)

    # bin to integer indices
    d["x_proj"] = pd.cut(d[lon_col], bins=lon_edges, labels=False, include_lowest=True).astype("Int64")
    d["y_proj"] = pd.cut(d[lat_col], bins=lat_edges, labels=False, include_lowest=True).astype("Int64")
    d["x_proj"] = d["x_proj"].fillna(x_size - 1)
    d["y_proj"] = d["y_proj"].fillna(y_size - 1)

    # centers (for mapping back if needed)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0

    grid = (
        pd.MultiIndex.from_product([range(y_size), range(x_size)], names=["y_proj","x_proj"])
        .to_frame(index=False)
        .assign(
            lat_min=lambda g: lat_edges[g["y_proj"]],
            lat_max=lambda g: lat_edges[g["y_proj"]+1],
            lon_min=lambda g: lon_edges[g["x_proj"]],
            lon_max=lambda g: lon_edges[g["x_proj"]+1],
            lat_center=lambda g: (g["lat_min"] + g["lat_max"]) / 2.0,
            lon_center=lambda g: (g["lon_min"] + g["lon_max"]) / 2.0,
        )
    ).astype({"x_proj":"Int64","y_proj":"Int64"})

    return d, grid, {
        "bbox_used": (float(lat_min), float(lat_max), float(lon_min), float(lon_max)),
        "x_size": x_size, "y_size": y_size
    }

# -----------------------
# 3) FULL PIPE + COUNTS
# -----------------------
def build_spacetime_counts(
    df: pd.DataFrame,
    *,
    datetime_col: str = "datetime",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    count_col: str = "active",          # indicator/weight to aggregate (0/1 or numeric)
    year: Optional[int] = None,
    x_size: int = 50,
    y_size: int = 50,
    t_size: int = 5,
    bbox: Optional[Tuple[float,float,float,float]] = None,
    time_binning: Literal["uniform","quantile"] = "uniform",
    agg_fn: Literal["sum","count","mean"] = "sum",
    fill_value: float = 0.0
):
    """
    Returns
    -------
    cube : DataFrame
        One row per (y_proj, x_proj, time_proj) with columns:
        ['y_proj','x_proj','time_proj','value'] plus lat/lon center/min/max.
    grid : DataFrame
        Spatial grid table (per cell).
    day_table : DataFrame
        Daily table with date and date_number.
    diag : dict
        Diagnostics from time and space steps.
    """
    # TIME
    dt_df, day_table, t_diag = _build_time_projection(
        df, datetime_col=datetime_col, year=year, t_size=t_size, binning=time_binning
    )

    # SPACE (on the time-processed rows)
    sp_df, grid, s_diag = _build_spatial_projection(
        dt_df, lat_col=lat_col, lon_col=lon_col, x_size=x_size, y_size=y_size, bbox=bbox
    )

    # ensure count_col exists; if not, default to 1 per row
    if count_col not in sp_df.columns:
        sp_df[count_col] = 1

    # aggregate on existing rows
    if agg_fn == "sum":
        agg_series = sp_df.groupby(["y_proj","x_proj","time_proj"], dropna=False)[count_col].sum()
    elif agg_fn == "count":
        agg_series = sp_df.groupby(["y_proj","x_proj","time_proj"], dropna=False)[count_col].count()
    elif agg_fn == "mean":
        agg_series = sp_df.groupby(["y_proj","x_proj","time_proj"], dropna=False)[count_col].mean()
    else:
        raise ValueError("agg_fn must be one of {'sum','count','mean'}")

    agg_df = agg_series.reset_index().rename(columns={count_col: "value"})

    # build full spatio-temporal index and reindex
    full_index = pd.MultiIndex.from_product(
        [range(y_size), range(x_size), range(t_size)],
        names=["y_proj","x_proj","time_proj"]
    )
    cube = (
        agg_df.set_index(["y_proj","x_proj","time_proj"])
             .reindex(full_index)
             .reset_index()
    )
    cube["value"] = cube["value"].fillna(fill_value)

    # attach spatial info
    cube = cube.merge(grid, on=["y_proj","x_proj"], how="left")

    diag = {"time": t_diag, "space": s_diag, "n_rows_after_agg": int(cube.shape[0])}
    return cube, grid, day_table, diag


def normalize_df(df, cols):
    norm_df = df.copy()
    for c in cols:
        min_val = df[c].min()
        max_val = df[c].max()
        if max_val > min_val:
            norm_df[c + "_norm"] = (df[c] - min_val) / (max_val - min_val)
        else:
            norm_df[c + "_norm"] = 0.0
    return norm_df


def normalize_columns(df, lat_col="latitude", lon_col="longitude", time_col="time"):
    """
    Min-max normalize latitude, longitude, and time into [0,1].
    - df: input DataFrame with columns lat_col, lon_col, time_col
    - time_col can be datetime or numeric
    
    Returns: new DataFrame with added *_norm columns
    """
    df = df.copy()
    
    # if time is datetime, convert to numeric (seconds since epoch)
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        t_numeric = df[time_col].astype("int64") // 10**9  # seconds
    else:
        t_numeric = df[time_col]
    
    for col, values in [(lat_col, df[lat_col]), (lon_col, df[lon_col]), (time_col, t_numeric)]:
        min_val, max_val = values.min(), values.max()
        df[col + "_norm"] = (values - min_val) / (max_val - min_val)
    
    return df.copy()

def discretize_and_aggregate_fast(
    df: pd.DataFrame,
    time_col: str,
    key_columns: list,
    columns_to_sum: list = None,
    columns_to_mean: list = None,
    time_discretization: int = 5,
    # Se i valori di time_norm rappresentano i CENTRI dei bin, usa 'centers'.
    # Se rappresentano gli ESTREMI [0..1] (passi regolari compresi gli estremi), usa 'edges'.
    time_mapping: str = "edges",   # 'edges' | 'centers'
    drop_na_time: bool = True
) -> pd.DataFrame:
    """
    Mappa time_norm già discretizzato in [0,1] a un indice di bin 0..(M-1)
    e aggrega per (key_columns, time_bin) sommando/mediando colonne specificate.
    """
    if columns_to_sum is None: columns_to_sum = []
    if columns_to_mean is None: columns_to_mean = []

    work = df.copy()
    if drop_na_time:
        work = work[work[time_col].notna()].copy()

    # robustezza numerica
    t = work[time_col].to_numpy(dtype=float)
    t = np.clip(t, 0.0, 1.0)

    M = int(time_discretization)
    if M < 1:
        raise ValueError("time_discretization must be >= 1")

    if time_mapping == "edges":
        # Valori in [0,1] agli estremi: 0 -> bin 0, 1 -> bin M-1
        # Eps per evitare che 1.0 cada fuori per floating point
        eps = 1e-12
        bin_idx = np.floor(t * M - eps).astype(int)
    elif time_mapping == "centers":
        # Valori già ai centri: {0, 1/(M-1), ..., 1}
        bin_idx = np.rint(t * (M - 1)).astype(int)
    else:
        raise ValueError("time_mapping must be 'edges' or 'centers'")

    # clamp finale di sicurezza
    bin_idx = np.clip(bin_idx, 0, M - 1)
    work["time_bin"] = bin_idx

    # dizionario aggregazioni
    agg = {c: "sum" for c in columns_to_sum}
    agg.update({c: "mean" for c in columns_to_mean})
    # contatore righe per gruppo
    #agg["n_records"] = ("time_bin", "size")

    grouped = (
        work
        .groupby(key_columns + ["time_bin"], dropna=False)
        .agg(agg)
        .reset_index()
        .sort_values(key_columns + ["time_bin"])
        .reset_index(drop=True)
    )

    # opzionale: aggiungi start/end/center del bin (utile per plotting sull'asse tempo)
    edges = np.linspace(0.0, 1.0, M + 1)
    starts = pd.Series(edges[:-1], name="time_bin_start")
    ends   = pd.Series(edges[1:],  name="time_bin_end")
    centers = ((starts + ends) / 2.0).rename("time_bin_center")
    meta = pd.concat([starts, ends, centers], axis=1)
    meta["time_bin"] = np.arange(M, dtype=int)

    grouped = grouped.merge(meta, on="time_bin", how="left")
    return grouped

def df_to_THW(
    df,
    time_col="time_proj_norm",
    y_col="y_proj_norm",     # lat (normalized)
    x_col="x_proj_norm",     # lon (normalized)
    count_col="n_shark",
    H=64,
    W=64,
    time_round=6             # how aggressively we snap time to discrete steps
):
    # ---- inputs as np arrays ----
    t = df[time_col].to_numpy(np.float64)
    y = df[y_col].to_numpy(np.float64)
    x = df[x_col].to_numpy(np.float64)
    c = df[count_col].to_numpy(np.float64)

    # guard NaNs / Infs
    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(x) & np.isfinite(c)
    t, y, x, c = t[mask], y[mask], x[mask], c[mask]

    # ---- discretize time: round then rank unique values ----
    t_rounded = np.round(t, time_round)
    uniq_t = np.unique(t_rounded)
    T = uniq_t.size
    t_to_idx = {tv: i for i, tv in enumerate(uniq_t)}
    ti = np.fromiter((t_to_idx[v] for v in t_rounded), count=t_rounded.size, dtype=np.int64)

    # ---- spatial bins in [0,1] → H×W grid ----
    # clip tiny numeric drifts
    y = np.clip(y, 0.0, 1.0)
    x = np.clip(x, 0.0, 1.0)
    y_edges = np.linspace(0.0, 1.0, H + 1)
    x_edges = np.linspace(0.0, 1.0, W + 1)
    yi = np.digitize(y, y_edges) - 1
    xi = np.digitize(x, x_edges) - 1
    # keep within valid [0, H-1]/[0, W-1]
    yi = np.clip(yi, 0, H - 1)
    xi = np.clip(xi, 0, W - 1)

    # ---- accumulate counts into THW ----
    THW = np.zeros((T, H, W), dtype=np.float32)
    # vectorized scatter-add
    np.add.at(THW, (ti, yi, xi), c.astype(np.float32, copy=False))

    # (optional) sanity checks
    # print("Sum in df:", c.sum(), "Sum in THW:", THW.sum())

    return torch.from_numpy(THW), uniq_t  # uniq_t keeps the mapping back to time


def df_to_THW_features(
    df,
    feature_cols,                 # list like ['chl','nppv','thetao', ...]
    time_col="time_norm",
    y_col="latitude_norm",        # in [0,1]
    x_col="longitude_norm",       # in [0,1]
    H=64,
    W=64,
    time_round=6,
    agg="mean"                    # 'mean' (for intensive vars) or 'sum' (for totals)
):
    # ---- base coords/time ----
    t = df[time_col].to_numpy(np.float64)
    y = df[y_col].to_numpy(np.float64)
    x = df[x_col].to_numpy(np.float64)
    feat_arrs = [df[c].to_numpy(np.float64) for c in feature_cols]

    # guard NaNs/Inf – require finite coords & time; features can be NaN (we'll absorb via counts)
    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(x)
    t, y, x = t[mask], y[mask], x[mask]
    feat_arrs = [f[mask] for f in feat_arrs]

    # ---- discretize time ----
    t_rounded = np.round(t, time_round)
    uniq_t = np.unique(t_rounded)
    T = uniq_t.size
    t_to_idx = {tv: i for i, tv in enumerate(uniq_t)}
    ti = np.fromiter((t_to_idx[v] for v in t_rounded), count=t_rounded.size, dtype=np.int64)

    # ---- spatial bins ----
    y = np.clip(y, 0.0, 1.0); x = np.clip(x, 0.0, 1.0)
    y_edges = np.linspace(0.0, 1.0, H + 1)
    x_edges = np.linspace(0.0, 1.0, W + 1)
    yi = np.digitize(y, y_edges) - 1
    xi = np.digitize(x, x_edges) - 1
    yi = np.clip(yi, 0, H - 1); xi = np.clip(xi, 0, W - 1)

    C = len(feature_cols)
    CTHW = np.zeros((C, T, H, W), dtype=np.float32)
    if agg == "mean":
        CNT = np.zeros((T, H, W), dtype=np.float32)

    # scatter-add each feature
    for c_idx, f in enumerate(feat_arrs):
        f32 = f.astype(np.float32, copy=False)
        np.add.at(CTHW[c_idx], (ti, yi, xi), f32)
    if agg == "mean":
        np.add.at(CNT, (ti, yi, xi), 1.0)
        CNT_safe = np.maximum(CNT, 1.0)
        for c_idx in range(C):
            CTHW[c_idx] = CTHW[c_idx] / CNT_safe

    return torch.from_numpy(CTHW), uniq_t


class CellSeriesMultiExoDataset(Dataset):
    """
    Inputs:
      counts_THW: [T,H,W]
      feats_CTHW: [C,T,H,W]  (C = number of exogenous features)
    Each sample:
      'seq' ∈ [K, 1+C]   where [:,0]=log1p(counts), [:,1:] = standardized features
      'y'   ∈ scalar count at time t
      't','i','j' indices for optional map reconstructions
    """
    def __init__(self, counts_THW, feats_CTHW, K=8,
                 log_input=True, standardize_feats=True,
                 f_mean=None, f_std=None):
        assert counts_THW.dim() == 3
        assert feats_CTHW.dim() == 4
        T, H, W = counts_THW.shape
        C, T2, H2, W2 = feats_CTHW.shape
        assert (T,H,W) == (T2,H2,W2), "counts and feats must align in [T,H,W]"

        self.X  = counts_THW.float()     # [T,H,W]
        self.F  = feats_CTHW.float()     # [C,T,H,W]
        self.T, self.H, self.W = T, H, W
        self.C  = C
        self.K  = K
        self.log_input = log_input
        self.standardize_feats = standardize_feats

        if self.standardize_feats:
            if (f_mean is None) or (f_std is None):
                # per-feature stats over all (T,H,W)
                f_mean = self.F.mean(dim=(1,2,3), keepdim=True)                 # [C,1,1,1]
                f_std  = self.F.std(dim=(1,2,3),  keepdim=True).clamp_min(1e-6) # [C,1,1,1]
            self.f_mean, self.f_std = f_mean, f_std
            self.F_in = (self.F - self.f_mean) / self.f_std
        else:
            self.f_mean = torch.zeros((self.C,1,1,1))
            self.f_std  = torch.ones((self.C,1,1,1))
            self.F_in   = self.F

        # build all valid (t,i,j) with t ≥ K
        self.triplets = [(t,i,j)
                         for t in range(self.K, self.T)
                         for i in range(self.H)
                         for j in range(self.W)]

    def __len__(self): return len(self.triplets)

    def __getitem__(self, idx):
        t,i,j = self.triplets[idx]
        seq_counts = self.X[t-self.K:t, i, j]                             # [K]
        if self.log_input:
            seq_counts = torch.log1p(seq_counts)

        # features per each of the K steps for this cell
        # F_in: [C,T,H,W] → take time slice [t-K:t] at (i,j) → [C,K]
        seq_feats = self.F_in[:, t-self.K:t, i, j].permute(1,0)           # [K,C]

        seq = torch.cat([seq_counts.unsqueeze(-1), seq_feats], dim=-1)    # [K,1+C]
        y   = self.X[t, i, j]                                             # scalar
        return {'seq': seq, 'y': y, 't': t, 'i': i, 'j': j}

def collate_series_multi(batch):
    seq = torch.stack([b['seq'] for b in batch], dim=0)      # [B,K,1+C]
    y   = torch.stack([b['y'] for b in batch], dim=0)        # [B]
    t   = torch.tensor([b['t'] for b in batch], dtype=torch.long)
    i   = torch.tensor([b['i'] for b in batch], dtype=torch.long)
    j   = torch.tensor([b['j'] for b in batch], dtype=torch.long)
    return {'seq': seq, 'y': y, 't': t, 'i': i, 'j': j}
