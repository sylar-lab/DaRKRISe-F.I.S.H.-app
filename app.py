#
# A web interface to evaluate the Shark Foraging Detection ML model created by DaRKRISe team
# during the 2025 Nasa Space Apps Challenge
#
# date: 4/10/2025
#

from typing import Tuple
import folium
import streamlit as st
from pathlib import Path
from streamlit_folium import st_folium
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import pickle
from torch import nn
from si_prefix import si_format, si_parse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io
from folium.plugins import HeatMap
from branca.colormap import LinearColormap
import matplotlib.colors as mcolors

class TinyTransformerMultiExo(nn.Module):
    """
    Input: batch['seq'] ∈ [B, K, C_in] with C_in = 1 (counts) + C (features)
    Output: log-rate for target y (Poisson)
    """
    def __init__(self, K, C_in, d=64, nhead=4, nlayers=2):
        super().__init__()
        self.pos   = nn.Parameter(torch.randn(1, K, d) * 0.01)
        self.embed = nn.Linear(C_in, d)
        enc_layer  = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, batch_first=True)
        self.enc   = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head  = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        x = batch['seq']                   # [B,K,C_in]
        h = self.embed(x) + self.pos       # [B,K,d]
        h = self.enc(h)                    # [B,K,d]
        h_last = h[:, -1, :]
        out = self.head(h_last).squeeze(-1)
        return self.alpha + out            # log-rate

st.set_page_config(
	page_title="Home",
	layout="wide",
	initial_sidebar_state="expanded",
	menu_items={
		"About": "Sharks Foraging Detection Version 1.0",
		"Get help": "https://github.com/sylar-lab/DaRKRISe-Sharks-App",
		"Report a bug": "https://github.com/sylar-lab/DaRKRISe-Sharks-App/issues"
	}
)


ROOT = Path(__file__).parent
IMAGES = ROOT / "data" / "images"
LAT_MIN, LAT_MAX = 28, 42
LON_MIN, LON_MAX = -81, -65
CENTER = [(LAT_MIN + LAT_MAX) / 2.0, (LON_MIN + LON_MAX) / 2.0]
ZOOM_START = 6

if 'model' not in st.session_state:
    st.session_state.model = pickle.load(open(ROOT/'stored'/'model.pkl', 'rb'))
if 'test_ds' not in st.session_state:
    st.session_state.test_ds = pickle.load(open(ROOT/'stored'/'test_ds.pkl', 'rb'))
if 'test_loader' not in st.session_state:
    st.session_state.test_loader = pickle.load(open(ROOT/'stored'/'test_loader.pkl', 'rb'))


class SiNumber(float):
    def __new__(cls, value, precision=2):
        instance = super().__new__(cls, value)
        instance._precision = precision
        return instance

    def __float__(self):
        return si_parse(self.__repr__())

    @classmethod
    def parse(cls, si_value, precision=2):
        instance = cls(si_parse(si_value), precision)
        return instance

    def __repr__(self):
        return si_format(self, precision=self._precision)


def _img(path: str):
    p = IMAGES / path
    if p.exists():
        st.image(p, width="stretch")

def folium_map():
    """Create a folium.Map centered and fitted"""

    # Use the zoom value from session state (set by the sidebar slider)
    # enforce zoom bounds
    ZOOM_MIN = 5
    ZOOM_MAX = 9
    zoom = int(st.session_state.get("zoom", ZOOM_START))
    if zoom < ZOOM_MIN:
        zoom = ZOOM_MIN
    if zoom > ZOOM_MAX:
        zoom = ZOOM_MAX

    m = folium.Map(
        location=CENTER,
        zoom_start=zoom,
        min_zoom=ZOOM_MIN,
        max_zoom=ZOOM_MAX,
        tiles='https://services.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles &copy; Esri &mdash; Source: Esri, GEBCO, NOAA, National Geographic, DeLorme, NAVTEQ, and other contributors',
        control_scale=True,
    )

    # Fit bounds using SW and NE corners: [[south, west], [north, east]]
    sw = [LAT_MIN, LON_MIN]
    ne = [LAT_MAX, LON_MAX]

    # Add a rectangle to show the bbox
    folium.Rectangle(bounds=[sw, ne], color="#ff7800", weight=2, fill=False, tooltip="BBox").add_to(m)

    if not st.session_state.show_bbox:
        # remove rectangles by filtering children (simple approach)
        to_remove = [k for k, v in list(m._children.items()) if v.__class__.__name__ == "Rectangle"]
        for k in to_remove:
            m._children.pop(k, None)

    with st.spinner("Computing prediction, please wait..."):
        smoothed_pred, smoothed_obs, (corr, mean_dev, obs_total, pred_total) = get_predictions()
    # Fix orientation: rotate arrays 90° clockwise (user requested fixed orientation)
    try:
        smoothed_pred = np.rot90(smoothed_pred, k=3)
        smoothed_obs = np.rot90(smoothed_obs, k=3)
    except Exception:
        pass

    metr_cols = st.columns([1,1,1])
    with metr_cols[0]:
        st.metric('Totals — observed sharks', SiNumber(obs_total))
    with metr_cols[1]:
        st.metric('Predicted sharks', SiNumber(pred_total))
    with metr_cols[2]:
        st.metric('True positives', f'{round(100*pred_total/max(1,obs_total), 1)}%')
    
    # --- render smoothed arrays as GeoJSON grid tiles (vector only) ---
    ds = int(st.session_state.get('overlay_downsample', 1))
    thresh_pct = float(st.session_state.get('overlay_threshold_pct', 5)) / 100.0
    H, W = smoothed_pred.shape

    # grid indices (downsampled)
    rows = np.arange(0, H, ds)
    cols = np.arange(0, W, ds)

    # edges for polygon coordinates (row 0 -> LAT_MAX top)
    lat_edges = np.linspace(LAT_MAX, LAT_MIN, H+1)
    lon_edges = np.linspace(LON_MIN, LON_MAX, W+1)

    # determine color scaling using robust percentile
    p_low = 2   # use 2nd percentile as vmin to avoid mapping tiny values to white
    p_high = 98
    vmax_pred = float(np.nanpercentile(smoothed_pred, p_high))
    vmax_obs = float(np.nanpercentile(smoothed_obs, p_high))
    vmin_pred = float(np.nanpercentile(smoothed_pred, p_low))
    vmin_obs = float(np.nanpercentile(smoothed_obs, p_low))
    if vmax_pred <= vmin_pred:
        vmax_pred = max(vmin_pred + 1e-6, float(smoothed_pred.max()) if smoothed_pred.max() > 0 else 1.0)
    if vmax_obs <= vmin_obs:
        vmax_obs = max(vmin_obs + 1e-6, float(smoothed_obs.max()) if smoothed_obs.max() > 0 else 1.0)

    # Use distinct warm/cool colormaps to avoid visual confusion when both layers are visible
    cmap_pred = cm.get_cmap('Reds')  # warm -> predicted
    cmap_obs = cm.get_cmap('Blues')  # cool -> observed
    norm_pred = mcolors.Normalize(vmin=vmin_pred, vmax=vmax_pred)
    norm_obs = mcolors.Normalize(vmin=vmin_obs, vmax=vmax_obs)

    features_pred = {"type": "FeatureCollection", "features": []}
    features_obs = {"type": "FeatureCollection", "features": []}

    for i in rows:
        for j in cols:
            # aggregate block values
            block_pred = smoothed_pred[i:i+ds, j:j+ds]
            block_obs = smoothed_obs[i:i+ds, j:j+ds]
            val_pred = float(np.nanmean(block_pred)) if block_pred.size else 0.0
            val_obs = float(np.nanmean(block_obs)) if block_obs.size else 0.0

            # skip both if below their respective thresholds
            keep_pred = (val_pred > 0) and (val_pred >= vmax_pred * thresh_pct)
            keep_obs = (val_obs > 0) and (val_obs >= vmax_obs * thresh_pct)
            if not (keep_pred or keep_obs):
                continue

            # polygon corners from edges: lat,lon pairs
            lat0 = lat_edges[i]
            lat1 = lat_edges[min(i+ds, H)]
            lon0 = lon_edges[j]
            lon1 = lon_edges[min(j+ds, W)]
            # polygon in lat,lon order then convert to GeoJSON lon,lat
            poly_latlon = [[lat0, lon0], [lat0, lon1], [lat1, lon1], [lat1, lon0], [lat0, lon0]]

            if keep_pred:
                rgba = cmap_pred(norm_pred(val_pred))
                hexcol = mcolors.to_hex(rgba)
                feat = {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in poly_latlon]]},
                    "properties": {"value": val_pred, "style": {"fillColor": hexcol, "color": hexcol, "weight": 0, "fillOpacity": float(st.session_state.get('overlay_opacity', 0.6))}}
                }
                features_pred["features"].append(feat)

            if keep_obs:
                rgba = cmap_obs(norm_obs(val_obs))
                hexcol = mcolors.to_hex(rgba)
                feat = {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in poly_latlon]]},
                    "properties": {"value": val_obs, "style": {"fillColor": hexcol, "color": hexcol, "weight": 0, "fillOpacity": float(st.session_state.get('overlay_opacity', 0.6))}}
                }
                features_obs["features"].append(feat)

    # Add GeoJSON layers if they contain features
    if features_pred["features"]:
        folium.GeoJson(features_pred, name='Predicted (grid)').add_to(m)
    if features_obs["features"]:
        folium.GeoJson(features_obs, name='Observed (grid)').add_to(m)

    # Add legends for both layers
    # Use explicit red/blue ramps for legends to match the tile colors
    ramp_pred = [mcolors.to_hex(cmap_pred(x)) for x in np.linspace(0.05, 1.0, 7)]
    ramp_obs = [mcolors.to_hex(cmap_obs(x)) for x in np.linspace(0.05, 1.0, 7)]
    pred_cmap = LinearColormap(ramp_pred, vmin=vmin_pred, vmax=vmax_pred, caption='Predicted (smoothed)')
    obs_cmap = LinearColormap(ramp_obs, vmin=vmin_obs, vmax=vmax_obs, caption='Observed (smoothed)')
    pred_cmap.add_to(m)
    obs_cmap.add_to(m)

    folium.LayerControl().add_to(m)

    # Capture interactive map state to preserve zoom/center between reruns and avoid jumpy zoom
    map_ret = st_folium(m, key="app_map", width="stretch")
    try:
        if map_ret and isinstance(map_ret, dict):
            zval = None
            if 'zoom' in map_ret:
                zval = map_ret.get('zoom')
            elif 'center' in map_ret and 'zoom' in map_ret.get('center', {}):
                zval = map_ret.get('center', {}).get('zoom')
            # clamp and store
            if zval is not None:
                try:
                    zv = int(zval)
                    zv = max(ZOOM_MIN, min(ZOOM_MAX, zv))
                    st.session_state.zoom = zv
                except Exception:
                    pass
    except Exception:
        # non-fatal: ignore map state update failures
        pass

@st.cache_data
def get_predictions():
    tx = st.session_state.model 
    test_ds = st.session_state.test_ds
    test_loader = st.session_state.test_loader
    tx.eval()
    all_y, all_pred, all_tij = [], [], []
    device = next(tx.parameters()).device

    with torch.no_grad():
        for batch in test_loader:
            y = batch['y'].float()
            log_rate = tx({k: (v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()})
            lam = torch.exp(log_rate).cpu().float()   # [B]

            all_y.append(y)
            all_pred.append(lam)
            all_tij.append(torch.stack([batch['t'], batch['i'], batch['j']], dim=1))  # [B,3]

    y_cat   = torch.cat(all_y, dim=0).numpy()           # [N_test_samples]
    pred_cat= torch.cat(all_pred, dim=0).numpy()        # [N_test_samples]
    tij_cat = torch.cat(all_tij, dim=0).numpy()         # [N_test_samples, 3] (t,i,j) in TEST local indexing

    # --- aggregate metrics across all test samples ---
    mask_pos = (y_cat > 0) | (pred_cat > 0)
    corr = (np.corrcoef(y_cat[mask_pos], pred_cat[mask_pos])[0,1] if mask_pos.any() else np.nan)

    eps = 1e-9
    lam = pred_cat + eps
    y_  = y_cat
    deviance = 2 * ( y_ * np.log((y_ + eps)/lam) - (y_ - lam) )
    mean_dev = float(np.nanmean(deviance))

    obs_total  = float(y_cat.sum())
    pred_total = float(pred_cat.sum())


    # --- reconstruct LAST test frame grid for mapping ---
    # In the TEST dataset, valid target times are local t ∈ [K, T_te_full-1]
    last_t_local = test_ds.T - 1
    pick = (tij_cat[:,0] == last_t_local)
    ii, jj = tij_cat[pick,1], tij_cat[pick,2]

    H, W = test_ds.H, test_ds.W
    last_obs_grid  = np.zeros((H,W), dtype=np.float32)
    last_pred_grid = np.zeros((H,W), dtype=np.float32)
    last_obs_grid[ii, jj]  = y_cat[pick]
    last_pred_grid[ii, jj] = pred_cat[pick]

    smoothed_pred = gaussian_filter(last_pred_grid, sigma=1.0)  # increase sigma for more smoothing
    smoothed_obs = gaussian_filter(last_obs_grid, sigma=1.0)
    return smoothed_pred, smoothed_obs, (corr, mean_dev, obs_total, pred_total)

@st.fragment
def map_fragment():
    """A fragment with the map and controls"""
    st.title("Shark Foraging ML Model Evaluation")
    # if st.button(type="primary", label="Reset view"):
    #    st.session_state.zoom = ZOOM_START
    folium_map()

# APP LAYOUT
_img("banner_dark_blue.png")

with st.expander("Interface Overview", expanded=False):
    st.link_button('Visit FULL project description website', 'https://www.spaceappschallenge.org/2025/find-a-team/darkrise/?tab=project')
    st.markdown('This web app is designed to evaluate the model inference created by the DaRKRISe team during the 2025 NASA Space Apps Challenge.')
    st.markdown('The app features an interactive map to explore the model\'s inference results.')
with st.sidebar:
    _img("darkrise_shark_logo.png")
    st.header("Map Controls")
    st.slider("Zoom start", min_value=5, max_value=9, value=max(5, min(9, ZOOM_START)), key="zoom")
    st.checkbox("Show selected training area", value=True, key="show_bbox")
    # Overlay display controls
    st.markdown("---")
    st.subheader("Overlay display")
    st.slider("Overlay opacity", 0.0, 1.0, value=0.85, key="overlay_opacity")
    st.slider("Hide low values (% of max)", 5, 50, value=14, key="overlay_threshold_pct")
    st.slider("Downsample factor (1 = no downsample)", 1, 8, value=1, key="overlay_downsample")


cols = st.columns([2,2])
with cols[0]:
    map_fragment()
with cols[1]:
    pass


