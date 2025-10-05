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
from src.models.transformer_model import TinyTransformerMultiExo
from src.helpers import database_helpers, model_helpers

if 'model' not in st.session_state:
    st.session_state.model = pickle.load(open(ROOT/'src'/'models'/'model.pkl', 'rb'))
if 'test_ds' not in st.session_state:
    st.session_state.test_ds = pickle.load(open(ROOT/'src'/'models'/'test_ds.pkl', 'rb'))
if 'test_loader' not in st.session_state:
    st.session_state.test_loader = pickle.load(open(ROOT/'src'/'models'/'test_loader.pkl', 'rb'))

def _img(path: str):
    p = IMAGES / path
    if p.exists():
        st.image(p, width="stretch")

def folium_map():
    """Create a folium.Map centered and fitted"""

    # Use the zoom value from session state (set by the sidebar slider)
    zoom = int(st.session_state.get("zoom", ZOOM_START))

    m = folium.Map(
        location=CENTER,
        zoom_start=zoom,
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

    st_folium(m, key="app_map", width="stretch")


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
    #print(f"[TEST] Totals — observed: {obs_total:.0f}  predicted: {pred_total:.1f}  ratio={pred_total/max(1,obs_total):.2f}")
    #print(f"[TEST] Per-sample corr (non-trivial): {corr:.3f}")
    #print(f"[TEST] Mean Poisson deviance: {mean_dev:.3f}")

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
    st.markdown('This web app is designed to evaluate the Shark Foraging Detection ML model created by the DaRKRISe team during the 2025 NASA Space Apps Challenge.')
    st.markdown('The app features an interactive map and various controls to explore the model\'s implementation and results.')
    st.markdown('Feel free to explore the map and adjust the settings in the sidebar to see how they affect the view!')
with st.sidebar:
    _img("darkrise_shark_logo.png")
    st.header("Map Controls")
    st.slider("Zoom start", min_value=3, max_value=12, value=ZOOM_START, key="zoom")
    st.checkbox("Show bbox rectangle", value=True, key="show_bbox")

map_fragment()
cols = st.columns([2,2])
"""with cols[0]:
    with st.container(height=1000):
        pass
with cols[1]:
    with st.container(height=1000):
        st.header("AI MODEL (IMPLEMENTATION AND RESULTS)")"""

print(get_predictions())

