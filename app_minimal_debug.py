
# app_minimal_debug.py
# -*- coding: utf-8 -*-
import io, base64
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image

st.set_page_config(page_title="Debug Pluma — Mínimo Viável", layout="wide")
st.title("Debug Pluma — Folium + Streamlit (MWE)")

# --- Estado ---
ss = st.session_state
ss.setdefault("source", None)        # (lat, lon)
ss.setdefault("overlay", None)       # (png_bytes, bounds)

# --- Sidebar ---
with st.sidebar:
    st.header("Controles")
    wind_dir = st.number_input("Direção do vento (graus de onde VEM)", 0, 359, 45, 1)
    wind_speed = st.number_input("Velocidade do vento (m/s)", 0.1, 50.0, 5.0, 0.1)
    Q = st.number_input("Q relativo", 0.1, 1e6, 100.0, 0.1)
    opacity = st.slider("Opacidade", 0.0, 1.0, 0.85, 0.01)
    clip_pct = st.slider("Clip (%)", 0, 95, 0, 1)
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    if c1.button("Atualizar overlay", type="primary", use_container_width=True):
        ss._update = True
    if c2.button("Selecionar novo ponto", use_container_width=True):
        ss.source = None; ss.overlay = None
    if c3.button("Overlay de teste", use_container_width=True):
        if ss.source is None:
            ss.source = (-22.9035, -43.2096)
        ss._update = True

# --- Funções ---
def jet_colormap():
    from matplotlib import cm
    cols = (cm.get_cmap('jet', 256)(np.linspace(0,1,256))[:,:3] * 255).astype(np.uint8)
    return cols

def make_overlay(lat, lon, wind_dir, wind_speed, Q, clip_pct, extent_km=2.0, px_per_km=150):
    # grade local (metros, equiretangular)
    R = 6371000.0
    lat_rad = np.deg2rad(lat)
    m_per_deg_lat = np.pi*R/180.0
    m_per_deg_lon = m_per_deg_lat*np.cos(lat_rad)
    half = extent_km*1000/2
    res = max(96, min(2048, int(extent_km*px_per_km)))
    x = np.linspace(-half, half, res); y = np.linspace(-half, half, res)
    X, Y = np.meshgrid(x, y)

    # orientação (pluma para onde vai: +180)
    theta = np.deg2rad((wind_dir + 180) % 360)
    Xp =  np.cos(theta)*X + np.sin(theta)*Y
    Yp = -np.sin(theta)*X + np.cos(theta)*Y
    mask = Xp > 0

    # gauss 2D + decaimento simples
    sigy = np.clip(100 + 0.2*Xp, 60, None)  # metros
    sigx = np.clip(200 + 0.4*Xp, 80, None)
    C = np.zeros_like(Xp, dtype=float)
    C[mask] = Q * np.exp(-0.5*((Xp[mask]/sigx[mask])**2 + (Yp[mask]/sigy[mask])**2)) * np.exp(-(Xp[mask]/1500.0))
    C = C / (C.max() + 1e-12)

    if clip_pct>0 and np.any(C>0):
        thr = np.percentile(C[C>0], clip_pct)
        C = np.clip((C - thr) / (C.max() - thr + 1e-12), 0, 1)

    lut = jet_colormap()
    idx = (C*255).astype(np.uint8)
    rgb = lut[idx]
    alpha = (np.sqrt(C)*255).astype(np.uint8)  # reforça valores baixos
    alpha[C<=0.003] = 0
    rgba = np.dstack([rgb, alpha])

    # bounds lat/lon
    dlat = half / m_per_deg_lat
    dlon = half / (m_per_deg_lon if m_per_deg_lon>0 else 111320.0)
    bounds = [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]

    im = Image.fromarray(rgba.astype(np.uint8), mode="RGBA")
    bio = io.BytesIO(); im.save(bio, format="PNG"); bio.seek(0)
    return bio.read(), bounds

# --- Mapa base ---
def base_map(center):
    m = folium.Map(location=center, zoom_start=15, tiles=None, control_scale=True)
    # HTTPS tiles only
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri — World Imagery', name='ESRI', overlay=False, control=True
    ).add_to(m)
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='© OpenStreetMap', name='OSM', overlay=False, control=True, show=False
    ).add_to(m)
    return m

# --- Fluxo ---
if ss.source is None:
    st.info("Clique no mapa para definir a fonte. Ou use 'Overlay de teste'.")
    m0 = base_map([-22.9035, -43.2096])
    state = st_folium(m0, height=700, width=None, returned_objects=["last_clicked"], use_container_width=True)
    if state and state.get("last_clicked"):
        ss.source = (state["last_clicked"]["lat"], state["last_clicked"]["lng"])
        st.success(f"Fonte: {ss.source[0]:.6f}, {ss.source[1]:.6f}")
else:
    lat, lon = ss.source
    if ss.get("_update", False) or ss.overlay is None:
        png, b = make_overlay(lat, lon, wind_dir, wind_speed, Q, clip_pct, extent_km=2.0, px_per_km=150)
        ss.overlay = (png, b)
        ss._update = False

    png, bounds = ss.overlay
    m1 = base_map([lat, lon])
    folium.CircleMarker([lat, lon], radius=6, color="#ff0000", fill=True, fill_opacity=1.0).add_to(m1)
    url = "data:image/png;base64," + base64.b64encode(png).decode("utf-8")
    folium.raster_layers.ImageOverlay(image=url, bounds=bounds, opacity=opacity, name="Overlay").add_to(m1)
    folium.LayerControl(collapsed=False).add_to(m1)
    st_folium(m1, height=700, width=None, use_container_width=True)
