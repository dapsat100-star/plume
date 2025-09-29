
# app_pro.py
# -*- coding: utf-8 -*-
import io
import base64
from pathlib import Path

import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image

st.set_page_config(page_title="Gaussian Plume — quick model", layout="wide")
st.title("Pluma Gaussiana (quase) — com altura da fonte e estabilidade")

st.caption("Modelo em plano (terreno plano) com pluma gaussiana estacionaria, reflexao no solo, altura efetiva (altura da chamine + plume rise Briggs aproximado), e estabilidade Pasquill-Gifford (A-F). Use para estimativa de ordem de grandeza. Nao substitui AERMOD/CALPUFF.")

# ==================== Sidebar controls ====================
with st.sidebar:
    st.header("Meteorologia")
    wind_dir_deg = st.number_input("Direção do vento (graus a partir do Norte, sentido horário)", 0, 359, 45, 1)
    wind_speed = st.number_input("Velocidade do vento a 10 m (m/s)", 0.1, 50.0, 5.0, 0.1, format="%.1f")
    stability = st.selectbox("Classe de estabilidade (Pasquill-Gifford)", ["A","B","C","D","E","F"], index=3)
    is_urban = st.checkbox("Condição urbana (↑ dispersão transversal)", value=False)

    st.header("Fonte / Chaminé")
    Q_gps = st.number_input("Taxa de emissão Q (g/s)", 0.001, 1e6, 100.0, 0.1, format="%.3f")
    H_stack = st.number_input("Altura geométrica da fonte Hs (m)", 0.0, 500.0, 10.0, 0.5)
    d_stack = st.number_input("Diâmetro da chaminé d (m)", 0.05, 10.0, 0.5, 0.05)
    V_exit = st.number_input("Velocidade de saída V (m/s)", 0.1, 120.0, 15.0, 0.1)
    Tamb = st.number_input("Temperatura do ar (K)", 230.0, 330.0, 298.0, 0.5)
    Tstack = st.number_input("Temperatura dos gases (K)", 230.0, 500.0, 320.0, 0.5)

    st.header("Domínio / Render")
    extent_km = st.slider("Extensão (km)", 1.0, 20.0, 5.0, 0.5)
    px = st.select_slider("Resolução (px/km)", [50, 75, 100, 150, 200, 300], value=150)
    clip_pct = st.slider("Corte inferior da paleta (%)", 0, 95, 10, 1)
    opacity = st.slider("Opacidade do overlay", 0.0, 1.0, 0.7, 0.01)

# ==================== Helpers ====================
def jet_colormap():
    from matplotlib import cm
    import numpy as np
    cmap = cm.get_cmap('jet', 256)
    cols = (cmap(np.linspace(0,1,256))[:,:3] * 255).astype(np.uint8)
    return cols

# Sigma_y, Sigma_z per Pasquill-Gifford (Turner, commonly used fits)
def sigma_yz(x_m, stability, is_urban=False):
    x_km = np.maximum(x_m/1000.0, 1e-6)
    s = stability.upper()
    # Rural fits
    if not is_urban:
        coefs = {
            "A": (0.22, 0.5, 0.20, 0.5),
            "B": (0.16, 0.5, 0.12, 0.5),
            "C": (0.11, 0.5, 0.08, 0.5),
            "D": (0.08, 0.5, 0.06, 0.5),
            "E": (0.06, 0.5, 0.03, 0.5),
            "F": (0.04, 0.5, 0.016,0.5),
        }
    else:
        # Urban fits (heuristic)
        coefs = {
            "A": (0.32, 0.5, 0.24, 0.5),
            "B": (0.22, 0.5, 0.18, 0.5),
            "C": (0.16, 0.5, 0.14, 0.5),
            "D": (0.12, 0.5, 0.10, 0.5),
            "E": (0.10, 0.5, 0.06, 0.5),
            "F": (0.08, 0.5, 0.04, 0.5),
        }
    a, by, c, bz = coefs.get(s, coefs["D"])
    sigy = a * (x_km ** by) * 1000.0
    sigz = c * (x_km ** bz) * 1000.0
    sigy = np.clip(sigy, 1.0, None)
    sigz = np.clip(sigz, 1.0, None)
    return sigy, sigz

def effective_height(H_stack, V_exit, d_stack, Tamb, Tstack, wind_speed):
    # Briggs plume rise (neutral, simplified final rise). This is an approximation.
    g = 9.80665
    F = g * V_exit * (d_stack**2) * (Tstack - Tamb) / (4.0 * Tstack + 1e-9)
    delta_m = 3.0 * d_stack * V_exit / max(wind_speed, 0.1)
    delta_b = 2.6 * (F ** (1.0/3.0)) / max(wind_speed, 0.1)
    delta_h = max(delta_m, delta_b, 0.0)
    return H_stack + delta_h

def make_overlay(lat, lon, params):
    wind_dir_deg = params["wind_dir_deg"]
    wind_speed   = params["wind_speed"]
    Q_gps        = params["Q_gps"]
    stability    = params["stability"]
    is_urban     = params["is_urban"]
    extent_km    = params["extent_km"]
    px_per_km    = params["px"]
    clip_pct     = params["clip_pct"]
    opacity      = params["opacity"]
    Hs           = params["H_eff"]

    # Local meters per degree
    R = 6371000.0
    lat_rad = np.deg2rad(lat)
    m_per_deg_lat = np.pi*R/180.0
    m_per_deg_lon = m_per_deg_lat*np.cos(lat_rad)

    half = extent_km*1000/2
    res = int(extent_km*px_per_km)
    res = max(96, min(3072, res))

    x = np.linspace(-half, half, res)
    y = np.linspace(-half, half, res)
    X, Y = np.meshgrid(x, y)

    theta = np.deg2rad((wind_dir_deg + 180) % 360)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    Xp =  cos_t*X + sin_t*Y
    Yp = -sin_t*X + cos_t*Y

    mask = Xp > 0.0
    Xp_eff = np.where(mask, Xp, np.nan)

    sigy, sigz = sigma_yz(np.nan_to_num(Xp_eff, nan=1.0), stability, is_urban=is_urban)

    u = max(wind_speed, 0.1)
    pref = Q_gps / (2.0 * np.pi * u * sigy * sigz + 1e-12)
    term_y = np.exp(-0.5 * (Yp**2) / (sigy**2 + 1e-12))
    term_z = np.exp(-0.5 * ((0.0 - Hs)**2) / (sigz**2 + 1e-12)) + np.exp(-0.5 * ((0.0 + Hs)**2) / (sigz**2 + 1e-12))
    C = pref * term_y * term_z
    C[~mask] = 0.0

    Cmax = float(np.nanmax(C))
    Cnorm = C / (Cmax + 1e-15)
    if clip_pct>0 and np.any(Cnorm>0):
        thr = np.nanpercentile(Cnorm[Cnorm>0], clip_pct)
        Cnorm = np.clip((Cnorm - thr) / (1e-9 + Cnorm.max() - thr), 0, 1)

    lut = jet_colormap()
    idx = (Cnorm*255).astype(np.uint8)
    rgb = lut[idx]
    alpha = (Cnorm*255).astype(np.uint8)
    alpha[Cnorm<=0.01] = 0
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)

    dlat = (half / m_per_deg_lat)
    dlon = (half / m_per_deg_lon if m_per_deg_lon>0 else half/111320.0)
    bounds = [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]

    im = Image.fromarray(rgba, mode="RGBA")
    bio = io.BytesIO(); im.save(bio, format="PNG"); bio.seek(0)
    return bio.read(), bounds, Cmax

# Base map
default_center = [-22.9035, -43.2096]
m = folium.Map(location=default_center, zoom_start=12, tiles=None, control_scale=True)
folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri — World Imagery', name='Esri WorldImagery', overlay=False, control=True).add_to(m)
folium.TileLayer(tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', attr='© OpenStreetMap', name='OSM', overlay=False, control=True, show=False).add_to(m)

map_state = st_folium(m, height=720, width=None, returned_objects=["last_clicked"], use_container_width=True)

st.markdown("### 1) Clique no mapa para definir a fonte")
if map_state and map_state.get("last_clicked"):
    lat = map_state["last_clicked"]["lat"]
    lon = map_state["last_clicked"]["lng"]
    st.success(f"Fonte em: {lat:.6f}, {lon:.6f}")

    H_eff = effective_height(H_stack, V_exit, d_stack, Tamb, Tstack, wind_speed)
    st.info(f"Altura efetiva estimada H = {H_eff:.1f} m (Hs + plume rise Briggs aproximado)")

    params = dict(
        wind_dir_deg=wind_dir_deg,
        wind_speed=wind_speed,
        Q_gps=Q_gps,
        stability=stability,
        is_urban=is_urban,
        extent_km=extent_km,
        px=px,
        clip_pct=clip_pct,
        opacity=opacity,
        H_eff=H_eff
    )

    data, bounds, Cmax = make_overlay(lat, lon, params)

    m2 = folium.Map(location=[lat, lon], zoom_start=13, tiles=None, control_scale=True)
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri — World Imagery', name='Esri WorldImagery', overlay=False, control=True).add_to(m2)
    folium.CircleMarker(location=[lat, lon], radius=5, color="#ff0000", fill=True, fill_opacity=1.0, tooltip="Fonte emissora").add_to(m2)

    b64 = base64.b64encode(data).decode("utf-8")
    url = f"data:image/png;base64,{b64}"
    folium.raster_layers.ImageOverlay(image=url, bounds=bounds, opacity=opacity, name="Pluma Gaussiana (solo)", zindex=3).add_to(m2)
    folium.LayerControl(collapsed=False).add_to(m2)
    st.markdown(f"### 2) Pluma gerada — Cmax relativo no solo ~ {Cmax:.3e} (g s^-1 m^-3) x constante")
    st.caption("A escala absoluta depende de densidade, altura do receptor e hipoteses do modelo.")

    st_folium(m2, height=720, width=None, use_container_width=True)
else:
    st.info("Aguardando clique...")

st.markdown("---")
st.write("**Notas**")
st.write("Fisica usada (simplificada): pluma gaussiana estacionaria com reflexao no solo; sigmas por Pasquill-Gifford; altura efetiva (Hs + delta h) via Briggs; vento uniforme; terreno plano. Para estudos regulatorios, utilize AERMOD/CALPUFF.")
