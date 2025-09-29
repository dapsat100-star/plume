
# app_gauss_robusto.py
# -*- coding: utf-8 -*-
import io, base64, json
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image

st.set_page_config(page_title="Pluma Gaussiana — robusto", layout="wide")
st.title("Pluma Gaussiana — robusto (altura, estabilidade, lock ~1 km)")

# ---- Sidebar scroll CSS ----
st.markdown('''
<style>
[data-testid="stSidebar"] { overflow-y: auto; max-height: 100vh; }
</style>
''', unsafe_allow_html=True)

# ---- Session state ----
ss = st.session_state
ss.setdefault("source", None)        # (lat, lon)
ss.setdefault("overlay", None)       # (png_bytes, bounds, meta)
ss.setdefault("_update", False)
ss.setdefault("locked", False)

# ---- Sidebar ----
with st.sidebar:
    st.header("Parâmetros")

    with st.expander("🌦 Meteorologia", expanded=True):
        wind_dir = st.number_input("Direção do vento (graus de onde VEM)", 0, 359, 45, 1)
        wind_speed = st.number_input("Velocidade do vento (m/s)", 0.1, 50.0, 5.0, 0.1, format="%.1f")
        stability = st.selectbox("Classe de estabilidade (Pasquill-Gifford)", ["A","B","C","D","E","F"], index=3)
        is_urban = st.checkbox("Condição urbana (↑ dispersão transversal)", value=False)

    with st.expander("🏭 Fonte / Chaminé", expanded=False):
        Q_gps = st.number_input("Taxa de emissão Q (g/s)", 0.001, 1e9, 100.0, 0.1, format="%.3f")
        H_stack = st.number_input("Altura geométrica Hs (m)", 0.0, 500.0, 10.0, 0.5)
        d_stack = st.number_input("Diâmetro da chaminé d (m)", 0.05, 10.0, 0.5, 0.05)
        V_exit = st.number_input("Velocidade de saída V (m/s)", 0.1, 120.0, 15.0, 0.1)
        Tamb = st.number_input("Temperatura do ar (K)", 230.0, 330.0, 298.0, 0.5)
        Tstack = st.number_input("Temperatura dos gases (K)", 230.0, 500.0, 320.0, 0.5)

    with st.expander("🖼 Renderização", expanded=False):
        px_per_km = st.select_slider("Resolução (px/km)", [50, 75, 100, 150, 200, 300], value=150)
        opacity = st.slider("Opacidade do overlay", 0.0, 1.0, 0.85, 0.01)
        clip_pct = st.slider("Corte inferior da paleta (%)", 0, 95, 0, 1)
        extent_km_unlocked = st.slider("Extensão simulada (km) — quando DEStravado", 1.0, 20.0, 5.0, 0.5)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Atualizar", type="primary", use_container_width=True):
        ss._update = True
    if c2.button("Selecionar novo ponto", use_container_width=True):
        ss.source = None; ss.overlay = None; ss.locked = False
    if c3.button("Overlay de teste", use_container_width=True):
        if ss.source is None:
            ss.source = (-22.9035, -43.2096); ss.locked = True
        ss._update = True
    # Downloads depois que houver overlay
    if ss.overlay is not None:
        png_bytes, bounds, meta = ss.overlay
        col = st.container()
        with col:
            st.download_button("Baixar PNG", data=png_bytes, file_name="pluma.png", mime="image/png", use_container_width=True)
            # Worldfile (.pgw) para georreferenciar o PNG em WGS84 (aprox equiretangular local)
            # worldfile ordem: pixel size in X, rotation, rotation, pixel size in Y (negativo), top-left X, top-left Y
            (lat0, lon0), (lat1, lon1) = (bounds[0], bounds[1])
            width_px, height_px = Image.open(io.BytesIO(png_bytes)).size
            pixelSizeX = (lon1 - lon0) / width_px
            pixelSizeY = -(lat1 - lat0) / height_px
            topLeftX = lon0
            topLeftY = lat1
            pgw = f"{pixelSizeX}\n0.0\n0.0\n{pixelSizeY}\n{topLeftX}\n{topLeftY}\n"
            st.download_button("Baixar PGW (worldfile)", data=pgw.encode("utf-8"), file_name="pluma.pgw", mime="text/plain", use_container_width=True)
            st.download_button("Baixar bounds.json", data=json.dumps({"bounds": bounds, "crs": "EPSG:4326"}).encode("utf-8"), file_name="bounds.json", mime="application/json", use_container_width=True)

# ---- Model helpers ----
def jet_colormap():
    from matplotlib import cm
    cols = (cm.get_cmap('jet', 256)(np.linspace(0,1,256))[:,:3] * 255).astype(np.uint8)
    return cols

def sigma_yz(x_m, stability, is_urban=False):
    x_km = np.maximum(x_m/1000.0, 1e-6)
    s = stability.upper()
    if not is_urban:
        coefs = {"A": (0.22, 0.5, 0.20, 0.5),"B": (0.16, 0.5, 0.12, 0.5),
                 "C": (0.11, 0.5, 0.08, 0.5),"D": (0.08, 0.5, 0.06, 0.5),
                 "E": (0.06, 0.5, 0.03, 0.5),"F": (0.04, 0.5, 0.016,0.5)}
    else:
        coefs = {"A": (0.32, 0.5, 0.24, 0.5),"B": (0.22, 0.5, 0.18, 0.5),
                 "C": (0.16, 0.5, 0.14, 0.5),"D": (0.12, 0.5, 0.10, 0.5),
                 "E": (0.10, 0.5, 0.06, 0.5),"F": (0.08, 0.5, 0.04, 0.5)}
    a, by, c, bz = coefs.get(s, coefs["D"])
    sigy = np.clip(a * (x_km ** by) * 1000.0, 1.0, None)
    sigz = np.clip(c * (x_km ** bz) * 1000.0, 1.0, None)
    return sigy, sigz

def effective_height(H_stack, V_exit, d_stack, Tamb, Tstack, wind_speed):
    g = 9.80665
    F = g * V_exit * (d_stack**2) * (Tstack - Tamb) / (4.0 * Tstack + 1e-9)
    delta_m = 3.0 * d_stack * V_exit / max(wind_speed, 0.1)
    delta_b = 2.6 * (F ** (1.0/3.0)) / max(wind_speed, 0.1)
    return H_stack + max(delta_m, delta_b, 0.0)

def make_overlay(lat, lon, wind_dir, wind_speed, Q_gps, stability, is_urban, clip_pct, px_per_km, locked=True, extent_km_unlocked=5.0):
    # domain
    extent_km = 2.0 if locked else extent_km_unlocked
    R = 6371000.0
    lat_rad = np.deg2rad(lat)
    m_per_deg_lat = np.pi*R/180.0
    m_per_deg_lon = m_per_deg_lat*np.cos(lat_rad)
    half = extent_km*1000/2
    res = max(96, min(3072, int(extent_km*px_per_km)))
    x = np.linspace(-half, half, res); y = np.linspace(-half, half, res)
    X, Y = np.meshgrid(x, y)

    # rotate coordinates (plume to where it goes: +180)
    theta = np.deg2rad((wind_dir + 180) % 360)
    Xp =  np.cos(theta)*X + np.sin(theta)*Y
    Yp = -np.sin(theta)*X + np.cos(theta)*Y

    # Pasquill-Gifford sigmas vs distance downwind
    mask = Xp > 0.0
    Xp_eff = np.where(mask, Xp, np.nan)
    sigy, sigz = sigma_yz(np.nan_to_num(Xp_eff, nan=1.0), stability, is_urban=is_urban)

    # ground-level (z=0) with ground reflection
    H_eff = effective_height(H_stack, V_exit, d_stack, Tamb, Tstack, wind_speed)
    u = max(wind_speed, 0.1)
    pref = Q_gps / (2.0 * np.pi * u * sigy * sigz + 1e-12)
    term_y = np.exp(-0.5 * (Yp**2) / (sigy**2 + 1e-12))
    term_z = np.exp(-0.5 * ((0.0 - H_eff)**2) / (sigz**2 + 1e-12)) + np.exp(-0.5 * ((0.0 + H_eff)**2) / (sigz**2 + 1e-12))
    C = pref * term_y * term_z
    C[~mask] = 0.0

    Cmax = float(np.nanmax(C))
    Cn = C / (Cmax + 1e-15)
    if clip_pct>0 and np.any(Cn>0):
        thr = np.nanpercentile(Cn[Cn>0], clip_pct)
        Cn = np.clip((Cn - thr) / (Cn.max() - thr + 1e-12), 0, 1)

    lut = jet_colormap()
    idx = (Cn*255).astype(np.uint8)
    rgb = lut[idx]
    alpha = (np.sqrt(Cn)*255).astype(np.uint8)
    alpha[Cn<=0.003] = 0
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)

    # bounds
    dlat = half / m_per_deg_lat
    dlon = half / (m_per_deg_lon if m_per_deg_lon>0 else 111320.0)
    bounds = [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]

    im = Image.fromarray(rgba, mode="RGBA")
    bio = io.BytesIO(); im.save(bio, format="PNG"); bio.seek(0)
    meta = {"Cmax_rel_g_s_m3": Cmax, "H_eff_m": H_eff, "extent_km": extent_km, "px_per_km": px_per_km}
    return bio.read(), bounds, meta

def base_map(center, bounds_fit=None):
    m = folium.Map(location=center, zoom_start=15, tiles=None, control_scale=True)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri — World Imagery', name='ESRI').add_to(m)
    folium.TileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', attr='© OpenStreetMap', name='OSM', show=False).add_to(m)
    if bounds_fit is not None:
        m.fit_bounds(bounds_fit)
    return m

# ---- Flow ----
if ss.source is None or not ss.locked:
    st.info("Clique no mapa para definir a fonte. (Depois a vista fica 'travada' num raio ~1 km)")
    m0 = base_map([-22.9035, -43.2096])
    state = st_folium(m0, height=720, returned_objects=["last_clicked"], use_container_width=True)
    if state and state.get("last_clicked"):
        ss.source = (state["last_clicked"]["lat"], state["last_clicked"]["lng"])
        ss.locked = True
        ss._update = True
        st.success(f"Fonte: {ss.source[0]:.6f}, {ss.source[1]:.6f}")
else:
    lat, lon = ss.source
    if ss._update or ss.overlay is None:
        png, bounds, meta = make_overlay(lat, lon, wind_dir, wind_speed, Q_gps, stability, is_urban, clip_pct, px_per_km, locked=True, extent_km_unlocked=extent_km_unlocked)
        ss.overlay = (png, bounds, meta)
        ss._update = False

    png, bounds, meta = ss.overlay
    st.markdown(f"**Altura efetiva (estimada)**: {meta['H_eff_m']:.1f} m  |  **Cmax relativo** ~ {meta['Cmax_rel_g_s_m3']:.3e} (g s^-1 m^-3) × const  |  Extensão: {meta['extent_km']:.1f} km")
    # fit to the overlay box (2 km) as soft-lock
    m1 = base_map([lat, lon], bounds_fit=bounds)
    folium.CircleMarker([lat, lon], radius=6, color="#ff0000", fill=True, fill_opacity=1.0, tooltip="Fonte").add_to(m1)
    url = "data:image/png;base64," + base64.b64encode(png).decode("utf-8")
    folium.raster_layers.ImageOverlay(image=url, bounds=bounds, opacity=opacity, name="Pluma").add_to(m1)
    folium.LayerControl(collapsed=False).add_to(m1)
    st_folium(m1, height=720, use_container_width=True)

st.markdown('---')
st.caption("Este app evita recursos JS que podem falhar em alguns deploys. O 'lock' é via ajuste da vista (soft lock).")
