
# app_gauss_abs.py
# -*- coding: utf-8 -*-
import io, base64, json, math
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image
import branca

st.set_page_config(page_title="Pluma Gaussiana — escala ABSOLUTA", layout="wide")
st.title("Pluma Gaussiana — escala ABSOLUTA (linear/log), altura & estabilidade")

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
        d_stack = st.number_input("Diâmetro d (m)", 0.05, 10.0, 0.5, 0.05)
        V_exit = st.number_input("Velocidade de saída V (m/s)", 0.1, 120.0, 15.0, 0.1)
        Tamb = st.number_input("Temperatura do ar (K)", 230.0, 330.0, 298.0, 0.5)
        Tstack = st.number_input("Temperatura dos gases (K)", 230.0, 500.0, 320.0, 0.5)

    with st.expander("🖼 Renderização", expanded=True):
        px_per_km = st.select_slider("Resolução (px/km)", [50, 75, 100, 150, 200, 300], value=150)
        opacity = st.slider("Opacidade do overlay", 0.0, 1.0, 0.90, 0.01)
        scale_mode = st.selectbox("Escala de cores", ["Relativa (normalizada)", "Absoluta (linear)", "Absoluta (log10)"], index=1)
        clip_pct = st.slider("Clip relativo (%) — só se modo relativo", 0, 95, 0, 1)

        st.markdown("---")
        st.caption("Faixa absoluta de concentração (g/m³)")
        auto_abs = st.checkbox("Usar Cmax calculado automaticamente", value=True)
        cmin = st.number_input("Cmin (g/m³)", value=0.0, min_value=0.0, format="%.3e")
        cmax_man = st.number_input("Cmax (g/m³) — manual", value=1.0, min_value=1e-12, format="%.3e")

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
    if ss.overlay is not None:
        png_bytes, bounds, meta = ss.overlay
        st.download_button("Baixar PNG", data=png_bytes, file_name="pluma_abs.png", mime="image/png", use_container_width=True)

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

def compute_conc(lat, lon, params):
    # domain (soft-lock ~1 km => 2 km box)
    extent_km = 2.0
    px_per_km = params["px_per_km"]
    R = 6371000.0
    lat_rad = np.deg2rad(lat)
    m_per_deg_lat = np.pi*R/180.0
    m_per_deg_lon = m_per_deg_lat*np.cos(lat_rad)
    half = extent_km*1000/2
    res = max(96, min(3072, int(extent_km*px_per_km)))
    x = np.linspace(-half, half, res); y = np.linspace(-half, half, res)
    X, Y = np.meshgrid(x, y)

    # rotate coords: plume to where it goes (+180)
    theta = np.deg2rad((params["wind_dir"] + 180) % 360)
    Xp =  np.cos(theta)*X + np.sin(theta)*Y
    Yp = -np.sin(theta)*X + np.cos(theta)*Y

    mask = Xp > 0.0
    Xp_eff = np.where(mask, Xp, np.nan)
    sigy, sigz = sigma_yz(np.nan_to_num(Xp_eff, nan=1.0), params["stability"], is_urban=params["is_urban"])

    H_eff = effective_height(params["H_stack"], params["V_exit"], params["d_stack"], params["Tamb"], params["Tstack"], params["wind_speed"])
    u = max(params["wind_speed"], 0.1)
    pref = params["Q_gps"] / (2.0 * np.pi * u * sigy * sigz + 1e-12)
    term_y = np.exp(-0.5 * (Yp**2) / (sigy**2 + 1e-12))
    term_z = np.exp(-0.5 * (H_eff**2 / (sigz**2 + 1e-12))) + np.exp(-0.5 * (H_eff**2 / (sigz**2 + 1e-12)))
    C = pref * term_y * term_z
    C[~mask] = 0.0

    # bounds
    dlat = half / m_per_deg_lat
    dlon = half / (m_per_deg_lon if m_per_deg_lon>0 else 111320.0)
    bounds = [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]
    return C, bounds, float(np.nanmax(C)), float(H_eff)

def render_rgba(C, mode, clip_pct, cmin, cmax):
    lut = jet_colormap()
    A = np.array(C, dtype=float)

    if mode == "Relativa (normalizada)":
        Cmax = A.max() + 1e-15
        N = A / Cmax
        if clip_pct>0 and np.any(N>0):
            thr = np.percentile(N[N>0], clip_pct)
            N = np.clip((N - thr) / (N.max() - thr + 1e-12), 0, 1)
        idx = (N*255).astype(np.uint8)
        alpha = (np.sqrt(N)*255).astype(np.uint8)
        alpha[N<=0.003] = 0

    elif mode == "Absoluta (linear)":
        vmin = float(cmin)
        vmax = float(cmax)
        if vmax <= vmin + 1e-15:
            vmax = vmin + 1e-12
        N = (A - vmin) / (vmax - vmin)
        N = np.clip(N, 0, 1)
        idx = (N*255).astype(np.uint8)
        alpha = (np.sqrt(N)*255).astype(np.uint8)
        alpha[N<=0.003] = 0

    else:  # "Absoluta (log10)"
        vmin = max(float(cmin), 1e-12)
        vmax = max(float(cmax), vmin * 10.0)
        logA = np.log10(np.maximum(A, 1e-12))
        N = (logA - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
        N = np.clip(N, 0, 1)
        idx = (N*255).astype(np.uint8)
        alpha = (N*255).astype(np.uint8)
        alpha[N<=0.02] = 0

    rgb = lut[idx]
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    return rgba

def base_map(center, bounds_fit=None):
    m = folium.Map(location=center, zoom_start=15, tiles=None, control_scale=True)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri — World Imagery', name='ESRI').add_to(m)
    folium.TileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', attr='© OpenStreetMap', name='OSM', show=False).add_to(m)
    if bounds_fit is not None:
        m.fit_bounds(bounds_fit)
    return m

def add_colorbar(m, mode, cmin, cmax):
    if mode == "Relativa (normalizada)":
        cm = branca.colormap.LinearColormap(colors=['blue','cyan','yellow','red'], vmin=0.0, vmax=1.0)
        cm.caption = "Concentração relativa (0–1)"
        cm.add_to(m)
    elif mode == "Absoluta (linear)":
        cm = branca.colormap.LinearColormap(['blue','cyan','yellow','red'], vmin=cmin, vmax=cmax)
        cm.caption = "Concentração (g/m³)"
        cm.add_to(m)
    else:  # log
        # Log colorbar: show ticks at 10^k
        ticks = []
        if cmin <= 0: cmin = 1e-12
        kmin = math.floor(math.log10(cmin))
        kmax = math.ceil(math.log10(max(cmax, cmin*10)))
        for k in range(kmin, kmax+1):
            ticks.append(10**k)
        cm = branca.colormap.LinearColormap(['blue','cyan','yellow','red'], vmin=np.log10(cmin), vmax=np.log10(max(cmax, cmin*10)))
        cm.caption = "Concentração (g/m³) [escala log]"
        cm.add_to(m)

# ---- Flow ----
if ss.source is None or not ss.locked:
    st.info("Clique no mapa para definir a fonte. (Depois a vista fica ~1 km ao redor)")
    m0 = base_map([-22.9035, -43.2096])
    state = st_folium(m0, height=720, returned_objects=["last_clicked"], use_container_width=True)
    if state and state.get("last_clicked"):
        ss.source = (state["last_clicked"]["lat"], state["last_clicked"]["lng"])
        ss.locked = True
        ss._update = True
        st.success(f"Fonte: {ss.source[0]:.6f}, {ss.source[1]:.6f}")
else:
    lat, lon = ss.source
    params = dict(
        wind_dir=wind_dir, wind_speed=wind_speed, stability=stability, is_urban=is_urban,
        Q_gps=Q_gps, H_stack=H_stack, d_stack=d_stack, V_exit=V_exit, Tamb=Tamb, Tstack=Tstack,
        px_per_km=px_per_km
    )

    if ss._update or ss.overlay is None:
        C, bounds, Cmax, H_eff = compute_conc(lat, lon, params)
        # decide cmax auto
        if scale_mode.startswith("Absoluta") and (auto_abs or cmax_man is None):
            vmax_use = Cmax
        else:
            vmax_use = cmax_man
        rgba = render_rgba(C, scale_mode, clip_pct, cmin, vmax_use)
        im = Image.fromarray(rgba, mode="RGBA")
        bio = io.BytesIO(); im.save(bio, format="PNG"); bio.seek(0)
        ss.overlay = (bio.read(), bounds, {"Cmax": Cmax, "H_eff": H_eff, "cmin": cmin, "cmax_used": vmax_use, "scale_mode": scale_mode})
        ss._update = False

    png_bytes, bounds, meta = ss.overlay
    st.markdown(f"**H efetiva**: {meta['H_eff']:.1f} m  |  **Cmax** ~ {meta['Cmax']:.3e} g/m³  |  Escala: {meta['scale_mode']} (cmin={cmin:.1e}, cmax={meta['cmax_used']:.1e})")

    m1 = base_map([lat, lon], bounds_fit=bounds)
    folium.CircleMarker([lat, lon], radius=6, color="#ff0000", fill=True, fill_opacity=1.0, tooltip="Fonte").add_to(m1)
    url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")
    folium.raster_layers.ImageOverlay(image=url, bounds=bounds, opacity=opacity, name="Pluma").add_to(m1)
    add_colorbar(m1, scale_mode, cmin, meta['cmax_used'])
    folium.LayerControl(collapsed=False).add_to(m1)
    st_folium(m1, height=720, use_container_width=True)

st.markdown('---')
st.caption("Agora a intensidade visual muda com Q: use 'Absoluta (linear)' ou 'Absoluta (log10)'.")
