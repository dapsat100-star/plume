# app.py — ppb, 25 m/pixel, legenda fixa à direita (MacroElement robusto) + Q em kg/h
# -*- coding: utf-8 -*-
import io, base64
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image
from folium import MacroElement
from jinja2 import Template

st.set_page_config(page_title="Pluma Gaussiana — ppb (25 m/pixel, kg/h)", layout="wide")
st.title("Pluma Gaussiana — ABSOLUTA (ppb), 25 m/pixel, emissão em kg CH₄/h")

# Sidebar scroll CSS
st.markdown("""
<style>
[data-testid="stSidebar"] { overflow-y: auto; max-height: 100vh; }
</style>
""", unsafe_allow_html=True)

# Session
ss = st.session_state
ss.setdefault("source", None)
ss.setdefault("overlay", None)
ss.setdefault("_update", False)
ss.setdefault("locked", False)

# Constantes
R_univ = 8.314462618  # J/mol/K
M_CH4  = 16.043       # g/mol
PX_PER_KM_FIXED = 40  # 25 m/pixel

# Sidebar
with st.sidebar:
    st.header("Parâmetros")

    with st.expander("🌦 Meteorologia", expanded=True):
        wind_dir   = st.number_input("Direção do vento (° de onde VEM)", 0, 359, 45, 1)
        wind_speed = st.number_input("Velocidade do vento (m/s)", 0.1, 50.0, 5.0, 0.1)
        stability  = st.selectbox("Classe de estabilidade", ["A","B","C","D","E","F"], index=3)
        is_urban   = st.checkbox("Condição urbana", value=False)
        P_hPa      = st.number_input("Pressão (hPa)", 800.0, 1050.0, 1013.25, 0.5)
        Tamb       = st.number_input("Temperatura (K)", 230.0, 330.0, 298.0, 0.5)

    with st.expander("🏭 Fonte / Chaminé", expanded=True):
        Q_kgph   = st.number_input("Taxa de emissão Q (kg CH₄/h)", 0.001, 1e6, 100.0, 0.1, help="100 kg/h ≈ 27,78 g/s")
        H_stack  = st.number_input("Altura geométrica (m)", 0.0, 500.0, 10.0, 0.5)
        d_stack  = st.number_input("Diâmetro (m)", 0.05, 10.0, 0.5, 0.05)
        V_exit   = st.number_input("Velocidade de saída (m/s)", 0.1, 120.0, 15.0, 0.1)
        Tstack   = st.number_input("Temperatura dos gases (K)", 230.0, 500.0, 320.0, 0.5)

    with st.expander("🖼 Renderização", expanded=True):
        st.markdown("**Resolução espacial:** `25 m/pixel` (fixa)")
        opacity    = st.slider("Opacidade do overlay", 0.0, 1.0, 0.90, 0.01)
        scale_mode = st.selectbox("Escala de cores", ["Absoluta (linear)", "Absoluta (log10)", "Relativa (normalizada)"], index=0)
        clip_pct   = st.slider("Clip relativo (%) — só no modo relativo", 0, 95, 0, 1)
        st.caption("Legenda fixa: 0, 150, 300, 450 ppb")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    if c1.button("Atualizar", type="primary", use_container_width=True):
        ss._update = True
    if c2.button("Selecionar novo ponto", use_container_width=True):
        ss.source = None; ss.overlay = None; ss.locked = False
    if c3.button("Overlay de teste", use_container_width=True):
        if ss.source is None:
            ss.source = (-22.9035, -43.2096); ss.locked = True
        ss._update = True

# Modelo
def sigma_yz(x_m, stability, is_urban=False):
    x_km = np.maximum(x_m/1000.0, 1e-6)
    s = stability.upper()
    if not is_urban:
        coefs = {"A":(0.22,0.5,0.20,0.5),"B":(0.16,0.5,0.12,0.5),
                 "C":(0.11,0.5,0.08,0.5),"D":(0.08,0.5,0.06,0.5),
                 "E":(0.06,0.5,0.03,0.5),"F":(0.04,0.5,0.016,0.5)}
    else:
        coefs = {"A":(0.32,0.5,0.24,0.5),"B":(0.22,0.5,0.18,0.5),
                 "C":(0.16,0.5,0.14,0.5),"D":(0.12,0.5,0.10,0.5),
                 "E":(0.10,0.5,0.06,0.5),"F":(0.08,0.5,0.04,0.5)}
    a, by, c, bz = coefs.get(s, coefs["D"])
    sigy = np.clip(a*(x_km**by)*1000.0, 1.0, None)
    sigz = np.clip(c*(x_km**bz)*1000.0, 1.0, None)
    return sigy, sigz

def effective_height(H, V, d, Tamb, Tstack, u):
    g = 9.80665
    F = g*V*(d**2)*(Tstack - Tamb)/(4.0*Tstack+1e-9)
    delta_m = 3*d*V/max(u,0.1)
    delta_b = 2.6*(F**(1/3))/max(u,0.1)
    return H + max(delta_m, delta_b, 0.0)

def compute_conc(lat, lon, p):
    extent_km = 2.0
    px_per_km = PX_PER_KM_FIXED
    R = 6371000.0
    lat_rad = np.deg2rad(lat)
    m_lat = np.pi*R/180
    m_lon = m_lat*np.cos(lat_rad)
    half = extent_km*1000/2
    res = int(extent_km*px_per_km)
    x = np.linspace(-half, half, res); y = np.linspace(-half, half, res)
    X,Y = np.meshgrid(x,y)

    theta = np.deg2rad((p["wind_dir"]+180)%360)  # para onde vai
    Xp = np.cos(theta)*X + np.sin(theta)*Y
    Yp = -np.sin(theta)*X + np.cos(theta)*Y
    mask = Xp>0
    sigy, sigz = sigma_yz(np.where(mask,Xp,1), p["stability"], p["is_urban"])
    H_eff = effective_height(p["H"], p["V"], p["d"], p["Tamb"], p["Tstack"], p["u"])
    pref = p["Q_gps"]/(2*np.pi*p["u"]*sigy*sigz+1e-12)
    C = pref*np.exp(-0.5*(Yp**2)/sigy**2)*(np.exp(-0.5*(H_eff**2)/sigz**2)+np.exp(-0.5*(H_eff**2)/sigz**2))
    C[~mask] = 0

    dlat = half/m_lat; dlon = half/m_lon if m_lon>0 else half/111320.0
    bounds = [[lat-dlat, lon-dlon],[lat+dlat, lon+dlon]]
    return C, bounds

def to_ppb(C, P_hPa, T_K):
    P_pa = P_hPa*100.0
    factor = (R_univ*T_K)/(M_CH4*P_pa)*1e9
    return C*factor

def render_ppb(A, vmin=0, vmax=450, log=False):
    from matplotlib import cm
    lut = (cm.get_cmap('jet',256)(np.linspace(0,1,256))[:,:3]*255).astype(np.uint8)
    if log:
        A = np.log10(np.maximum(A,1e-12))
        vmin, vmax = np.log10(vmin+1e-12), np.log10(vmax)
    N = np.clip((A - vmin)/(vmax - vmin + 1e-12), 0, 1)
    idx = (N*255).astype(np.uint8)
    alpha = (np.sqrt(N)*255).astype(np.uint8); alpha[N<=0.003]=0
    return np.dstack([lut[idx], alpha]).astype(np.uint8)

# --- Legenda robusta (MacroElement) ---
LEGEND_TEMPLATE = Template("""
{% macro html(this, kwargs) %}
<div style="position:absolute; top:80px; right:10px; width:36px; height:340px; z-index:10000;">
  <div style="position:relative; width:36px; height:340px;">
    <div style="position:absolute; right:8px; top:10px; width:16px; height:300px;
                background:linear-gradient(to top, purple, blue, cyan, green, yellow, red);
                border-radius:6px; box-shadow:0 0 6px rgba(0,0,0,0.3);"></div>
    <div style="position:absolute; right:4px; top:-6px; font-size:12px; font-weight:600;
                background:rgba(255,255,255,0.85); padding:1px 4px; border-radius:4px;">[ppb]</div>
    <div style="position:absolute; right:30px; top:310px;">0</div>
    <div style="position:absolute; right:30px; top:210px;">150</div>
    <div style="position:absolute; right:30px; top:110px;">300</div>
    <div style="position:absolute; right:30px; top:10px;">450</div>
  </div>
</div>
{% endmacro %}
""")

class RightLegend(MacroElement):
    def __init__(self):
        super().__init__()
        self._template = LEGEND_TEMPLATE

# Badge de resolução
BADGE_TEMPLATE = Template("""
{% macro html(this, kwargs) %}
<div style="position:absolute; top:430px; right:10px; z-index:10000;
            background:rgba(255,255,255,0.85); padding:2px 6px; border-radius:6px;
            font-size:12px; font-weight:600; color:#333; box-shadow:0 0 4px rgba(0,0,0,0.2);">
  Resolução: 25 m/pixel · Emissão em kg/h
</div>
{% endmacro %}
""")

class ResolutionBadge(MacroElement):
    def __init__(self):
        super().__init__()
        self._template = BADGE_TEMPLATE

# Fluxo
if ss.source is None or not ss.locked:
    st.info("Clique no mapa para definir a fonte.")
    m0 = folium.Map(location=[-22.9035, -43.2096], zoom_start=15)
    state = st_folium(m0, height=720, returned_objects=["last_clicked"], use_container_width=True)
    if state and state.get("last_clicked"):
        ss.source = (state["last_clicked"]["lat"], state["last_clicked"]["lng"])
        ss.locked = True
        ss._update = True
else:
    lat, lon = ss.source
    Q_gps = (Q_kgph * 1000.0) / 3600.0  # kg/h -> g/s
    p = dict(
        wind_dir=wind_dir, wind_speed=wind_speed, stability=stability, is_urban=is_urban,
        Q_gps=Q_gps, H=H_stack, d=d_stack, V=V_exit, Tamb=Tamb, Tstack=Tstack, u=wind_speed
    )

    if ss._update or ss.overlay is None:
        C, bounds = compute_conc(lat, lon, p)
        Cppb = to_ppb(C, P_hPa, Tamb)
        rgba = render_ppb(Cppb, 0, 450, log=(scale_mode=="Absoluta (log10)"))
        im = Image.fromarray(rgba, "RGBA")
        bio = io.BytesIO(); im.save(bio, "PNG"); bio.seek(0)
        ss.overlay = (bio.read(), bounds)
        ss._update = False

    png_bytes, bounds = ss.overlay
    m1 = folium.Map(location=[lat, lon], zoom_start=15)
    folium.raster_layers.ImageOverlay(
        image="data:image/png;base64,"+base64.b64encode(png_bytes).decode("utf-8"),
        bounds=bounds, opacity=opacity, name="Pluma").add_to(m1)
    folium.CircleMarker([lat,lon], radius=6, color="#f00", fill=True).add_to(m1)

    # adiciona legenda e badge como MacroElement (robusto em iframe)
    m1.get_root().add_child(RightLegend())
    m1.get_root().add_child(ResolutionBadge())

    folium.LayerControl(collapsed=False).add_to(m1)
    st_folium(m1, height=720, use_container_width=True)
