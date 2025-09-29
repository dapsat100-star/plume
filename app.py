# app.py — Pluma CH4 + Footprint GHGSat 5x5 km (TLE de arquivo) — clique simples + anti-rerun
# -*- coding: utf-8 -*-
import io, base64, os
import numpy as np
import streamlit as st
import folium
from folium.plugins import Draw, MeasureControl
try:
    from folium.plugins import ScaleBar
except Exception:
    ScaleBar = None
# Geocoder (busca estilo Google Earth)
try:
    from folium.plugins import Geocoder
except Exception:
    Geocoder = None
# Seta de direção (texto ao longo da linha)
try:
    from folium.plugins import PolyLineTextPath as PolyArrow
except Exception:
    PolyArrow = None
# Marcador triangular (fallback)
try:
    from folium.features import RegularPolygonMarker
except Exception:
    RegularPolygonMarker = None
from streamlit_folium import st_folium
from PIL import Image
import matplotlib
# CSS dentro do iframe do Folium
try:
    from branca.element import Element
except Exception:
    Element = None
# Requisições HTTP (para logo via URL/GitHub raw)
try:
    import requests
except Exception:
    requests = None
from geopy.geocoders import Nominatim, Photon, ArcGIS
import datetime as dt

# ================== CONFIG ==================
st.set_page_config(page_title="Pluma CH₄ + GHGSat Footprint (TLE do arquivo)", layout="wide")
st.title("Pluma Gaussiana (CH₄) · 25 m/pixel · ppb 0–450 + Footprint GHGSat 5×5 km (via TLE)")

st.markdown(
    """
<style>
/* Sidebar layout */
[data-testid="stSidebar"] { overflow-y: auto; max-height: 100vh; }
.stButton>button { height: 40px; }

/* ⬆️ Aumentar fonte dos campos de BUSCA (sidebar) */
[data-testid="stSidebar"] .stTextInput input {
  font-size: 1.15rem !important;   /* ~18px */
  height: 48px !important;
}
[data-testid="stSidebar"] .stTextInput input::placeholder {
  font-size: 1.05rem !important;
  opacity: 0.85;
}

/* ⬆️ Aumentar fonte do Geocoder no mapa (Leaflet) */
.leaflet-control-geocoder,
.leaflet-control-geocoder * {
  font-size: 20px !important;  /* era 16px */
}
.leaflet-control-geocoder-form input {
  font-size: 20px !important;
  height: 48px !important;
  padding: 10px 14px !important;
}
.leaflet-control-geocoder-form input::placeholder {
  font-size: 20px !important;
}
/* Resultados (autocomplete) */
.leaflet-control-geocoder-alternatives {
  font-size: 18px !important;  /* itens da lista */
}
.leaflet-control-geocoder-alternatives li a {
  line-height: 1.35 !important;
  padding: 8px 10px !important;
}

/* 🔖 Logo fixo (topo direito) */
.branding-fixed {
  position: fixed;
  top: 8px;
  right: 16px;
  z-index: 1000;
  pointer-events: none; /* não intercepta cliques */
}
.branding-fixed img { max-height: 56px; height: auto; }
@media (max-width: 768px) {
  .branding-fixed img { max-height: 44px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ================== ESTADO (antes de qualquer uso!) ==================
ss = st.session_state
ss.setdefault("source", None)
ss.setdefault("pending_click", None)
ss.setdefault("overlay", None)
ss.setdefault("_update", False)
ss.setdefault("locked", False)
ss.setdefault("tle_cache", {})
ss.setdefault("tle_path_loaded", "")
ss.setdefault("search_results", [])
ss.setdefault("logo_bytes", None)
ss.setdefault("logo_w", 140)

# Tenta pré-carregar logo padrão do repo, se existir
DEFAULT_LOGO_PATH = "images/logomavipe.jpeg"
if ss.get("logo_bytes") is None and os.path.exists(DEFAULT_LOGO_PATH):
    try:
        with open(DEFAULT_LOGO_PATH, "rb") as _f:
            ss.logo_bytes = _f.read()
    except Exception:
        pass
# NÃO defina/atribua ss["tle_choice"] depois que o widget com key="tle_choice" existir

# defaults ESTÁVEIS para data/hora (evita loop com utcnow)
if "obs_date" not in ss or "obs_time" not in ss:
    _now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    ss.obs_date = _now.date()
    ss.obs_time = _now.time()

# ================== GEOCODING (opcional) ==================
@st.cache_resource
def _geocoder():
    return Nominatim(user_agent="plume_streamlit_app")

@st.cache_data(show_spinner=False, ttl=3600)
def reverse_geocode(lat, lon):
    try:
        g = _geocoder()
        loc = g.reverse((lat, lon), language="pt", zoom=18, exactly_one=True, timeout=5)
        return loc.address if loc else None
    except Exception:
        return None

# Baixar logo por URL (ex.: GitHub raw)
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_logo_url(url: str) -> bytes | None:
    try:
        if not url:
            return None
        if requests is None:
            return None
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

# ---- Geocoders extra para AUTOCOMPLETE ----
@st.cache_resource
def _geocoder_photon():
    try:
        return Photon(user_agent="plume_streamlit_app")
    except Exception:
        return None

@st.cache_resource
def _geocoder_arcgis():
    try:
        return ArcGIS(timeout=5)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=1800)
def geocode_multi(q: str, limit: int = 8):
    """Busca multi-provedor (Photon → Nominatim → ArcGIS) com deduplicação.
    Retorna lista de dicts: {label, lat, lon, provider}. Aceita também "lat,lon" direto."""
    results = []
    qs = (q or "").strip()
    # coordenadas diretas
    try:
        if "," in qs:
            a,b = qs.split(",",1)
            lat = float(a.strip()); lon = float(b.strip())
            results.append({"label": f"{lat:.6f}, {lon:.6f}", "lat": lat, "lon": lon, "provider": "manual"})
    except Exception:
        pass
    # Photon
    try:
        gp = _geocoder_photon()
        if gp:
            locs = gp.geocode(qs, exactly_one=False, limit=limit, timeout=5)
            if locs:
                for L in locs:
                    results.append({"label": getattr(L, "address", qs), "lat": L.latitude, "lon": L.longitude, "provider": "photon"})
    except Exception:
        pass
    # Nominatim
    try:
        gn = _geocoder()
        if gn:
            remain = max(0, limit - len(results))
            if remain:
                locs = gn.geocode(qs, exactly_one=False, limit=remain, timeout=5)
                if locs:
                    for L in locs:
                        results.append({"label": getattr(L, "address", qs), "lat": L.latitude, "lon": L.longitude, "provider": "nominatim"})
    except Exception:
        pass
    # ArcGIS
    try:
        ga = _geocoder_arcgis()
        if ga:
            remain = max(0, limit - len(results))
            if remain:
                locs = ga.geocode(qs, out_fields='*', maxRows=remain)
                if locs:
                    iterlocs = locs if isinstance(locs, list) else [locs]
                    for L in iterlocs:
                        results.append({"label": getattr(L, "address", qs), "lat": L.latitude, "lon": L.longitude, "provider": "arcgis"})
    except Exception:
        pass
    # dedup
    seen = set(); dedup = []
    for r in results:
        key = (round(r["lat"],5), round(r["lon"],5), r["label"].lower())
        if key in seen: continue
        seen.add(key); dedup.append(r)
    return dedup[:limit]

# ================== CONSTANTES ==================
PX_PER_KM_FIXED = 40            # 25 m/pixel
V_ABS_MIN, V_ABS_MAX = 0.0, 450.0
FOOTPRINT_SIZE_KM = 5.0         # 5×5 km

# ================== SIDEBAR ==================
with st.sidebar:
    # Logo no topo da sidebar (canto superior esquerdo)
    try:
        if ss.get("logo_bytes"):
            _b64 = base64.b64encode(ss.logo_bytes).decode("utf-8")
            _w = int(ss.get("logo_w", 140))
            st.markdown(
                f"<img src='data:image/png;base64,{_b64}' style='width:{_w}px; margin:6px 0 12px 6px;' />",
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    st.header("Parâmetros")

    with st.expander("🌦 Meteorologia", expanded=True):
        wind_dir   = st.number_input("Direção do vento (° de onde VEM)", 0, 359, 45, 1)
        wind_speed = st.number_input("Velocidade do vento (m/s)", 0.1, 50.0, 5.0, 0.1)
        stability  = st.selectbox("Classe de estabilidade (Pasquill–Gifford)", ["A","B","C","D","E","F"], index=3)
        is_urban   = st.checkbox("Condição urbana (↑ σᵧ/σ𝑧)", value=False)
        P_hPa      = st.number_input("Pressão (hPa)", 800.0, 1050.0, 1013.25, 0.5)
        Tamb       = st.number_input("Temperatura do ar (K)", 230.0, 330.0, 298.0, 0.5)

    with st.expander("🏭 Fonte / Chaminé", expanded=True):
        Q_kgph   = st.number_input("Taxa de emissão Q (kg CH₄/h)", 0.001, 1e9, 100.0, 0.1, help="100 kg/h ≈ 27,78 g/s")
        H_stack  = st.number_input("Altura geométrica Hs (m)", 0.0, 500.0, 10.0, 0.5)
        d_stack  = st.number_input("Diâmetro d (m)", 0.05, 10.0, 0.5, 0.05)
        V_exit   = st.number_input("Vel. de saída V (m/s)", 0.1, 120.0, 15.0, 0.1)
        Tstack   = st.number_input("Temp. dos gases (K)", 230.0, 500.0, 320.0, 0.5)

    with st.expander("🖼 Renderização", expanded=True):
        st.markdown("**Resolução:** `25 m/pixel` (fixa)")
        opacity    = st.slider("Opacidade do overlay", 0.0, 1.0, 0.90, 0.01)
        scale_mode = st.selectbox("Escala de cores", ["Absoluta (linear)", "Absoluta (log10)"], index=0)
        st.caption("Faixa absoluta fixa: **0 · 150 · 300 · 450 ppb**")

    with st.expander("🛰 GHGSat — TLE (do arquivo)", expanded=True):
        def load_tle_file(path):
            sats = {}
            if not os.path.exists(path):
                return sats
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            i = 0
            while i + 2 < len(lines):
                name, l1, l2 = lines[i], lines[i+1], lines[i+2]
                if l1.startswith("1 ") and l2.startswith("2 "):
                    sats[name] = (l1, l2); i += 3
                else:
                    i += 1
            return sats

        tle_path = st.text_input(
            "Caminho do arquivo TLE no repo",
            value="data/ghgsat.tle",
            help="Formato: blocos Nome / Linha1 / Linha2.",
        )
        reload_tle = st.button("Recarregar TLE")

        # (1) Recarrega cache se mudou o caminho ou clicou no botão
        if reload_tle or (ss.tle_path_loaded != tle_path):
            ss.tle_cache = load_tle_file(tle_path)
            ss.tle_path_loaded = tle_path
            # invalida escolha anterior para recalcular default ANTES de instanciar o selectbox
            ss.pop("tle_choice", None)

        options = list(ss.tle_cache.keys())
        if not options:
            st.error("Arquivo TLE não encontrado ou vazio. Coloque um arquivo no formato Nome/L1/L2.")
        else:
            # (2) Define default em session_state **antes** de criar o widget
            if "tle_choice" not in ss or ss.tle_choice not in options:
                ss.tle_choice = options[0]
            # (3) Cria o widget sem atribuições posteriores ao mesmo key
            st.selectbox(
                "Satélite",
                options=options,
                index=options.index(ss.tle_choice),
                key="tle_choice",
            )

        # Data/hora de observação
        obs_date = st.date_input("Data (UTC)", value=ss.obs_date, key="obs_date")
        obs_time = st.time_input("Hora (UTC)", value=ss.obs_time, key="obs_time")

        # Forçar modo automático: não pedir horas/controles ao usuário
        ad_hours = 0   # deixa o algoritmo expandir sozinho (72→120→168→240 h)
        sep_deg = 160  # separação mínima entre headings (quase opostos)

        # (sem controles) — busca automática do par ASC/DESC à frente",
            value=True,
            help="Procura o primeiro caso ascendente e o primeiro descendente em ±90 min do horário acima.",
        )

    # 🔎 Busca com AUTOCOMPLETE (Photon/Nominatim/ArcGIS)
    with st.expander("🔎 Buscar lugar (autocomplete)", expanded=False):
        q = st.text_input("Digite um lugar, endereço ou lat,lon", key="query_text", placeholder="Ex.: 'Cabiúnas, Macaé' ou -22.91,-41.42")
        do_search = st.button("Procurar", key="btn_search")
        if do_search and q:
            ss.search_results = geocode_multi(q, limit=8)
        results = ss.get("search_results", [])
        if results:
            labels = [f"{r['label']} — {r['provider']}" for r in results]
            idx = st.selectbox("Resultados", list(range(len(results))), format_func=lambda i: labels[i], key="search_sel")
            lat_s, lon_s = results[idx]["lat"], results[idx]["lon"]
            c1, c2 = st.columns(2)
            if c1.button("Centralizar no mapa", use_container_width=True):
                ss.pending_click = (lat_s, lon_s)
            if c2.button("Fixar como fonte", type="primary", use_container_width=True):
                ss.source = (lat_s, lon_s); ss.locked = True; ss._update = True

    st.markdown("---")
    # 🎨 Marca (uploader/path e tamanho)
    with st.expander("🎨 Marca (logo no topo)", expanded=False):
        up = st.file_uploader("Logo MAVIPE (PNG/JPG)", type=["png","jpg","jpeg"], key="logo_upl")
        logo_path = st.text_input("Ou caminho do logo local", value="images/logomavipe.jpeg", key="logo_path")
        logo_url = st.text_input("Ou URL do logo (GitHub raw)", value="", key="logo_url",
                                 help="Ex.: https://raw.githubusercontent.com/usuario/repositorio/branch/images/logomavipe.png")
        st.slider("Largura do logo (px)", 80, 320, int(ss.get("logo_w", 140)), 2, key="logo_w")
        if up is not None:
            ss.logo_bytes = up.read()
        elif logo_url:
            data = fetch_logo_url(logo_url)
            if data:
                ss.logo_bytes = data
            else:
                st.warning('Não foi possível baixar o logo pela URL. Verifique o link raw e permissões.')
        elif logo_path:
            try:
                if os.path.exists(logo_path):
                    with open(logo_path, "rb") as f:
                        ss.logo_bytes = f.read()
            except Exception:
                st.warning("Logo local não encontrado.")

    st.markdown("---")
    co, cr = st.columns(2)
    if co.button("Atualizar pluma", type="primary", use_container_width=True):
        ss._update = True
    if cr.button("Selecionar outro ponto", use_container_width=True):
        ss.source = None; ss.overlay = None; ss.locked = False; ss.pending_click = None

# ================== CONVERSÃO ==================
# kg/h -> g/s usando diretamente o valor do widget
Q_gps = (float(Q_kgph) * 1000.0) / 3600.0

# ================== MODELO PLUMA ==================
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
    extent_km = 2.0                 # ~1 km de raio
    px_per_km = PX_PER_KM_FIXED     # 25 m/pixel
    R_earth = 6371000.0
    lat_rad = np.deg2rad(lat)
    m_lat = np.pi*R_earth/180
    m_lon = m_lat*np.cos(lat_rad)
    half  = extent_km*1000/2
    res   = int(extent_km*px_per_km)
    x = np.linspace(-half, half, res); y = np.linspace(-half, half, res)
    X, Y = np.meshgrid(x, y)

    theta = np.deg2rad((p["wind_dir"]+180)%360)  # para onde VAI
    Xp =  np.cos(theta)*X + np.sin(theta)*Y
    Yp = -np.sin(theta)*X + np.cos(theta)*Y
    mask = Xp > 0.0

    sigy, sigz = sigma_yz(np.where(mask, Xp, 1.0), p["stability"], p["is_urban"])
    H_eff = effective_height(p["H"], p["V"], p["d"], p["Tamb"], p["Tstack"], p["u"])
    pref  = p["Q_gps"]/(2*np.pi*p["u"]*sigy*sigz + 1e-12)
    C = pref * np.exp(-0.5*(Yp**2)/(sigy**2 + 1e-12)) * (
        2.0*np.exp(-0.5*(H_eff**2)/(sigz**2 + 1e-12))
    )
    C[~mask] = 0.0

    dlat = half / m_lat
    dlon = half / (m_lon if m_lon > 0 else 111320.0)
    bounds = [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]
    return C, bounds

def to_ppb(C_gm3, P_hPa, T_K):
    R = 8.314462618; M = 16.043; P_pa = float(P_hPa)*100.0
    return C_gm3 * (R*float(T_K)) / (M*P_pa) * 1e9

def render_ppb(A_ppb, vmin=V_ABS_MIN, vmax=V_ABS_MAX, log=False):
    cmap = matplotlib.colormaps.get_cmap('jet')
    lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    if log:
        A = np.log10(np.maximum(A_ppb, 1e-12))
        vmin_, vmax_ = np.log10(max(vmin, 1e-12) + 1e-12), np.log10(max(vmax, 1e-12))
    else:
        A = A_ppb; vmin_, vmax_ = vmin, vmax
    N = np.clip((A - vmin_) / (vmax_ - vmin_ + 1e-12), 0, 1)
    idx = (N * 255).astype(np.uint8)
    alpha = (np.sqrt(N) * 255).astype(np.uint8); alpha[N <= 0.003] = 0
    rgb = lut[idx]
    return np.dstack([rgb, alpha]).astype(np.uint8)

# ---- CSS no iframe do Folium (Geocoder maior) ----
def _inject_geocoder_css(m, font_px: int = 20, result_px: int = 18, width_px: int = 520, input_h: int = 52):
    """Injeta CSS diretamente no HTML do mapa (iframe) para ampliar o Geocoder."""
    try:
        if Element is None:
            return
        css = f"""
        <style>
        .leaflet-control-geocoder, .leaflet-control-geocoder * {{
          font-size: {font_px}px !important;
        }}
        .leaflet-control-geocoder-form input {{
          font-size: {font_px}px !important;
          height: {input_h}px !important;
          padding: 10px 14px !important;
          width: {width_px}px !important;
        }}
        .leaflet-control-geocoder-expanded {{
          width: {width_px + 40}px !important;
        }}
        .leaflet-control-geocoder-alternatives {{
          font-size: {result_px}px !important;
        }}
        .leaflet-control-geocoder-alternatives li a {{
          line-height: 1.4 !important;
          padding: 8px 10px !important;
        }}
        </style>
        """
        m.get_root().html.add_child(Element(css))
    except Exception:
        pass

# --- encontrar exemplos ascendente/descendente próximos ---
def _haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, asin
    R = 6371.0088
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ = φ2 - φ1; dλ = λ2 - λ1
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return 2*R*asin(min(1, sqrt(a)))

def find_pass_example_headings(tle_l1, tle_l2, center_dt, site_lat, site_lon,
                                window_min: int = 120, step_s: int = 30, max_dist_km: int = 500):
    """Retorna dict com 'asc' e/ou 'desc' → (heading_deg, datetime_utc, dist_km)
    procurando por instantes em que o subponto do satélite esteja **perto do local** (≤ max_dist_km).
    Varre ±window_min minutos a partir de center_dt em passos de step_s segundos.
    """
    found = {}
    try:
        from skyfield.api import load, EarthSatellite
        ts = load.timescale()
        sat = EarthSatellite(tle_l1.strip(), tle_l2.strip(), "GHGSat", ts)
    except Exception:
        return found

    max_steps = int((window_min * 60) // step_s)
    for k in range(max_steps + 1):
        offsets = [0] if k == 0 else [k * step_s, -k * step_s]
        for off in offsets:
            t = center_dt + dt.timedelta(seconds=off)
            try:
                sp = sat.at(ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second)).subpoint()
                slat, slon = sp.latitude.degrees, sp.longitude.degrees
                dist = _haversine_km(site_lat, site_lon, slat, slon)
                if dist > max_dist_km:
                    continue
                # perto o suficiente: classifica
                h, asc = tle_heading_and_sense(tle_l1, tle_l2, t)
                key = 'asc' if asc else 'desc'
                if key not in found:
                    found[key] = (h, t, dist)
                if 'asc' in found and 'desc' in found:
                    return found
            except Exception:
                continue
    return found
    return found

# --- busca simples (visual): próximo ASC e próximo DESC à frente ---
def find_next_direction_pair_vec(sat, ts, start_dt, max_hours: int = 72, step_s: int = 60, min_sep_deg: int = 160):
    """Varredura vetorizada: pega o primeiro heading e o primeiro subsequente
    com direção oposta e separação angular >= min_sep_deg.
    """
    # grade temporal
    N = int((max_hours*3600)//step_s) + 2
    dts = [start_dt + dt.timedelta(seconds=i*step_s) for i in range(N)]
    tarr = ts.from_datetimes(dts)
    sp = sat.at(tarr).subpoint()
    lat = sp.latitude.degrees
    lon = sp.longitude.degrees
    # bearings entre passos consecutivos
    b = _bearing_deg_vec(lat[:-1], lon[:-1], lat[1:], lon[1:])
    # classe asc/desc (asc se indo ao norte)
    asc = (b <= 90.0) | (b >= 270.0)
    # índice do primeiro
    if len(b) == 0:
        return {}
    i0 = 0
    asc0 = bool(asc[i0])
    h0 = float(b[i0]); t0 = dts[i0+1]
    # procurar oposto com separação suficiente
    for j in range(i0+1, len(b)):
        if bool(asc[j]) != asc0 and _ang_sep(float(b[j]), h0) >= float(min_sep_deg):
            return {
                'first': (h0, t0, asc0),
                'opposite': (float(b[j]), dts[j+1], bool(asc[j]))
            }
    # não achou par
    return {'first': (h0, t0, asc0)}

    t = start_dt
    end = start_dt + dt.timedelta(hours=max_hours)
    h1 = None; asc1 = None
    while t <= end and ('opposite' not in out):
        try:
            h, asc = tle_heading_and_sense(tle_l1, tle_l2, t)
            if 'first' not in out:
                out['first'] = (h, t, asc)
                h1, asc1 = h, asc
            else:
                if asc != asc1 and _ang_sep(h, h1) >= float(min_sep_deg):
                    out['opposite'] = (h, t, asc)
                    break
        except Exception:
            pass
        t += dt.timedelta(seconds=step_s)
    return out

    t = start_dt
    end = start_dt + dt.timedelta(hours=max_hours)
    first_dir = None
    while t <= end and ('opposite' not in out):
        try:
            h, asc = tle_heading_and_sense(tle_l1, tle_l2, t)
            if 'first' not in out:
                out['first'] = (h, t, asc)
                first_dir = asc
            else:
                if asc != first_dir:
                    out['opposite'] = (h, t, asc)
                    break
        except Exception:
            pass
        t += dt.timedelta(seconds=step_s)
    return out

# --- legenda ASC/DESC ---
def add_ad_legend(m, font_px: int = 18):
    """Adiciona legenda fixa (ASC verde, DESC laranja) dentro do iframe do mapa."""
    try:
        if Element is None:
            return
        html = f"""
        <div id="legend-ad" style="
          position: fixed; bottom: 16px; right: 16px; z-index: 1000;
          background: rgba(0,0,0,0.65); color: #fff; padding: 8px 12px;
          border-radius: 10px; font-size: {font_px}px; line-height: 1.3;
          box-shadow: 0 2px 6px rgba(0,0,0,0.35);
        ">
          <div style="display:flex; align-items:center; gap:8px; margin:2px 0;">
            <span style="display:inline-block; width:14px; height:14px; background:#00c853; border:2px solid #00c853;"></span>
            <span>Órbita ascendente</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px; margin:2px 0;">
            <span style="display:inline-block; width:14px; height:14px; background:#ff6d00; border:2px solid #ff6d00;"></span>
            <span>Órbita descendente</span>
          </div>
        </div>
        """
        m.get_root().html.add_child(Element(html))
    except Exception:
        pass

# ================== TLE / FOOTPRINT ==================
def _meters_per_deg(lat_deg: float):
    R = 6371000.0
    m_per_deg_lat = np.pi * R / 180.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.deg2rad(lat_deg))
    return m_per_deg_lat, max(m_per_deg_lon, 1e-6)

def _square_poly(lat0, lon0, size_km, rot_deg=0.0):
    """Quadrado size_km×size_km centrado em (lat0,lon0), rotacionado (N=0°, horário)."""
    half_m = (size_km * 1000.0) / 2.0
    pts = np.array([[-half_m, -half_m],
                    [ half_m, -half_m],
                    [ half_m,  half_m],
                    [-half_m,  half_m],
                    [-half_m, -half_m]], dtype=float)
    theta = np.deg2rad(rot_deg)
    # Rotação com 0° = Norte, sentido horário (x=Este, y=Norte)
    Rz = np.array([[ np.sin(theta),  np.cos(theta)],
                   [ np.cos(theta), -np.sin(theta)]])
    pts_rot = pts @ Rz.T
    m_lat, m_lon = _meters_per_deg(lat0)
    lats = lat0 + (pts_rot[:,1] / m_lat)
    lons = lon0 + (pts_rot[:,0] / m_lon)
    return list(map(lambda xy: [xy[0], xy[1]], zip(lats, lons)))

def _bearing_deg(lat1, lon1, lat2, lon2):
    from math import atan2, radians, degrees, cos, sin
    φ1, φ2 = radians(lat1), radians(lat2)
    Δλ = radians(lon2 - lon1)
    x = sin(Δλ) * cos(φ2)
    y = cos(φ1)*sin(φ2) - sin(φ1)*cos(φ2)*cos(Δλ)
    return (degrees(atan2(x, y)) + 360.0) % 360.0

def _bearing_deg_vec(lat0, lon0, lat1, lon1):
    """Versão vetorizada (numpy) do bearing 0°=N, 90°=E."""
    lat0 = np.deg2rad(lat0); lat1 = np.deg2rad(lat1)
    dlon = np.deg2rad(lon1 - lon0)
    x = np.sin(dlon) * np.cos(lat1)
    y = np.cos(lat0)*np.sin(lat1) - np.sin(lat0)*np.cos(lat1)*np.cos(dlon)
    ang = np.degrees(np.arctan2(x, y))
    return (ang + 360.0) % 360.0

def tle_heading_and_sense(tle_l1, tle_l2, dt_utc):
    """Heading (0°=N, 90°=E, horário) e sentido (asc/desc) no instante dt_utc via Skyfield.
    """
    from skyfield.api import load, EarthSatellite
    ts = load.timescale()
    sat = EarthSatellite(tle_l1.strip(), tle_l2.strip(), "GHGSat", ts)
    # amostra centrada para derivada temporal estável
    t0 = ts.utc(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, dt_utc.minute, dt_utc.second - 1)
    t1 = ts.utc(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, dt_utc.minute, dt_utc.second + 1)
    sp0 = sat.at(t0).subpoint(); sp1 = sat.at(t1).subpoint()
    lat0, lon0 = sp0.latitude.degrees, sp0.longitude.degrees
    lat1, lon1 = sp1.latitude.degrees, sp1.longitude.degrees
    heading = _bearing_deg(lat0, lon0, lat1, lon1)  # 0°=N, 90°=E
    # Critério robusto: norte = heading ∈ [270°,360°) ∪ [0°,90°]; sul = (90°,270°)
    if 90.0 < heading < 270.0:
        is_asc = False  # descendente (rumo ao sul)
    elif 85.0 < heading < 95.0 or 265.0 < heading < 275.0:
        # quase leste/oeste: decide pelo sinal de Δlat
        is_asc = (lat1 > lat0)
    else:
        is_asc = True   # ascendente (rumo ao norte)
    return heading, is_asc

# --- util: ponto destino dado azimute e distância ---
@st.cache_resource(show_spinner=False)
def get_sat_cached(name: str, l1: str, l2: str):
    """Cacheia Timescale e EarthSatellite para evitar recriação a cada passo."""
    from skyfield.api import load, EarthSatellite
    ts = load.timescale()
    sat = EarthSatellite(l1.strip(), l2.strip(), name, ts)
    return ts, sat
from math import radians, degrees, sin, cos, asin, atan2, pi

def dest_point(lat, lon, bearing_deg, dist_m):
    R = 6371000.0
    φ1, λ1 = radians(lat), radians(lon)
    θ = radians(bearing_deg)
    δ = dist_m / R
    φ2 = asin(sin(φ1)*cos(δ) + cos(φ1)*sin(δ)*cos(θ))
    λ2 = λ1 + atan2(sin(θ)*sin(δ)*cos(φ1), cos(δ) - sin(φ1)*sin(φ2))
    lon2 = (degrees(λ2) + 540) % 360 - 180
    return degrees(φ2), lon2

# separação angular mais curta entre dois azimutes (0–180°)
def _ang_sep(a_deg: float, b_deg: float) -> float:
    d = abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)
    return d

# wrapper: expande janela automaticamente até achar o oposto (até ~10 dias)
def find_dir_pair_auto(sat, ts, start_dt, base_hours: int = 72, min_sep_deg: int = 160):
    hours_list = []
    if base_hours and base_hours > 0:
        hours_list.append(int(base_hours))
    hours_list += [72, 120, 168, 240]
    seen = set(); ordered = []
    for h in hours_list:
        if h in seen:
            continue
        seen.add(h); ordered.append(h)
    last_pair = {}
    for h in ordered:
        step = 60 if h <= 72 else 120
        pair = find_next_direction_pair_vec(sat, ts, start_dt, max_hours=h, step_s=step, min_sep_deg=min_sep_deg)
        if 'opposite' in pair:
            return pair, h
        last_pair = pair
    return last_pair, ordered[-1]

# --- adiciona seta de direção no centro do footprint ---
def add_heading_arrow(m, lat, lon, heading_deg, color="#00c853"):
    try:
        lat2, lon2 = dest_point(lat, lon, heading_deg, 3000.0)  # 3 km
        pl = folium.PolyLine(locations=[(lat, lon), (lat2, lon2)], color=color, weight=4, opacity=1.0)
        pl.add_to(m)
        if PolyArrow is not None:
            PolyArrow(
                pl,
                "▶",
                repeat=True,
                offset=8,
                attributes={"fill": color, "font-weight": "bold", "font-size": "24"},
            ).add_to(m)
        elif RegularPolygonMarker is not None:
            # marcador triangular apontando na direção do heading
            RegularPolygonMarker(location=[lat2, lon2], number_of_sides=3, radius=10,
                                 rotation=heading_deg, color=color, fill=True, fill_color=color,
                                 fill_opacity=1.0).add_to(m)
    except Exception:
        pass

# ================== SELEÇÃO (clique simples + marcador provisório) ==================
if ss.source is None or not ss.locked:
    st.info("🖱️ **Clique uma vez no mapa** para escolher o ponto **(não precisa Save)**. "
            "Se preferir, use o ícone de marcador e clique em **Save**.")

    center0 = ss.pending_click or (-22.9035, -43.2096)
    m_sel = folium.Map(location=center0, zoom_start=16, control_scale=True, zoom_control=True)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri, Maxar, Earthstar Geographics, and the GIS User Community',
        name='Esri World Imagery'
    ).add_to(m_sel)
    # 🔎 Busca no mapa (Nominatim)
    if Geocoder:
        Geocoder(collapsed=False, add_marker=True, position='topleft',
                 placeholder='Buscar lugar…').add_to(m_sel)
    # CSS maior dentro do mapa (iframe)
    _inject_geocoder_css(m_sel, font_px=22, result_px=20, width_px=560, input_h=56)

    if ScaleBar: ScaleBar(position="bottomleft", imperial=False).add_to(m_sel)
    m_sel.add_child(MeasureControl(primary_length_unit='meters',
                                   secondary_length_unit='kilometers',
                                   position='topleft'))

    Draw(draw_options={"polyline": False, "polygon": False, "circle": False,
                       "circlemarker": False, "rectangle": False, "marker": True},
         edit_options={"edit": True, "remove": True}).add_to(m_sel)

    folium.LatLngPopup().add_to(m_sel)

    if ss.pending_click is not None:
        folium.CircleMarker(location=list(ss.pending_click), radius=6,
                            color="#e91e63", fill=True, fill_opacity=0.9,
                            tooltip="Ponto provisório").add_to(m_sel)

    ret = st_folium(
        m_sel,
        height=560,
        returned_objects=["all_drawings", "last_draw", "last_clicked"],
        key="map_select",
        use_container_width=True,
    )

    # captura do ponto (clique simples OU marcador salvo)
    new_pt = None
    lc = ret.get("last_clicked") if ret else None
    if lc and "lat" in lc and "lng" in lc:
        new_pt = (float(lc["lat"]), float(lc["lng"]))
    if new_pt is None:
        drawings = ret.get("all_drawings") if ret else None
        if drawings:
            for feat in drawings[::-1]:
                try:
                    if feat and feat["geometry"]["type"] == "Point":
                        lon, lat = feat["geometry"]["coordinates"]; new_pt = (float(lat), float(lon)); break
                except Exception:
                    pass
        if new_pt is None:
            ld = ret.get("last_draw") if ret else None
            if ld:
                try:
                    if ld["geometry"]["type"] == "Point":
                        lon, lat = ld["geometry"]["coordinates"]; new_pt = (float(lat), float(lon))
                except Exception:
                    pass

    # salva apenas se realmente mudou
    if new_pt is not None:
        if ss.pending_click is None or tuple(np.round(ss.pending_click, 7)) != tuple(np.round(new_pt, 7)):
            ss.pending_click = new_pt

    # painel e botões
    if ss.pending_click is not None:
        lat_p, lon_p = ss.pending_click
        addr = reverse_geocode(lat_p, lon_p)
        st.markdown(
            f"📍 **Ponto selecionado:** `{lat_p:.6f}, {lon_p:.6f}`" + (f"<br/>🏠 {addr}" if addr else ""),
            unsafe_allow_html=True,
        )
    else:
        st.caption("Clique uma vez no mapa para habilitar o botão.")

    c_ok, c_rm = st.columns(2)
    ok_btn = c_ok.button("✅ Confirmar este ponto", use_container_width=True, disabled=(ss.pending_click is None))
    rm_btn = c_rm.button("🗑 Remover/limpar seleção", use_container_width=True, disabled=(ss.pending_click is None))

    if ok_btn and ss.pending_click is not None:
        ss.source = ss.pending_click; ss.locked = True; ss._update = True; ss.pending_click = None
        st.success("Fonte confirmada. Gerando pluma…")

    if rm_btn and ss.pending_click is not None:
        ss.pending_click = None; st.info("Seleção limpa. Clique novamente no mapa.")

# ================== MAPA FINAL ==================
else:
    lat, lon = ss.source
    params = dict(
        wind_dir=wind_dir, wind_speed=wind_speed, stability=stability, is_urban=is_urban,
        Q_gps=Q_gps, H=H_stack, d=d_stack, V=V_exit, Tamb=Tamb, Tstack=Tstack, u=wind_speed
    )

    # calcula/atualiza a pluma (cache em ss.overlay)
    if ss._update or ss.overlay is None:
        C, bounds = compute_conc(lat, lon, params)
        C_ppb = to_ppb(C, P_hPa, Tamb)
        rgba = render_ppb(C_ppb, 0.0, 450.0, log=(scale_mode == "Absoluta (log10)"))
        im = Image.fromarray(rgba, "RGBA")
        bio = io.BytesIO(); im.save(bio, "PNG"); bio.seek(0)
        ss.overlay = (bio.read(), bounds)
        ss._update = False

    png_bytes, bounds = ss.overlay

    # mapa final
    m1 = folium.Map(location=[lat, lon], zoom_start=15, control_scale=True)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri, Maxar, Earthstar Geographics, and the GIS User Community',
        name='Esri World Imagery'
    ).add_to(m1)
    # 🔎 Busca no mapa (Nominatim)
    if Geocoder:
        Geocoder(collapsed=False, add_marker=True, position='topleft',
                 placeholder='Buscar lugar…').add_to(m1)
    # CSS maior dentro do mapa (iframe)
    _inject_geocoder_css(m1, font_px=22, result_px=20, width_px=560, input_h=56)
    if ScaleBar: ScaleBar(position="bottomleft", imperial=False).add_to(m1)
    m1.add_child(MeasureControl(primary_length_unit='meters',
                                secondary_length_unit='kilometers', position='topleft'))

    folium.raster_layers.ImageOverlay(
        image="data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8"),
        bounds=bounds, opacity=opacity, name="Pluma (ppb)"
    ).add_to(m1)
    folium.CircleMarker([lat,lon], radius=6, color="#f00", fill=True, tooltip="Fonte").add_to(m1)

    # Footprint GHGSat 5×5 km orientado por TLE do arquivo
    if ss.tle_cache and ("tle_choice" in ss):
        try:
            sat_name = ss.tle_choice
            l1, l2 = ss.tle_cache[sat_name]
            t_center = dt.datetime.combine(ss.obs_date, ss.obs_time).replace(tzinfo=dt.timezone.utc)

            # Sempre mostrar dois footprints (visual): primeiro instante e o primeiro com direção oposta
            ts_obj, sat_obj = get_sat_cached(sat_name, l1, l2)
            pair, used_h = find_dir_pair_auto(sat_obj, ts_obj, t_center, base_hours=int(ad_hours), min_sep_deg=160)

            if 'first' in pair:
                h1, t1, asc1 = pair['first']
                label1 = 'Ascendente' if asc1 else 'Descendente'
                color1 = '#00c853' if asc1 else '#ff6d00'
                name1 = ('ASC' if asc1 else 'DESC') + f" (visual, {sat_name})"
                fg1 = folium.FeatureGroup(name=name1, show=True)
                poly1 = _square_poly(lat, lon, FOOTPRINT_SIZE_KM, rot_deg=h1)
                folium.Polygon(locations=poly1, color=color1, weight=4, opacity=1.0,
                               fill=True, fill_color=color1, fill_opacity=0.12,
                               tooltip=f"{sat_name} • {label1} (visual) • heading {h1:.1f}° @ {t1.isoformat()}").add_to(fg1)
                add_heading_arrow(fg1, lat, lon, h1, color=color1)
                fg1.add_to(m1)
                st.caption(f"🛰 {sat_name} • {label1} (visual) • heading {h1:.1f}° @ {t1.isoformat()}")

            if 'opposite' in pair:
                h2, t2, asc2 = pair['opposite']
                label2 = 'Ascendente' if asc2 else 'Descendente'
                color2 = '#00c853' if asc2 else '#ff6d00'
                name2 = ('ASC' if asc2 else 'DESC') + f" (visual, {sat_name})"
                fg2 = folium.FeatureGroup(name=name2, show=True)
                poly2 = _square_poly(lat, lon, FOOTPRINT_SIZE_KM, rot_deg=h2)
                folium.Polygon(locations=poly2, color=color2, weight=4, opacity=1.0,
                               fill=True, fill_color=color2, fill_opacity=0.10,
                               tooltip=f"{sat_name} • {label2} (visual) • heading {h2:.1f}° (Δ≈{_ang_sep(h2, h1):.0f}°) @ {t2.isoformat()}").add_to(fg2)
                add_heading_arrow(fg2, lat, lon, h2, color=color2)
                fg2.add_to(m1)
                st.caption(f"🛰 {sat_name} • {label2} (visual) • heading {h2:.1f}° (Δ≈{_ang_sep(h2, h1):.0f}°) @ {t2.isoformat()}")

            if 'first' in pair and 'opposite' not in pair:
                # desenha orientação oposta sintética (+180°) para garantir as duas órbitas
                h2 = (h1 + 180.0) % 360.0
                asc2 = not asc1
                label2 = 'Ascendente' if asc2 else 'Descendente'
                color2 = '#00c853' if asc2 else '#ff6d00'
                name2 = ('ASC' if asc2 else 'DESC') + f" (visual, {sat_name})"
                fg2 = folium.FeatureGroup(name=name2, show=True)
                poly2 = _square_poly(lat, lon, FOOTPRINT_SIZE_KM, rot_deg=h2)
                folium.Polygon(
                    locations=poly2,
                    color=color2, weight=4, opacity=1.0,
                    fill=True, fill_color=color2, fill_opacity=0.10,
                    tooltip=f"{sat_name} • {label2} (visual) • heading {h2:.1f}° (oposto sintético)"
                ).add_to(fg2)
                add_heading_arrow(fg2, lat, lon, h2, color=color2)
                fg2.add_to(m1)
                st.caption(f"🛰 {sat_name} • {label2} (visual) • heading {h2:.1f}° (oposto sintético)") dentro do limite definido.')
        except Exception as e:
            st.warning(f"Footprint via TLE falhou: {e}")

    add_ad_legend(m1, font_px=18)
    folium.LayerControl(collapsed=False).add_to(m1)
    st_folium(m1, height=720, key="map_final", use_container_width=True)

