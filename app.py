# app.py — Pluma Gaussiana + Mira arrastável + Régua + GHGSat Footprint (TLE opcional)
# -*- coding: utf-8 -*-
import io, base64, datetime as dt
import numpy as np
import streamlit as st
import folium
from folium.plugins import Draw, MeasureControl
try:
    from folium.plugins import ScaleBar  # pode não existir em versões bem antigas
except Exception:
    ScaleBar = None
from streamlit_folium import st_folium
from PIL import Image
from matplotlib import cm
from geopy.geocoders import Nominatim

# ============ CONFIG ============
st.set_page_config(page_title="Pluma CH₄ + GHGSat Footprint", layout="wide")
st.title("Pluma Gaussiana (CH₄) · 25 m/pixel · ppb 0–450 + GHGSat Footprint 5×5 km")

st.markdown("""
<style>
[data-testid="stSidebar"] { overflow-y: auto; max-height: 100vh; }
.stButton>button { height: 40px; }
</style>
""", unsafe_allow_html=True)

# ============ ESTADO ============
ss = st.session_state
ss.setdefault("source", None)         # (lat, lon) confirmado
ss.setdefault("pending_click", None)  # último ponto selecionado (lat, lon)
ss.setdefault("overlay", None)
ss.setdefault("_update", False)
ss.setdefault("locked", False)

# ============ CONSTANTES ============
PX_PER_KM_FIXED = 40        # 25 m/pixel
V_ABS_MIN, V_ABS_MAX = 0.0, 450.0  # ppb

# ============ GEOCODING ============
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

# ============ SIDEBAR ============
with st.sidebar:
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

    with st.expander("🛰 GHGSat — Footprint", expanded=True):
        ghg_enable   = st.checkbox("Mostrar footprint", value=True)
        ghg_size_km  = st.number_input("Tamanho (km)", 1.0, 20.0, 5.0, 0.5)
        ghg_rot_deg  = st.number_input("Rotação manual (° a partir do Norte, horário)", -180.0, 180.0, 0.0, 1.0)
        ghg_shift_km = st.number_input("Deslocamento ao longo da trilha (km)", -10.0, 10.0, 0.0, 0.5,
                                       help="Usado para separar visualmente ascendente/descendente.")

    with st.expander("🛰 GHGSat — Orientação pelo TLE (opcional)", expanded=False):
        use_tle = st.checkbox("Auto-orientar footprint com TLE", value=False,
                              help="Alinha o footprint ao ground-track real no instante escolhido.")
        tle_l1 = st.text_input("TLE linha 1", value="", placeholder="1 XXXXXU 20XXXA ...")
        tle_l2 = st.text_input("TLE linha 2", value="", placeholder="2 XXXXX  XX.XXXXX ...")
        # default = agora em UTC
        now_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        obs_date = st.date_input("Data (UTC)", value=now_utc.date())
        obs_time = st.time_input("Hora (UTC)", value=now_utc.time().replace(microsecond=0))

    st.markdown("---")
    c1, c2 = st.columns(2)
    if c1.button("Atualizar pluma", type="primary", use_container_width=True):
        ss._update = True
    if c2.button("Selecionar outro ponto", use_container_width=True):
        ss.source = None; ss.overlay = None; ss.locked = False; ss.pending_click = None

# ============ CONVERSÃO ============
Q_gps = (float(Q_kgph) * 1000.0) / 3600.0  # kg/h -> g/s

# ============ MODELO PLUMA ============
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
        np.exp(-0.5*(H_eff**2)/(sigz**2 + 1e-12)) + np.exp(-0.5*(H_eff**2)/(sigz**2 + 1e-12))
    )
    C[~mask] = 0.0

    dlat = half / m_lat
    dlon = half / (m_lon if m_lon > 0 else 111320.0)
    bounds = [[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]]
    return C, bounds

def to_ppb_safe(C_val, pres_hPa, temp_K):
    R_local = 8.314462618
    M_CH4_local = 16.043
    P_pa = float(pres_hPa) * 100.0
    return C_val * (R_local * float(temp_K)) / (M_CH4_local * P_pa) * 1e9

def render_ppb(A_ppb, vmin=V_ABS_MIN, vmax=V_ABS_MAX, log=False):
    lut = (cm.get_cmap('jet', 256)(np.linspace(0,1,256))[:,:3]*255).astype(np.uint8)
    if log:
        A = np.log10(np.maximum(A_ppb, 1e-12))
        vmin_, vmax_ = np.log10(max(vmin,1e-12)+1e-12), np.log10(max(vmax,1e-12))
    else:
        A = A_ppb
        vmin_, vmax_ = vmin, vmax
    N = np.clip((A - vmin_) / (vmax - vmin + 1e-12), 0, 1)
    idx = (N*255).astype(np.uint8)
    alpha = (np.sqrt(N)*255).astype(np.uint8); alpha[N<=0.003]=0
    rgb = lut[idx]
    return np.dstack([rgb, alpha]).astype(np.uint8)

# ======== UTILITÁRIOS FOOTPRINT ========
def _meters_per_deg(lat_deg: float):
    R = 6371000.0
    m_per_deg_lat = np.pi * R / 180.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.deg2rad(lat_deg))
    return m_per_deg_lat, max(m_per_deg_lon, 1e-6)

def _square_poly(lat0, lon0, size_km, rot_deg=0.0, shift_km=0.0):
    """Quadrado size_km×size_km centrado em (lat0,lon0), rotacionado e deslocado ao longo do eixo principal (N)."""
    half_m = (size_km * 1000.0) / 2.0
    pts = np.array([[-half_m, -half_m],
                    [ half_m, -half_m],
                    [ half_m,  half_m],
                    [-half_m,  half_m],
                    [-half_m, -half_m]], dtype=float)
    # Rotação: referência N=0°, sentido horário
    theta = np.deg2rad(rot_deg)
    # Conversão local EN (E,N) — matriz ajustada para referência Norte
    Rz = np.array([[ np.sin(theta),  np.cos(theta)],
                   [ np.cos(theta), -np.sin(theta)]])
    pts_rot = pts @ Rz.T
    # Desloca ao longo do eixo principal (N)
    pts_rot[:,1] += shift_km * 1000.0
    # Para lat/lon
    m_lat, m_lon = _meters_per_deg(lat0)
    lats = lat0 + (pts_rot[:,1] / m_lat)
    lons = lon0 + (pts_rot[:,0] / m_lon)
    return list(map(lambda xy: [xy[0], xy[1]], zip(lats, lons)))

# ---- Bearing geodésico e heading via TLE ----
def _bearing_deg(lat1, lon1, lat2, lon2):
    # N=0°, horário
    from math import atan2, radians, degrees, cos, sin
    φ1, φ2 = radians(lat1), radians(lat2)
    Δλ = radians(lon2 - lon1)
    x = sin(Δλ) * cos(φ2)
    y = cos(φ1)*sin(φ2) - sin(φ1)*cos(φ2)*cos(Δλ)
    brng = (degrees(atan2(x, y)) + 360.0) % 360.0
    return brng

def tle_heading_deg_at_time(tle_l1, tle_l2, dt_utc):
    """Heading do ground-track no instante dt_utc (N=0°, horário) via Skyfield."""
    from skyfield.api import load, EarthSatellite
    ts = load.timescale()
    sat = EarthSatellite(tle_l1.strip(), tle_l2.strip(), "GHGSat", ts)
    t0 = ts.utc(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, dt_utc.minute, dt_utc.second - 1)
    t1 = ts.utc(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, dt_utc.minute, dt_utc.second + 1)
    sp0 = sat.at(t0).subpoint(); sp1 = sat.at(t1).subpoint()
    lat0, lon0 = sp0.latitude.degrees, sp0.longitude.degrees
    lat1, lon1 = sp1.latitude.degrees, sp1.longitude.degrees
    return _bearing_deg(lat0, lon0, lat1, lon1)

# ============ SELEÇÃO: MIRA ARRASTÁVEL + TEMPO REAL + CLIQUE + RÉGUA ============
if ss.source is None or not ss.locked:
    st.info("🎯 Na barra do mapa, clique no **marcador (alvo)**, posicione/ARRASTE a mira e **clique em Save** na barrinha cinza. "
            "Se preferir, um **clique simples** no mapa também habilita o botão.")

    center0 = ss.pending_click or (-22.9035, -43.2096)
    m_sel = folium.Map(location=center0, zoom_start=16, control_scale=True, zoom_control=True)
    folium.TileLayer("OpenStreetMap").add_to(m_sel)

    if ScaleBar: ScaleBar(position="bottomleft", imperial=False).add_to(m_sel)
    m_sel.add_child(MeasureControl(primary_length_unit='meters',
                                   secondary_length_unit='kilometers',
                                   position='topleft'))

    # Toolbar: só Marker; edição/remoção habilitadas
    Draw(
        draw_options={"polyline": False, "polygon": False, "circle": False,
                      "circlemarker": False, "rectangle": False, "marker": True},
        edit_options={"edit": True, "remove": True}
    ).add_to(m_sel)

    # Ícone "mira" + arrastar
    custom_js = """
    <script>
    (function(){
      var map = window._leaflet_map_instance = window._leaflet_map_instance || (function(){
        var k = Object.keys(window).find(k=>window[k] && window[k].setView && window[k].hasLayer && window[k].on && window[k].eachLayer);
        return window[k];
      })();
      if (!map) return;

      if (L && L.Draw && L.Draw.Marker) {
        L.Draw.Marker.prototype.options.icon = L.divIcon({
          className: 'crosshair-marker',
          html: '<div style="font-size:28px;font-weight:700;color:red;text-shadow:1px 1px 2px #fff;">＋</div>',
          iconSize: [20,20], iconAnchor: [10,10]
        });
      }
      map.on('draw:created', function (e) {
        if (e.layer && e.layer.dragging) e.layer.dragging.enable();
      });
      map.on('draw:editstart', function(){
        map.eachLayer(function(layer){
          if (layer && layer.dragging && layer.getLatLng) { try { layer.dragging.enable(); } catch(_){} }
        });
      });
    })();
    </script>
    """
    m_sel.get_root().html.add_child(folium.Element(custom_js))

    # Render do mapa e captura (inclui last_active_drawing)
    ret = st_folium(
        m_sel,
        height=560,
        returned_objects=["all_drawings", "last_active_drawing", "last_draw", "last_clicked"],
        use_container_width=True
    )

    # Extrai (lat, lon): 1) enquanto arrasta; 2) após Save; 3) criação; 4) clique simples
    def _extract_point(ret_obj):
        lad = ret_obj.get("last_active_drawing") if ret_obj else None
        if lad:
            try:
                if lad["geometry"]["type"] == "Point":
                    lon, lat = lad["geometry"]["coordinates"]; return float(lat), float(lon)
            except Exception: pass
        drawings = ret_obj.get("all_drawings") if ret_obj else None
        if drawings:
            for feat in drawings[::-1]:
                try:
                    if feat and feat["geometry"]["type"] == "Point":
                        lon, lat = feat["geometry"]["coordinates"]; return float(lat), float(lon)
                except Exception: pass
        ld = ret_obj.get("last_draw") if ret_obj else None
        if ld:
            try:
                if ld["geometry"]["type"] == "Point":
                    lon, lat = ld["geometry"]["coordinates"]; return float(lat), float(lon)
            except Exception: pass
        lc = ret_obj.get("last_clicked") if ret_obj else None
        if lc and "lat" in lc and "lng" in lc:
            return float(lc["lat"]), float(lc["lng"])
        return None

    new_pt = _extract_point(ret)
    if new_pt is not None:
        ss.pending_click = new_pt

    # Painel e botões
    if ss.pending_click is not None:
        lat_p, lon_p = ss.pending_click
        addr = reverse_geocode(lat_p, lon_p)
        st.markdown(f"📍 **Ponto selecionado:** `{lat_p:.6f}, {lon_p:.6f}`" + (f"<br/>🏠 {addr}" if addr else ""), unsafe_allow_html=True)
    else:
        st.caption("Posicione/arraste a mira e clique em **Save** (ou clique no mapa) para habilitar o botão.")

    c_ok, c_rm = st.columns(2)
    ok_btn = c_ok.button("✅ Confirmar este ponto", use_container_width=True, disabled=(ss.pending_click is None))
    rm_btn = c_rm.button("🗑 Remover/limpar seleção", use_container_width=True, disabled=(ss.pending_click is None))

    if ok_btn and ss.pending_click is not None:
        ss.source = ss.pending_click
        ss.locked = True
        ss._update = True
        ss.pending_click = None
        st.success("Fonte confirmada. Gerando pluma…")

    if rm_btn and ss.pending_click is not None:
        ss.pending_click = None
        st.info("Seleção limpa. Posicione a mira novamente.")

else:
    # ====== SIMULAÇÃO / RENDER ======
    lat, lon = ss.source
    params = dict(
        wind_dir=wind_dir, wind_speed=wind_speed, stability=stability, is_urban=is_urban,
        Q_gps=Q_gps, H=H_stack, d=d_stack, V=V_exit, Tamb=Tamb, Tstack=Tstack, u=wind_speed
    )

    if ss._update or ss.overlay is None:
        C, bounds = compute_conc(lat, lon, params)
        C_ppb = to_ppb_safe(C, P_hPa, Tamb)
        rgba = render_ppb(C_ppb, V_ABS_MIN, V_ABS_MAX, log=(scale_mode == "Absoluta (log10)"))
        im = Image.fromarray(rgba, "RGBA")
        bio = io.BytesIO(); im.save(bio, "PNG"); bio.seek(0)
        ss.overlay = (bio.read(), bounds)
        ss._update = False

    png_bytes, bounds = ss.overlay
    m1 = folium.Map(location=[lat, lon], zoom_start=15, control_scale=True)
    if ScaleBar: ScaleBar(position="bottomleft", imperial=False).add_to(m1)
    m1.add_child(MeasureControl(primary_length_unit='meters',
                                secondary_length_unit='kilometers',
                                position='topleft'))

    # ---- Overlay da pluma ----
    folium.raster_layers.ImageOverlay(
        image="data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8"),
        bounds=bounds, opacity=opacity, name="Pluma (ppb)"
    ).add_to(m1)
    folium.CircleMarker([lat,lon], radius=6, color="#f00", fill=True, tooltip="Fonte").add_to(m1)

    # ---- GHGSat footprints (Asc/Desc) ----
    if ghg_enable:
        rot_use = ghg_rot_deg  # padrão manual
        if use_tle and tle_l1.strip() and tle_l2.strip():
            try:
                dt_utc = dt.datetime.combine(obs_date, obs_time).replace(tzinfo=dt.timezone.utc)
                rot_use = float(tle_heading_deg_at_time(tle_l1, tle_l2, dt_utc))
                st.caption(f"🔁 Rotação pelo TLE: {rot_use:.1f}° (N=0°, sentido horário).")
            except Exception as e:
                st.warning(f"Não foi possível calcular orientação pelo TLE: {e}. Usando rotação manual.")

        # Ascendente: +shift ao longo da trilha
        poly_asc = _square_poly(lat, lon, ghg_size_km, rot_deg=rot_use, shift_km=+ghg_shift_km)
        folium.Polygon(
            locations=poly_asc, color="#1f77b4", weight=2, fill=True, fill_opacity=0.10,
            tooltip="GHGSat Footprint — Ascendente"
        ).add_to(m1)

        # Descendente: -shift para separar visualmente
        poly_desc = _square_poly(lat, lon, ghg_size_km, rot_deg=rot_use, shift_km=-ghg_shift_km)
        folium.Polygon(
            locations=poly_desc, color="#ff7f0e", weight=2, fill=True, fill_opacity=0.10,
            tooltip="GHGSat Footprint — Descendente"
        ).add_to(m1)

    folium.LayerControl(collapsed=False).add_to(m1)
    st_folium(m1, height=720, use_container_width=True)
