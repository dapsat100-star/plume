# app.py — build mínimo/robusto com legenda e debug
import io, base64
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
from PIL import Image
from matplotlib import cm
from branca.colormap import LinearColormap
from branca.element import Element  # importante no Edge

st.set_page_config(page_title="Pluma (debug) — ppb 0–450", layout="wide")
st.title("Pluma Gaussiana — DEBUG (ppb 0–450, Edge-safe)")

# ---- estado
ss = st.session_state
ss.setdefault("source", None)
ss.setdefault("_update", False)
ss.setdefault("overlay", None)

# ---- constantes
R = 8.314462618
M = 16.043
PX_PER_KM = 40     # 25 m/pixel
V_MIN, V_MAX = 0.0, 450.0

# ---- sidebar
with st.sidebar:
    st.header("Parâmetros")
    wind_dir   = st.number_input("Direção do vento (° de onde VEM)", 0, 359, 45, 1)
    wind_speed = st.number_input("Velocidade do vento (m/s)", 0.1, 50.0, 5.0, 0.1)
    stability  = st.selectbox("Estabilidade", ["A","B","C","D","E","F"], index=3)
    is_urban   = st.checkbox("Urbano", value=False)
    P_hPa      = st.number_input("Pressão (hPa)", 800.0, 1050.0, 1013.25, 0.5)
    Tamb       = st.number_input("Temperatura (K)", 230.0, 330.0, 298.0, 0.5)
    Q_kgph     = st.number_input("Q (kg CH₄/h)", 0.001, 1e9, 100.0, 0.1)

    st.markdown("---")
    col1, col2 = st.columns(2)
    if col1.button("Atualizar", type="primary", use_container_width=True):
        ss["_update"] = True
    if col2.button("Selecionar novo ponto", use_container_width=True):
        ss["source"] = None; ss["overlay"] = None

# ---- helpers simples
def sigmas(x_m, stab, urb):
    xkm = np.maximum(x_m/1000.0, 1e-6)
    if not urb:
        tbl = {"A":(0.22,0.5,0.20,0.5),"B":(0.16,0.5,0.12,0.5),"C":(0.11,0.5,0.08,0.5),
               "D":(0.08,0.5,0.06,0.5),"E":(0.06,0.5,0.03,0.5),"F":(0.04,0.5,0.016,0.5)}
    else:
        tbl = {"A":(0.32,0.5,0.24,0.5),"B":(0.22,0.5,0.18,0.5),"C":(0.16,0.5,0.14,0.5),
               "D":(0.12,0.5,0.10,0.5),"E":(0.10,0.5,0.06,0.5),"F":(0.08,0.5,0.04,0.5)}
    a,by,c,bz = tbl.get(stab,"D")
    return np.clip(a*(xkm**by)*1000,1,None), np.clip(c*(xkm**bz)*1000,1,None)

def H_eff(H, V, d, Ta, Ts, u):
    g=9.80665
    F = g*V*(d**2)*(Ts-Ta)/(4*Ts+1e-9)
    return H + max(3*d*V/max(u,0.1), 2.6*(F**(1/3))/max(u,0.1), 0.0)

def calc(lat, lon, pars):
    extent_km=2.0; px=PX_PER_KM
    R_earth=6371000.0
    latr=np.deg2rad(lat)
    mlat=np.pi*R_earth/180; mlon=mlat*np.cos(latr)
    half=extent_km*1000/2; res=int(extent_km*px)
    x=np.linspace(-half,half,res); y=np.linspace(-half,half,res)
    X,Y=np.meshgrid(x,y)
    th=np.deg2rad((pars["wd"]+180)%360)
    Xp=np.cos(th)*X + np.sin(th)*Y
    Yp=-np.sin(th)*X + np.cos(th)*Y
    mask=Xp>0
    sy,sz = sigmas(np.where(mask,Xp,1), pars["stab"], pars["urb"])
    He = H_eff(pars["H"], pars["V"], pars["d"], pars["Ta"], pars["Ts"], pars["u"])
    pref = pars["Q"]/(2*np.pi*pars["u"]*sy*sz + 1e-12)
    C = pref*np.exp(-0.5*(Yp**2)/(sy**2+1e-12))*(np.exp(-0.5*(He**2)/(sz**2+1e-12))+np.exp(-0.5*(He**2)/(sz**2+1e-12)))
    C[~mask]=0
    dlat=half/mlat; dlon=half/(mlon if mlon>0 else 111320)
    b=[[lat-dlat,lon-dlon],[lat+dlat,lon+dlon]]
    return C,b

def to_ppb(C,P_hPa,T):
    P = P_hPa*100.0
    return C * (R*T)/(M*P) * 1e9

def rgba_from_ppb(A, vmin=0, vmax=450, log=False):
    lut=(cm.get_cmap('jet',256)(np.linspace(0,1,256))[:,:3]*255).astype(np.uint8)
    if log:
        A=np.log10(np.maximum(A,1e-12)); vmin=np.log10(max(vmin,1e-12)+1e-12); vmax=np.log10(max(vmax,1e-12))
    N=np.clip((A-vmin)/(vmax-vmin+1e-12),0,1)
    idx=(N*255).astype(np.uint8)
    alpha=(np.sqrt(N)*255).astype(np.uint8); alpha[N<=0.003]=0
    rgb=lut[idx]
    return np.dstack([rgb,alpha]).astype(np.uint8)

# ---- passo 1: selecionar ponto
if ss["source"] is None:
    st.info("Clique no mapa para definir a **fonte**.")
    m0 = folium.Map(location=[-22.9035,-43.2096], zoom_start=15, control_scale=True)
    r = st_folium(m0, height=720, returned_objects=["last_clicked"], use_container_width=True)
    st.write("DEBUG last_clicked:", r.get("last_clicked") if r else None)
    if r and r.get("last_clicked"):
        ss["source"] = (r["last_clicked"]["lat"], r["last_clicked"]["lng"])
        ss["_update"] = True
else:
    lat, lon = ss["source"]
    st.success(f"Fonte selecionada: {lat:.6f}, {lon:.6f}")

    # kg/h -> g/s
    Q_gps = (Q_kgph*1000.0)/3600.0

    pars=dict(wd=wind_dir, u=wind_speed, stab=stability, urb=is_urban,
              Q=Q_gps, H=H_stack, d=d_stack, V=V_exit, Ta=Tamb, Ts=Tstack)

    if ss["_update"] or ss["overlay"] is None:
        C,bounds = calc(lat,lon,pars)
        Cppb = to_ppb(C, P_hPa, Tamb)
        RGBA = rgba_from_ppb(Cppb, V_MIN, V_MAX, log=False)
        img = Image.fromarray(RGBA, "RGBA")
        bio=io.BytesIO(); img.save(bio,"PNG"); bio.seek(0)
        ss["overlay"] = (bio.read(), bounds)
        ss["_update"] = False

    png, bounds = ss["overlay"]

    # mapa
    m1 = folium.Map(location=[lat,lon], zoom_start=15, control_scale=True)

    # overlay
    folium.raster_layers.ImageOverlay(
        image="data:image/png;base64,"+base64.b64encode(png).decode("utf-8"),
        bounds=bounds, opacity=0.9, name="Pluma").add_to(m1)
    folium.CircleMarker([lat,lon], radius=6, color="#f00", fill=True).add_to(m1)

    # --- LEGENDA ROBUSTA (LinearColormap) ---
    cm = LinearColormap(['purple','blue','cyan','green','yellow','red'], vmin=V_MIN, vmax=V_MAX)
    cm.caption = "[ppb]  (0 — 150 — 300 — 450)"
    cm.add_to(m1)

    folium.LayerControl(collapsed=False).add_to(m1)
    ret = st_folium(m1, height=720, use_container_width=True)
    # logs simples
    st.write("DEBUG map return keys:", list(ret.keys()) if ret else None)
