import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import MeasureControl

# ==== CONFIG ====
st.set_page_config(page_title="Teste de Seleção de Ponto", layout="wide")
st.title("Teste: selecionar ponto no mapa (clique simples)")

# ==== ESTADO (sempre ANTES de usar ss) ====
ss = st.session_state
ss.setdefault("pending_click", None)
ss.setdefault("source", None)
ss.setdefault("locked", False)

# ==== SIDEBAR (cria widgets ANTES de usar os valores) ====
with st.sidebar:
    st.header("Controles")
    if st.button("Selecionar outro ponto"):
        ss.source = None
        ss.locked = False
        ss.pending_click = None

# ==== BLOCO DE SELEÇÃO ====
if ss.source is None or not ss.locked:
    st.info("🖱️ Clique UMA vez no mapa. Depois, clique em **Confirmar**.")

    center = ss.pending_click or (-22.90, -43.20)
    m = folium.Map(location=center, zoom_start=14, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)
    m.add_child(MeasureControl(primary_length_unit='meters', secondary_length_unit='kilometers'))
    folium.LatLngPopup().add_to(m)

    # marcador provisório
    if ss.pending_click is not None:
        folium.CircleMarker(ss.pending_click, radius=6, color="red", fill=True).add_to(m)

    ret = st_folium(m, height=520, key="map_select",
                    returned_objects=["last_clicked", "all_drawings", "last_draw"])

    # captura por clique simples
    new_pt = None
    lc = ret.get("last_clicked") if ret else None
    if lc and "lat" in lc and "lng" in lc:
        new_pt = (float(lc["lat"]), float(lc["lng"]))

    if new_pt is not None:
        if (ss.pending_click is None) or (tuple(round(x,7) for x in ss.pending_click) != tuple(round(x,7) for x in new_pt)):
            ss.pending_click = new_pt

    if ss.pending_click:
        lat, lon = ss.pending_click
        st.success(f"Ponto selecionado: {lat:.6f}, {lon:.6f}")
        confirm = st.button("✅ Confirmar este ponto", type="primary")
        if confirm:
            ss.source = ss.pending_click
            ss.locked = True
            ss.pending_click = None
            st.rerun()
    else:
        st.caption("Clique no mapa para habilitar o botão.")
else:
    # ==== PÓS-SELEÇÃO: só mostra o ponto fixo ====
    lat, lon = ss.source
    st.success(f"Fonte confirmada em: {lat:.6f}, {lon:.6f}")
    m = folium.Map(location=[lat, lon], zoom_start=15, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)
    m.add_child(MeasureControl(primary_length_unit='meters', secondary_length_unit='kilometers'))
    folium.CircleMarker([lat, lon], radius=8, color="#f00", fill=True, tooltip="Fonte").add_to(m)
    st_folium(m, height=560, key="map_final")

