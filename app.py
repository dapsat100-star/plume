# ================== SELEÇÃO (clique simples + marcador opcional) ==================
if ss.source is None or not ss.locked:
    st.info("🖱️ **Clique uma vez no mapa** para escolher o ponto **(não precisa Save)**. "
            "Se preferir, use o ícone de marcador e clique em **Save**.")

    center0 = ss.pending_click or (-22.9035, -43.2096)
    m_sel = folium.Map(location=center0, zoom_start=16, control_scale=True, zoom_control=True)
    folium.TileLayer("OpenStreetMap").add_to(m_sel)

    # Régua + Medição
    if ScaleBar: ScaleBar(position="bottomleft", imperial=False).add_to(m_sel)
    m_sel.add_child(MeasureControl(primary_length_unit='meters',
                                   secondary_length_unit='kilometers',
                                   position='topleft'))

    # Opcional: toolbar de marcador (continua funcionando para quem quiser)
    Draw(draw_options={"polyline": False, "polygon": False, "circle": False,
                       "circlemarker": False, "rectangle": False, "marker": True},
         edit_options={"edit": True, "remove": True}).add_to(m_sel)

    # Popup com lat/lon ao clicar (feedback instantâneo)
    folium.LatLngPopup().add_to(m_sel)

    # Se já havia ponto pendente, mostra um marcador provisório
    if ss.pending_click is not None:
        folium.CircleMarker(location=list(ss.pending_click), radius=6,
                            color="#e91e63", fill=True, fill_opacity=0.9,
                            tooltip="Ponto provisório").add_to(m_sel)

    # Render — use um key fixo para estabilizar e capturar 'last_clicked'
    ret = st_folium(
        m_sel,
        height=560,
        returned_objects=["all_drawings", "last_draw", "last_clicked"],
        key="map_select",
        use_container_width=True,
    )

    # -------- captura do ponto --------
    new_pt = None

    # (1) clique simples no mapa
    lc = ret.get("last_clicked") if ret else None
    if lc and "lat" in lc and "lng" in lc:
        new_pt = (float(lc["lat"]), float(lc["lng"]))

    # (2) marcador via Leaflet.draw (após Save)
    if new_pt is None:
        drawings = ret.get("all_drawings") if ret else None
        if drawings:
            for feat in drawings[::-1]:
                try:
                    if feat and feat["geometry"]["type"] == "Point":
                        lon, lat = feat["geometry"]["coordinates"]
                        new_pt = (float(lat), float(lon))
                        break
                except Exception:
                    pass
        if new_pt is None:
            ld = ret.get("last_draw") if ret else None
            if ld:
                try:
                    if ld["geometry"]["type"] == "Point":
                        lon, lat = ld["geometry"]["coordinates"]
                        new_pt = (float(lat), float(lon))
                except Exception:
                    pass

    # Salva só se realmente mudou (evita “piscar”)
    if new_pt is not None:
        if ss.pending_click is None or tuple(np.round(ss.pending_click, 7)) != tuple(np.round(new_pt, 7)):
            ss.pending_click = new_pt

    # Painel e botões
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
        ss.source = ss.pending_click
        ss.locked = True
        ss._update = True
        ss.pending_click = None
        st.success("Fonte confirmada. Gerando pluma…")

    if rm_btn and ss.pending_click is not None:
        ss.pending_click = None
        st.info("Seleção limpa. Clique novamente no mapa.")
