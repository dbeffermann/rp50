import streamlit as st
import pandas as pd

st.set_page_config(page_title="RP50 Assistant", layout="wide")
st.title("🎛️ RP50 Preset Assistant")
st.caption("App guía paso a paso + badges con mapa de color + shortcodes + explorador de datos")

# ===========================
# Helpers
# ===========================
def nz(x):
    return "" if pd.isna(x) or x is None else str(x)

def load_all(file):
    """Lee las hojas principales desde XLSX/CSV.
       CSV: solo 'User Presets'. 
       XLSX: User Presets + UI Hints + Category Map + Lookups + Shortcodes (si existen).
    """
    ui = catmap = lookups = shorts = None
    if file.name.endswith(".csv"):
        df = pd.read_csv(file).fillna("")
    else:
        xls = file
        df = pd.read_excel(xls, "User Presets", engine="openpyxl").fillna("")
        for name in ["UI Hints", "Category Map", "Lookups", "Shortcodes"]:
            try:
                obj = pd.read_excel(xls, name, engine="openpyxl").fillna("")
            except Exception:
                obj = None
            if name == "UI Hints": ui = obj
            if name == "Category Map": catmap = obj
            if name == "Lookups": lookups = obj
            if name == "Shortcodes": shorts = obj

    # normalizar Pos
    if "Pos" in df.columns:
        try:
            df["Pos"] = df["Pos"].astype(float).astype(int)
        except Exception:
            pass

    # claves mínimas
    if "Remarks" not in df.columns:
        df["Remarks"] = ""
    if "Shortcode" not in df.columns:
        df["Shortcode"] = ""
    if "Labels" not in df.columns:
        df["Labels"] = ""

    return df, ui, catmap, lookups, shorts

def color_of(field, ui):
    """ColorHex definido en UI Hints para el campo."""
    if ui is None or "Field" not in ui.columns or "ColorHex" not in ui.columns:
        return "#333333"
    row = ui[ui["Field"] == field]
    if row.empty:
        return "#333333"
    return str(row.iloc[0]["ColorHex"])

def explain_of(field, ui):
    """Explicación corta del campo (opcional, de UI Hints)."""
    if ui is None or "Field" not in ui.columns or "Explicación UI" not in ui.columns:
        return ""
    row = ui[ui["Field"] == field]
    return "" if row.empty else str(row.iloc[0]["Explicación UI"])

def badge(label, value, bg="#333333", big=False, help_text=""):
    """Cajita con color + valor + ayuda optativa."""
    v = "—" if str(value).strip() == "" else str(value)
    size = "1.25rem" if big else "1rem"
    html = f"""
    <div style="background:{bg};color:white;padding:10px 12px;border-radius:12px;margin:6px 0;font-size:{size};
                border:1px solid rgba(0,0,0,.15);">
        <b>{label}:</b> <code style="color:white">{v}</code>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    if help_text:
        st.caption(help_text)

def split_labels(series):
    """Devuelve lista de etiquetas únicas y contador por etiqueta."""
    from collections import Counter
    tokens = []
    for s in series.fillna("").astype(str).tolist():
        parts = [p.strip() for p in s.split(",") if p.strip()]
        tokens.extend(parts)
    cnt = Counter(tokens)
    # ordenar por frecuencia desc
    return [t for t,_ in cnt.most_common()], cnt

# ===========================
# Carga de archivo (en sidebar)
# ===========================
with st.sidebar:
    st.subheader("Archivo")
    uploaded = st.file_uploader(
        "Sube tu archivo (CSV o XLSX)",
        type=["csv", "xlsx"],
        label_visibility="visible",
        help="Usa el Excel/CSV que generamos con User Presets, UI Hints, etc."
    )

if "progress" not in st.session_state:
    st.session_state.progress = {}  # progreso por Pos

if not uploaded:
    st.info("Sube tu archivo para comenzar. Puedes usar el Excel/CSV que generamos.")
    st.stop()

df, ui, catmap, lookups, shorts = load_all(uploaded)

# ===========================
# Tabs
# ===========================
tab_assistant, tab_explore, tab_stats, tab_labels, tab_docs = st.tabs(
    ["🧭 Asistente", "🔎 Explorar datos", "📊 Estadísticas", "🏷️ Etiquetas / Búsqueda", "📘 Documentación"]
)

# ===========================
# 🧭 ASISTENTE
# ===========================
with tab_assistant:
    
    # --------- Vista del preset ----------
    #with col_right:
    pos=4
    cur = df[df["Pos"] == int(pos)]
    if cur.empty:
        st.error("No se encontró la fila para ese Pos.")
        st.stop()
    row = cur.iloc[0]

    st.subheader(f"Preset P{int(row['Pos']):02d} — {row.get('Remarks','')}")
    #st.code(row.get("Shortcode", ""), language="text")

    #with st.expander("🧭 Flujo de entrenamiento RP50 (orden de edición)"):
    #    st.markdown("""
    #    1. **Level** → Volumen  
    #    2. **Pickup/Wah**  
    #    3. **Compressor**  
    #    4. **Amp Model**  
    #    5. **Noise Gate**  
    #    6. **EQ** (Bass → Mid → Treble)  
    #    7. **Chorus/Mod** (Type → Var)  
    #    8. **Delay** (Type/Var → Time)  
    #    9. **Reverb**  
    #    10. **Expression** (Vol / Wah / Whammy / Off)
    #    """)

    st.subheader("Paso a paso")
    c1, c2, c3 = st.columns(3)
    with c1:
        only_filled = st.checkbox("Solo campos con valor", True)
    with c2:
        hide_off = st.checkbox("Ocultar módulos 'Off'", True)
    with c3:
        big_text = st.checkbox("Modo letra grande", False)

    fields = [
        "Level","Pickup/Wah","Compressor (0-15)","Amp Model","Noise Gate",
        "EQ Bass (b1-b9)","EQ Mid (d1-d9)","EQ Treble (t1-t9)",
        "Chorus/Mod Type","Chorus/Mod Var",
        "Delay Type/Var","Delay Time (1-99,1.0,2.0)","Reverb",
        "Expression (Vol/Wah/Whammy/Off)",
    ]

    def keep(label, value):
        v = str(value).strip()
        if only_filled and v == "": 
            return False
        if hide_off and label in ["Chorus/Mod Type","Expression (Vol/Wah/Whammy/Off)"] and v.lower()=="off":
            return False
        return True

    fields_to_show = [(f, row.get(f,"")) for f in fields if keep(f, row.get(f,""))]
    cols = st.columns(4)
    for i, (label, value) in enumerate(fields_to_show):
        with cols[i % 4]:
            badge(label, value, bg=color_of(label, ui), big=big_text, help_text=explain_of(label, ui))

    st.markdown("### Resumen tonal")
    tcols = st.columns(3)
    with tcols[0]:
        badge("Gain Profile", row.get("Gain Profile",""), bg=color_of("Gain Profile", ui), big=big_text, help_text=explain_of("Gain Profile", ui))
    with tcols[1]:
        badge("Delay Class", row.get("Delay Class",""), bg=color_of("Delay Class", ui), big=big_text, help_text=explain_of("Delay Class", ui))
    with tcols[2]:
        badge("Reverb Size", row.get("Reverb Size",""), bg=color_of("Reverb", ui), big=big_text, help_text=explain_of("Reverb", ui))

    st.markdown("### Complejidad")
    badge("Complexity Score (0-7)", row.get("Complexity Score (0-7)",""), bg=color_of("Complexity Score (0-7)", ui), big=big_text, help_text=explain_of("Complexity Score (0-7)", ui))

# ===========================
# 🔎 EXPLORAR DATOS
# ===========================
with tab_explore:
    #st.subheader("Hojas disponibles")
    tabs_inner = st.tabs(["User Presets", "UI Hints", "Category Map", "Lookups", "Shortcodes"])
    with tabs_inner[0]:
        st.markdown("**User Presets**")
        st.dataframe(df, use_container_width=True, height=520)
    with tabs_inner[1]:
        st.markdown("**UI Hints** (colores, criticidad y ayudas de UI)")
        if ui is not None:
            st.dataframe(ui, use_container_width=True, height=520)
        else:
            st.info("No se encontró la hoja 'UI Hints' en este archivo.")
    with tabs_inner[2]:
        st.markdown("**Category Map** (índice por slot)")
        if catmap is not None:
            st.dataframe(catmap, use_container_width=True, height=520)
        else:
            st.info("No se encontró la hoja 'Category Map'.")
    with tabs_inner[3]:
        st.markdown("**Lookups** (valores válidos + explicación práctica)")
        if lookups is not None:
            st.dataframe(lookups, use_container_width=True, height=520)
        else:
            st.info("No se encontró la hoja 'Lookups'.")
    with tabs_inner[4]:
        st.markdown("**Shortcodes** (recetas compactas)")
        if shorts is not None:
            st.dataframe(shorts, use_container_width=True, height=520)
        else:
            st.info("No se encontró la hoja 'Shortcodes'.")

# ===========================
# 📊 ESTADÍSTICAS
# ===========================
with tab_stats:
    st.subheader("Distribuciones clave")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Gain Profile**")
        st.bar_chart(df["Gain Profile"].value_counts(), use_container_width=True, height=260)
    with cols[1]:
        st.markdown("**Delay Class**")
        st.bar_chart(df["Delay Class"].value_counts(), use_container_width=True, height=260)
    with cols[2]:
        st.markdown("**Reverb Size**")
        st.bar_chart(df["Reverb Size"].value_counts(), use_container_width=True, height=260)

    st.markdown("**Complejidad (0–7)**")
    st.bar_chart(df["Complexity Score (0-7)"].value_counts().sort_index(), use_container_width=True, height=260)

    # Heatmap Gain x Delay con Altair (si disponible)
    try:
        import altair as alt
        st.markdown("**Heatmap: Gain Profile × Delay Class (conteo)**")
        heat_df = (
            df.groupby(["Gain Profile","Delay Class"])
              .size().reset_index(name="count")
        )
        chart = (
            alt.Chart(heat_df)
            .mark_rect()
            .encode(
                x=alt.X("Gain Profile:N", sort="-y"),
                y=alt.Y("Delay Class:N"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["Gain Profile","Delay Class","count"]
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.info("Instala 'altair' para ver el heatmap (pip install altair).")

# ===========================
# 🏷️ ETIQUETAS / BÚSQUEDA
# ===========================
with tab_labels:
    st.subheader("Explorar por etiquetas (Labels)")
    if "Labels" not in df.columns:
        st.info("Este archivo no tiene columna 'Labels'.")
    else:
        all_tags, counts = split_labels(df["Labels"])

        # --- Selectbox rápido (Top N) ---
        top_n = st.slider("Ver top N etiquetas", 10, min(100, len(all_tags) if all_tags else 10), 20)
        top_tags = all_tags[:top_n]

        row1 = st.columns([1, 1, 2])
        with row1[0]:
            st.markdown("**Etiqueta rápida (Top)**")
            quick_tag = st.selectbox(
                "Elegir (Top)",
                options=["(ninguna)"] + top_tags,
                index=0
            )
        with row1[1]:
            if st.button("➕ Añadir etiqueta rápida", use_container_width=True, disabled=(quick_tag == "(ninguna)")):
                sel = st.session_state.get("selected_tags", set())
                if quick_tag != "(ninguna)":
                    sel.add(quick_tag)
                st.session_state["selected_tags"] = sel
        with row1[2]:
            st.markdown("**Etiquetas más usadas (chips)**")
            tag_cols = st.columns(5)
            selected = st.session_state.get("selected_tags", set())
            for i, t in enumerate(top_tags):
                with tag_cols[i % 5]:
                    if st.button(f"{t} ({counts[t]})", key=f"chip_{i}"):
                        # toggle
                        if t in selected:
                            selected.remove(t)
                        else:
                            selected.add(t)
            st.session_state["selected_tags"] = selected

        # --- Controles de selección y limpieza ---
        st.caption("Seleccionadas: " + (", ".join(sorted(st.session_state.get('selected_tags', set()))) or "ninguna"))
        csel1, csel2 = st.columns([3, 1])
        with csel1:
            extra = st.multiselect(
                "Agregar/editar etiquetas manualmente",
                options=all_tags,
                default=list(st.session_state.get("selected_tags", set()))
            )
        with csel2:
            st.markdown("&nbsp;")
            if st.button("🧹 Limpiar filtros", use_container_width=True):
                st.session_state["selected_tags"] = set()
                extra = []  # vaciar también el multiselect

        st.session_state["selected_tags"] = set(extra)

        # --- Filtrado final ---
        selected = st.session_state["selected_tags"]
        if selected:
            mask = df["Labels"].fillna("").apply(lambda s: all(tag in s for tag in selected))
            st.markdown(f"**Resultados que contienen**: {', '.join(sorted(selected))}")
            st.dataframe(
                df[mask][["Pos","Remarks","Labels","Shortcode"]],
                use_container_width=True, height=500
            )
        else:
            st.info("Elige etiquetas para filtrar. Consejo: prueba 'gilmour', 'wah', 'ambient', 'spring', 'metallica'…")

# ===========================
# 📘 DOCUMENTACIÓN
# ===========================
with tab_docs:
    st.subheader("Guías y referencias")
    st.markdown("""
    - **Diagrama del panel/luces (PDF)**: si descargaste el paquete, ábrelo junto a la app para referencia rápida.
    - **Lookups**: tabla con códigos y explicaciones prácticas por campo.
    - **UI Hints**: metadatos de interfaz (colores, criticidad) que esta app usa para las cajitas.
    """)
    st.markdown("> Tip: puedes editar 'UI Hints' en el Excel para cambiar colores o prioridades **sin tocar el código**.")
