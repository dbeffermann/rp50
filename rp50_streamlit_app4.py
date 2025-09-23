import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="RP50 Assistant", layout="wide")
st.title("🎛️ RP50 Preset Assistant")

# ===========================
# Helpers de datos/UI (solo Streamlit)
# ===========================
def nz(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)

def valid_url(s: str) -> bool:
    s = (s or "").strip()
    return s.startswith("http://") or s.startswith("https://")

def load_all(file):
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

    if "Pos" in df.columns:
        df["Pos"] = pd.to_numeric(df["Pos"], errors="coerce").astype("Int64")

    for col in ["Remarks", "Shortcode", "Labels"]:
        if col not in df.columns:
            df[col] = ""
    if "Artist Image URL" not in df.columns:
        df["Artist Image URL"] = ""
    if "Artist Style" not in df.columns:
        df["Artist Style"] = ""
    return df, ui, catmap, lookups, shorts

def parse_gate(val):
    s = str(val or "").strip()
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else 0

def parse_eq_token(tok, default=5):
    if tok is None:
        return default
    s = str(tok).strip().lower()
    if s == "":
        return default
    m = re.search(r'(\d+)', s)
    if not m:
        return default
    n = int(m.group(1))
    return max(1, min(n, 9))

def to_float(x, default=0.0):
    try:
        return float(str(x).strip())
    except Exception:
        return default

def features_vector(row):
    """
    Vector numérico normalizado (0..1) para comparaciones:
    [Level, Comp, Gate, Bass, Mid, Treble]
    """
    level = to_float(row.get("Level", 0)) / 99.0
    comp  = to_float(row.get("Compressor (0-15)", 0)) / 15.0
    gate  = parse_gate(row.get("Noise Gate", 0)) / 99.0
    b = (parse_eq_token(row.get("EQ Bass (b1-b9)"), 5) - 1) / 8.0
    m = (parse_eq_token(row.get("EQ Mid (d1-d9)"), 5) - 1) / 8.0
    t = (parse_eq_token(row.get("EQ Treble (t1-t9)"), 5) - 1) / 8.0
    return np.array([level, comp, gate, b, m, t])

def preset_label(row):
    pos = int(row["Pos"]) if pd.notna(row["Pos"]) else -1
    rem = nz(row.get("Remarks", ""))
    return f"P{pos:02d} — {rem}"

def draw_radar_single(row):
    level = to_float(row.get("Level", 0)) / 99.0
    comp  = to_float(row.get("Compressor (0-15)", 0)) / 15.0
    gate  = parse_gate(row.get("Noise Gate", 0)) / 99.0
    b = (parse_eq_token(row.get("EQ Bass (b1-b9)"), 5) - 1) / 8.0
    m = (parse_eq_token(row.get("EQ Mid (d1-d9)"), 5) - 1) / 8.0
    t = (parse_eq_token(row.get("EQ Treble (t1-t9)"), 5) - 1) / 8.0

    vals = np.array([level, comp, gate, b, m, t])
    labels = ["Level", "Comp", "Gate", "Bass", "Mid", "Treble"]
    vals = np.append(vals, vals[0])
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.append(angles, angles[0])

    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_yticklabels([]); ax.set_ylim(0, 1)
    st.pyplot(fig, clear_figure=True)

def draw_radar_multi(rows, highlight_idx=None, label_suffix=None, thin_alpha=0.15):
    """
    Radar multi-traza: dibuja muchas curvas (0..N).
    - highlight_idx: índice en la lista 'rows' que se resalta.
    - label_suffix: función que dado un row retorna texto extra para etiqueta.
    """
    labels = ["Level", "Comp", "Gate", "Bass", "Mid", "Treble"]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.append(angles, angles[0])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # Trazo para cada fila
    for i, r in enumerate(rows):
        v = features_vector(r)
        v = np.append(v, v[0])
        if highlight_idx is not None and i == highlight_idx:
            ax.plot(angles, v, linewidth=2.5)
            ax.fill(angles, v, alpha=0.20)
        else:
            ax.plot(angles, v, linewidth=1.0, alpha=thin_alpha)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_yticklabels([]); ax.set_ylim(0, 1)
    st.pyplot(fig, clear_figure=True)

def draw_radar_compare(current_row, compare_rows):
    """
    Radar comparador: actual + N selecciones.
    """
    labels = ["Level", "Comp", "Gate", "Bass", "Mid", "Treble"]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.append(angles, angles[0])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # Actual (destacado)
    v0 = features_vector(current_row)
    v0 = np.append(v0, v0[0])
    ax.plot(angles, v0, linewidth=2.5)
    ax.fill(angles, v0, alpha=0.25)

    # Comparados
    for r in compare_rows:
        v = features_vector(r)
        v = np.append(v, v[0])
        ax.plot(angles, v, linewidth=1.5, alpha=0.7)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_yticklabels([]); ax.set_ylim(0, 1)
    st.pyplot(fig, clear_figure=True)

def table_diffs(current_row, compare_rows):
    """
    Tabla de diferencias por parámetro del vector normalizado.
    (valor_comp - valor_actual), para cada seleccionado.
    """
    labels = ["Level", "Comp", "Gate", "Bass", "Mid", "Treble"]
    base = features_vector(current_row)
    records = []
    for r in compare_rows:
        v = features_vector(r)
        dif = (v - base)
        rec = {"Preset": preset_label(r)}
        for i, lb in enumerate(labels):
            rec[lb] = round(float(dif[i]), 3)
        records.append(rec)
    if records:
        df_d = pd.DataFrame(records, columns=["Preset"] + labels)
    else:
        df_d = pd.DataFrame(columns=["Preset"] + labels)
    return df_d

def preset_table_dict(row):
    return {
        "Level": nz(row.get("Level", "")),
        "Compressor (0-15)": nz(row.get("Compressor (0-15)", "")),
        "Noise Gate": nz(row.get("Noise Gate", "")),
        "Pickup/Wah": nz(row.get("Pickup/Wah", "")),
        "Amp Model": nz(row.get("Amp Model", "")),
        "EQ Bass (b1-b9)": nz(row.get("EQ Bass (b1-b9)", "")),
        "EQ Mid (d1-d9)": nz(row.get("EQ Mid (d1-d9)", "")),
        "EQ Treble (t1-t9)": nz(row.get("EQ Treble (t1-t9)", "")),
        "Chorus/Mod Type": nz(row.get("Chorus/Mod Type", "")),
        "Chorus/Mod Var": nz(row.get("Chorus/Mod Var", "")),
        "Delay Type/Var": nz(row.get("Delay Type/Var", "")),
        "Delay Time (1-99,1.0,2.0)": nz(row.get("Delay Time (1-99,1.0,2.0)", "")),
        "Reverb": nz(row.get("Reverb", "")),
        "Expression": nz(row.get("Expression (Vol/Wah/Whammy/Off)", "")),
        "Gain Profile": nz(row.get("Gain Profile", "")),
        "Delay Class": nz(row.get("Delay Class", "")),
        "Reverb Size": nz(row.get("Reverb Size", "")),
        "Shortcode": nz(row.get("Shortcode", "")),
    }

# ===========================
# Carga de archivo (sidebar)
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
    st.session_state.progress = {}

if not uploaded:
    st.info("Sube tu archivo para comenzar. Puedes usar el Excel/CSV que generamos.")
    st.stop()

df, ui, catmap, lookups, shorts = load_all(uploaded)

# Índice / navegación
try:
    pos_list = sorted(pd.to_numeric(df["Pos"], errors="coerce").dropna().astype(int).unique().tolist())
except Exception:
    pos_list = []
st.session_state.pos_list = pos_list
if "pos_idx" not in st.session_state:
    st.session_state.pos_idx = 0
if not st.session_state.pos_list:
    st.error("No hay valores válidos en la columna 'Pos'."); st.stop()
st.session_state.pos_idx = max(0, min(st.session_state.pos_idx, len(st.session_state.pos_list)-1))

# ===========================
# Tabs
# ===========================
tab_assistant, tab_compare, tab_allradial, tab_rp50, tab_explore, tab_stats, tab_labels, tab_docs = st.tabs(
    ["🧭 Asistente", "🧮 Comparador", "🌐 Radial (todos)", "🎚️ RP50", "🔎 Explorar datos", "📊 Estadísticas", "🏷️ Etiquetas / Búsqueda", "📘 Documentación"]
)

# ===========================
# 🧭 ASISTENTE (Layout Artista)
# ===========================
with tab_assistant:
    top = st.columns([1,1,6,3])
    with top[0]:
        if st.button("⟵ Anterior", use_container_width=True):
            st.session_state.pos_idx = (st.session_state.pos_idx - 1) % len(st.session_state.pos_list)
    with top[1]:
        if st.button("Siguiente ⟶", use_container_width=True):
            st.session_state.pos_idx = (st.session_state.pos_idx + 1) % len(st.session_state.pos_list)

    pos_actual = st.session_state.pos_list[st.session_state.pos_idx]

    with top[2]:
        df_titles = df.copy()
        df_titles["Pos"] = pd.to_numeric(df_titles["Pos"], errors="coerce").astype("Int64")
        df_titles = df_titles[df_titles["Pos"].isin(st.session_state.pos_list)].drop_duplicates("Pos").sort_values("Pos")
        opciones = [f"P{int(p):02d} — {str(r)}" for p, r in zip(df_titles["Pos"], df_titles.get("Remarks",""))]
        try:
            idx_actual = df_titles.index[df_titles["Pos"] == pos_actual][0]
            idx_in_options = list(df_titles.index).index(idx_actual)
        except Exception:
            idx_in_options = 0 if len(opciones)>0 else 0
        label_sel = st.selectbox("Ir por nombre (Remarks):", options=opciones, index=idx_in_options)
        if opciones:
            chosen_idx = list(df_titles.index)[opciones.index(label_sel)]
            pos_from_label = int(df_titles.loc[chosen_idx,"Pos"])
            if pos_from_label != pos_actual:
                st.session_state.pos_idx = st.session_state.pos_list.index(pos_from_label)
                pos_actual = pos_from_label
    with top[3]:
        st.caption("Layout: Imagen + Reseña + Tabla + Radar")

    cur = df[pd.to_numeric(df["Pos"], errors="coerce").astype("Int64") == pd.Series([pos_actual], dtype="Int64").iloc[0]]
    if cur.empty:
        st.error("No se encontró la fila para ese Pos."); st.stop()
    row = cur.iloc[0]

    st.subheader(f"Preset P{int(row['Pos']):02d} — {row.get('Remarks','')}")
    col_left, col_right = st.columns([2, 2])

    with col_left:
        st.markdown("**Artista / Referencia**")
        img_url = nz(row.get("Artist Image URL", "")).strip()
        if not valid_url(img_url):
            img_url = "https://placekitten.com/800/1100"
        st.image(img_url, use_column_width=True, caption="Artista")

    style_text = nz(row.get("Artist Style", ""))
    if not style_text:
        style_text = (
            "Sonido orientado a tonos limpios con compresión ligera, "
            "delays rítmicos y reverbs espaciosas. Modulación sutil para abrir el estéreo."
        )
    data_dict = preset_table_dict(row)
    df_view = pd.DataFrame({"Parámetro": list(data_dict.keys()), "Valor": list(data_dict.values())})

    with col_right:
        with st.expander("#### Reseña del estilo", expanded=True):
            st.write(style_text)
        with st.expander("#### Valores del preset", expanded=True):
            st.dataframe(df_view, use_container_width=True, height=320)
        with st.expander("#### Visualización (radial)", expanded=True):
            draw_radar_single(row)

# ===========================
# 🧮 COMPARADOR (multi-select)
# ===========================
with tab_compare:
    # Actual
    pos_actual = st.session_state.pos_list[st.session_state.pos_idx]
    cur = df[pd.to_numeric(df["Pos"], errors="coerce") == pos_actual]
    row = cur.iloc[0]

    st.subheader(f"Comparar contra P{int(row['Pos']):02d} — {row.get('Remarks','')}")
    # Opciones
    df_opts = df.sort_values("Pos")
    labels_opts = [preset_label(r) for _, r in df_opts.iterrows()]
    default_idx = []
    # Multiselect
    selection = st.multiselect(
        "Elige uno o más presets para comparar",
        options=labels_opts,
        default=[labels_opts[min(st.session_state.pos_idx+1, len(labels_opts)-1)]] if labels_opts else []
    )
    # Map a rows
    label_to_row = {preset_label(r): r for _, r in df_opts.iterrows()}
    compare_rows = [label_to_row[s] for s in selection if s in label_to_row]

    cols = st.columns([1,1])
    with cols[0]:
        st.markdown("**Radar: Actual vs Seleccionados**")
        draw_radar_compare(row, compare_rows)
    with cols[1]:
        st.markdown("**Diferencias normalizadas (comp − actual)**")
        st.caption("Rango −1..+1 aprox. (0 = igual).")
        diffs_df = table_diffs(row, compare_rows)
        st.dataframe(diffs_df, use_container_width=True, height=420)

# ===========================
# 🌐 RADIAL (TODOS)
# ===========================
with tab_allradial:
    st.subheader("Vista radial de todo el dataset")
    pos_actual = st.session_state.pos_list[st.session_state.pos_idx]
    # Ordenar por Pos y armar lista de dicts con las filas (para el radar multi)
    df_sorted = df.sort_values("Pos")
    rows_all = [r for _, r in df_sorted.iterrows()]
    # Encontrar índice del actual en la lista ordenada
    try:
        idx_high = [int(r["Pos"]) for r in rows_all].index(int(pos_actual))
    except Exception:
        idx_high = None
    draw_radar_multi(rows_all, highlight_idx=idx_high)

# ===========================
# 🎚️ RP50 (Guía específica)
# ===========================
with tab_rp50:
    st.subheader("Digitech RP50 — Guía rápida")
    st.write(
        "Sugerencia de **flujo de edición** para entrenar/ajustar un preset en el RP50:"
    )
    st.markdown(
        "- **Level** → equilibrar volumen\n"
        "- **Pickup/Wah** → OFF / Wah / AutoYa / Envelope\n"
        "- **Compressor** → control de dinámica (0–15)\n"
        "- **Amp Model** → carácter del drive (c* combos, s* stacks, t* tweed, o* boutique, r* rectifier, G* high-gain, AC acoustic, etc.)\n"
        "- **Noise Gate** → limpia ruido si subes el gain\n"
        "- **EQ** → Bass (b1..b9), Mid (d1..d9), Treble (t1..t9)\n"
        "- **Chorus/Mod** → Chorus/Phaser/Flanger/Rotary/Detune/Vibrato/Univibe/Whammy/Off\n"
        "- **Delay** → analógico/‘p’ ping-pong/‘d’ digital; **Time**: 1–99 pasos o 1.0/2.0 (s)\n"
        "- **Reverb** → Hall/Plate/Spring/Arena/Church…\n"
        "- **Expression** → Vol / Wah / Whammy / Off"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Rangos útiles**")
        st.table(pd.DataFrame({
            "Parámetro": ["Level", "Comp", "Gate", "Bass/Mid/Treble", "Delay Time", "Expr"],
            "Rango": ["0–99", "0–15", "G0–G9 (0–99)", "b/d/t 1–9", "1–99 / 1.0 / 2.0 (s)", "Vol/Wah/Whammy/Off"]
        }))
    with c2:
        st.markdown("**Consejos**")
        st.write(
            "- Empieza con **modulación Off** y **EQ en 5/5/5**; mueve de a poco.\n"
            "- Si usas Whammy, pon Gate moderado y compensa Level.\n"
            "- Para leads largos: **Delay 400–450 ms** y **Hall/Plate** suave.\n"
            "- Para funk: **Comp 4–6**, **Spring**, **AutoYa/Envelope**.\n"
        )

# ===========================
# 🔎 EXPLORAR DATOS
# ===========================
with tab_explore:
    tabs_inner = st.tabs(["User Presets", "UI Hints", "Category Map", "Lookups", "Shortcodes"])
    with tabs_inner[0]:
        st.markdown("**User Presets**")
        st.dataframe(df, use_container_width=True, height=520)
    with tabs_inner[1]:
        st.markdown("**UI Hints** (colores, criticidad y ayudas de UI)")
        if ui is not None:
            st.dataframe(ui, use_container_width=True, height=520)
        else:
            st.info("No se encontró la hoja 'UI Hints'.")
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
    try:
        st.bar_chart(df["Complexity Score (0-7)"].value_counts().sort_index(), use_container_width=True, height=260)
    except Exception:
        st.info("No se encontró la columna 'Complexity Score (0-7)'.")

    try:
        import altair as alt
        st.markdown("**Heatmap: Gain Profile × Delay Class (conteo)**")
        heat_df = df.groupby(["Gain Profile","Delay Class"]).size().reset_index(name="count")
        chart = (
            alt.Chart(heat_df)
            .mark_rect()
            .encode(
                x=alt.X("Gain Profile:N", sort="-y"),
                y=alt.Y("Delay Class:N"),
                color=alt.Color("count:Q"),
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
        from collections import Counter
        tokens = []
        for s in df["Labels"].fillna("").astype(str).tolist():
            parts = [p.strip() for p in s.split(",") if p.strip()]
            tokens.extend(parts)
        cnt = Counter(tokens)
        all_tags = [t for t,_ in cnt.most_common()]

        top_n = st.slider("Ver top N etiquetas", 10, min(100, len(all_tags) if all_tags else 10), 20)
        top_tags = all_tags[:top_n]

        row1 = st.columns([1, 1, 2])
        with row1[0]:
            st.markdown("**Etiqueta rápida (Top)**")
            quick_tag = st.selectbox("Elegir (Top)", options=["(ninguna)"] + top_tags, index=0)
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
                    if st.button(f"{t} ({cnt[t]})", key=f"chip_{i}"):
                        if t in selected:
                            selected.remove(t)
                        else:
                            selected.add(t)
            st.session_state["selected_tags"] = selected

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
                extra = []
        st.session_state["selected_tags"] = set(extra)

        selected = st.session_state["selected_tags"]
        if selected:
            mask = df["Labels"].fillna("").apply(lambda s: all(tag in s for tag in selected))
            st.markdown(f"**Resultados que contienen**: {', '.join(sorted(selected))}")
            st.dataframe(df[mask][["Pos","Remarks","Labels","Shortcode"]], use_container_width=True, height=500)
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
    st.markdown("> Tip: agrega columnas **'Artist Image URL'** y **'Artist Style'** en tu Excel para completar la vista del artista.")
