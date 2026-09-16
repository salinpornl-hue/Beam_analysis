# input_handler.py  ── Fixed & Production-Ready Version
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def render_all_sidebar_inputs():
    """
    Renders sidebar inputs for RC Beam Analysis.
    Units: mm for dimensions, ksc for material strength, kg(f) for loads.
    Internally converts all values to SI (MPa, kN, m) before returning.
    """
    st.sidebar.markdown("### 1. Material & Section")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        fc_ksc = st.number_input("f'c (ksc)", 100.0, 800.0, 240.0, step=10.0,
                                  help="กำลังอัดคอนกรีต (1 ksc ≈ 0.0981 MPa)")
        b = st.number_input("Width b (mm)", 100.0, 1000.0, 200.0, step=50.0)
    with col2:
        fy_ksc = st.number_input("fy main (ksc)", 2000.0, 6000.0, 4000.0, step=100.0,
                                  help="กำลังครากเหล็กเสริมหลัก SD40 = 4000 ksc")
        h = st.number_input("Depth h (mm)", 200.0, 2000.0, 400.0, step=50.0)

    # [FIX 1] Stirrups are a different grade from the main bars.  Thai RB
    # (SR24) is 2400 ksc; using the SD40 main-bar value overstated phi*Vn
    # by about 30%.
    fyt_ksc = st.sidebar.number_input(
        "fyt stirrup (ksc)", 2000.0, 5000.0, 2400.0, step=100.0,
        help="RB/SR24 = 2400 ksc, SD30 = 3000 ksc. ACI 20.2.2.4 caps fyt at 420 MPa")
    d_agg = st.sidebar.number_input(
        "Max aggregate size (mm)", 10.0, 40.0, 20.0, step=5.0,
        help="ACI 25.2.1 minimum clear bar spacing includes (4/3) x d_agg")

    # Unit conversion: ksc → MPa  (1 ksc = 0.0980665 MPa)
    KSC_TO_MPA = 0.0980665
    fc_mpa  = fc_ksc  * KSC_TO_MPA
    fy_mpa  = fy_ksc  * KSC_TO_MPA
    fyt_mpa = fyt_ksc * KSC_TO_MPA

    # Show converted values for reference
    st.sidebar.caption(
        f"ℹ️ f'c = {fc_mpa:.2f} MPa | fy = {fy_mpa:.2f} MPa | fyt = {fyt_mpa:.2f} MPa")
    if fyt_mpa > 420.0:
        st.sidebar.warning("fyt will be capped at 420 MPa for shear (ACI 20.2.2.4)")

    # Stiffness: Ec (kN/m²) and Ig (m⁴) for solver
    Ec_mpa  = 4700 * np.sqrt(fc_mpa)          # MPa
    E_kNm2  = Ec_mpa * 1000.0                  # kN/m²
    b_m, h_m = b / 1000.0, h / 1000.0
    I_g = (b_m * h_m**3) / 12.0               # m⁴

    params = {
        'fc':  fc_mpa,    # MPa  – used by design engine
        'fy':  fy_mpa,    # MPa  – main reinforcement
        'fyt': fyt_mpa,   # MPa  – [FIX 1] stirrups, separate grade
        'b':   b,         # mm   – used for drawing and design
        'h':   h,         # mm
        'E':   E_kNm2,    # kN/m²  – used by solver
        'I':   I_g,       # m⁴     – used by solver
        'd_agg': d_agg,   # mm   – [FIX 10] ACI 25.2.1 spacing
    }

    # --- 2. Geometry ---
    st.sidebar.markdown("### 2. Geometry")
    n_spans = st.sidebar.number_input("Number of Spans", 1, 10, 1)
    spans = []
    st_cols = st.sidebar.columns(min(n_spans, 4))
    for i in range(n_spans):
        with st_cols[i % 4]:
            l_val = st.number_input(f"L{i+1} (m)", 1.0, 20.0, 4.0, key=f"span_{i}")
            spans.append(l_val)

    # --- 3. Supports ---
    st.sidebar.markdown("### 3. Supports")
    node_coords = [0] + list(np.cumsum(spans))
    n_nodes = len(node_coords)
    default_sups = ["Pin"] + ["Roller"] * (n_nodes - 1)

    sup_data = []
    for i in range(n_nodes):
        stype = st.sidebar.selectbox(
            f"Node {i} (@{node_coords[i]:.2f}m)",
            ["None", "Pin", "Roller", "Fixed"],
            index=["None", "Pin", "Roller", "Fixed"].index(default_sups[i]),
            key=f"sup_{i}"
        )
        if stype != "None":
            sup_data.append({"id": i, "x": node_coords[i], "type": stype})
    sup_df = pd.DataFrame(sup_data)

    # --- 4. Loads ---
    st.sidebar.markdown("### 4. Loads")
    if 'load_list' not in st.session_state:
        st.session_state.load_list = []

    with st.sidebar.expander("➕ Add New Load", expanded=True):
        l_case = st.radio("Load Case", ["DL (Dead)", "LL (Live)"], horizontal=True)
        l_type = st.selectbox("Load Type", ["Point Load (P)", "Uniform Load (U)"])

        span_opts = [f"Span {i+1}" for i in range(n_spans)]
        l_span_idx = span_opts.index(st.selectbox("Select Span", span_opts))
        max_l = spans[l_span_idx]

        l_mag_kg = st.number_input(
            "Magnitude (kg or kg/m)", 0.0, 200000.0, 1000.0, step=100.0,
            help="กิโลกรัม สำหรับ Point Load, กิโลกรัม/เมตร สำหรับ Uniform Load"
        )

        d_start, d_end = 0.0, max_l
        if l_type == "Point Load (P)":
            d_start = st.slider("Position (m)", 0.0, max_l, max_l / 2)
            d_end = d_start
        else:
            c1, c2 = st.columns(2)
            with c1: d_start = st.number_input("Start (m)", 0.0, max_l, 0.0)
            with c2: d_end   = st.number_input("End (m)", d_start, max_l, max_l)

        if st.button("✅ Confirm & Add Load"):
            # 1 kgf = 9.80665 N = 0.00980665 kN
            mag_kN = l_mag_kg * 0.00980665
            st.session_state.load_list.append({
                "id":         len(st.session_state.load_list),
                "case":       "DL" if "DL" in l_case else "LL",
                "type":       "P" if "Point" in l_type else "U",
                "span_index": l_span_idx,
                "mag_kg":     l_mag_kg,      # display only
                "mag":        mag_kN,         # kN — used by solver
                "d_start":    d_start,
                "d_end":      d_end,
                "dist":       d_end - d_start
            })
            st.rerun()

    # --- Display load table ---
    loads_df = pd.DataFrame(st.session_state.load_list)
    required_cols = ['case', 'type', 'span_index', 'mag_kg', 'mag', 'd_start', 'd_end']
    display_cols  = ['case', 'type', 'span_index', 'mag_kg', 'd_start', 'd_end']

    if not loads_df.empty:
        if all(c in loads_df.columns for c in required_cols):
            st.sidebar.markdown("#### Active Loads")
            st.sidebar.dataframe(loads_df[display_cols], hide_index=True)
        else:
            st.sidebar.warning("⚠️ Data structure changed. Clearing loads…")
            st.session_state.load_list = []
            st.rerun()

        if st.sidebar.button("🗑️ Clear All Loads"):
            st.session_state.load_list = []
            st.rerun()

    # --- 5. Stability check ---
    fixed_dof = 0
    for s in sup_data:
        if s['type'] == 'Pin':    fixed_dof += 2
        elif s['type'] == 'Roller': fixed_dof += 1
        elif s['type'] == 'Fixed':  fixed_dof += 3
    stable = fixed_dof >= 3

    return params, n_spans, spans, sup_df, loads_df, stable
