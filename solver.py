# solver.py  ── Fixed & Production-Ready Version
import numpy as np
import pandas as pd

def safe_float(val, default=0.0):
    try:
        if pd.isna(val) or str(val).strip() == '': return default
        return float(val)
    except:
        return default

def _fem_udl_hermite(w, a, b_e, L):
    """
    Closed-form fixed-end forces for partial UDL w from x=a to x=b_e on span L.
    Uses Hermite shape function integration (Euler-Bernoulli beam).
    Sign convention: upward force +, CCW moment +.
    """
    def F1(x): return w * (x - x**3 / L**2 + x**4 / (2 * L**3))
    def F2(x): return w * (x**2 / 2 - 2 * x**3 / (3 * L) + x**4 / (4 * L**2))
    def F3(x): return w * (x**3 / L**2 - x**4 / (2 * L**3))
    def F4(x): return w * (x**4 / (4 * L**2) - x**3 / (3 * L))
    return (F1(b_e) - F1(a), F2(b_e) - F2(a),
            F3(b_e) - F3(a), F4(b_e) - F4(a))


def solve_beam(spans, sup_df, loads_df, params):
    # --- 0.1 Safety Check ---
    if loads_df is None or loads_df.empty or 'span_index' not in loads_df.columns:
        loads_df = pd.DataFrame(columns=['span_index', 'type', 'mag', 'dist', 'd_start'])
    if sup_df is None or sup_df.empty:
        sup_df = pd.DataFrame([{'id': 0, 'type': 'Pinned'}, {'id': len(spans), 'type': 'Pinned'}])

    # --- 0.2 Parameter Calculation ---
    b_raw = safe_float(params.get('b', 300), 300)
    h_raw = safe_float(params.get('h', 500), 500)
    b = b_raw / 1000.0 if b_raw >= 10 else b_raw
    h = h_raw / 1000.0 if h_raw >= 10 else h_raw

    if 'I' in params:
        I = safe_float(params['I'], (b * h**3) / 12.0)
        if I > 1: I = I / 1e12
    else:
        I = (b * h**3) / 12.0

    if 'fc' in params:
        fc_mpa = safe_float(params['fc'])
        E_mpa  = 4700 * np.sqrt(fc_mpa) if fc_mpa > 0 else 25000.0
        E      = E_mpa * 1000.0   # kN/m²
    else:
        E = safe_float(params.get('E', 25e6), 25e6)
        if E > 1e8:
            E = E / 1000.0

    nu       = 0.2
    G        = E / (2.0 * (1.0 + nu))
    k_factor = 5.0 / 6.0
    As_shear = k_factor * b * h

    n_spans     = len(spans)
    n_nodes     = n_spans + 1
    node_coords = [0] + list(np.cumsum(spans))
    n_dof       = 2 * n_nodes
    K_global    = np.zeros((n_dof, n_dof))
    F_global    = np.zeros(n_dof)

    # --- 1. Build Stiffness Matrix (Timoshenko Beam) ---
    for i in range(n_spans):
        L   = safe_float(spans[i], 1.0)
        Phi = (12 * E * I) / (G * As_shear * L**2)
        const = (E * I) / ((1 + Phi) * L**3)
        k11 = 12; k12 = 6*L; k22 = (4+Phi)*L**2; k24 = (2-Phi)*L**2
        k_ele = const * np.array([
            [ k11,  k12, -k11,  k12],
            [ k12,  k22, -k12,  k24],
            [-k11, -k12,  k11, -k12],
            [ k12,  k24, -k12,  k22]
        ])
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        for r in range(4):
            for c in range(4):
                K_global[idx[r], idx[c]] += k_ele[r, c]

    fea_local = [np.zeros(4) for _ in range(n_spans)]

    # --- 2. Process Loads → Fixed-End Actions ---
    for _, load in loads_df.iterrows():
        try:
            span_idx = int(safe_float(load.get('span_index', -1)))
            if span_idx < 0 or span_idx >= n_spans: continue
            L   = spans[span_idx]
            mag = safe_float(load.get('mag', 0.0))
            if mag == 0.0: continue

            idx    = [2*span_idx, 2*span_idx+1, 2*(span_idx+1), 2*(span_idx+1)+1]
            fea    = np.zeros(4)
            l_type = str(load.get('type', 'P')).strip().upper()

            if l_type in ['P', 'POINT', 'POINT LOAD']:
                P       = mag
                a_loc   = safe_float(load.get('d_start', 0.0))
                a_loc   = max(0.0, min(L, a_loc))
                b_dist  = L - a_loc
                fea[0]  =  (P * b_dist**2 * (3*a_loc + b_dist)) / L**3
                fea[1]  =  (P * a_loc * b_dist**2) / L**2
                fea[2]  =  (P * a_loc**2 * (a_loc + 3*b_dist)) / L**3
                fea[3]  = -(P * a_loc**2 * b_dist) / L**2

            elif l_type in ['U', 'UNIFORM', 'DISTRIBUTED', 'LINE']:
                w      = mag
                a_load = safe_float(load.get('d_start', 0.0))
                b_load = a_load + safe_float(load.get('dist', L))
                a_load = max(0.0, min(L, a_load))
                b_load = max(a_load, min(L, b_load))
                if b_load > a_load:
                    r1, m1, r2, m2 = _fem_udl_hermite(w, a_load, b_load, L)
                    fea[0] = r1; fea[1] = m1; fea[2] = r2; fea[3] = m2

            fea_local[span_idx] += fea
            F_global[idx[0]] -= fea[0]; F_global[idx[1]] -= fea[1]
            F_global[idx[2]] -= fea[2]; F_global[idx[3]] -= fea[3]
        except Exception as exc:
            # [FIX] a dropped load silently changes the answer.  Fail loudly.
            raise ValueError(
                f"Load row could not be processed: {dict(load)} ({exc})"
            ) from exc

    # --- 3. Apply Boundary Conditions ---
    fixed_dofs = []
    for i, row in sup_df.iterrows():
        node_idx = int(safe_float(row.get('id', i)))
        if node_idx >= n_nodes: continue
        fixed_dofs.append(2 * node_idx)
        if str(row.get('type', '')).strip().title() == 'Fixed':
            fixed_dofs.append(2 * node_idx + 1)

    free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_ff = F_global[free_dofs]

    try:
        d_free = np.linalg.solve(K_ff, F_ff)
    except np.linalg.LinAlgError:
        dummy_x = np.linspace(0, sum(spans), 10)
        return dummy_x, np.zeros(10), np.zeros(10), np.zeros(10), {"Error": "Support ไม่สมบูรณ์"}

    d_all              = np.zeros(n_dof)
    d_all[free_dofs]   = d_free

    # --- 4. Post-Processing ---
    x_total, moment_total, shear_total, def_total = [], [], [], []
    has_any_load = not loads_df.empty and (loads_df.get('mag', pd.Series([0])) != 0).any()

    for i in range(n_spans):
        L   = spans[i]
        x0  = node_coords[i]
        u_ele = d_all[[2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]]

        # Collect critical x-points
        points     = [0.0, L]
        span_loads = loads_df[loads_df['span_index'] == i]
        for _, load in span_loads.iterrows():
            mag = safe_float(load.get('mag', 0.0))
            if mag == 0: continue
            l_type = str(load.get('type', 'P')).strip().upper()
            if l_type in ['P', 'POINT', 'POINT LOAD']:
                p_loc = safe_float(load.get('d_start', 0.0))
                points += [max(0, p_loc - 1e-6), p_loc, min(L, p_loc + 1e-6)]
            elif l_type in ['U', 'UNIFORM', 'DISTRIBUTED', 'LINE']:
                s = safe_float(load.get('d_start', 0.0))
                e = s + safe_float(load.get('dist', L))
                points += [max(0, s), min(L, e)]

        x_dense = np.linspace(0, L, 101)
        x_local = np.sort(np.unique(np.concatenate([x_dense, points])))

        # Element stiffness for internal forces
        Phi   = (12 * E * I) / (G * As_shear * L**2)
        const = (E * I) / ((1 + Phi) * L**3)
        k_ele_local = const * np.array([
            [12,   6*L,   -12,  6*L],
            [6*L,  (4+Phi)*L**2, -6*L, (2-Phi)*L**2],
            [-12, -6*L,    12, -6*L],
            [6*L,  (2-Phi)*L**2, -6*L, (4+Phi)*L**2]
        ])
        f_int        = np.dot(k_ele_local, u_ele) + fea_local[i]
        V_start      = f_int[0]
        M_beam_start = -f_int[1]

        m_x_list, v_x_list = [], []
        for x in x_local:
            if not has_any_load:
                V_curr, M_curr = 0.0, 0.0
            else:
                V_curr = V_start
                M_curr = M_beam_start + V_start * x
                for _, load in span_loads.iterrows():
                    mag    = safe_float(load.get('mag', 0.0))
                    if mag == 0.0: continue
                    l_type = str(load.get('type', 'P')).strip().upper()
                    if l_type in ['P', 'POINT', 'POINT LOAD']:
                        p_loc = safe_float(load.get('d_start', 0.0))
                        if x > p_loc:
                            V_curr -= mag
                            M_curr -= mag * (x - p_loc)
                    elif l_type in ['U', 'UNIFORM', 'DISTRIBUTED', 'LINE']:
                        u_start = safe_float(load.get('d_start', 0.0))
                        u_len   = safe_float(load.get('dist', L))
                        u_end   = u_start + u_len
                        if x > u_start:
                            eff_end  = min(x, u_end)
                            eff_len  = eff_end - u_start
                            load_force = mag * eff_len
                            V_curr  -= load_force
                            M_curr  -= load_force * (x - (u_start + eff_len / 2))
            m_x_list.append(M_curr)
            v_x_list.append(V_curr)

        M_arr, V_arr = np.array(m_x_list), np.array(v_x_list)

        # Deflection by numerical integration
        theta_b = np.zeros_like(x_local)
        v_b     = np.zeros_like(x_local)
        v_s     = np.zeros_like(x_local)
        for j in range(1, len(x_local)):
            dx         = x_local[j] - x_local[j-1]
            theta_b[j] = theta_b[j-1] + 0.5 * (M_arr[j-1] + M_arr[j]) / (E * I) * dx
            v_b[j]     = v_b[j-1]     + 0.5 * (theta_b[j-1] + theta_b[j]) * dx
            # [FIX] shear deflection sign.  With v positive upward and M
            # sagging-positive, the bending term already yields downward
            # deflection; the shear term must subtract, not add.  Previously
            # this reduced total deflection by ~2*shear component.
            v_s[j]     = v_s[j-1]     - 0.5 * (V_arr[j-1] + V_arr[j]) / (G * As_shear) * dx

        v_total_int = v_b + v_s
        C2     = u_ele[0]
        C1     = (u_ele[2] - v_total_int[-1] - C2) / L if L > 0 else 0
        v_def_m = v_total_int + C1 * x_local + C2
        if not has_any_load:
            v_def_m = np.zeros_like(x_local)

        x_total.extend(x0 + x_local)
        moment_total.extend(m_x_list)
        shear_total.extend(v_x_list)
        def_total.extend(v_def_m)

    # --- 5. Reactions ---
    FEA_R = np.zeros(n_dof)
    for i in range(n_spans):
        f   = fea_local[i]
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        FEA_R[idx[0]] += f[0]; FEA_R[idx[1]] += f[1]
        FEA_R[idx[2]] += f[2]; FEA_R[idx[3]] += f[3]

    R_final  = np.dot(K_global, d_all) + FEA_R
    reactions = {}
    for i, row in sup_df.iterrows():
        n_idx = int(safe_float(row.get('id', i)))
        if n_idx < n_nodes:
            reactions[f"R{n_idx}"] = 0.0 if not has_any_load else float(R_final[2 * n_idx])

    return np.array(x_total), np.array(moment_total), np.array(shear_total), np.array(def_total), reactions
