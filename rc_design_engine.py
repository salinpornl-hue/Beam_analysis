# rc_design_engine.py  ── ACI 318-19 corrected build
#
# Changes vs previous version are marked [FIX n].  See review document.
#   FIX 1  separate stirrup yield fyt, capped at 420 MPa (ACI 20.2.2.4)
#   FIX 2  maximum stirrup spacing check (ACI 9.7.6.2.2)
#   FIX 3  Av,min gated on Vu > phi*Vc (ACI 9.6.3.1)
#   FIX 4  rho_max enforced; tension-controlled state returned (ACI 21.2.2)
#   FIX 5  compression-steel concrete area deducted (no double count)
#   FIX 6  eps_ty = fy/Es instead of hard-coded 0.002
#   FIX 7  tension As classified by neutral axis, not by h/2
#   FIX 8  ACI 24.3.2 bar-spacing crack control (Gergely-Lutz kept as info only)
#   FIX 9  long-term multiplier applied to sustained load only
#   FIX 10 bar arrangement honours (4/3)*d_agg and reports genuine non-fit

import numpy as np
import math
from rc_utils import get_beta1

ES = 200000.0          # MPa, ACI 20.2.2.2
EPS_CU = 0.003         # ACI 22.2.2.1
FYT_MAX = 420.0        # ACI 20.2.2.4 limit on fyt for shear


# ══════════════════════════════════════════════════════════════════════
# 1.  SECTION GEOMETRY
# ══════════════════════════════════════════════════════════════════════
def get_centroid_and_d(layers, h_mm, cover_mm, stir_db, clear_spacing=25.0):
    """Total As, centroid y_bar from the nearest face, and effective depth d."""
    valid = [l for l in layers if l.get('n', 0) > 0 and l.get('db', 0) > 0]
    if not valid:
        return 0.0, 0.0, 0.0

    total_area = 0.0
    sum_area_y = 0.0
    current_y = cover_mm + stir_db

    for i, layer in enumerate(valid):
        n, db = layer['n'], layer['db']
        area = n * (math.pi * db ** 2 / 4.0)
        if i == 0:
            current_y += db / 2.0
        else:
            current_y += valid[i - 1]['db'] / 2.0 + clear_spacing + db / 2.0
        total_area += area
        sum_area_y += area * current_y

    y_bar = sum_area_y / total_area
    return h_mm - y_bar, total_area, y_bar


def arrange_bars_into_layers(total_n, db, b, cover, stir_db, d_agg=20.0):
    """
    Arrange bars into layers per ACI 25.2.1.
    [FIX 10] clear spacing now includes (4/3)*d_agg, and a section too narrow
    for two bars raises instead of silently returning an unbuildable layout.
    """
    if total_n <= 0:
        return [], True, ""

    min_spacing = max(25.0, db, (4.0 / 3.0) * d_agg)
    inner_w = b - 2 * cover - 2 * stir_db
    max_per_layer = int((inner_w + min_spacing) // (db + min_spacing))

    if max_per_layer < 2:
        need = 2 * db + min_spacing + 2 * cover + 2 * stir_db
        return [], False, (f"Width {b:.0f} mm cannot fit 2no. DB{db:.0f} "
                           f"(needs {need:.0f} mm). Widen the beam or reduce bar size.")

    layers, rem = [], int(total_n)
    while rem > 0:
        take = min(rem, max_per_layer)
        layers.append({'n': take, 'db': db})
        rem -= take
    return layers, True, ""


# ══════════════════════════════════════════════════════════════════════
# 2.  FLEXURE — REQUIRED STEEL
# ══════════════════════════════════════════════════════════════════════
def get_as_req(Mu_kNm, d_eff_mm, fc, fy, b_mm):
    """
    Required tension steel, ACI 318-19.
    Returns (as_final, rho, needs_comp_steel, details).

    [FIX 4] rho_max is now ENFORCED.  Previously it was computed, stored in
    'details', and never compared to anything, so a compression-controlled
    section was designed at phi = 0.9 with no warning.
    """
    if Mu_kNm == 0 or d_eff_mm <= 0:
        return 0.0, 0.0, False, {}

    Mu = abs(Mu_kNm) * 1e6
    phi = 0.9
    Rn = Mu / (phi * b_mm * d_eff_mm ** 2)
    term = 1 - (2 * Rn) / (0.85 * fc)

    beta1 = get_beta1(fc)
    eps_ty = fy / ES                                       # [FIX 6]
    eps_tc = eps_ty + 0.003                                # tension-controlled limit
    rho_max = (0.85 * fc * beta1 / fy) * (EPS_CU / (EPS_CU + eps_tc))

    as_min = max((0.25 * np.sqrt(fc) / fy) * b_mm * d_eff_mm,
                 (1.4 / fy) * b_mm * d_eff_mm)
    rho_min = as_min / (b_mm * d_eff_mm)

    base = dict(Mu=Mu, phi=phi, Rn=Rn, rho_min=rho_min, rho_max=rho_max,
                as_min=as_min, as_max=rho_max * b_mm * d_eff_mm,
                eps_ty=eps_ty, eps_tc=eps_tc, beta1=beta1)

    # Concrete crushes before the equation can balance
    if term < 0:
        base.update(reason="Rn exceeds the singly-reinforced capacity of the section")
        return 0.0, 0.0, True, base

    rho = (0.85 * fc / fy) * (1 - np.sqrt(term))
    as_calc = rho * b_mm * d_eff_mm

    # [FIX 4] over rho_max -> section is not tension-controlled as a singly
    # reinforced member.  Caller must add compression steel or deepen it.
    if rho > rho_max:
        base.update(rho_req=rho, as_req_calc=as_calc,
                    reason=f"rho_req={rho:.5f} exceeds rho_max={rho_max:.5f} "
                           f"(ACI 21.2.2 tension-controlled limit)")
        return float(as_calc), float(rho), True, base

    as_final = max(as_calc, as_min)
    base.update(rho_req=rho, as_req_calc=as_calc, as_final=as_final)
    return float(as_final), float(rho), False, base


# ══════════════════════════════════════════════════════════════════════
# 3.  FLEXURE — CAPACITY BY STRAIN COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════
def get_phi_Mn_details_multi(bot_layers, top_layers, b, h, fc, fy,
                             cover, stir_db, is_top_tension=False,
                             clear_spacing=25.0):
    """
    Strain compatibility, ACI 318-19 Ch. 22.

    is_top_tension=False  positive moment, compression face = TOP
    is_top_tension=True   negative moment, compression face = BOTTOM

    All bar depths d_i are measured from whichever face is in compression, so
    the two cases are the same calculation on a mirrored section.  A section
    with equal top and bottom steel returns identical phiMn either way.

    Returns (phi_Mn, As_tension, a, Mn, c, eps_t, layer_results)
    """
    beta1 = get_beta1(fc)
    eps_ty = fy / ES                     # [FIX 6]
    eps_tc = eps_ty + 0.003
    all_bars = []

    def add_bars(layers, is_bottom_bars):
        y = cover + stir_db
        for i, layer in enumerate(layers):
            n = layer.get('n', 0)
            db = layer.get('db', 0)
            if n <= 0 or db <= 0:
                continue
            if i == 0:
                y_c = y + db / 2.0
            else:
                y_c = y + db / 2.0
            # depth from the compression face
            near_comp_face = (is_bottom_bars == is_top_tension)
            d_i = y_c if near_comp_face else h - y_c
            all_bars.append({'area': n * np.pi * (db / 2) ** 2, 'd_i': d_i})
            y += db + clear_spacing

    add_bars(bot_layers, is_bottom_bars=True)
    add_bars(top_layers, is_bottom_bars=False)

    if not all_bars:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []

    dt = max(bar['d_i'] for bar in all_bars)

    # ---- bisection on the neutral axis --------------------------------
    def net_force(c):
        a = min(beta1 * c, h)
        Cc = 0.85 * fc * a * b
        Fs = 0.0
        for bar in all_bars:
            fs = max(-fy, min(fy, ES * EPS_CU * (bar['d_i'] - c) / c))
            # [FIX 5] a compression bar inside the stress block displaces
            # concrete already counted in Cc; remove the duplicate 0.85*fc.
            if fs < 0 and bar['d_i'] < a:
                fs += 0.85 * fc
            Fs += bar['area'] * fs
        return Fs - Cc, a, Cc

    c_low, c_high = 1e-3, h
    c = h / 2
    for _ in range(200):
        c = 0.5 * (c_low + c_high)
        nf, a, Cc = net_force(c)
        if abs(nf) < 1.0:
            break
        if nf > 0:
            c_low = c
        else:
            c_high = c

    a = min(beta1 * c, h)
    Cc = 0.85 * fc * a * b

    # ---- moment about the compression face ----------------------------
    Mn_Nmm = -Cc * (a / 2.0)
    As_tension = 0.0                                        # [FIX 7]
    layer_results = []

    for idx, bar in enumerate(sorted(all_bars, key=lambda x: x['d_i'])):
        eps_s = EPS_CU * (bar['d_i'] - c) / c
        fs = max(-fy, min(fy, eps_s * ES))
        if fs < 0 and bar['d_i'] < a:
            fs += 0.85 * fc
        Force = bar['area'] * fs
        Mn_Nmm += Force * bar['d_i']
        # [FIX 7] tension is defined by position relative to the neutral
        # axis, not by sitting below mid-depth.
        if bar['d_i'] > c:
            As_tension += bar['area']
        layer_results.append({
            'layer_idx': idx + 1,
            'd_i': bar['d_i'],
            'area': bar['area'],
            'eps_s': eps_s,
            'fs': fs,
            'is_yielded': abs(eps_s) >= eps_ty,
            'type': "Tension" if bar['d_i'] > c else "Compression",
        })

    Mn_kNm = Mn_Nmm / 1e6

    # ---- strength reduction factor, ACI Table 21.2.2 -------------------
    eps_t = EPS_CU * (dt - c) / c if c > 0 else 999.0
    if eps_t >= eps_tc:
        phi, section_state = 0.90, "Tension-controlled"
    elif eps_t <= eps_ty:
        phi, section_state = 0.65, "Compression-controlled"
    else:
        phi = 0.65 + 0.25 * (eps_t - eps_ty) / 0.003
        section_state = "Transition"

    for r in layer_results:
        r['section_state'] = section_state

    return (float(phi * Mn_kNm), float(As_tension), float(a), float(Mn_kNm),
            float(c), float(eps_t), layer_results)


# ══════════════════════════════════════════════════════════════════════
# 4.  SHEAR
# ══════════════════════════════════════════════════════════════════════
def check_shear_details(Vu_kN, b, d, fc, fyt, stir_db, spacing, n_legs=2):
    """
    ACI 318-19 Ch. 22.5 and 9.7.6.2.

    [FIX 1] fyt is the STIRRUP yield strength, passed separately from the main
            bar fy.  Thai RB (SR24) is 235 MPa, not the 392 MPa of SD40 main
            bars.  Capped at 420 MPa per ACI 20.2.2.4.
    [FIX 2] maximum spacing check added.
    [FIX 3] Av,min only required where Vu > phi*Vc.

    Returns (status, phi_Vn, phi_Vc, phi_Vs, Vc, Vs, info)
    """
    info = {}
    if d <= 0 or b <= 0:
        return "FAIL (invalid section)", 0.0, 0.0, 0.0, 0.0, 0.0, info

    fyt_used = min(float(fyt), FYT_MAX)
    info['fyt_used'] = fyt_used
    info['fyt_capped'] = fyt > FYT_MAX

    Vu = abs(Vu_kN) * 1000.0                       # N
    phi = 0.75

    Vc = 0.17 * np.sqrt(fc) * b * d                # N, ACI 22.5.5.1, lambda=1.0
    Av = n_legs * np.pi * (stir_db / 2) ** 2       # mm^2
    s = max(float(spacing), 1.0)
    Vs = (Av * fyt_used * d) / s                   # N

    Vs_max = 0.66 * np.sqrt(fc) * b * d            # ACI 22.5.8.5.3
    phi_Vc = phi * Vc
    phi_Vs = phi * min(Vs, Vs_max)
    phi_Vn = (phi_Vc + phi_Vs) / 1000.0            # kN

    # [FIX 2] maximum spacing, ACI 9.7.6.2.2
    if Vs > 0.33 * np.sqrt(fc) * b * d:
        s_max = min(d / 4.0, 300.0)
    else:
        s_max = min(d / 2.0, 600.0)

    # [FIX 3] minimum area, ACI 9.6.3.1 — only where Vu > phi*Vc
    Av_min_s = max(0.062 * np.sqrt(fc) * b / fyt_used, 0.35 * b / fyt_used)
    av_min_required = Vu > phi_Vc
    av_min_ok = (Av / s) >= Av_min_s or not av_min_required

    info.update(Vs_max_kN=Vs_max / 1000.0, s_max=s_max,
                Av=Av, Av_s=Av / s, Av_min_s=Av_min_s,
                av_min_required=av_min_required)

    if Vs > Vs_max:
        status = (f"FAIL (Vs={Vs/1000:.1f} > Vs,max={Vs_max/1000:.1f} kN — "
                  f"ACI 22.5.8.5.3, enlarge the section)")
    elif s > s_max:
        status = f"FAIL (s={s:.0f} > s,max={s_max:.0f} mm — ACI 9.7.6.2.2)"
    elif not av_min_ok:
        status = (f"FAIL (Av/s={Av/s:.3f} < {Av_min_s:.3f} mm²/mm — "
                  f"ACI 9.6.3.4)")
    elif phi_Vn * 1000 < Vu:
        status = f"FAIL (Vu={abs(Vu_kN):.1f} > φVn={phi_Vn:.1f} kN)"
    else:
        status = "OK"

    return (status, float(phi_Vn), float(phi_Vc / 1000), float(phi_Vs / 1000),
            float(Vc), float(Vs), info)


# ══════════════════════════════════════════════════════════════════════
# 5.  SERVICEABILITY
# ══════════════════════════════════════════════════════════════════════
def check_serviceability(Ma_kNm, delta_elastic_mm, b, h, d_eff,
                         Ast_bot, Ast_top, fc, sustained_frac=1.0,
                         duration_years=5.0, Es=ES):
    """
    ACI 318-19 24.2, Bischoff effective inertia.

    [FIX 9] sustained_frac is the fraction of the service deflection that is
    sustained (dead load plus the sustained part of live load).  The long-term
    multiplier applies only to that part, per ACI 24.2.4.1.1 — previously it
    was applied to the whole D+L deflection.

    Returns (delta_immediate, delta_total_longterm, Ie, Icr, lambda_delta)
    """
    if Ma_kNm == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    Ma = abs(Ma_kNm) * 1e6
    Ec = 4700 * np.sqrt(fc)
    n = Es / Ec
    Ig = (b * h ** 3) / 12.0
    fr = 0.62 * np.sqrt(fc)
    Mcr = fr * Ig / (h / 2.0)

    if d_eff > 0 and Ast_bot > 0:
        rho = Ast_bot / (b * d_eff)
        k = np.sqrt(2 * rho * n + (rho * n) ** 2) - rho * n
        kd = k * d_eff
        Icr = (b * kd ** 3) / 3.0 + n * Ast_bot * (d_eff - kd) ** 2
    else:
        Icr = 0.35 * Ig

    limit = (2.0 / 3.0) * Mcr
    if Ma <= limit:
        Ie = Ig
    else:
        denom = max(1 - (limit / Ma) ** 2 * (1 - Icr / Ig), 1e-6)
        Ie = Icr / denom
    Ie = min(Ie, Ig)

    delta_immediate = delta_elastic_mm * (Ig / Ie)

    xi = {0.25: 1.2, 0.5: 1.4, 1.0: 1.4, 3.0: 1.8}.get(duration_years, 2.0)
    rho_prime = Ast_top / (b * d_eff) if d_eff > 0 else 0.0
    lambda_delta = xi / (1 + 50 * rho_prime)

    # [FIX 9] creep and shrinkage act on the sustained portion only
    delta_sustained = delta_immediate * sustained_frac
    delta_total = delta_immediate + lambda_delta * delta_sustained

    return (float(delta_immediate), float(delta_total),
            float(Ie), float(Icr), float(lambda_delta))


def deflection_limits(span_mm):
    """ACI Table 24.2.2.  Returns the four limits in mm."""
    return {
        "L/180 (roof, no attached elements)": span_mm / 180.0,
        "L/360 (floor, no attached elements)": span_mm / 360.0,
        "L/480 (supporting damageable elements)": span_mm / 480.0,
        "L/240 (not supporting damageable elements)": span_mm / 240.0,
    }


# ══════════════════════════════════════════════════════════════════════
# 6.  CRACK CONTROL
# ══════════════════════════════════════════════════════════════════════
def check_crack_spacing(b, cover, stir_db, main_db, n_bars_in_layer, fy,
                        fs_service=None):
    """
    [FIX 8] ACI 318-19 24.3.2 — the current crack-control provision, a limit
    on BAR SPACING.  Gergely-Lutz crack width left ACI at 318-99.

    Returns (status, s_provided, s_max, fs_used)
    """
    fs = fs_service if fs_service else (2.0 / 3.0) * fy      # ACI 24.3.2.1
    cc = cover + stir_db                                     # to the main bar

    s_max = min(380.0 * (280.0 / fs) - 2.5 * cc,
                300.0 * (280.0 / fs))

    if n_bars_in_layer <= 1:
        return "OK (single bar)", 0.0, float(s_max), float(fs)

    clear_w = b - 2 * cover - 2 * stir_db - main_db
    s_prov = clear_w / (n_bars_in_layer - 1)

    status = "OK" if s_prov <= s_max else \
        f"FAIL (s={s_prov:.0f} > s,max={s_max:.0f} mm — ACI 24.3.2)"
    return status, float(s_prov), float(s_max), float(fs)


def check_crack_width(Ma_svc, b, h, d, As, n_bars, fc, dc_actual=None, Es=ES):
    """
    Gergely-Lutz.  INFORMATIONAL ONLY — not an ACI 318-19 provision.
    dc_actual: distance to the centre of the CLOSEST bar.  If omitted, h-d is
    used, which over-reports width for multi-layer steel.
    """
    if Ma_svc <= 0 or As <= 0 or n_bars == 0:
        return 0.0, 0.0

    Ec = 4700 * np.sqrt(fc)
    n = Es / Ec
    rho = As / (b * d)
    k = np.sqrt((rho * n) ** 2 + 2 * rho * n) - rho * n
    j = 1 - k / 3
    fs = (Ma_svc * 1e6) / (As * j * d)

    x = k * d
    dc = dc_actual if dc_actual else (h - d)
    if dc < 0:
        dc = 40.0
    beta = (h - x) / (d - x) if (d - x) > 0 else 1.2
    A_eff = (2 * dc * b) / n_bars

    w_mm = (0.076 * beta * (fs / 6.895) *
            ((dc / 25.4) * (A_eff / 645.16)) ** (1 / 3) / 1000.0) * 25.4
    return float(max(w_mm, 0.0)), float(fs)


# ══════════════════════════════════════════════════════════════════════
# 7.  AUTO DESIGN
# ══════════════════════════════════════════════════════════════════════
def design_flexure_auto(Mu_kNm, b, h, cover, stir_db, main_db, fc, fy,
                        d_agg=20.0):
    """As_req -> layers -> real d -> iterate until As_prov >= As_req."""
    d_assume = h - cover - stir_db - main_db / 2.0
    if Mu_kNm == 0:
        return [], float(d_assume), 0.0, 0.0, "OK", {}

    as_req, _, needs_comp, details = get_as_req(Mu_kNm, d_assume, fc, fy, b)
    if needs_comp:
        return [], float(d_assume), float(as_req), 0.0, \
            f"FAIL ({details.get('reason', 'compression steel required')})", details

    a_bar = np.pi * (main_db / 2) ** 2
    n_bars = max(2, int(np.ceil(as_req / a_bar)))
    layers, d_actual, as_prov = [], d_assume, 0.0

    for _ in range(15):
        layers, fits, msg = arrange_bars_into_layers(n_bars, main_db, b, cover,
                                                     stir_db, d_agg)
        if not fits:
            return [], float(d_assume), float(as_req), 0.0, f"FAIL ({msg})", details
        d_actual, as_prov, _ = get_centroid_and_d(layers, h, cover, stir_db)
        as_req, _, needs_comp, details = get_as_req(Mu_kNm, d_actual, fc, fy, b)
        if needs_comp:
            return layers, float(d_actual), float(as_req), float(as_prov), \
                f"FAIL ({details.get('reason', '')})", details
        if as_prov >= as_req:
            break
        n_bars += 1

    return layers, float(d_actual), float(as_req), float(as_prov), "OK", details
