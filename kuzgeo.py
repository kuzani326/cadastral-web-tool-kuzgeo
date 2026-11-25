import math
import io
import streamlit as st
import folium
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from fpdf import FPDF

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Zambian Cadastral Survey Computation Engine",
    layout="wide"
)

st.title("Zambian Cadastral Survey Computation Engine")
st.caption("GE500 Topic 1.4: Bearings, Distances, Bowditch Adjustment, Area & 2D Helmert Transformation")

# --- MATHEMATICAL HELPER FUNCTIONS ---

def dms_to_deg(d, m, s):
    """Converts Degrees, Minutes, Seconds to Decimal Degrees."""
    sign = -1 if d < 0 else 1
    return sign * (abs(d) + m / 60.0 + s / 3600.0)

def deg_to_dms(deg):
    """Converts Decimal Degrees to Degrees, Minutes, Seconds string."""
    if deg is None or math.isnan(deg) or math.isinf(deg):
        return "N/A"
    deg_norm = (deg + 360) % 360
    d = int(deg_norm)
    m_full = (deg_norm - d) * 60
    m = int(m_full)
    s = round((m_full - m) * 60, 1)
    if s >= 60.0:
        s = 0.0
        m += 1
    if m >= 60:
        m = 0
        d = (d + 1) % 360
    return f"{d}° {m:02d}' {s:04.1f}\""

def calculate_join(e1, n1, e2, n2):
    """Computes Join between two points: dE, dN, Distance, WCB."""
    if any(v is None or math.isnan(v) or math.isinf(v) for v in (e1, n1, e2, n2)):
        return float("nan"), float("nan"), float("nan"), float("nan"), "N/A"
    de = e2 - e1
    dn = n2 - n1
    dist = math.hypot(de, dn)
    if dist == 0:
        return 0.0, 0.0, 0.0, 0.0, "0° 00' 00.0\""
    rad = math.atan2(de, dn)
    wcb = (math.degrees(rad) + 360) % 360
    return de, dn, dist, wcb, deg_to_dms(wcb)

def calculate_shoelace_area(coords):
    """Computes polygon area using Gauss Shoelace Formula."""
    valid_coords = [
        (float(x), float(y)) for x, y in coords
        if x is not None and y is not None and not (math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y))
    ]
    n = len(valid_coords)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += valid_coords[i][0] * valid_coords[j][1]
        area -= valid_coords[j][0] * valid_coords[i][1]
    return abs(area) / 2.0

def bowditch_traverse_adjustment(df_traverse, start_e, start_n):
    """
    Performs Bowditch adjustment given leg distances and bearings.
    df_traverse columns: ['Leg', 'Distance_m', 'WCB_deg']
    """
    records = []
    total_dist = df_traverse['Distance_m'].sum()
    
    # 1. Compute raw departures (dE) and latitudes (dN)
    raw_de = []
    raw_dn = []
    for _, row in df_traverse.iterrows():
        rad = math.radians(row['WCB_deg'])
        de = row['Distance_m'] * math.sin(rad)
        dn = row['Distance_m'] * math.cos(rad)
        raw_de.append(de)
        raw_dn.append(dn)
        
    misclosure_e = sum(raw_de)
    misclosure_n = sum(raw_dn)
    linear_misclosure = math.hypot(misclosure_e, misclosure_n)
    if math.isnan(total_dist) or math.isnan(linear_misclosure) or total_dist == 0:
        relative_precision = float('nan')
    elif linear_misclosure == 0:
        relative_precision = float('inf')
    else:
        relative_precision = total_dist / linear_misclosure
    
    # Legal Class Assessment (Zambian Standards)
    if math.isnan(relative_precision):
        precision_class = "Incomplete / Invalid Data"
    elif math.isinf(relative_precision) or relative_precision >= 12000:
        precision_class = "Class A (Primary Cadastral >= 1:12,000)"
    elif relative_precision >= 8000:
        precision_class = "Class B (Secondary Cadastral >= 1:8,000)"
    elif relative_precision >= 4000:
        precision_class = "Class C (Tertiary/Suburban >= 1:4,000)"
    else:
        precision_class = "Unacceptable (Below Legal Precision < 1:4,000)"

    # 2. Compute Bowditch Corrections & Adjusted Coordinates
    curr_e, curr_n = start_e, start_n
    coords = [(curr_e, curr_n)]
    
    for i, row in df_traverse.iterrows():
        dist = row['Distance_m']
        if total_dist and not math.isnan(total_dist):
            corr_e = -misclosure_e * (dist / total_dist)
            corr_n = -misclosure_n * (dist / total_dist)
        else:
            corr_e = 0.0
            corr_n = 0.0
        adj_de = raw_de[i] + corr_e
        adj_dn = raw_dn[i] + corr_n
        
        next_e = curr_e + adj_de
        next_n = curr_n + adj_dn
        
        records.append({
            'Leg': row['Leg'],
            'Distance (m)': dist,
            'WCB': deg_to_dms(row['WCB_deg']),
            'Raw ΔE (m)': raw_de[i],
            'Raw ΔN (m)': raw_dn[i],
            'Corr ΔE (m)': corr_e,
            'Corr ΔN (m)': corr_n,
            'Adj ΔE (m)': adj_de,
            'Adj ΔN (m)': adj_dn,
            'Easting (m)': next_e,
            'Northing (m)': next_n
        })
        curr_e, curr_n = next_e, next_n
        coords.append((curr_e, curr_n))
        
    res_df = pd.DataFrame(records)
    return res_df, misclosure_e, misclosure_n, linear_misclosure, relative_precision, precision_class, coords

def helmert_2d_transformation(src_p1, tgt_p1, src_p2, tgt_p2, points_to_transform):
    """
    Computes 2D Helmert 4-Parameter Transformation parameters and transforms source points.
    E' = cE + a*E - b*N
    N' = cN + b*E + a*N
    """
    x1, y1 = src_p1
    x2, y2 = src_p2
    X1, Y1 = tgt_p1
    X2, Y2 = tgt_p2
    
    dx = x2 - x1
    dy = y2 - y1
    dX = X2 - X1
    dY = Y2 - Y1
    
    denom = dx**2 + dy**2
    if denom == 0:
        return None, None
    
    a = (dx * dX + dy * dY) / denom
    b = (dx * dY - dy * dX) / denom
    cE = X1 - a * x1 + b * y1
    cN = Y1 - b * x1 - a * y1
    
    scale = math.hypot(a, b)
    rotation_deg = math.degrees(math.atan2(b, a))
    
    transformed_points = []
    for px, py in points_to_transform:
        Tx = cE + a * px - b * py
        Ty = cN + b * px + a * py
        transformed_points.append((Tx, Ty))
        
    params = {'a': a, 'b': b, 'cE': cE, 'cN': cN, 'scale': scale, 'rotation_deg': rotation_deg}
    return params, transformed_points

def calculate_utm_scale_factor(e_left_km, e_right_km, hm_km, r_km=6358.0):
    """
    Calculates UTM Combined Scale Factor (all units in km, unrounded float):
    Em = (EA + EB) / 2
    E'm = Em - 500
    SF = 0.9996 * [1 + (E'm^2) / (2 * R^2) - hm / R]
    """
    e_m = (e_left_km + e_right_km) / 2.0
    e_prime_m = e_m - 500.0
    sf = 0.9996 * (1.0 + ((e_prime_m ** 2) / (2.0 * (r_km ** 2))) - (hm_km / r_km))
    return sf, e_m, e_prime_m

def calculate_lo_scale_factor(y_left_km, y_right_km, hm_km, r_km=6358.0):
    """
    Calculates Longitude Zero (Lo) Combined Scale Factor (all units in km, unrounded float):
    Ym = (YA + YB) / 2
    SF = [1 + (Ym^2) / (2 * R^2) - hm / R]
    """
    y_m = (y_left_km + y_right_km) / 2.0
    sf = 1.0 + ((y_m ** 2) / (2.0 * (r_km ** 2))) - (hm_km / r_km)
    return sf, y_m

def round_table_to_4_decimals(df, enabled=True):
    """Rounds float columns in an output DataFrame to 4 decimal places."""
    if df is None or not enabled:
        return df
    rounded_df = df.copy()
    float_cols = rounded_df.select_dtypes(include=["float", "float32", "float64"]).columns
    rounded_df[float_cols] = rounded_df[float_cols].round(4)
    return rounded_df

def load_csv_with_schema(uploaded_file, expected_cols, numeric_cols, default_df):
    """Reads an uploaded CSV and validates/coerces required columns."""
    if uploaded_file is None:
        return default_df, None
    try:
        csv_df = pd.read_csv(uploaded_file)
        csv_df.columns = [str(c).strip() for c in csv_df.columns]
        missing = [c for c in expected_cols if c not in csv_df.columns]
        if missing:
            return default_df, f"CSV is missing required columns: {', '.join(missing)}. Expected order: {', '.join(expected_cols)}"
        ordered_df = csv_df[expected_cols].copy()
        for col in numeric_cols:
            ordered_df[col] = pd.to_numeric(ordered_df[col], errors="coerce")
        if ordered_df[numeric_cols].isna().any().any():
            return default_df, f"CSV contains non-numeric values in numeric columns: {', '.join(numeric_cols)}"
        return ordered_df, None
    except Exception as exc:
        return default_df, f"Error reading CSV file: {exc}"

def render_csv_column_note(columns_spec, example_row):
    """Displays a note clearly listing the exact column order and format for CSV import."""
    with st.expander("Note: Required CSV Column Order & Format"):
        st.markdown("Your `.csv` file must include a header row with columns in this **exact order**:")
        for idx, (col_name, desc) in enumerate(columns_spec, start=1):
            st.markdown(f"{idx}. **`{col_name}`** — {desc}")
        st.markdown("**Header line:**")
        st.code(",".join([c[0] for c in columns_spec]), language="text")
        st.markdown("**Sample CSV rows:**")
        st.code(example_row, language="text")

def _pdf_safe_text(val):
    """Sanitizes strings for core Latin-1 PDF fonts."""
    text = str(val)
    replacements = {
        "°": " deg ",
        "Δ": "d",
        "²": "^2",
        "θ": "theta",
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "’": "'",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")

def _format_cell(val):
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return _pdf_safe_text(val)
        return f"{val:.4f}"
    return _pdf_safe_text(val)

def generate_full_pdf_report(sections):
    """
    Builds a multi-section PDF containing all provided summary lines and DataFrames.
    sections: list of dicts with keys:
- 'title': str
- 'summary': list[str] (optional)
- 'tables': list[tuple[str, pd.DataFrame]]
    """
    class SurveyPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 13)
            self.cell(0, 8, "ZAMBIAN CADASTRAL SURVEY COMPUTATION REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
            self.set_font("Helvetica", "I", 9)
            self.cell(0, 5, "GE500 Topic 1.4: Traverse, Shoelace Area & 2D Helmert Transformation", new_x="LMARGIN", new_y="NEXT", align="C")
            self.line(10, 24, 287, 24)
            self.ln(5)

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 8, f"Page {self.page_no()}", align="C")

    pdf = SurveyPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    usable_w = pdf.w - pdf.l_margin - pdf.r_margin

    def write_dataframe_table(table_title, df):
        if df is None or df.empty:
            return
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, _pdf_safe_text(table_title), new_x="LMARGIN", new_y="NEXT")

        cols = [str(c) for c in df.columns]
        n_cols = max(len(cols), 1)
        col_w = usable_w / n_cols

        def draw_table_header():
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_fill_color(230, 235, 242)
            for c in cols:
                pdf.cell(col_w, 6, _pdf_safe_text(c)[:26], border=1, fill=True)
            pdf.ln()

        draw_table_header()
        pdf.set_font("Helvetica", "", 7.5)
        for _, row in df.iterrows():
            if pdf.get_y() + 6 > pdf.h - pdf.b_margin:
                pdf.add_page()
                draw_table_header()
                pdf.set_font("Helvetica", "", 7.5)
            for c in df.columns:
                cell_txt = _format_cell(row[c])
                pdf.cell(col_w, 5.5, cell_txt[:28], border=1)
            pdf.ln()
        pdf.ln(4)

    for sec in sections:
        if pdf.get_y() + 16 > pdf.h - pdf.b_margin:
            pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, _pdf_safe_text(sec["title"]), new_x="LMARGIN", new_y="NEXT")

        for line in sec.get("summary", []):
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 5.5, _pdf_safe_text(line), new_x="LMARGIN", new_y="NEXT")
        if sec.get("summary"):
            pdf.ln(2)

        for t_title, t_df in sec.get("tables", []):
            write_dataframe_table(t_title, t_df)

        pdf.ln(3)

    return bytes(pdf.output())

# --- APP LAYOUT (TABS) ---
tab1, tab2, tab3 = st.tabs([
    "Traverse Computation & Bowditch",
    "Shoelace Area Engine",
    "2D Helmert Transformation"
])

report_sections = []
sf_value = None
sf_mode_label = None
sf_summary_lines = []

# ==========================================
# TAB 1: TRAVERSE COMPUTATION & BOWDITCH
# ==========================================
with tab1:
    st.header("Traverse Computation & Bowditch Adjustment")
    
    # INPUT MODE SELECTION (Cycle options without removing coordinates!)
    input_mode = st.radio(
        "Select Input Data Mode:",
        ["Mode A: Coordinate Inputs (Direct Easting/Northing)",
         "Mode B: Bearings & Distances Inputs",
         "Mode C: Interior Angles & Distances Inputs"],
        horizontal=True
    )
    
    st.markdown("---")
    tab1_tables = []
    tab1_summary = [f"Input Mode: {input_mode}"]
    
    if "Mode A" in input_mode:
        st.subheader("1. Enter Station Coordinates (Closed Loop)")
        default_coords = pd.DataFrame({
            "Station": ["STN_1", "STN_2", "STN_3", "STN_4"],
            "Easting (m)": [1000.000, 1250.000, 1250.000, 1000.000],
            "Northing (m)": [2000.000, 2000.000, 2300.000, 2300.000]
        })
        mode_a_source = st.radio(
            "Input Method for Station Coordinates:",
            ["Type / Edit Table", "Import from CSV"],
            horizontal=True,
            key="mode_a_source"
        )
        render_csv_column_note(
            [
                ("Station", "Station name / ID (text)"),
                ("Easting (m)", "Easting coordinate in meters (number)"),
                ("Northing (m)", "Northing coordinate in meters (number)")
            ],
            "Station,Easting (m),Northing (m)\nSTN_1,1000.000,2000.000\nSTN_2,1250.000,2000.000"
        )
        if mode_a_source == "Import from CSV":
            csv_a = st.file_uploader("Import Station Coordinates CSV", type=["csv"], key="csv_mode_a")
            base_a_df, err_a = load_csv_with_schema(
                csv_a,
                ["Station", "Easting (m)", "Northing (m)"],
                ["Easting (m)", "Northing (m)"],
                default_coords
            )
            if err_a:
                st.error(err_a)
        else:
            base_a_df = default_coords

        edited_df = st.data_editor(base_a_df, num_rows="dynamic", width="stretch", key="editor_mode_a")
        tab1_tables.append(("Input Station Coordinates (Mode A)", round_table_to_4_decimals(edited_df)))
        
        if len(edited_df) >= 3:
            coords_list = list(zip(edited_df["Easting (m)"], edited_df["Northing (m)"]))
            # Derive bearings and distances for Bowditch evaluation
            legs = []
            for i in range(len(coords_list)):
                nxt = (i + 1) % len(coords_list)
                e1, n1 = coords_list[i]
                e2, n2 = coords_list[nxt]
                stn1 = edited_df["Station"].iloc[i]
                stn2 = edited_df["Station"].iloc[nxt]
                _, _, dist, wcb, _ = calculate_join(e1, n1, e2, n2)
                legs.append({"Leg": f"{stn1}->{stn2}", "Distance_m": dist, "WCB_deg": wcb})
            df_legs = pd.DataFrame(legs)
            start_e, start_n = coords_list[0][0], coords_list[0][1]
            
    elif "Mode B" in input_mode:
        st.subheader("1. Enter Leg Bearings & Distances")
        col_start1, col_start2 = st.columns(2)
        start_e = col_start1.number_input("Starting Easting E0 (m)", value=1000.000, format="%.3f")
        start_n = col_start2.number_input("Starting Northing N0 (m)", value=2000.000, format="%.3f")
        
        default_legs = pd.DataFrame({
            "Leg": ["1->2", "2->3", "3->4", "4->1"],
            "Distance_m": [250.000, 300.000, 250.000, 300.000],
            "WCB_deg": [90.0, 0.0, 270.0, 180.0]
        })
        mode_b_source = st.radio(
            "Input Method for Leg Bearings & Distances:",
            ["Type / Edit Table", "Import from CSV"],
            horizontal=True,
            key="mode_b_source"
        )
        render_csv_column_note(
            [
                ("Leg", "Traverse leg label, e.g. 1->2 (text)"),
                ("Distance_m", "Leg horizontal distance in meters (number)"),
                ("WCB_deg", "Whole Circle Bearing in decimal degrees, 0 to 360 (number)")
            ],
            "Leg,Distance_m,WCB_deg\n1->2,250.000,90.0\n2->3,300.000,0.0"
        )
        if mode_b_source == "Import from CSV":
            csv_b = st.file_uploader("Import Bearings & Distances CSV", type=["csv"], key="csv_mode_b")
            base_b_df, err_b = load_csv_with_schema(
                csv_b,
                ["Leg", "Distance_m", "WCB_deg"],
                ["Distance_m", "WCB_deg"],
                default_legs
            )
            if err_b:
                st.error(err_b)
        else:
            base_b_df = default_legs

        df_legs = st.data_editor(base_b_df, num_rows="dynamic", width="stretch", key="editor_mode_b")
        tab1_summary.append(f"Starting Coordinates: E0 = {start_e:.3f} m, N0 = {start_n:.3f} m")
        tab1_tables.append(("Input Leg Bearings & Distances (Mode B)", round_table_to_4_decimals(df_legs)))
        
    else:  # Mode C
        st.subheader("1. Enter Baseline Bearing, Interior Angles & Distances")
        col_c1, col_c2, col_c3 = st.columns(3)
        start_e = col_c1.number_input("Start Easting E0 (m)", value=1000.000, format="%.3f", key="c_e")
        start_n = col_c2.number_input("Start Northing N0 (m)", value=2000.000, format="%.3f", key="c_n")
        init_wcb = col_c3.number_input("Initial Reference Bearing (°)", value=90.0, format="%.4f")
        
        default_angles = pd.DataFrame({
            "Station": ["STN_1", "STN_2", "STN_3", "STN_4"],
            "Distance_to_Next (m)": [250.000, 300.000, 250.000, 300.000],
            "Interior_Angle (deg)": [90.0, 90.0, 90.0, 90.0]
        })
        mode_c_source = st.radio(
            "Input Method for Angles & Distances:",
            ["Type / Edit Table", "Import from CSV"],
            horizontal=True,
            key="mode_c_source"
        )
        render_csv_column_note(
            [
                ("Station", "Station name / ID (text)"),
                ("Distance_to_Next (m)", "Distance to next station in meters (number)"),
                ("Interior_Angle (deg)", "Interior angle in decimal degrees (number)")
            ],
            "Station,Distance_to_Next (m),Interior_Angle (deg)\nSTN_1,250.000,90.0\nSTN_2,300.000,90.0"
        )
        if mode_c_source == "Import from CSV":
            csv_c = st.file_uploader("Import Angles & Distances CSV", type=["csv"], key="csv_mode_c")
            base_c_df, err_c = load_csv_with_schema(
                csv_c,
                ["Station", "Distance_to_Next (m)", "Interior_Angle (deg)"],
                ["Distance_to_Next (m)", "Interior_Angle (deg)"],
                default_angles
            )
            if err_c:
                st.error(err_c)
        else:
            base_c_df = default_angles

        df_angles = st.data_editor(base_c_df, num_rows="dynamic", width="stretch", key="editor_mode_c")
        tab1_summary.append(f"Start: E0 = {start_e:.3f} m, N0 = {start_n:.3f} m | Initial Bearing: {init_wcb:.4f} deg")
        tab1_tables.append(("Input Angles & Distances (Mode C)", round_table_to_4_decimals(df_angles)))
        
        # Propagate bearings from angles
        legs = []
        curr_bearing = init_wcb
        for i in range(len(df_angles)):
            dist = df_angles["Distance_to_Next (m)"].iloc[i]
            stn1 = df_angles["Station"].iloc[i]
            nxt_idx = (i + 1) % len(df_angles)
            stn2 = df_angles["Station"].iloc[nxt_idx]
            legs.append({"Leg": f"{stn1}->{stn2}", "Distance_m": dist, "WCB_deg": curr_bearing})
            angle = df_angles["Interior_Angle (deg)"].iloc[i]
            curr_bearing = (curr_bearing + 180 - angle + 360) % 360
        df_legs = pd.DataFrame(legs)

    # PROCESS BOWDITCH COMPUTATION
    if 'df_legs' in locals() and len(df_legs) >= 3:
        st.markdown("---")
        st.subheader("2. Bowditch Adjustment Results")
        
        res_df, mis_e, mis_n, lin_mis, rel_prec, prec_cls, adjusted_coords = bowditch_traverse_adjustment(df_legs, start_e, start_n)

        if math.isnan(rel_prec):
            rel_prec_str = "N/A"
        elif math.isinf(rel_prec):
            rel_prec_str = "Exact (No Misclosure)"
        else:
            rel_prec_str = f"1 : {int(rel_prec):,}"
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Linear Misclosure (e)", f"{lin_mis:.4f} m" if not math.isnan(lin_mis) else "N/A")
        m2.metric("Relative Precision", rel_prec_str)
        m3.metric("ΔE Misclosure", f"{mis_e:+.4f} m" if not math.isnan(mis_e) else "N/A")
        m4.metric("ΔN Misclosure", f"{mis_n:+.4f} m" if not math.isnan(mis_n) else "N/A")
        
        st.info(f"**Zambian Legal Standards Evaluation:** {prec_cls}")
        display_res_df = round_table_to_4_decimals(res_df)
        st.dataframe(display_res_df, width="stretch")
        
        # Calculate Parcel Area
        area_sqm = calculate_shoelace_area(adjusted_coords)
        area_ha = area_sqm / 10000.0

        tab1_summary.extend([
            f"Linear Misclosure (e): {lin_mis:.4f} m | Relative Precision: {rel_prec_str}",
            f"dE Misclosure: {mis_e:+.4f} m | dN Misclosure: {mis_n:+.4f} m",
            f"Zambian Legal Standards Evaluation: {prec_cls}",
            f"Adjusted Parcel Area: {area_sqm:,.3f} m^2 ({area_ha:,.4f} ha)"
        ])
        tab1_tables.append(("Bowditch Adjusted Traverse Table", display_res_df))

    report_sections.append({
        "title": "1. Traverse Computation & Bowditch Adjustment",
        "summary": tab1_summary,
        "tables": tab1_tables
    })

# ==========================================
# TAB 2: SHOELACE AREA ENGINE
# ==========================================
with tab2:
    st.header("Polygon Area Engine (Gauss Shoelace Formula)")
    st.write("Compute the rectilinear area of any parcel directly from coordinate pairs.")

    default_area_df = pd.DataFrame({
        "Corner Point": ["P1", "P2", "P3", "P4", "P5"],
        "Easting (m)": [4500.0, 4750.0, 4800.0, 4600.0, 4450.0],
        "Northing (m)": [8000.0, 8050.0, 7850.0, 7700.0, 7800.0]
    })
    area_source = st.radio(
        "Input Method for Parcel Coordinates:",
        ["Type / Edit Table", "Import from CSV"],
        horizontal=True,
        key="area_source"
    )
    render_csv_column_note(
        [
            ("Corner Point", "Parcel corner / boundary point label (text)"),
            ("Easting (m)", "Easting coordinate in meters (number)"),
            ("Northing (m)", "Northing coordinate in meters (number)")
        ],
        "Corner Point,Easting (m),Northing (m)\nP1,4500.0,8000.0\nP2,4750.0,8050.0"
    )
    if area_source == "Import from CSV":
        csv_area = st.file_uploader("Import Parcel Coordinates CSV", type=["csv"], key="csv_area")
        base_area_df, err_area = load_csv_with_schema(
            csv_area,
            ["Corner Point", "Easting (m)", "Northing (m)"],
            ["Easting (m)", "Northing (m)"],
            default_area_df
        )
        if err_area:
            st.error(err_area)
    else:
        base_area_df = default_area_df
    
    area_df = st.data_editor(
        base_area_df,
        num_rows="dynamic",
        width="stretch",
        key="shoelace_editor"
    )

    display_area_df = round_table_to_4_decimals(area_df)
    tab2_summary = []
    tab2_tables = [("Shoelace Parcel Coordinates", display_area_df)]
    
    sh_coords = list(zip(area_df["Easting (m)"], area_df["Northing (m)"]))
    if len(sh_coords) >= 3:
        calc_area_sqm = calculate_shoelace_area(sh_coords)
        calc_area_ha = calc_area_sqm / 10000.0
        
        res1, res2 = st.columns(2)
        res1.metric("Calculated Area (m²)", f"{calc_area_sqm:,.2f} m²")
        res2.metric("Calculated Area (Hectares)", f"{calc_area_ha:,.4f} ha")
        tab2_summary.append(f"Calculated Area: {calc_area_sqm:,.2f} m^2 ({calc_area_ha:,.4f} ha)")

    report_sections.append({
        "title": "2. Shoelace Area Engine",
        "summary": tab2_summary,
        "tables": tab2_tables
    })

# ==========================================
# TAB 3: 2D HELMERT TRANSFORMATION
# ==========================================
with tab3:
    hdr_col1, hdr_col2 = st.columns([2, 2], vertical_alignment="center")
    with hdr_col1:
        st.header("2D Helmert Transformation")
    with hdr_col2:
        calc_sf_toggle = st.toggle("Calculate Scale Factor (UTM / Lo Projection Formulas)", value=False)

    if calc_sf_toggle:
        with st.container(border=True):
            st.subheader("Projection Combined Scale Factor Calculator (All Units in km)")
            use_utm_toggle = st.toggle(
                "Scale Factor Type: UTM Combined Scale Factor (ON) / Lo Combined Scale Factor (OFF)",
                value=True
            )

            r_km = st.number_input("Radius of Curvature R (km)", value=6358.0, format="%.6f")
            hm_km = st.number_input("Mean Site Elevation hm (km)", value=1.2000, format="%.6f")

            if use_utm_toggle:
                sf_mode_label = "UTM Combined Scale Factor"
                c_ea, c_eb = st.columns(2)
                ea_km = c_ea.number_input("Leftmost Point Easting EA (km)", value=500.0000, format="%.6f")
                eb_km = c_eb.number_input("Rightmost Point Easting EB (km)", value=501.5000, format="%.6f")

                sf_value, e_m_km, e_prime_m_km = calculate_utm_scale_factor(ea_km, eb_km, hm_km, r_km)
                st.write(f"**Mean Easting (Em):** `{repr(e_m_km)} km` | **Reduced Easting (E'm):** `{repr(e_prime_m_km)} km`")
                st.write(f"**Unrounded UTM Combined Scale Factor (SF):** `{repr(sf_value)}`")
                sf_summary_lines = [
                    "Scale Factor Mode: UTM Combined Scale Factor",
                    f"EA = {repr(ea_km)} km, EB = {repr(eb_km)} km, Em = {repr(e_m_km)} km, E'm = {repr(e_prime_m_km)} km",
                    f"hm = {repr(hm_km)} km, R = {repr(r_km)} km",
                    f"Unrounded Scale Factor (SF): {repr(sf_value)}"
                ]
            else:
                sf_mode_label = "Lo Combined Scale Factor"
                c_ya, c_yb = st.columns(2)
                ya_km = c_ya.number_input("Leftmost Point Y Coordinate YA (km)", value=100.0000, format="%.6f")
                yb_km = c_yb.number_input("Rightmost Point Y Coordinate YB (km)", value=101.5000, format="%.6f")

                sf_value, y_m_km = calculate_lo_scale_factor(ya_km, yb_km, hm_km, r_km)
                st.write(f"**Mean Y Coordinate (Ym):** `{repr(y_m_km)} km`")
                st.write(f"**Unrounded Lo Combined Scale Factor (SF):** `{repr(sf_value)}`")
                sf_summary_lines = [
                    "Scale Factor Mode: Lo Combined Scale Factor",
                    f"YA = {repr(ya_km)} km, YB = {repr(yb_km)} km, Ym = {repr(y_m_km)} km",
                    f"hm = {repr(hm_km)} km, R = {repr(r_km)} km",
                    f"Unrounded Scale Factor (SF): {repr(sf_value)}"
                ]

    st.write("Transform local or baseline survey coordinates into National Grid / WGS84 coordinates.")
    
    st.subheader("1. Enter Common Control Points (Minimum 2 Points)")
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        st.markdown("**Control Point 1**")
        src_e1 = st.number_input("Source E1", value=100.000, format="%.3f")
        src_n1 = st.number_input("Source N1", value=200.000, format="%.3f")
        tgt_e1 = st.number_input("Target E1", value=500105.200, format="%.3f")
        tgt_n1 = st.number_input("Target N1", value=8580210.500, format="%.3f")
        
    with h_col2:
        st.markdown("**Control Point 2**")
        src_e2 = st.number_input("Source E2", value=350.000, format="%.3f")
        src_n2 = st.number_input("Source N2", value=450.000, format="%.3f")
        tgt_e2 = st.number_input("Target E2", value=500355.800, format="%.3f")
        tgt_n2 = st.number_input("Target N2", value=8580460.900, format="%.3f")

    st.subheader("2. Enter Source Points to Transform")
    default_src_pts_df = pd.DataFrame({
        "Point Name": ["Corner_A", "Corner_B", "Corner_C"],
        "Source Easting (m)": [150.000, 220.000, 300.000],
        "Source Northing (m)": [250.000, 310.000, 400.000]
    })
    helmert_source = st.radio(
        "Input Method for Source Points to Transform:",
        ["Type / Edit Table", "Import from CSV"],
        horizontal=True,
        key="helmert_source"
    )
    render_csv_column_note(
        [
            ("Point Name", "Point identifier (text)"),
            ("Source Easting (m)", "Local/source Easting coordinate in meters (number)"),
            ("Source Northing (m)", "Local/source Northing coordinate in meters (number)")
        ],
        "Point Name,Source Easting (m),Source Northing (m)\nCorner_A,150.000,250.000\nCorner_B,220.000,310.000"
    )
    if helmert_source == "Import from CSV":
        csv_helm = st.file_uploader("Import Source Points CSV", type=["csv"], key="csv_helmert")
        base_src_pts_df, err_helm = load_csv_with_schema(
            csv_helm,
            ["Point Name", "Source Easting (m)", "Source Northing (m)"],
            ["Source Easting (m)", "Source Northing (m)"],
            default_src_pts_df
        )
        if err_helm:
            st.error(err_helm)
    else:
        base_src_pts_df = default_src_pts_df

    src_pts_df = st.data_editor(
        base_src_pts_df,
        num_rows="dynamic",
        width="stretch",
        key="helmert_editor"
    )

    control_pts_df = pd.DataFrame({
        "Control Point": ["CP1", "CP2"],
        "Source E (m)": [src_e1, src_e2],
        "Source N (m)": [src_n1, src_n2],
        "Target E (m)": [tgt_e1, tgt_e2],
        "Target N (m)": [tgt_n1, tgt_n2]
    })
    display_cp_df = round_table_to_4_decimals(control_pts_df)
    display_src_pts_df = round_table_to_4_decimals(src_pts_df)

    tab3_summary = []
    tab3_tables = [
        ("Common Control Points", display_cp_df),
        ("Source Points to Transform", display_src_pts_df)
    ]
    
    pts_to_tx = list(zip(src_pts_df["Source Easting (m)"], src_pts_df["Source Northing (m)"]))
    params, tx_results = helmert_2d_transformation((src_e1, src_n1), (tgt_e1, tgt_n1), (src_e2, src_n2), (tgt_e2, tgt_n2), pts_to_tx)
    
    if params:
        st.markdown("---")
        st.subheader("3. Transformation Parameters & Results")
        p1, p2, p3 = st.columns(3)
        p1.metric("Scale Factor (s)", f"{params['scale']:.6f}")
        p2.metric("Rotation Angle (θ)", f"{params['rotation_deg']:.4f}°")
        p3.metric("Shifts (cE, cN)", f"{params['cE']:,.2f} m, {params['cN']:,.2f} m")
        
        out_records = []
        for i, (tx_e, tx_n) in enumerate(tx_results):
            out_records.append({
                "Point Name": src_pts_df["Point Name"].iloc[i],
                "Source Easting": src_pts_df["Source Easting (m)"].iloc[i],
                "Source Northing": src_pts_df["Source Northing (m)"].iloc[i],
                "Transformed Easting (m)": tx_e,
                "Transformed Northing (m)": tx_n
            })
        tx_out_df = pd.DataFrame(out_records)
        display_tx_out_df = round_table_to_4_decimals(tx_out_df)
        st.dataframe(display_tx_out_df, width="stretch")

        tab3_summary.append(
            f"Scale Factor: {params['scale']:.6f} | Rotation: {params['rotation_deg']:.4f} deg | "
            f"Shifts (cE, cN): {params['cE']:,.2f} m, {params['cN']:,.2f} m"
        )
        tab3_tables.append(("Transformed Coordinates", display_tx_out_df))

    report_sections.append({
        "title": "3. 2D Helmert Transformation",
        "summary": tab3_summary,
        "tables": tab3_tables
    })

if calc_sf_toggle and sf_value is not None:
    report_sections.insert(0, {
        "title": f"Projection Scale Factor ({sf_mode_label})",
        "summary": sf_summary_lines,
        "tables": []
    })

# --- PDF REPORT EXPORT ---
st.markdown("---")
st.subheader("Export Full PDF Report")
st.write("Generate a PDF report containing all computation tables and summary results across the tabs.")
pdf_bytes = generate_full_pdf_report(report_sections)
st.download_button(
    label="Download Full Survey Report (PDF)",
    data=pdf_bytes,
    file_name="Zambian_Cadastral_Survey_Report.pdf",
    mime="application/pdf"
)
