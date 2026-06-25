import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Adaptive UWB Open-Pit Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛰️ Adaptive UWB RTLS Deployment Simulator")
st.markdown("Based on the *Adaptive UWB Open-Pit Simulation Architecture Specification*.")

# --- Sidebar Inputs (Simulation Parameters) ---
st.sidebar.header("1. Mine Geometry")
pit_radius = st.sidebar.slider("Pit Radius (m)", 100, 1000, 400, step=50)
pit_depth = st.sidebar.slider("Pit Depth (m)", 20, 300, 120, step=10)
grid_resolution = st.sidebar.slider("Grid Resolution (m)", 10, 50, 25)

st.sidebar.header("2. Primary Rim Anchors")
num_rim_anchors = st.sidebar.slider("Initial Rim Anchors", 4, 24, 8, step=1)
pole_height = st.sidebar.slider("Default Tower Height (m)", 8, 30, 12, step=1)

st.sidebar.header("3. Antenna Pattern & Orientation")
downtilt = st.sidebar.slider("Mechanical Downtilt (°)", 0, 60, 25, step=1)
h_beamwidth = st.sidebar.selectbox("Default Horizontal Beamwidth (°)", [30, 60, 90, 120, 150], index=2)
max_range = st.sidebar.slider("Base UWB Effective Range (m)", 50, 600, 300, step=25)

st.sidebar.header("4. Optimization Settings")
auto_insert_interior = st.sidebar.checkbox("Auto-insert Interior Anchors (Adaptive)", value=True)

# --- Visualization Toggles ---
st.sidebar.header("5. Visualization Toggles")
show_heatmap = st.sidebar.checkbox("Show Coverage Heatmap", value=True)
show_cones = st.sidebar.checkbox("Show Conical Rays (Directional)", value=True)
show_spheres = st.sidebar.checkbox("Show Spheres (Omnidirectional)", value=True)
all_cones_off = st.sidebar.checkbox("ALL OFF: Conical Ranges", value=False)

# --- Manual Interior Anchors ---
st.sidebar.header("6. Manual Interior Anchors")
num_manual_anchors = st.sidebar.number_input(
    "Number of Manual Interior Anchors",
    min_value=0,
    max_value=50,
    value=0,
    step=1
)

# Initialize editable dataframe in session state
if "manual_anchor_df" not in st.session_state:
    st.session_state.manual_anchor_df = pd.DataFrame(
        [{
            "x": 0.0, "y": 0.0, "z": 0.0, "azimuth": 0.0,
            "beamwidth": float(h_beamwidth), "base_range": float(max_range),
            "is_omni": True, "enabled": True, "show_range": True
        }]
    )

target_rows = int(num_manual_anchors)
df = st.session_state.manual_anchor_df.copy()

# Ensure required columns exist
required_cols = ["x", "y", "z", "azimuth", "beamwidth", "base_range", "is_omni", "enabled", "show_range"]
for c in required_cols:
    if c not in df.columns:
        if c in ["is_omni", "enabled", "show_range"]:
            df[c] = True
        elif c == "beamwidth":
            df[c] = float(h_beamwidth)
        elif c == "base_range":
            df[c] = float(max_range)
        else:
            df[c] = 0.0

# Resize rows to match requested count
if target_rows == 0:
    df = pd.DataFrame(columns=required_cols)
elif len(df) < target_rows:
    add_n = target_rows - len(df)
    add_df = pd.DataFrame([{
        "x": 0.0, "y": 0.0, "z": 0.0, "azimuth": 0.0,
        "beamwidth": float(h_beamwidth), "base_range": float(max_range),
        "is_omni": True, "enabled": True, "show_range": True
    }] * add_n)
    df = pd.concat([df, add_df], ignore_index=True)
elif len(df) > target_rows:
    df = df.iloc[:target_rows].reset_index(drop=True)

edited_df = st.sidebar.data_editor(
    df,
    key="manual_anchor_editor",
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    column_config={
        "x": st.column_config.NumberColumn("X (m)", min_value=-float(pit_radius), max_value=float(pit_radius), step=1.0),
        "y": st.column_config.NumberColumn("Y (m)", min_value=-float(pit_radius), max_value=float(pit_radius), step=1.0),
        "z": st.column_config.NumberColumn("Tower Height Z (m)", min_value=-float(pit_depth), max_value=float(pole_height + 80), step=1.0),
        "azimuth": st.column_config.NumberColumn("Azimuth (°)", min_value=0.0, max_value=359.9, step=1.0),
        "beamwidth": st.column_config.NumberColumn("Beamwidth (°)", min_value=10.0, max_value=180.0, step=1.0),
        "base_range": st.column_config.NumberColumn("Base Range (m)", min_value=10.0, max_value=1000.0, step=5.0),
        "is_omni": st.column_config.CheckboxColumn("Omnidirectional"),
        "enabled": st.column_config.CheckboxColumn("Enabled"),
        "show_range": st.column_config.CheckboxColumn("Show Range"),
    },
)

# Persist edited values
st.session_state.manual_anchor_df = edited_df

# ---------------------------
# Simulation / Physics Helpers
# ---------------------------

def generate_pit_mesh(radius, depth, resolution):
    """Generates a simplified 3D paraboloid bowl to represent the open-pit mine."""
    x = np.arange(-radius, radius + resolution, resolution)
    y = np.arange(-radius, radius + resolution, resolution)
    X, Y = np.meshgrid(x, y)

    # Keep it within a circular boundary
    mask = (X ** 2 + Y ** 2) <= radius ** 2

    # Paraboloid formula for depth profile
    Z = -depth * (1 - (X ** 2 + Y ** 2) / radius ** 2)
    Z[~mask] = 0  # Surface level outside the pit

    return X, Y, Z, mask


def effective_range(base_range, height_z, beamwidth, h_ref=12.0, bw_ref=90.0, r_min=20.0, r_cap=1200.0):
    """
    Practical coupling:
      - Higher tower -> moderate range improvement (sqrt law)
      - Narrower beamwidth -> higher effective range (sqrt law)
      - Wider beamwidth -> lower effective range
    """
    h = max(1.0, float(height_z))
    bw = max(10.0, float(beamwidth))
    base = max(1.0, float(base_range))

    r_eff = base * np.sqrt(h / h_ref) * np.sqrt(bw_ref / bw)
    return float(np.clip(r_eff, r_min, r_cap))


def place_rim_anchors(radius, num_anchors, pole_h, default_bw, default_range):
    """Places anchors uniformly around the top rim facing inward."""
    angles = np.linspace(0, 2 * np.pi, num_anchors, endpoint=False)
    anchors = []
    for idx, theta in enumerate(angles):
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = pole_h
        azimuth = np.degrees(theta + np.pi) % 360  # inward

        anchors.append({
            "id": f"Rim_{idx+1}",
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "azimuth": float(azimuth),
            "beamwidth": float(default_bw),
            "base_range": float(default_range),
            "is_omni": False,
            "enabled": True,
            "show_range": True,
            "type": "Rim"
        })
    return anchors


def calculate_coverage(X, Y, Z, mask, anchors, default_h_bw, tilt, default_r_max):
    """Evaluates beam/range intersection for every grid node."""
    rows, cols = X.shape
    flat_X = X.flatten()
    flat_Y = Y.flatten()
    flat_Z = Z.flatten()
    flat_mask = mask.flatten()

    visible_counts = np.zeros_like(flat_X, dtype=int)

    for a in anchors:
        if not bool(a.get("enabled", True)):
            continue

        a_bw = float(a.get("beamwidth", default_h_bw))
        a_base_r = float(a.get("base_range", default_r_max))
        a_h = float(a.get("z", 1.0))
        is_omni = bool(a.get("is_omni", a.get("type") in {"Interior", "Manual"}))

        a_range = effective_range(a_base_r, a_h, a_bw, r_cap=max(1200.0, default_r_max * 2.0))

        dx = flat_X - a["x"]
        dy = flat_Y - a["y"]
        dz = flat_Z - a["z"]
        distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        in_range = distances <= a_range

        if is_omni:
            in_h_beam = np.ones_like(in_range, dtype=bool)
        else:
            angle_to_points = np.arctan2(dy, dx)
            azimuth_rad = np.radians(a["azimuth"])
            angle_diff = np.arctan2(
                np.sin(angle_to_points - azimuth_rad),
                np.cos(angle_to_points - azimuth_rad),
            )
            in_h_beam = np.abs(np.degrees(angle_diff)) <= (a_bw / 2.0)

        is_visible = in_range & in_h_beam & flat_mask
        visible_counts += is_visible.astype(int)

    return visible_counts.reshape(rows, cols)


def add_omni_sphere(fig, a, default_h_bw, default_r_max):
    """Draw omnidirectional range as sphere."""
    a_bw = float(a.get("beamwidth", default_h_bw))
    a_base_r = float(a.get("base_range", default_r_max))
    a_h = float(a.get("z", 1.0))
    r = effective_range(a_base_r, a_h, a_bw, r_cap=max(1200.0, default_r_max * 2.0))
    if r <= 0:
        return

    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 18)
    xs = a["x"] + r * np.outer(np.cos(u), np.sin(v))
    ys = a["y"] + r * np.outer(np.sin(u), np.sin(v))
    zs = a["z"] + r * np.outer(np.ones_like(u), np.cos(v))

    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.10,
        showscale=False,
        hoverinfo="skip",
        name=f'{a["id"]} Omni Range'
    ))


def add_conical_rays(fig, a, tilt_deg, default_h_bw, default_r_max):
    """Draw directional range as conical rays using practical tilt projection."""
    a_bw = float(a.get("beamwidth", default_h_bw))
    a_base_r = float(a.get("base_range", default_r_max))
    a_h = float(a.get("z", 1.0))
    r = effective_range(a_base_r, a_h, a_bw, r_cap=max(1200.0, default_r_max * 2.0))

    if r <= 0 or a_bw <= 0:
        return

    az = np.radians(float(a.get("azimuth", 0.0)))
    half = np.radians(a_bw / 2.0)
    tilt = np.radians(float(tilt_deg))

    # Horizontal projected length due to tilt
    r_xy = r * np.cos(tilt)
    z_drop = r * np.sin(tilt)

    ray_angles = np.linspace(-half, half, 11)
    for da in ray_angles:
        th = az + da
        x2 = a["x"] + r_xy * np.cos(th)
        y2 = a["y"] + r_xy * np.sin(th)
        z2 = a["z"] - z_drop
        fig.add_trace(go.Scatter3d(
            x=[a["x"], x2],
            y=[a["y"], y2],
            z=[a["z"], z2],
            mode="lines",
            line=dict(width=3),
            showlegend=False,
            hoverinfo="skip"
        ))

# ---------------------------
# Build simulation entities
# ---------------------------

X, Y, Z, mask = generate_pit_mesh(pit_radius, pit_depth, grid_resolution)
anchors = place_rim_anchors(pit_radius, num_rim_anchors, pole_height, h_beamwidth, max_range)

# Build manual anchors from editable table
manual_anchors = []
for i, r in edited_df.iterrows():
    mx = float(r["x"])
    my = float(r["y"])
    mz = float(r["z"])
    maz = float(r["azimuth"])
    mbw = float(r["beamwidth"])
    mbr = float(r["base_range"])
    mis_omni = bool(r["is_omni"])
    menabled = bool(r["enabled"])
    mshow = bool(r["show_range"])

    if (mx ** 2 + my ** 2) <= (pit_radius ** 2):
        manual_anchors.append({
            "id": f"Manual_{i+1}",
            "x": mx,
            "y": my,
            "z": mz,
            "azimuth": maz,
            "beamwidth": mbw,
            "base_range": mbr,
            "is_omni": mis_omni,
            "enabled": menabled,
            "show_range": mshow,
            "type": "Manual"
        })
    else:
        st.sidebar.warning(f"Manual Anchor {i+1}: (x,y) outside pit boundary; ignored.")

anchors.extend(manual_anchors)

# First coverage run
coverage_map = calculate_coverage(X, Y, Z, mask, anchors, h_beamwidth, downtilt, max_range)

# Adaptive interior anchors
interior_anchors_added = []
if auto_insert_interior:
    weak_zones = (coverage_map < 4) & mask
    if np.any(weak_zones):
        weak_indices = np.argwhere(weak_zones)
        n_add = min(2, max(1, len(weak_indices) // 10 + 1))
        for i in range(n_add):
            idx = weak_indices[int(len(weak_indices) * (i + 1) / (i + 2))]
            int_x, int_y = float(X[idx[0], idx[1]]), float(Y[idx[0], idx[1]])
            int_z = float(Z[idx[0], idx[1]] + pole_height)

            interior_anchors_added.append({
                "id": f"Interior_{i+1}",
                "x": int_x,
                "y": int_y,
                "z": int_z,
                "azimuth": 0.0,
                "beamwidth": 360.0,
                "base_range": float(max_range * 0.8),
                "is_omni": True,
                "enabled": True,
                "show_range": True,
                "type": "Interior"
            })

        anchors.extend(interior_anchors_added)
        coverage_map = calculate_coverage(X, Y, Z, mask, anchors, h_beamwidth, downtilt, max_range)

# ---------------------------
# Per-anchor controls (all anchors)
# ---------------------------
st.sidebar.header("7. Per-Anchor Controls (All Anchors)")
anchor_df = pd.DataFrame(anchors)

# Ensure columns exist
for c, default in [
    ("beamwidth", float(h_beamwidth)),
    ("base_range", float(max_range)),
    ("is_omni", False),
    ("enabled", True),
    ("show_range", True),
]:
    if c not in anchor_df.columns:
        anchor_df[c] = default

editable_cols = ["id", "type", "x", "y", "z", "azimuth", "beamwidth", "base_range", "is_omni", "enabled", "show_range"]
anchor_df = anchor_df[editable_cols].copy()

edited_anchor_df = st.sidebar.data_editor(
    anchor_df,
    key="all_anchor_editor",
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    column_config={
        "x": st.column_config.NumberColumn("X (m)", min_value=-float(pit_radius), max_value=float(pit_radius), step=1.0),
        "y": st.column_config.NumberColumn("Y (m)", min_value=-float(pit_radius), max_value=float(pit_radius), step=1.0),
        "z": st.column_config.NumberColumn("Tower Height Z (m)", min_value=-float(pit_depth), max_value=float(pole_height + 120), step=1.0),
        "azimuth": st.column_config.NumberColumn("Azimuth (°)", min_value=0.0, max_value=359.9, step=1.0),
        "beamwidth": st.column_config.NumberColumn("Beamwidth (°)", min_value=10.0, max_value=360.0, step=1.0),
        "base_range": st.column_config.NumberColumn("Base Range (m)", min_value=10.0, max_value=1500.0, step=5.0),
        "is_omni": st.column_config.CheckboxColumn("Omnidirectional"),
        "enabled": st.column_config.CheckboxColumn("Enabled"),
        "show_range": st.column_config.CheckboxColumn("Show Range"),
    },
)

anchors = edited_anchor_df.to_dict(orient="records")

# Recompute coverage after per-anchor edits
coverage_map = calculate_coverage(X, Y, Z, mask, anchors, h_beamwidth, downtilt, max_range)

# ---------------------------
# Layout / Plots
# ---------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("3D Interactive Coverage Heatmap & Anchor Ranges")

    display_coverage = np.copy(coverage_map).astype(float)
    display_coverage[~mask] = np.nan

    fig = go.Figure()

    # Heatmap surface toggle
    if show_heatmap:
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=display_coverage,
            colorscale="RdYlGn",
            cmin=0, cmax=6,
            colorbar=dict(title="Visible Anchors", x=1.05),
            opacity=0.85,
            name="Mine Pit"
        ))
    else:
        # show pit geometry lightly if heatmap off
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=np.zeros_like(Z),
            colorscale=[[0, "#808080"], [1, "#808080"]],
            showscale=False,
            opacity=0.20,
            name="Mine Pit (No Heatmap)"
        ))

    # Anchor markers by type
    for t, color, symbol, size in [
        ("Rim", "blue", "diamond", 6),
        ("Manual", "orange", "square", 7),
        ("Interior", "purple", "cross", 7),
    ]:
        subset = [a for a in anchors if a.get("type") == t and bool(a.get("enabled", True))]
        if subset:
            fig.add_trace(go.Scatter3d(
                x=[a["x"] for a in subset],
                y=[a["y"] for a in subset],
                z=[a["z"] for a in subset],
                mode="markers+text",
                text=[a["id"] for a in subset],
                textposition="top center",
                marker=dict(size=size, color=color, symbol=symbol),
                name=f"{t} Anchors"
            ))

    # Range visuals per anchor
    for a in anchors:
        if not bool(a.get("enabled", True)):
            continue
        if not bool(a.get("show_range", True)):
            continue

        is_omni = bool(a.get("is_omni", a.get("type") in {"Interior", "Manual"}))

        if is_omni:
            if show_spheres:
                add_omni_sphere(fig, a, h_beamwidth, max_range)
        else:
            if show_cones and (not all_cones_off):
                add_conical_rays(fig, a, downtilt, h_beamwidth, max_range)

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Depth / Elevation (m)",
            aspectratio=dict(x=1, y=1, z=0.45)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Deployment Analytics")

    total_valid_points = int(np.sum(mask))
    covered_points = int(np.sum((coverage_map >= 4) & mask))
    coverage_pct = (covered_points / total_valid_points) * 100 if total_valid_points > 0 else 0
    blind_points = int(np.sum((coverage_map == 0) & mask))

    enabled_anchors = [a for a in anchors if bool(a.get("enabled", True))]
    manual_enabled = [a for a in enabled_anchors if a.get("type") == "Manual"]

    st.metric("Total Anchors (Enabled)", len(enabled_anchors))
    st.metric("Manual Interior Anchors (Enabled)", len(manual_enabled))
    st.metric("Target Coverage Met (≥4 Anchors)", f"{coverage_pct:.1f}%")
    st.metric("Blind Points", blind_points)

    if coverage_pct >= 95:
        st.success("✅ Architecture Target Satisfied (Centimeter Accuracy Ready)")
    elif coverage_pct >= 80:
        st.warning("⚠️ Borderline Coverage. Consider adding more rim anchors or manual nodes.")
    else:
        st.error("❌ High Risk of Blind Spots. System requires adjustment.")

    st.markdown("---")
    st.markdown("### Active Anchor Node Manifest")

    manifest_df = pd.DataFrame(enabled_anchors).copy()
    if not manifest_df.empty:
        manifest_df["effective_range_m"] = manifest_df.apply(
            lambda r: effective_range(
                r.get("base_range", max_range),
                r.get("z", pole_height),
                r.get("beamwidth", h_beamwidth),
                r_cap=max(1200.0, max_range * 2.0)
            ),
            axis=1
        )
        st.dataframe(manifest_df, use_container_width=True)
    else:
        st.info("No enabled anchors.")
