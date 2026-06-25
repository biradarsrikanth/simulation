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
pole_height = st.sidebar.slider("Pole Height (m)", 8, 15, 12, step=1)

st.sidebar.header("3. Antenna Pattern & Orientation")
downtilt = st.sidebar.slider("Mechanical Downtilt (°)", 10, 45, 25, step=5)
h_beamwidth = st.sidebar.selectbox("Horizontal Beamwidth", [60, 90, 120], index=1)
max_range = st.sidebar.slider("Max UWB Effective Range (m)", 100, 500, 300, step=25)

st.sidebar.header("4. Optimization Settings")
auto_insert_interior = st.sidebar.checkbox("Auto-insert Interior Anchors (Adaptive)", value=True)

# --- Manual Interior Anchors ---
st.sidebar.header("5. Manual Interior Anchors")
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
        [{"x": 0.0, "y": 0.0, "z": 0.0, "azimuth": 0.0}]
    )

target_rows = int(num_manual_anchors)
df = st.session_state.manual_anchor_df.copy()

# Resize rows to match requested count
if target_rows == 0:
    df = pd.DataFrame(columns=["x", "y", "z", "azimuth"])
elif len(df) < target_rows:
    add_n = target_rows - len(df)
    add_df = pd.DataFrame([{"x": 0.0, "y": 0.0, "z": 0.0, "azimuth": 0.0}] * add_n)
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
        "z": st.column_config.NumberColumn("Z (m)", min_value=-float(pit_depth), max_value=float(pole_height + 50), step=1.0),
        "azimuth": st.column_config.NumberColumn("Azimuth (°)", min_value=0.0, max_value=359.9, step=1.0),
    },
)

# Persist edited values
st.session_state.manual_anchor_df = edited_df

# --- Simulation Engine ---
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


def place_rim_anchors(radius, num_anchors, pole_h):
    """Places anchors uniformly around the top rim facing inward."""
    angles = np.linspace(0, 2 * np.pi, num_anchors, endpoint=False)
    anchors = []
    for idx, theta in enumerate(angles):
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = pole_h  # Top rim is z=0, plus the pole height
        # Azimuth points exactly inward (flip the angle)
        azimuth = np.degrees(theta + np.pi) % 360
        anchors.append(
            {"id": f"Rim_{idx+1}", "x": x, "y": y, "z": z, "azimuth": azimuth, "type": "Rim"}
        )
    return anchors


def calculate_coverage(X, Y, Z, mask, anchors, h_bw, tilt, r_max):
    """Evaluates line-of-sight visibility and beam intersection for every grid node."""
    rows, cols = X.shape
    flat_X = X.flatten()
    flat_Y = Y.flatten()
    flat_Z = Z.flatten()
    flat_mask = mask.flatten()

    visible_counts = np.zeros_like(flat_X, dtype=int)

    for a in anchors:
        # Distance from anchor to all points
        dx = flat_X - a["x"]
        dy = flat_Y - a["y"]
        dz = flat_Z - a["z"]
        distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # 1. Range Check
        in_range = distances <= r_max

        # 2. Horizontal Beamwidth Check (Azimuth alignment)
        angle_to_points = np.arctan2(dy, dx)  # Vector from anchor to point
        azimuth_rad = np.radians(a["azimuth"])

        # Difference angle normalized to [-pi, pi]
        angle_diff = np.arctan2(
            np.sin(angle_to_points - azimuth_rad),
            np.cos(angle_to_points - azimuth_rad),
        )
        in_h_beam = np.abs(np.degrees(angle_diff)) <= (h_bw / 2)

        # Combine constraints
        is_visible = in_range & in_h_beam & flat_mask
        visible_counts += is_visible.astype(int)

    return visible_counts.reshape(rows, cols)


# Run Base Simulation
X, Y, Z, mask = generate_pit_mesh(pit_radius, pit_depth, grid_resolution)
anchors = place_rim_anchors(pit_radius, num_rim_anchors, pole_height)

# Build manual anchor list from editable table
manual_anchors = []
for i, r in edited_df.iterrows():
    mx, my, mz, maz = float(r["x"]), float(r["y"]), float(r["z"]), float(r["azimuth"])
    if (mx ** 2 + my ** 2) <= (pit_radius ** 2):
        manual_anchors.append({
            "id": f"Manual_{i+1}",
            "x": mx,
            "y": my,
            "z": mz,
            "azimuth": maz,
            "type": "Manual"
        })
    else:
        st.sidebar.warning(f"Manual Anchor {i+1}: (x,y) outside pit boundary; ignored.")

# Include manual anchors before first coverage run
anchors.extend(manual_anchors)
coverage_map = calculate_coverage(X, Y, Z, mask, anchors, h_beamwidth, downtilt, max_range)

# Adaptive Algorithm Layer (Section 6 of Spec)
interior_anchors_added = []
if auto_insert_interior:
    # Identify coordinates where visible anchors < 4 (Blind / Weak zones)
    weak_zones = (coverage_map < 4) & mask
    if np.any(weak_zones):
        weak_indices = np.argwhere(weak_zones)
        for i in range(min(2, max(1, len(weak_indices) // 10 + 1))):
            idx = weak_indices[int(len(weak_indices) * (i + 1) / (i + 2))]
            int_x, int_y = X[idx[0], idx[1]], Y[idx[0], idx[1]]
            int_z = Z[idx[0], idx[1]] + pole_height

            interior_anchors_added.append({
                "id": f"Interior_{i+1}",
                "x": int_x,
                "y": int_y,
                "z": int_z,
                "azimuth": 0,
                "type": "Interior"
            })

        anchors.extend(interior_anchors_added)
        coverage_map = calculate_coverage(X, Y, Z, mask, anchors, h_beamwidth, downtilt, max_range)

# --- Layout Columns ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("3D Interactive Coverage Heatmap")

    display_coverage = np.copy(coverage_map).astype(float)
    display_coverage[~mask] = np.nan

    fig = go.Figure()

    # Pit surface map
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=display_coverage,
        colorscale="RdYlGn",
        cmin=0, cmax=6,
        colorbar=dict(title="Visible Anchors", x=1.05),
        opacity=0.85,
        name="Mine Pit"
    ))

    # Rim Anchors
    rim_x = [a["x"] for a in anchors if a["type"] == "Rim"]
    rim_y = [a["y"] for a in anchors if a["type"] == "Rim"]
    rim_z = [a["z"] for a in anchors if a["type"] == "Rim"]
    fig.add_trace(go.Scatter3d(
        x=rim_x, y=rim_y, z=rim_z,
        mode="markers",
        marker=dict(size=6, color="blue", symbol="diamond"),
        name="Rim Anchors"
    ))

    # Manual Anchors
    if manual_anchors:
        man_x = [a["x"] for a in manual_anchors]
        man_y = [a["y"] for a in manual_anchors]
        man_z = [a["z"] for a in manual_anchors]
        fig.add_trace(go.Scatter3d(
            x=man_x, y=man_y, z=man_z,
            mode="markers",
            marker=dict(size=8, color="orange", symbol="square"),
            name="Manual Interior Anchors"
        ))

    # Adaptive Interior Anchors
    if interior_anchors_added:
        int_x = [a["x"] for a in interior_anchors_added]
        int_y = [a["y"] for a in interior_anchors_added]
        int_z = [a["z"] for a in interior_anchors_added]
        fig.add_trace(go.Scatter3d(
            x=int_x, y=int_y, z=int_z,
            mode="markers",
            marker=dict(size=8, color="purple", symbol="cross"),
            name="Adaptive Interior Anchors"
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Depth (m)",
            aspectratio=dict(x=1, y=1, z=0.4)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Deployment Analytics")

    total_valid_points = int(np.sum(mask))
    covered_points = int(np.sum((coverage_map >= 4) & mask))
    coverage_pct = (covered_points / total_valid_points) * 100 if total_valid_points > 0 else 0
    blind_points = int(np.sum((coverage_map == 0) & mask))

    st.metric("Total Deployed Anchors", len(anchors))
    st.metric("Manual Interior Anchors", len(manual_anchors))
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
    st.dataframe(pd.DataFrame(anchors), use_container_width=True)

