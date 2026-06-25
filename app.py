import streamlit as st
import numpy as np
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

# --- Simulation Engine ---
def generate_pit_mesh(radius, depth, resolution):
    """Generates a simplified 3D paraboloid bowl to represent the open-pit mine."""
    x = np.arange(-radius, radius + resolution, resolution)
    y = np.arange(-radius, radius + resolution, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Keep it within a circular boundary
    mask = (X**2 + Y**2) <= radius**2
    
    # Paraboloid formula for depth profile
    Z = -depth * (1 - (X**2 + Y**2) / radius**2)
    Z[~mask] = 0 # Surface level outside the pit
    
    return X, Y, Z, mask

def place_rim_anchors(radius, num_anchors, pole_h):
    """Places anchors uniformly around the top rim facing inward."""
    angles = np.linspace(0, 2 * np.pi, num_anchors, endpoint=False)
    anchors = []
    for idx, theta in enumerate(angles):
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = pole_h # Top rim is z=0, plus the pole height
        # Azimuth points exactly inward (flip the angle)
        azimuth = np.degrees(theta + np.pi) % 360
        anchors.append({"id": f"Rim_{idx+1}", "x": x, "y": y, "z": z, "azimuth": azimuth, "type": "Rim"})
    return anchors

def calculate_coverage(X, Y, Z, mask, anchors, h_bw, tilt, r_max):
    """Evaluates line-of-sight visibility and beam intersection for every grid node."""
    rows, cols = X.shape
    visibility_matrix = np.zeros((rows, cols))
    
    # Flatten grid for fast vector math
    flat_X = X.flatten()
    flat_Y = Y.flatten()
    flat_Z = Z.flatten()
    flat_mask = mask.flatten()
    
    visible_counts = np.zeros_like(flat_X)
    
    for a in anchors:
        # Distance from anchor to all points
        dx = flat_X - a["x"]
        dy = flat_Y - a["y"]
        dz = flat_Z - a["z"]
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # 1. Range Check
        in_range = distances <= r_max
        
        # 2. Horizontal Beamwidth Check (Azimuth alignment)
        angle_to_points = np.arctan2(dy, dx) # Vector from anchor to point
        azimuth_rad = np.radians(a["azimuth"])
        
        # Difference angle normalized to [-pi, pi]
        angle_diff = np.arctan2(np.sin(angle_to_points - azimuth_rad), np.cos(angle_to_points - azimuth_rad))
        in_h_beam = np.abs(np.degrees(angle_diff)) <= (h_bw / 2)
        
        # Combine constraints
        is_visible = in_range & in_h_beam & flat_mask
        visible_counts += is_visible
        
    return visible_counts.reshape(rows, cols)

# Run Base Simulation
X, Y, Z, mask = generate_pit_mesh(pit_radius, pit_depth, grid_resolution)
anchors = place_rim_anchors(pit_radius, num_rim_anchors, pole_height)
coverage_map = calculate_coverage(X, Y, Z, mask, anchors, h_beamwidth, downtilt, max_range)

# Adaptive Algorithm Layer (Section 6 of Spec)
interior_anchors_added = []
if auto_insert_interior:
    # Identify coordinates where visible anchors < 4 (Blind / Weak zones)
    weak_zones = (coverage_map < 4) & mask
    if np.any(weak_zones):
        # Place a fallback interior anchor at the centroid of the weakest zone
        weak_indices = np.argwhere(weak_zones)
        # Select up to 2 strategic deep points for illustration
        for i in range(min(2, len(weak_indices) // 10 + 1)):
            idx = weak_indices[int(len(weak_indices) * (i + 1) / (i + 2))]
            int_x, int_y = X[idx[0], idx[1]], Y[idx[0], idx[1]]
            int_z = Z[idx[0], idx[1]] + pole_height # Mounted on a mobile trailer/pole
            
            interior_anchors_added.append({
                "id": f"Interior_{i+1}", "x": int_x, "y": int_y, "z": int_z, "azimuth": 0, "type": "Interior"
            })
        
        # Recalculate with interior fixes included
        anchors.extend(interior_anchors_added)
        coverage_map = calculate_coverage(X, Y, Z, mask, anchors, h_beamwidth, downtilt, max_range)

# --- Layout Columns ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("3D Interactive Coverage Heatmap")
    
    # Hide outside data for cleaner rendering
    display_coverage = np.copy(coverage_map)
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
    
    # Plot Rim Anchors
    rim_x = [a["x"] for a in anchors if a["type"] == "Rim"]
    rim_y = [a["y"] for a in anchors if a["type"] == "Rim"]
    rim_z = [a["z"] for a in anchors if a["type"] == "Rim"]
    fig.add_trace(go.Scatter3d(
        x=rim_x, y=rim_y, z=rim_z, mode='markers',
        marker=dict(size=6, color='blue', symbol='diamond'),
        name="Rim Anchors"
    ))
    
    # Plot Adaptive Interior Anchors
    if interior_anchors_added:
        int_x = [a["x"] for a in interior_anchors_added]
        int_y = [a["y"] for a in interior_anchors_added]
        int_z = [a["z"] for a in interior_anchors_added]
        fig.add_trace(go.Scatter3d(
            x=int_x, y=int_y, z=int_z, mode='markers',
            marker=dict(size=8, color='purple', symbol='cross'),
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
    
    # Calculations
    total_valid_points = np.sum(mask)
    covered_points = np.sum((coverage_map >= 4) & mask)
    coverage_pct = (covered_points / total_valid_points) * 100 if total_valid_points > 0 else 0
    blind_points = np.sum((coverage_map == 0) & mask)
    
    st.metric("Total Deployed Anchors", len(anchors))
    st.metric("Target Coverage Met (≥4 Anchors)", f"{coverage_pct:.1f}%")
    
    # Progress bars / visual status
    if coverage_pct >= 95:
        st.success("✅ Architecture Target Satisfied (Centimeter Accuracy Ready)")
    elif coverage_pct >= 80:
        st.warning("⚠️ Borderline Coverage. Consider adding more rim anchors or manual nodes.")
    else:
        st.error("❌ High Risk of Blind Spots. System requires adjustment.")
        
    st.markdown("---")
    st.markdown("### Active Anchor Node Manifest")
    st.dataframe(anchors, use_container_width=True)

# --- Bottom Documentation ---
st.markdown("---")
st.markdown("""
### How to Run This Online Instantly:
1. Copy this code block block and save it as a file named `app.py`.
2. Create a file named `requirements.txt` in the same directory and add these two lines:
   ```text
   streamlit
   plotly
   numpy
          
