import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import plotly.io as pio
pio.renderers.default = 'browser'

# Parameters
threshold = 0.7  # 70% water coverage
sigma = 1 # gaussian noise smoothing
file_path = "data/world/etopo10_bedrock.xyz"
R = 6378.14e3  # meters

# Load data
df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["lon", "lat", "alt"])

# Prepare grid
lon = np.sort(df["lon"].unique())
lat = np.sort(df["lat"].unique())

Z_raw = df.pivot(index="lat", columns="lon", values="alt").values
Z = gaussian_filter(Z_raw, sigma=sigma)

if lat[0] > lat[-1]:
    lat = lat[::-1]
    Z = Z[::-1, :]

# Sea level
sea_level = df["alt"].quantile(threshold)

lon_grid, lat_grid = np.meshgrid(np.radians(lon), np.radians(lat))

# Radius + elevation (no exaggeration)
r = R + (Z - sea_level)

# Convert to Cartesian coordinates
x = r * np.cos(lat_grid) * np.cos(lon_grid)
y = r * np.cos(lat_grid) * np.sin(lon_grid)
z = r * np.sin(lat_grid)

# Elevation range
zmin = Z.min()
zmax = Z.max()

# Compute water volume
dlon = np.abs(lon[1] - lon[0])
dlat = np.abs(lat[1] - lat[0])

# Mean grid cell area approximation on a sphere
deg2rad = np.pi / 180
mean_cell_area = (
    (R**2) * deg2rad * dlon *  # longitude width
    (deg2rad * dlat)           # latitude height
)

# Adjust for latitude compression (cos(latitude))
df["weight"] = np.cos(np.radians(df["lat"]))
df["water_depth"] = np.maximum(0, sea_level - df["alt"])
df["cell_volume"] = df["water_depth"] * mean_cell_area * df["weight"]

# Total volume in m³ → km³
total_volume_km3 = df["cell_volume"].sum() / 1e9

# Earth's ocean volume for reference
earth_ocean_volume_km3 = 1.332e9
ratio = total_volume_km3 / earth_ocean_volume_km3

print(f"Earth water volume: ~{total_volume_km3:.1f} km³")
print(f"Ratio to Earth's oceans: ~{ratio:.3f}×")

# Normalize helper
def norm(val):
    return (val - zmin) / (zmax - zmin)

breakpoint = norm(sea_level)

# Colorscale: water (blue) below sea level, land (green-brown-white) above
colorscale = [
    [0.0, 'rgb(0, 70, 140)'],               # dark blue deep water
    [breakpoint, 'rgb(173, 216, 230)'],     # light blue sea level
    [breakpoint + 1e-6, 'rgb(36, 94, 43)'],   # dark green just above sea level
    [norm(zmax) * 0.4 + breakpoint * 0.6, 'rgb(160, 120, 90)'],  # brown mid land
    [1.0, 'rgb(255, 255, 255)']             # white peak
]

hovertemplate = (
    "Latitude: %{customdata[0]:.2f}°<br>" +
    "Longitude: %{customdata[1]:.2f}°<br>" +
    "Altitude: %{customdata[2]:.1f} m<br>" +
    "<extra></extra>"
)

customdata_fixed = np.dstack((lat_grid*180/np.pi, lon_grid*180/np.pi, Z))

surface = go.Surface(
    x=x, y=y, z=z,
    surfacecolor=Z,
    colorscale=colorscale,
    cmin=zmin,
    cmax=zmax,
    showscale=False,
    lighting=dict(ambient=0.5, diffuse=1, roughness=0.9),
    customdata=customdata_fixed,
    hovertemplate=hovertemplate
)

pole_axis = go.Scatter3d(
    x=[0, 0],
    y=[0, 0],
    z=[R * -1.1, R * 1.1],
    mode='lines',
    line=dict(color='white', width=4),
    name='Pole axis'
)

fig = go.Figure(data=[surface, pole_axis])

fig.update_layout(
    scene=dict(
        xaxis=dict(showbackground=False, visible=False, zeroline=False, showticklabels=False, showgrid=False, showline=False, showspikes=False),
        yaxis=dict(showbackground=False, visible=False, zeroline=False, showticklabels=False, showgrid=False, showline=False, showspikes=False),
        zaxis=dict(showbackground=False, visible=False, zeroline=False, showticklabels=False, showgrid=False, showline=False, showspikes=False, range=[R * -1.2, R * 1.2]),
        bgcolor='black',
        aspectmode='data',
    ),
    hovermode=False,
    paper_bgcolor='black',
    font=dict(color='white'),
)

fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[R],
    mode='markers+text',
    text=['North Pole'],
    textposition='top center',
    marker=dict(size=4, color='white')
))

fig.show()
