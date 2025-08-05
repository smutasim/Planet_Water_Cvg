import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from scipy.ndimage import gaussian_filter

# Threshold
threshold = 0.5 # 50% water coverage
sigma = 1 # gaussian noise smoothing
R = 3389.5e3  # Radius of Mars in meters

file_path = "data/mars/megt90n000cb.xyz"
df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["lon", "lat", "alt"])
df["lon"] = np.degrees(df["lon"] / R)
df["lat"] = np.degrees(df["lat"] / R)

# Reshape to grid
lon = np.sort(df["lon"].unique())
lat = np.sort(df["lat"].unique())
Z_raw = df.pivot(index="lat", columns="lon", values="alt").values
Z = gaussian_filter(Z_raw, sigma=sigma)

# Define sea level
sea_level = df["alt"].quantile(threshold)
water_mask = np.ma.masked_where(Z >= sea_level, Z)
land_mask = np.ma.masked_where(Z < sea_level, Z)

# Custom green→brown→white colormap
terrain_colors = [
    (0.0, "#245e2b"),   # dark green
    (0.4, "#a0785a"),   # brown
    (1.0, "#ffffff")    # white
]
custom_terrain = LinearSegmentedColormap.from_list("green_brown_white", terrain_colors)

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

print(f"Mars water volume: ~{total_volume_km3:.1f} km³")
print(f"Ratio to Earth's oceans: ~{ratio:.3f}×")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(
    water_mask,
    extent=[lon[0], lon[-1], lat[0], lat[-1]],
    cmap='Blues_r',
    origin='lower'
)
ax.imshow(
    land_mask,
    extent=[lon[0], lon[-1], lat[0], lat[-1]],
    cmap=custom_terrain,
    origin='lower'
)
ax.contour(lon, lat, Z, levels=[sea_level], colors='black', linewidths=0.4)

ax.set_title("Mars Topography")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.show()




