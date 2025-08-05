This project includes data from (https://github.com/ahstat/topography) licensed under the Apache License 2.0.

Latitude, Longitude, Height (LLH) data for Venus, Earth, and Mars.

We simulate what the planets would look like with XX% water coverage. For example, ~70% of the Earth is covered by water. We can define this in earth_2d.py and earth_3d.py to get a map of Earth with the lower 70% of data being "blue". We can adjust this value to see what the world would look like with more or less water.

planet_2d.py plots a map of the planet. However, this map is misrepresentative at high latitudes. planet_3d.py was built to create a 3D sphere that represents the planet. By running this code, a window will pop up in your browser with a 3D model. You can rotate and zoom in to see finer details.

All scripts will output in the terminal:
print(f"Planet water volume: ~{total_volume_km3:.1f} km³")
print(f"Ratio to Earth's oceans: ~{ratio:.3f}×")

This represents the volume of water needed to cover x% of the surface, and how much that is relative to Earth's oceans. This computation is an estimate based on each point's surface area and "depth". The surface area is scaled with latitude (higher latitude bins have a lower surface area than those at the equator).

