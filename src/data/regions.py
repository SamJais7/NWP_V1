"""
Subdivision and Region Spatial Manager.
Generates GeoJSON boundaries and gridded spatial masks for the 36 IMD Subdivisions
and the 4 Homogeneous Monsoon Regions.
"""

from typing import Dict, List, Any, Tuple
import json
import numpy as np
from src.config import IMD_SUBDIVISIONS, HOMOGENEOUS_REGIONS, TARGET_DOMAIN


# Subdivision approximate polygon boundary definitions (lat/lon rings)
# Centered around the official IMD subdivision centroids
def _build_box_poly(lat: float, lon: float, dlat: float = 1.2, dlon: float = 1.2) -> List[List[float]]:
    """Generates a closed polygon ring [lon, lat] for GeoJSON."""
    return [
        [round(lon - dlon, 3), round(lat - dlat, 3)],
        [round(lon + dlon, 3), round(lat - dlat, 3)],
        [round(lon + dlon, 3), round(lat + dlat, 3)],
        [round(lon - dlon, 3), round(lat + dlat, 3)],
        [round(lon - dlon, 3), round(lat - dlat, 3)],
    ]


class SubdivisionManager:
    """Manages IMD Subdivisions, GeoJSON generation, and gridded spatial masks."""

    def __init__(self):
        self.subdivisions = IMD_SUBDIVISIONS
        self.homogeneous_regions = HOMOGENEOUS_REGIONS

    def get_geojson(self) -> Dict[str, Any]:
        """Generates a valid GeoJSON FeatureCollection of 36 IMD Subdivisions."""
        features = []
        for sub_id, info in self.subdivisions.items():
            # Adaptive bounding size based on geographical extent
            dlat = 1.8 if sub_id in ["SUB_16", "SUB_17", "SUB_19", "SUB_20"] else 1.0
            dlon = 2.0 if sub_id in ["SUB_16", "SUB_17"] else 1.2
            
            coords = _build_box_poly(info["lat"], info["lon"], dlat, dlon)
            feature = {
                "type": "Feature",
                "id": sub_id,
                "properties": {
                    "sub_id": sub_id,
                    "name": info["name"],
                    "abbr": info["abbr"],
                    "region": info["region"],
                    "centroid_lat": info["lat"],
                    "centroid_lon": info["lon"]
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features
        }

    def get_subdivision_mask(
        self,
        sub_id: str,
        lats: np.ndarray,
        lons: np.ndarray
    ) -> np.ndarray:
        """
        Creates a 2D boolean mask (lat_dim, lon_dim) for a given subdivision.
        Points within centroid radius / bounding box are marked True.
        """
        if sub_id not in self.subdivisions:
            raise ValueError(f"Unknown subdivision ID: {sub_id}")

        info = self.subdivisions[sub_id]
        c_lat, c_lon = info["lat"], info["lon"]
        dlat = 1.8 if sub_id in ["SUB_16", "SUB_17", "SUB_19", "SUB_20"] else 1.2
        dlon = 2.0 if sub_id in ["SUB_16", "SUB_17"] else 1.4

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        mask = (
            (lat_grid >= c_lat - dlat) & (lat_grid <= c_lat + dlat) &
            (lon_grid >= c_lon - dlon) & (lon_grid <= c_lon + dlon)
        )

        # Fallback to nearest point if mask is completely empty (e.g. tiny island)
        if not np.any(mask):
            dist = (lat_grid - c_lat) ** 2 + (lon_grid - c_lon) ** 2
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            mask[min_idx] = True

        return mask

    def get_homogeneous_subdivisions(self, region_code: str) -> List[str]:
        """Returns list of subdivision IDs for a given homogeneous monsoon region."""
        if region_code not in self.homogeneous_regions:
            raise ValueError(f"Unknown region code: {region_code}. Must be one of {list(self.homogeneous_regions.keys())}")
        return self.homogeneous_regions[region_code]["subdivisions"]
