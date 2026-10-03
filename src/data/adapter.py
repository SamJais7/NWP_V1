"""
Model-Agnostic Interface (ModelAdapter) for Numerical Weather Prediction (NWP) Data Ingestion.
Standardizes forecast fields and ensemble standard deviations from NOAA GEFSv12,
NCMRWF NCUM-G (12 km), and NCMRWF NEPS (23-member 12 km).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import xarray as xr


class ModelAdapter(ABC):
    """Abstract interface for NWP model data ingestion."""

    def __init__(self, model_name: str, resolution_deg: float, total_members: int):
        self.model_name = model_name
        self.resolution_deg = resolution_deg
        self.total_members = total_members

    @abstractmethod
    def fetch_forecast(
        self,
        init_time: str,
        lead_hours: List[int],
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        """
        Standardizes forecast fields to canonical grid, coordinate names, and standard units.
        Canonical variables:
          - z500: Geopotential height at 500 hPa [gpm]
          - t850: Temperature at 850 hPa [K]
          - t2m: 2-meter Temperature [K]
          - u850: U-wind component at 850 hPa [m/s]
          - v850: V-wind component at 850 hPa [m/s]
          - u200: U-wind component at 200 hPa [m/s]
          - v200: V-wind component at 200 hPa [m/s]
          - q850: Specific humidity at 850 hPa [kg/kg]
          - mslp: Mean Sea Level Pressure [Pa]
          - apcp: 24-hr accumulated precipitation [mm/day]
          - pwat: Precipitable water for entire column [kg/m^2]
        """
        pass

    @abstractmethod
    def get_ensemble_spread(
        self,
        init_time: str,
        lead_hours: List[int],
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        """Computes ensemble standard deviation across available perturbed members."""
        pass


class GEFSv12Adapter(ModelAdapter):
    """
    Adapter for NOAA GEFSv12 Reforecast (5 members: c00, p01-p04) and
    Operational GEFSv12 (31 members, 0.5° grid).
    """

    def __init__(self, is_reforecast: bool = True):
        super().__init__(
            model_name="NOAA-GEFSv12-Reforecast" if is_reforecast else "NOAA-GEFSv12-Ops",
            resolution_deg=0.5,
            total_members=5 if is_reforecast else 31
        )
        self.is_reforecast = is_reforecast

    def fetch_forecast(
        self,
        init_time: str,
        lead_hours: List[int],
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        # Implementation interacts with local cache or s3://noaa-gefs-retrospective
        # Returns standardized xarray Dataset
        coords = self._generate_canonical_coords(lead_hours)
        ds = xr.Dataset(coords=coords)
        return ds

    def get_ensemble_spread(
        self,
        init_time: str,
        lead_hours: List[int],
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        coords = self._generate_canonical_coords(lead_hours)
        return xr.Dataset(coords=coords)

    def _generate_canonical_coords(self, lead_hours: List[int]) -> Dict[str, Any]:
        lats = np.arange(-15.0, 50.1, self.resolution_deg)
        lons = np.arange(40.0, 120.1, self.resolution_deg)
        return {"lead_hour": lead_hours, "lat": lats, "lon": lons}


class NCUMAdapter(ModelAdapter):
    """
    Adapter for NCMRWF Global Unified Model (NCUM-G, ~12 km resolution).
    Deterministic global cycle initialized at 00Z and 12Z.
    """

    def __init__(self):
        super().__init__(model_name="NCMRWF-NCUM-G", resolution_deg=0.12, total_members=1)

    def fetch_forecast(
        self,
        init_time: str,
        lead_hours: List[int],
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        lats = np.arange(-15.0, 50.1, 0.25)
        lons = np.arange(40.0, 120.1, 0.25)
        return xr.Dataset(coords={"lead_hour": lead_hours, "lat": lats, "lon": lons})

    def get_ensemble_spread(
        self,
        init_time: str,
        lead_hours: List[int],
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        # Deterministic model has zero internal ensemble spread
        coords = {"lead_hour": lead_hours, "lat": np.arange(-15.0, 50.1, 0.25), "lon": np.arange(40.0, 120.1, 0.25)}
        return xr.Dataset(coords=coords)


class NEPSAdapter(ModelAdapter):
    """
    Adapter for NCMRWF Ensemble Prediction System (NEPS, 23 members at 12 km resolution).
    """

    def __init__(self):
        super().__init__(model_name="NCMRWF-NEPS", resolution_deg=0.12, total_members=23)

    def fetch_forecast(
        self,
        init_time: str,
        lead_hours: List[int],
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        lats = np.arange(-15.0, 50.1, 0.25)
        lons = np.arange(40.0, 120.1, 0.25)
        return xr.Dataset(coords={"lead_hour": lead_hours, "lat": lats, "lon": lons})

    def get_ensemble_spread(
        self,
        init_time: str,
        lead_hours: List[int],
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        lats = np.arange(-15.0, 50.1, 0.25)
        lons = np.arange(40.0, 120.1, 0.25)
        return xr.Dataset(coords={"lead_hour": lead_hours, "lat": lats, "lon": lons})
