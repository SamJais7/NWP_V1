"""
Atmospheric Dynamics Preprocessor for Project Pratyay.
Implements spherical metric spatial derivatives:
  - Scaled Relative Vorticity (ζ_850)
  - True Moisture Flux Convergence -∇·(q V_850)
  - Vertical Wind Shear (200-850 hPa)
  - Training-only P90 Error Standardization
"""

from typing import Tuple, Dict, Any
import numpy as np
import xarray as xr
from src.config import EARTH_RADIUS


class AtmosphericDynamicsPreprocessor:
    """Computes physical atmospheric dynamic fields using spherical metric coordinates."""

    def __init__(self, earth_radius: float = EARTH_RADIUS):
        self.R = earth_radius

    def compute_relative_vorticity(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray
    ) -> np.ndarray:
        """
        Computes Scaled Relative Vorticity (ζ_850) in spherical coordinates:
          ζ = (1 / (R * cos(φ))) * (∂v/∂λ - ∂(u * cos(φ))/∂φ) * 10^5 [s^-1]
        
        Args:
            u: U-wind [m/s], 2D (n_lat, n_lon) or 3D (leads, n_lat, n_lon)
            v: V-wind [m/s]
            lats: Latitude array in degrees
            lons: Longitude array in degrees
        """
        phi = np.radians(lats)
        lam = np.radians(lons)
        dlam = np.gradient(lam)  # radians
        dphi = np.gradient(phi)  # radians

        cos_phi = np.cos(phi)
        # Avoid division by zero at poles
        cos_phi_safe = np.where(np.abs(cos_phi) < 1e-4, 1e-4, cos_phi)

        is_3d = (u.ndim == 3)
        if not is_3d:
            u_in = u[np.newaxis, ...]
            v_in = v[np.newaxis, ...]
        else:
            u_in, v_in = u, v

        n_leads, n_lat, n_lon = u_in.shape
        zeta = np.zeros_like(u_in)

        for t in range(n_leads):
            # ∂v/∂λ
            dv_dlam = np.gradient(v_in[t], axis=1) / dlam[np.newaxis, :]

            # ∂(u * cos(φ))/∂φ
            u_cos = u_in[t] * cos_phi_safe[:, np.newaxis]
            du_cos_dphi = np.gradient(u_cos, axis=0) / dphi[:, np.newaxis]

            # ζ = (1 / (R * cos(φ))) * (dv_dlam - du_cos_dphi) * 10^5
            term = (dv_dlam - du_cos_dphi) / (self.R * cos_phi_safe[:, np.newaxis])
            zeta[t] = term * 1e5

        return zeta if is_3d else zeta[0]

    def compute_moisture_flux_convergence(
        self,
        q: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray
    ) -> np.ndarray:
        """
        Computes True Moisture Flux Convergence (MFC):
          MFC = -∇ · (q V) = - [ (1/(R*cos(φ))) * ( ∂(q*u)/∂λ + ∂(q*v*cos(φ))/∂φ ) ]
        
        Args:
            q: Specific humidity [kg/kg]
            u: U-wind [m/s]
            v: V-wind [m/s]
            lats: Latitude array in degrees
            lons: Longitude array in degrees
        Returns:
            MFC scaled by 10^5 [kg/(kg·s) * 10^5]
        """
        phi = np.radians(lats)
        lam = np.radians(lons)
        dlam = np.gradient(lam)
        dphi = np.gradient(phi)

        cos_phi = np.cos(phi)
        cos_phi_safe = np.where(np.abs(cos_phi) < 1e-4, 1e-4, cos_phi)

        is_3d = (q.ndim == 3)
        if not is_3d:
            q_in = q[np.newaxis, ...]
            u_in = u[np.newaxis, ...]
            v_in = v[np.newaxis, ...]
        else:
            q_in, u_in, v_in = q, u, v

        n_leads, n_lat, n_lon = q_in.shape
        mfc = np.zeros_like(q_in)

        for t in range(n_leads):
            qu = q_in[t] * u_in[t]
            qv_cos = q_in[t] * v_in[t] * cos_phi_safe[:, np.newaxis]

            d_qu_dlam = np.gradient(qu, axis=1) / dlam[np.newaxis, :]
            d_qv_dphi = np.gradient(qv_cos, axis=0) / dphi[:, np.newaxis]

            div = (d_qu_dlam + d_qv_dphi) / (self.R * cos_phi_safe[:, np.newaxis])
            # Convergence is negative divergence
            mfc[t] = -div * 1e5

        return mfc if is_3d else mfc[0]

    def compute_vertical_wind_shear(
        self,
        u200: np.ndarray,
        v200: np.ndarray,
        u850: np.ndarray,
        v850: np.ndarray
    ) -> np.ndarray:
        """
        Computes 200-850 hPa Deep Vertical Wind Shear:
          Shear = sqrt((u200 - u850)^2 + (v200 - v850)^2) [m/s]
        """
        return np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Enriches an input xarray Dataset with derived dynamic fields."""
        lats = ds["lat"].values
        lons = ds["lon"].values

        zeta = self.compute_relative_vorticity(
            ds["u850"].values, ds["v850"].values, lats, lons
        )
        mfc = self.compute_moisture_flux_convergence(
            ds["q850"].values, ds["u850"].values, ds["v850"].values, lats, lons
        )
        shear = self.compute_vertical_wind_shear(
            ds["u200"].values, ds["v200"].values,
            ds["u850"].values, ds["v850"].values
        )

        dims = ["lead_hour", "lat", "lon"]
        ds = ds.assign({
            "zeta850": (dims, zeta),
            "mfc": (dims, mfc),
            "shear": (dims, shear)
        })
        return ds
