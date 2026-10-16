"""Published WD01 R_V=3.1, b_C=60 ppm distributions, equations (2)--(6).

Population number densities are dn/da per H nucleus per micron. No total
abundance set or gas depletion is implied. Original and 0.93-rescaled variants
are distinct; neither is silently fitted to the Wilms elemental budget.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .material_definitions import (
    ATOMIC_MASS_UNIT_G,
    ATOMIC_WEIGHTS_U,
    ElementalAbundances,
    GrainComponent,
    MaterialDefinition,
)

WD01_REFERENCE = "Weingartner & Draine 2001, ApJ 548, 296; Table 1, R_V=3.1, b_C=6e-5, case A; https://arxiv.org/abs/astro-ph/0008146"
MINIMUM_RADIUS_UM = 0.00035
# alpha, beta, a_t [um], a_c [um], C. The silicate cutoff is fixed in Table 1.
WD01_PARAMETERS = {
    "carbonaceous": (-1.54, -0.165, 0.0107, 0.428, 9.99e-12),
    "silicate": (-2.21, 0.300, 0.164, 0.100, 1.00e-13),
}
DENSITIES_G_CM3 = {"carbonaceous": 2.24, "silicate": 3.5}
D03_DENSITIES_G_CM3 = {"carbonaceous": 2.2, "silicate": 3.8}
STOICHIOMETRY = {
    "carbonaceous": {"C": 1},
    "silicate": {"Mg": 1, "Fe": 1, "Si": 1, "O": 4},
}


@dataclass(frozen=True)
class WD01Population:
    """A named normalization of the published distribution on a finite domain.

    Carbonaceous particles, including the log-normal populations, use a bulk
    graphite sphere approximation in the X-ray producer. This is not a full
    Li--Draine PAH optical model. Radius cutoffs are explicit numerical inputs.
    """

    normalization: str = "wd01_original"
    carbon_scale: float = 1.0
    silicate_scale: float = 1.0
    maximum_radius_um: float = 2.0

    def __post_init__(self):
        if self.normalization not in ("wd01_original", "d03_0p93"):
            raise ValueError("normalization must be wd01_original or d03_0p93")
        for name in ("carbon_scale", "silicate_scale", "maximum_radius_um"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.maximum_radius_um <= MINIMUM_RADIUS_UM:
            raise ValueError("maximum radius must exceed 3.5 Angstrom")

    @property
    def base_scale(self):
        return 1.0 if self.normalization == "wd01_original" else 0.93

    def as_dict(self):
        return {
            "model": "WD01-RV3.1-bC60-caseA",
            "reference": WD01_REFERENCE,
            "normalization": self.normalization,
            "base_scale": self.base_scale,
            "carbon_scale": self.carbon_scale,
            "silicate_scale": self.silicate_scale,
            "minimum_radius_um": MINIMUM_RADIUS_UM,
            "maximum_radius_um": self.maximum_radius_um,
            "parameters": WD01_PARAMETERS,
            "densities_g_cm3": DENSITIES_G_CM3,
            "d03_optical_material_densities_g_cm3": D03_DENSITIES_G_CM3,
            "carbon_atom_mass_g": ATOMIC_WEIGHTS_U["C"] * ATOMIC_MASS_UNIT_G,
            "carbon_optics": "D03 bulk graphite spheres, 1/3 parallel + 2/3 perpendicular; PAH-specific optics not included",
        }

    @property
    def sha256(self):
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def dn_da_per_h_per_um(self, radius_um, component, *, lognormal_only=False):
        """Evaluate the published law with its absolute per-H normalization."""
        if component not in WD01_PARAMETERS:
            raise ValueError("component must be carbonaceous or silicate")
        radius = np.asarray(radius_um, dtype=np.float64)
        if np.any(~np.isfinite(radius)) or np.any(radius <= 0):
            raise ValueError("grain radii must be finite and positive")
        alpha, beta, transition_um, cutoff_um, amplitude = WD01_PARAMETERS[component]
        ratio = radius / transition_um
        curvature = 1 + beta * ratio if beta >= 0 else 1 / (1 - beta * ratio)
        # The factor 1/a and output unit conversion cancel when a is in um.
        tail = np.maximum((radius - transition_um) / cutoff_um, 0)
        distribution = (
            amplitude / radius * ratio**alpha * curvature * np.exp(-(tail**3))
        )
        if lognormal_only:
            distribution = np.zeros_like(radius)
        if component == "carbonaceous":
            import math

            sigma = 0.4
            density = DENSITIES_G_CM3[component]
            carbon_mass = ATOMIC_WEIGHTS_U["C"] * ATOMIC_MASS_UNIT_G
            for center_um, carbon_abundance in (
                (0.00035, 0.75 * 6e-5),
                (0.003, 0.25 * 6e-5),
            ):
                center_cm = center_um * 1e-4
                argument = (
                    3 * sigma + np.log(center_um / MINIMUM_RADIUS_UM) / sigma
                ) / np.sqrt(2)
                b = (
                    3
                    / (2 * np.pi) ** 1.5
                    * np.exp(-4.5 * sigma**2)
                    * carbon_abundance
                    * carbon_mass
                    / (density * center_cm**3 * sigma * (1 + math.erf(argument)))
                )
                distribution += (
                    b
                    / radius
                    * np.exp(-0.5 * (np.log(radius / center_um) / sigma) ** 2)
                )
        scale = (
            self.carbon_scale if component == "carbonaceous" else self.silicate_scale
        )
        return np.where(
            (radius >= MINIMUM_RADIUS_UM) & (radius <= self.maximum_radius_um),
            distribution * self.base_scale * scale,
            0.0,
        )

    def quadrature(self, component, order=24):
        """Piecewise Gauss--Legendre in log(radius), including law transitions.

        Returns radius in um and number of grains/H represented by each node.
        Refinement changes the quadrature order rather than the physical law.
        """
        if not isinstance(order, int) or order < 8 or order > 256:
            raise ValueError("quadrature order must be an integer in [8, 256]")
        if component not in WD01_PARAMETERS:
            raise ValueError("component must be carbonaceous or silicate")
        breaks = np.unique(
            np.clip(
                [
                    MINIMUM_RADIUS_UM,
                    0.003,
                    0.0107,
                    0.03,
                    0.1,
                    0.164,
                    0.3,
                    0.7,
                    self.maximum_radius_um,
                ],
                MINIMUM_RADIUS_UM,
                self.maximum_radius_um,
            )
        )
        nodes, weights = np.polynomial.legendre.leggauss(order)
        radii, counts = [], []
        for lo, hi in zip(np.log(breaks[:-1]), np.log(breaks[1:]), strict=True):
            a = np.exp((lo + hi) / 2 + (hi - lo) / 2 * nodes)
            radii.append(a)
            counts.append(
                weights * (hi - lo) / 2 * a * self.dn_da_per_h_per_um(a, component)
            )
        return np.concatenate(radii), np.concatenate(counts)

    def inventory(self, order=32, *, density_model="wd01"):
        """Calculate volume, mass, and element counts from the actual size law."""
        if density_model not in ("wd01", "d03"):
            raise ValueError("density_model must be wd01 or d03")
        densities = DENSITIES_G_CM3 if density_model == "wd01" else D03_DENSITIES_G_CM3
        result = {}
        for name in WD01_PARAMETERS:
            radius, count = self.quadrature(name, order)
            volume = float(np.sum(4 * np.pi / 3 * (radius * 1e-4) ** 3 * count))
            mass = volume * densities[name]
            formula_mass = (
                sum(ATOMIC_WEIGHTS_U[e] * n for e, n in STOICHIOMETRY[name].items())
                * ATOMIC_MASS_UNIT_G
            )
            result[name] = {
                "volume_cm3_per_h": volume,
                "mass_g_per_h": mass,
                "density_g_cm3": densities[name],
                "atoms_per_h": {
                    e: n * mass / formula_mass for e, n in STOICHIOMETRY[name].items()
                },
            }
        return result

    def material_definition(
        self,
        abundances: ElementalAbundances,
        *,
        method="mie",
        molecular_hydrogen_fraction=0.2,
    ):
        """Join a total abundance set explicitly, failing on elemental overdraw."""
        # Number-versus-radius is WD01; chemical masses must use the densities
        # associated with the actual D03 optical constants, not WD01's older
        # mass-conversion densities. Both inventories remain separately available.
        inventory = self.inventory(density_model="d03")
        grains = tuple(
            GrainComponent(
                component_id=name,
                stoichiometry=STOICHIOMETRY[name],
                density_g_cm3=D03_DENSITIES_G_CM3[name],
                mass_per_h_g=inventory[name]["mass_g_per_h"],
                size_distribution_id=f"WD01-RV3.1-bC60-{self.normalization}",
                optical_constants_id=f"Draine2003-{name}-Xray-spheres-v1",
                reference=WD01_REFERENCE,
                size_parameters={
                    "minimum_radius_um": MINIMUM_RADIUS_UM,
                    "maximum_radius_um": self.maximum_radius_um,
                    "normalization_scale": self.base_scale
                    * (
                        self.carbon_scale
                        if name == "carbonaceous"
                        else self.silicate_scale
                    ),
                },
            )
            for name in WD01_PARAMETERS
        )
        definition = MaterialDefinition(
            material_id=f"WD01-RV3.1-bC60-Xray-{self.sha256[:12]}",
            abundances=abundances,
            grains=grains,
            molecular_hydrogen_fraction=molecular_hydrogen_fraction,
            scattering_method=method,
            reference=WD01_REFERENCE,
        )
        definition.audit()
        return definition
