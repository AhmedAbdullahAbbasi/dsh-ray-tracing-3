"""Version-2 physical halo response: initial and outgoing energies stay distinct."""

from dataclasses import dataclass

import numpy as np

from .response import _ARRAYS, SpectralResponse, response_from_events

INELASTIC_RESPONSE_ARRAYS = _ARRAYS + (
    "source_energy_kev",
    "incident_energy_kev",
    "dust_order",
    "compton_order",
    "interaction_type",
)
INELASTIC_RESPONSE_CONVENTION = (
    "inelastic halo G/(N*q(E_source)); observed energy separate; extinction included"
)


class IncompleteInelasticResponse(ValueError):
    def __init__(self, diagnostics):
        super().__init__("response requires completed, uncapped, valid histories")
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class InelasticSpectralResponse(SpectralResponse):
    source_energy_kev: np.ndarray
    incident_energy_kev: np.ndarray
    dust_order: np.ndarray
    compton_order: np.ndarray
    interaction_type: np.ndarray

    def __post_init__(self):
        # Reuse all legacy provenance/history/weight validation at INITIAL
        # energy; the observed energy may legitimately lie below the proposal.
        base = SpectralResponse(
            **{name: getattr(self, name) for name in _ARRAYS if name != "energy_kev"},
            energy_kev=self.source_energy_kev,
            n_histories=self.n_histories,
            energy_support_kev=self.energy_support_kev,
            provenance=self.provenance,
        )
        for name in base.__dataclass_fields__:
            if name != "energy_kev":
                object.__setattr__(self, name, getattr(base, name))
        object.__setattr__(self, "source_energy_kev", base.energy_kev)
        for name in (
            "energy_kev",
            "incident_energy_kev",
            "dust_order",
            "compton_order",
            "interaction_type",
        ):
            values = np.asarray(getattr(self, name))
            integer = name in {"dust_order", "compton_order", "interaction_type"}
            if (
                values.shape != base.energy_kev.shape
                or not np.all(np.isfinite(values))
                or (integer and values.dtype.kind not in "iu")
            ):
                raise ValueError(
                    "inelastic response energies/orders must be matching finite vectors; orders/types must be integers"
                )
            values = values.astype(np.int64 if integer else np.float64, copy=True)
            values.flags.writeable = False
            object.__setattr__(self, name, values)
        if (
            np.any(self.energy_kev <= 0)
            or np.any(self.energy_kev > self.incident_energy_kev * (1 + 2e-6))
            or np.any(self.incident_energy_kev > self.source_energy_kev * (1 + 2e-6))
            or np.any(self.dust_order < 0)
            or np.any(self.compton_order < 0)
            or np.any(self.scattering_order != self.dust_order + self.compton_order)
            or np.any(~np.isin(self.interaction_type, [1, 3]))
            or np.any((self.interaction_type == 1) & (self.dust_order < 1))
            or np.any((self.interaction_type == 3) & (self.compton_order < 1))
            or np.any(
                (self.interaction_type == 1)
                & ~np.isclose(
                    self.energy_kev, self.incident_energy_kev, rtol=2e-6, atol=0
                )
            )
        ):
            raise ValueError(
                "inelastic response has inconsistent energies, prefix orders, or interaction types"
            )

    @property
    def component(self):
        return np.where(
            (self.dust_order > 0) & (self.compton_order > 0),
            3,
            np.where(self.compton_order > 0, 2, 1),
        )


def inelastic_response_from_events(
    scored, packets, proposal, transport_status, *, provenance
):
    if np.any(scored.unsupported_escape) or np.any(scored.unsupported_phase):
        raise ValueError("response requires completed, uncapped, valid histories")
    events = scored.events
    base = response_from_events(
        events._replace(energy_kev=scored.source_energy_kev),
        packets,
        proposal,
        transport_status,
        provenance=provenance,
    )
    selected = np.asarray(events.valid)
    arrays = {name: getattr(base, name) for name in _ARRAYS}
    arrays["energy_kev"] = np.asarray(events.energy_kev)[selected]
    return InelasticSpectralResponse(
        **arrays,
        source_energy_kev=base.energy_kev,
        incident_energy_kev=np.asarray(scored.incident_energy_kev)[selected],
        dust_order=np.asarray(scored.dust_order)[selected],
        compton_order=np.asarray(scored.compton_order)[selected],
        interaction_type=np.asarray(scored.interaction_type)[selected],
        n_histories=base.n_histories,
        energy_support_kev=base.energy_support_kev,
        provenance=base.provenance,
    )
