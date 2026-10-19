"""Small configured-run inputs; no transport or numerical imports at import time."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def config_text():
    data = ROOT / "dsh/data"
    return f'''config_version = 1
[run]
name = "test"
[scene]
kind = "example_four_cloud"
source_distance_kpc = 10.0
[source]
kind = "representative"
time_edges_days = [0.0, 0.01]
effective_energy_kev = [3.3, 4.9, 6.9]
band_flux = [[0.02, 0.012, 0.006]]
[materials]
kind = "table_pair"
scattering_file = "{(data / "newdust/mrn_rg_drude_v1.npz").as_posix()}"
absorption_file = "{(data / "absorption/tbabs_wilm_vern_v1.npz").as_posix()}"
[execution]
packets = 37
chunk_size = 16
max_interactions = 8
seed = 2026
device = "cpu"
[observer]
energy_edges_kev = [2.5, 4.1, 5.9, 7.9]
[observer.arrival]
kind = "uniform"
start_days = 0.0
stop_days = 2.5
step_days = 1.0
[output]
directory = "results with spaces"
formats = ["npz", "fits"]
'''


def write_config(directory, text=None, *, filename="run.toml", fits_scene=False):
    directory = Path(directory)
    text = config_text() if text is None else text
    if fits_scene:
        import numpy as np
        from astropy.io import fits

        hdu = fits.PrimaryHDU(np.full((2, 3, 3), 1e21))
        hdu.header["BUNIT"] = "cm-2"
        for i, (kind, unit, start, step) in enumerate(
            (
                ("XOFFSET", "arcsec", -10.0, 10.0),
                ("YOFFSET", "arcsec", -10.0, 10.0),
                ("DISTANCE", "kpc", 2.0, 0.1),
            ),
            1,
        ):
            for key, value in {
                "CTYPE": kind,
                "CUNIT": unit,
                "CRVAL": start,
                "CDELT": step,
                "CRPIX": 1.0,
            }.items():
                hdu.header[f"{key}{i}"] = value
        hdu.writeto(directory / "cloud.fits", overwrite=True)
        text = text.replace(
            'kind = "example_four_cloud"', 'kind = "fits"\ncloud_file = "cloud.fits"'
        )
    path = directory / filename
    path.write_text(text, encoding="utf-8")
    return path
