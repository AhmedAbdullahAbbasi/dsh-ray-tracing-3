"""Small CPU smoke run of the optional run_simulation.py material hook.

Overrides configuration in memory, preserving the user's sim_config.py.
The compact cloud is for integration testing, not the external benchmark.
"""
import argparse
import json
import inspect
import sys
from pathlib import Path

import numpy as np

REPOSITORY=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(REPOSITORY))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    import sim_config as cfg
    values=dict(MATERIAL_MODEL="reflex",REFLEX_MATERIAL_DIRECTORY=str(Path(__file__).parent/"materials"),
        REFLEX_ABSORPTION_ENABLED=True,ENERGY_EDGES_KEV=None,DEVICE="cpu",SOURCE_DISTANCE_KPC=10.0,
        CLOUD_FITS_PATH=None,GRID_X_ARCSEC=(-147.5,150,5),GRID_Y_ARCSEC=(-147.5,150,5),
        GRID_Z_KPC=(4.98,5.021,.01),DISC_CENTER_Z_KPC=5.0,DISC_THICKNESS_KPC=.03,
        DISC_RADIUS_ARCSEC=1000,DISC_N_H_CM3=108.025976,SOURCE_MODEL="custom",
        CUSTOM_TIME_EDGES_DAYS=(0,1/86400),CUSTOM_BAND_FLUXES=((0,1,0),),
        PACKETS=8192,CHUNK_SIZE=1024,MAX_INTERACTIONS=8,SEED=8461,LAUNCH_SAMPLER="radial",
        ARRIVAL_DAYS=1000000,TIME_BIN_DAYS=1000000,IMAGE_EPOCH_DAYS=None,IMAGE_ENERGY_KEV=4.9,
        IMAGE_FOV_ARCSEC=None,IMAGE_SCALES=("linear",),IMAGE_DPI=70,
        PROGRESS_LOG=str(root/"progress.log"),OUTPUT_PNG=str(root/"dsh_image.png"),
        OUTPUT_COUNT_PNG=str(root/"dsh_count.png"),OUTPUT_COUNT_NPY=str(root/"counts.npy"))
    for name,value in values.items():setattr(cfg,name,value)
    import run_simulation as driver
    assert "load_reflex_materials" in inspect.getsource(driver.main), "apply driver_hook.patch first"
    original=driver.run_tabulated_source_to_observer_chunked
    report={}
    import reflex_adapter.reflex_materials as provider
    original_loader=provider.load_reflex_materials
    def checked_loader(*a,**kw):
        result=original_loader(*a,**kw)
        sigma=np.asarray(result[2].scattering_cross_section_cm2_per_h)
        assert np.isclose(float(sigma[1]),1.9772030633480432e-24,rtol=1e-6,atol=0)
        report.update(material_model="reflex",sigma_sca_4p9=float(sigma[1]))
        return result
    provider.load_reflex_materials=checked_loader
    def capture(*a,**kw):
        result=original(*a,**kw)
        statuses=np.asarray(result.diagnostics.transport_status_count)
        assert int(statuses[[0,4,5,6]].sum())==0
        assert float(result.diagnostics.scored_observer_fluence)>0
        report.update(statuses=statuses.tolist(),packets=int(result.diagnostics.source_packet_count),
            scored_fluence=float(result.diagnostics.scored_observer_fluence))
        return result
    driver.run_tabulated_source_to_observer_chunked=capture
    driver.main()
    assert report.get("material_model")=="reflex"
    counts=np.load(root/"counts.npy")
    assert counts.shape==(60,60) and counts.sum()>0
    report.update(passed=True,image_shape=list(counts.shape),binned_scoring_events=int(counts.sum()))
    (root/"driver_smoke.json").write_text(json.dumps(report,indent=2)+'\n')


if __name__=="__main__":main()
