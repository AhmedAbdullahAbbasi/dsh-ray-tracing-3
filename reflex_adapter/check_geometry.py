"""Paired-history check of the 16-to-32-cell cloud boundary refinement.

Both scenes receive the same physical source, optical-depth and scattering
variates and the same camera-aperture offsets. The resulting paired difference
can be much more precise than two independent image estimates.
"""
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from dsh.observer.scoring import score_peeloff_events
from dsh.sources.launch import sample_source_launches
from dsh.sources.models import SourcePackets
from dsh.sources.radial_launch import build_radial_cloud_launch_geometry
from dsh.transport.kernel import transport_photon_batch
from reflex_adapter.compare_reflex import make_cloud, normalize_camera_counts
from reflex_adapter.reflex_materials import load_reflex_materials


def main():
    output=Path(sys.argv[1]);config=json.loads((output/"frozen_config.json").read_text())
    _,_,physics=load_reflex_materials()
    coarse=make_cloud(config);fine=make_cloud(dict(config,cloud_sky_cells=32))
    geometry=build_radial_cloud_launch_geometry(coarse,physics)
    packets=262144;chunk=4096;guard=config["max_interactions"]
    @jax.jit
    def batch(key):
        lk,tk,ak=random.split(key,3)
        source=SourcePackets(jnp.full(chunk,config["energy_kev"]),jnp.zeros(chunk),
            jnp.full(chunk,1.0/packets),jnp.zeros(chunk,dtype=jnp.int32),jnp.zeros(chunk,dtype=jnp.int32))
        launched=sample_source_launches(lk,source,geometry)
        u=random.uniform(ak,(chunk,guard,2))
        def score(cloud):
            history=transport_photon_batch(tk,launched.position_pc,launched.momentum_kev,
                                           cloud,physics,max_interactions=guard)
            ev=score_peeloff_events(launched,history,cloud,physics)
            distance=jnp.linalg.norm(history.interactions.position_pc,axis=-1)
            radius=jnp.sqrt(u[:,:,0])*config["camera_aperture_radius_pc"]/jnp.maximum(distance,1)/(jnp.pi/648000)
            x=ev.sky_x_arcsec+radius*jnp.cos(2*jnp.pi*u[:,:,1])
            y=ev.sky_y_arcsec+radius*jnp.sin(2*jnp.pi*u[:,:,1])
            keep=ev.valid&(jnp.abs(x)<config["image_half_width_arcsec"])&(jnp.abs(y)<config["image_half_width_arcsec"])
            return jnp.sum(jnp.where(keep,ev.weight_observer_fluence,0),axis=1),history.status
        a,sa=score(coarse);b,sb=score(fine)
        return a,b,sa,sb
    sums=np.zeros(2);difference=0.0;squares=0.0;bad=0;start=time.monotonic()
    for i in range(packets//chunk):
        a,b,sa,sb=[np.asarray(v) for v in batch(random.fold_in(random.PRNGKey(8659),i))]
        a=a.astype(float);b=b.astype(float);delta=b-a
        sums += [a.sum(),b.sum()];difference+=delta.sum();squares+=np.dot(delta,delta)
        bad += int(np.isin(sa,[0,4,5,6]).sum()+np.isin(sb,[0,4,5,6]).sum())
        if i==0 or i%16==0:print(f"geometry pairs: {(i+1)*chunk:,}/{packets:,}, {time.monotonic()-start:.1f}s",flush=True)
    se=np.sqrt(packets/(packets-1)*max(squares-difference**2/packets,0))
    bound=(abs(difference)+4*se)/sums[0]
    result=dict(passed=bool(bound<=config["geometry_response_fraction_allowance"] and bad==0),
        paired_histories=packets,coarse_cells=16,fine_cells=32,coarse_aperture_fluence=sums[0],
        fine_aperture_fluence=sums[1],paired_difference=difference,paired_se=float(se),
        four_se_relative_upper_bound=float(bound),fixed_fractional_allowance=config["geometry_response_fraction_allowance"],
        invalid_or_incomplete_histories=bad,elapsed_seconds=time.monotonic()-start,
        rare_path_bound_established=False,
        qualification="Zero sample variance is not a confidence bound on unsampled rare paths.",
        scope="transverse boundary discretization at fixed 30-pc radius; not a general cylinder-versus-frustum bound")
    (output/"geometry_refinement.json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
