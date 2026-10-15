"""Independent finite-volume first-order integral for the matched Draine law.

Uses NumPy geometry and the analytic deflection density, not production ray
traversal, scoring, CDF inversion or launch sampling. This is a transport check
with shared material coefficients, not an independent grain-physics validation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ARCSEC = np.pi / 648000
PC_CM = 3.0856775814913673e18


def first_order_image(config, scale, sigma_sca, sigma_abs, *, radial_step, depth_order, pixel_order):
    distance = config["source_distance_kpc"] * 1000
    center = config["cloud_center_kpc"] * 1000
    half = config["cloud_thickness_pc"] / 2
    r1, r2 = center-half, center+half
    density = config["nh_cm2"] / (2*half*PC_CM)
    ext = sigma_sca+sigma_abs
    maximum = np.sqrt(2)*config["image_half_width_arcsec"]+10
    theta_grid = np.arange(0, maximum+radial_step, radial_step)
    theta = theta_grid * ARCSEC
    nodes, weights = np.polynomial.legendre.leggauss(depth_order)
    profile = np.zeros_like(theta)
    for node, weight in zip(nodes, weights):
        r = center+half*node
        length = np.sqrt(distance**2+r**2-2*distance*r*np.cos(theta))
        alpha = np.arctan2(distance*np.sin(theta), distance*np.cos(theta)-r)
        correction = np.ones_like(alpha)
        np.divide(alpha, np.sin(alpha), out=correction, where=alpha != 0)
        phase = correction/(np.pi*scale**2)/(1+(alpha/scale)**2)**2
        phase /= np.pi**2/(np.pi**2+scale**2)
        projection = distance*(r*np.cos(theta)-distance)/length
        incoming_pc = length+projection+np.sqrt(np.maximum(projection**2-distance**2+r2**2,0))
        outgoing_pc = r-r1
        transmission = np.exp(-ext*density*(incoming_pc+outgoing_pc)*PC_CM)
        profile += weight*half*density*PC_CM*(distance/length)**2*sigma_sca*phase*transmission*ARCSEC**2
    # Circular aperture convolution. Radius^2 and azimuth are uniform.
    nodes, weights = np.polynomial.legendre.leggauss(8)
    blur = config["camera_aperture_radius_pc"]/center/ARCSEC
    blurred = np.zeros_like(theta_grid)
    for node, weight in zip(nodes,weights):
        radius = blur*np.sqrt((node+1)/2)
        for phi in (np.arange(64)+0.5)*(2*np.pi/64):
            shifted = np.sqrt(theta_grid**2+radius**2+2*theta_grid*radius*np.cos(phi))
            blurred += weight/2/64*np.interp(shifted,theta_grid,profile)
    pixel=config["pixel_arcsec"]; half_width=config["image_half_width_arcsec"]
    edges=np.linspace(-half_width,half_width,round(2*half_width/pixel)+1)
    centers=(edges[1:]+edges[:-1])/2
    x,y=np.meshgrid(centers,centers)
    image=np.zeros_like(x)
    nodes,weights=np.polynomial.legendre.leggauss(pixel_order)
    for a,wa in zip(nodes,weights):
        for b,wb in zip(nodes,weights):
            tx=np.tan((x+a*pixel/2)*ARCSEC);ty=np.tan((y+b*pixel/2)*ARCSEC)
            radius=np.arctan(np.hypot(tx,ty))/ARCSEC
            jac=(1+tx**2)*(1+ty**2)/(1+tx**2+ty**2)**1.5
            image += wa*wb*pixel**2/4*np.interp(radius,theta_grid,blurred)*jac
    return image*config["source_fluence_ph_cm2"]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("output",type=Path);args=p.parse_args()
    root=Path(__file__).parent
    result=json.loads((args.output/"comparison_results.json").read_text());config=result["config"]
    meta=json.loads((root/"materials/reflex_scattering_v1.json").read_text())
    with np.load(root/"materials/reflex_scattering_v1.npz") as d:
        ei=int(np.argmin(abs(d["energy_kev"]-config["energy_kev"])));sigma=float(d["scattering_cross_section_cm2_per_h"][ei])
    with np.load(root/"materials/reflex_absorption_v1.npz") as d:absorption=float(d["absorption_cross_section_cm2_per_h"][ei])
    scale=meta["angular_scale_rad"][ei]
    coarse=first_order_image(config,scale,sigma,absorption,radial_step=.05,depth_order=12,pixel_order=4)
    fine=first_order_image(config,scale,sigma,absorption,radial_step=.025,depth_order=24,pixel_order=8)
    numerical_error=float(np.abs(fine-coarse).sum())
    measured=result["order_fluences"][0];expected=float(fine.sum())
    se=float(np.sqrt(result["order_covariance"][0][0]));difference=measured-expected
    report=dict(passed=abs(difference)<=3*se+numerical_error+0.001*expected,
                dsh_first_order_fluence=measured,reference_first_order_fluence=expected,
                dsh_mc_se=se,dsh_over_reference=measured/expected,z=difference/se,
                numerical_l1_error=numerical_error,numerical_relative_l1_error=numerical_error/expected,
                fixed_relative_regression_allowance=.001,shared_material_coefficients=True,
                scope="first-order, absorbed, finite-volume, aperture-integrated; not individual higher-order validation")
    (args.output/"first_order_check.json").write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/"first_order_reference.npz",reference=fine,coarse_reference=coarse)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
