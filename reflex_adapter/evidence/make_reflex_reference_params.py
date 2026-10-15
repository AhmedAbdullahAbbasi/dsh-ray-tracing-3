"""Write params.par for the RefleX vs DSH comparison test. Edit the constants and rerun."""
import numpy as np

PC_CM = 3.0856775814913673e18
ARCSEC = np.pi / (180 * 3600)

# ---- source, cloud, photons ----
ENERGY_EV = 4900.0
NPHOTS = 100_000_000
THREADS = 16                      # = --cpus-per-task in run_reflex.sh
SEED = 2026
SOURCE_TO_CAMERA_PC = 10000.0     # DSH SOURCE_DISTANCE_KPC = 10
CLOUD_FROM_SOURCE_PC = 5000.0     # DSH DISC_CENTER_Z_KPC = 5 (from the observer)
CLOUD_THICKNESS_PC = 30.0         # three 10 pc DSH distance slices
CLOUD_RADIUS_PC = 30.0
NH_CM2 = 1.0e22
CONE_HALF_ANGLE_DEG = 1.0         # the EMGEOM value IS the half-angle; < 1 deg gives a pencil beam

# ---- cameras ----
APERTURE_PC = 0.1                 # radius
N_SIDE = 100                      # N_SIDE x N_SIDE cameras, spacing 2 x aperture
PIXEL_ARCSEC = 5.0
N_PIX = 60                        # even: RefleX puts the source on the corner of the 4 middle pixels

DENSITY_CM3 = NH_CM2 / (CLOUD_THICKNESS_PC * PC_CM)
FOV_DEG = N_PIX * PIXEL_ARCSEC / 3600.0
spacing = 2 * APERTURE_PC
coords = (np.arange(N_SIDE) - (N_SIDE - 1) / 2) * spacing
grid = [(x, y) for x in coords for y in coords]
grid_half = coords[-1] + APERTURE_PC
cloud_to_camera = SOURCE_TO_CAMERA_PC - CLOUD_FROM_SOURCE_PC
lit_radius = CLOUD_FROM_SOURCE_PC * np.tan(np.radians(CONE_HALF_ANGLE_DEG))
needed_radius = (grid_half * np.sqrt(2) * CLOUD_FROM_SOURCE_PC / SOURCE_TO_CAMERA_PC
                 + cloud_to_camera * np.tan(N_PIX / 2 * PIXEL_ARCSEC * np.sqrt(2) * ARCSEC))
collect_frac = len(grid) * APERTURE_PC**2 / (
    SOURCE_TO_CAMERA_PC * np.tan(np.radians(CONE_HALF_ANGLE_DEG))) ** 2
offaxis_deg = np.degrees(grid_half * np.sqrt(2) / SOURCE_TO_CAMERA_PC)
assert CLOUD_RADIUS_PC >= needed_radius, "disc too small for the camera fields"
assert lit_radius >= CLOUD_RADIUS_PC, "cone does not light the whole disc"
assert offaxis_deg < CONE_HALF_ANGLE_DEG, "outer cameras outside the source cone"
assert CONE_HALF_ANGLE_DEG >= 1.0, "RefleX emits a pencil beam below 1 deg"


def camera_block(fname, selection):
    x0, y0 = grid[0]
    out = [f"IMAGE NEW fits_images/{fname} {x0:.4f} {y0:.4f} {SOURCE_TO_CAMERA_PC:.1f} "
           f"{APERTURE_PC} {FOV_DEG:.7f} {N_PIX}"]
    if selection:
        out.append(selection)
    out += [f"IMAGE REPEAT {x:.4f} {y:.4f} {SOURCE_TO_CAMERA_PC:.1f}" for x, y in grid[1:]]
    return "\n".join(out)


blur_halo = APERTURE_PC / cloud_to_camera / ARCSEC
blur_direct = APERTURE_PC / SOURCE_TO_CAMERA_PC / ARCSEC
text = f"""# ======================================================================
# RefleX vs DSH comparison test: dust-scattering halo of a uniform slab at
# {ENERGY_EV / 1000:.1f} keV, set up to match what the DSH code (run_simulation.py,
# V1 mono-energetic mode) can simulate. Written by make_params.py.
#
# Run (from this folder):
#   sbatch run_reflex.sh
#
# ---- RefleX behaviour this file relies on (tested 2026-10-14) ----
#   * EMGEOM POINT ... a: a is the cone HALF-angle (photons fill
#     theta <= a, uniformly in solid angle).
#   * a < 1 deg (0.5, 0.6, 0.8 tried) emits every photon exactly on the
#     axis: a pencil beam, not a cone. 1 deg is the narrowest usable cone.
#   * Each IMAGE selection applies only to the image it follows.
#   * Test run, 2e6 photons on 8 threads: 655 s; direct beam 6218 counts
#     (exp(-tau_ext) = 0.947), halo/direct inside the field 0.019.
#     So 1e8 photons take about 9 h on 8 threads (about half that on 16) and give ~6,000 halo
#     counts; for more, run several copies with different RANDOM seeds
#     and add the images.
#
# ---- Geometry (RefleX frame: source at origin, everything along +Z) ----
#   Source   point at (0,0,0), cone half-angle {CONE_HALF_ANGLE_DEG} deg along +Z.
#   Cloud    face-on disc at z = {CLOUD_FROM_SOURCE_PC:.0f} pc, {CLOUD_THICKNESS_PC:.0f} pc thick, radius {CLOUD_RADIUS_PC:.0f} pc,
#            n_H = {DENSITY_CM3:.4f} cm^-3, so N_H = {NH_CM2:.2e} cm^-2.
#            The cone lights out to {lit_radius:.1f} pc, so the whole disc is lit.
#            The camera fields need cloud out to {needed_radius:.1f} pc from the axis, so
#            every camera sees an effectively infinite uniform slab, like the
#            DSH observer.
#   Cameras  {N_SIDE} x {N_SIDE} = {len(grid)} pinholes at z = {SOURCE_TO_CAMERA_PC:.0f} pc, aperture radius {APERTURE_PC} pc,
#            spacing {spacing:.2f} pc (touching, no overlap, so no photon is counted
#            twice), grid +/-{grid_half:.1f} pc. Each points at the source. The outermost
#            camera sees the source {offaxis_deg:.3f} deg off axis, inside the cone.
#            Aperture blur (radius): {blur_halo:.1f}" for halo light, {blur_direct:.1f}" for the direct beam.
#            Blur the DSH image with the same kernel (uniform disc, radius
#            {blur_halo:.1f}") before a pixel-by-pixel comparison.
#   Image    {N_PIX} x {N_PIX} pixels of {PIXEL_ARCSEC:.0f}" (FOV {FOV_DEG:.7f} deg). RefleX puts the source at
#            pixel edge N/2, i.e. on the corner of the four middle pixels (checked:
#            also true for odd N), so N is even and the DSH grid uses the same edges.
#   Expected direct-beam counts ~ NPHOTS x {collect_frac:.2e} x exp(-tau_ext).
#
# ---- Matching DSH settings (sim_config.py) ----
#   SOURCE_DISTANCE_KPC = 10.0 ; CLOUD_FITS_PATH = None
#   GRID_X_ARCSEC = GRID_Y_ARCSEC = (-147.5, 150.0, 5.0)   (60 x 60, 5" pixels, edges +/-150")
#   GRID_Z_KPC = (4.5, 5.5, 0.01)                         (10 pc slices)
#   DISC_CENTER_Z_KPC = 5.0 ; DISC_THICKNESS_KPC = 0.03   (slices 4.99/5.00/5.01)
#   DISC_RADIUS_ARCSEC = 1000.0                           (fills the sky grid)
#   DISC_N_H_CM3 = {DENSITY_CM3:.4f}
#   ENERGY_EDGES_KEV = None ; PEAK_BAND_FLUXES = (0.0, <flux>, 0.0)   (4.9 keV only)
#   The arrival window must hold every delay: the field corner (216") arrives
#   about 6.4 days late, so e.g. a 1-day flare with ARRIVAL_DAYS >= 10.
#
# ---- Physics: only what DSH V1 simulates ----
#   Dust scattering and photoabsorption in gas and dust; Compton, Rayleigh and
#   fluorescence OFF. Wilms abundances, 20% H2 (as in TBabs), Fe depletion
#   0.691 (the maximum RefleX allows for wilm).
#   Intended difference: RefleX dust is Draine 2003 / WD01 (Mie + ADT); DSH
#   dust is MRN with Rayleigh-Gans/Drude. Median scattering angle at 4.9 keV:
#   RefleX 74.9" (measured), DSH 84.7" (table).
#
# ---- Outputs (raw photon counts; no LUMINOSITY is set) ----
#   fits_images/allphotons_e4p9.fits   halo + direct beam (direct beam in the 4 middle pixels)
#   fits_images/dust_e4p9.fits         photons with >= 1 dust scattering (the halo)
#   Compare the halo/direct ratio and normalised radial profiles.
# ======================================================================

# --- Control ---
VERBOSE NONE
PERCENT ON
RANDOM {SEED}
NPHOTS {NPHOTS}
THREAD {THREADS}

# --- Physics ---
PHYSICS COMPTON OFF
PHYSICS RAYLEIGH OFF
PHYSICS FLUOR OFF
PHYSICS PHOTO ON
PHYSICS DUST_PHOTO ON
PHYSICS DUST_SCAT ON
TEMPERATURE 0

# --- Source ---
EMSPEC MONO {ENERGY_EV:.0f}
EGEN 4000 6000
ECUT 100
LENGTH Parsec
EMGEOM POINT 0.0 0.0 0.0 0 0 {CONE_HALF_ANGLE_DEG}

# --- Composition ---
MATTER wilm
METALLICITY 1
H2FRACTION 0.2
DUST 0.691

# --- World and cloud ---
LENGTH Parsec
OBJECT WORLD 12000
LENGTH Centimeter
DENSITY {DENSITY_CM3:.6f}
LENGTH Parsec
OBJECT DISC DUST_CLOUD 0.0 0.0 {CLOUD_FROM_SOURCE_PC:.1f} {CLOUD_RADIUS_PC:.1f} {CLOUD_THICKNESS_PC:.1f}

# --- Cameras: all photons ---
{camera_block("allphotons_e4p9.fits", None)}

# --- Cameras: dust-scattered photons only ---
{camera_block("dust_e4p9.fits", "IMAGE DUSTSCAT > 0")}
"""
with open("params.par", "w") as f:
    f.write(text)
print(f"params.par written: {len(grid)} cameras per image, collecting fraction {collect_frac:.3e}, "
      f"cloud needed to {needed_radius:.1f} pc, lit to {lit_radius:.1f} pc")
