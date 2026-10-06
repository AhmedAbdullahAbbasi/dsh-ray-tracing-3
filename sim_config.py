"""Parameters for run_simulation.py -- edit these, then submit submit_simulation.sh."""

# --- geometry ---
SOURCE_DISTANCE_KPC = 11.5
CLOUD_FITS_PATH = "clouds/4u1630_NH_111111111100001_DSH_11p5kpc.fits"  # e.g. "/path/to/delta_nh_cube.fits"; None uses the disc below

# --- uniform-density disc cloud (used when CLOUD_FITS_PATH is None) ---
GRID_X_ARCSEC = (-150.0, 155.0, 5.0)  # (start, stop, step) of sky-x cell centers
GRID_Y_ARCSEC = (-150.0, 155.0, 5.0)  # (start, stop, step) of sky-y cell centers
GRID_Z_KPC = (1.0, 9.0, 0.05)  # (start, stop, step) of distance cell centers
DISC_CENTER_X_ARCSEC = 0.0
DISC_CENTER_Y_ARCSEC = 0.0
DISC_CENTER_Z_KPC = 5.0  # distance from observer
DISC_RADIUS_ARCSEC = 80.0  # in-plane radius
DISC_THICKNESS_KPC = 0.3  # thickness along the line of sight
DISC_N_H_CM3 = 10.0  # hydrogen density inside the disc [cm^-3]

# --- source model ---
SOURCE_MODEL = "flux-file"  # "constant-flare", "exponential-decay", "custom" or "flux-file"
PEAK_BAND_FLUXES = (0.0, 1.2e-2, 0.0)  # ph cm^-2 s^-1 at 3.3/4.9/6.9 keV
BASELINE_BAND_FLUXES = (0.0, 0.0, 0.0)  # exponential-decay only
DECAY_TIME_DAYS = 25.0  # exponential-decay only
DECAY_START_DAYS = 0.0  # exponential-decay only
DECAY_DURATION_DAYS = 120.0  # exponential-decay only
SOURCE_TIME_BIN_DAYS = 0.25  # exponential-decay only

# --- light curve from a text file (SOURCE_MODEL = "flux-file" only) ---
FLUX_FILE = "fluxes/1630_allflux.txt"  # columns: MJD, then flux at 3.3/4.9/6.9 keV
FLUX_SCALE = 1.0  # multiply file values by this to get ph cm^-2 s^-1

# --- energy bands (SOURCE_MODEL = "flux-file" only) ---
# None: photons are emitted at exactly 3.3/4.9/6.9 keV, from flux columns 1-3.
# Edges: photon energies are drawn from a power law inside each band, using the
# 2-10 keV dust tables. N + 1 edges need N flux columns; all edges within 2-10 keV.
ENERGY_EDGES_KEV = (2.25, 3.15)  # E2
FLUX_COLUMNS = (2,)  # FLUX_FILE flux column for each band (1 = first after MJD)
PHOTON_INDEX = 2.0  # dN/dE ~ E^-PHOTON_INDEX inside each band

# --- custom light curve (SOURCE_MODEL = "custom" only) ---
# Flux is constant inside each time bin: N + 1 edges give N rows of fluxes.
CUSTOM_TIME_EDGES_DAYS = (0.0, 1.0, 2.0, 3.0)
CUSTOM_BAND_FLUXES = (  # ph cm^-2 s^-1 at 3.3/4.9/6.9 keV, one row per time bin
    (0.0, 1.2e-2, 0.0),
    (0.0, 6.0e-3, 0.0),
    (0.0, 3.0e-3, 0.0),
)

# --- Monte Carlo run ---
PACKETS = 1_000_000_000
CHUNK_SIZE = 10_000  # GPU memory per packet grows with the cloud's distance slices (~1 MB at 1500)
MAX_INTERACTIONS = 4
SEED = 2026
DEVICE = "gpu"  # "cpu" or "gpu"; submit_simulation.sh requests a GPU to match

# --- observer binning ---
ARRIVAL_DAYS = 200.0  # must be longer than the source light curve
TIME_BIN_DAYS = 1.0

# --- progress log ---
PROGRESS_LOG = "logs/progress.log"  # overwritten by each run
PROGRESS_EVERY_CHUNKS = 5  # write a progress line every this many chunks

# --- output image ---
OUTPUT_PNG = "outputs/dsh_image.png"  # saved as dsh_image_log.png, dsh_image_linear.png
OUTPUT_COUNT_PNG = "outputs/dsh_count_image.png"  # scored events per pixel, same suffixes
OUTPUT_COUNT_NPY = "outputs/dsh_count_image.npy"  # the count image as a (y, x) matrix
# None: fluence summed over all arrival times. (start, stop) in days since the
# first source time (first FLUX_FILE row): mean intensity I_nu over that epoch.
IMAGE_EPOCH_DAYS = None  # e.g. (100.0, 101.0)
IMAGE_ENERGY_KEV = 4.9  # nearest energy band is plotted; None sums all bands
IMAGE_FOV_ARCSEC = None  # e.g. 300.0 crops to a centered field of view; None keeps all
IMAGE_SCALES = ("log", "linear")  # one set of images is saved per colour scale
IMAGE_DPI = 200
