import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm, AsinhStretch, MinMaxInterval
from scipy.ndimage import median_filter, center_of_mass
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.optimize import curve_fit

# ----------------------------
# 1. Load & Basic Calibration
# ----------------------------
fits_file = "hlsp_orion_hst_acs_strip0l_f775w_v1_drz.fits" 

try:
    with fits.open(fits_file) as hdu:
        # Get the data (assuming standard HST/MAST format)
        image_data = hdu[0].data if hdu[0].data is not None else hdu[1].data
        header = hdu[0].header if hdu[0].data is not None else hdu[1].header
        
        # Try to get PHOTFLAM directly from the header
        PHOTFLAM = header.get('PHOTFLAM', 1.5074510E-19)
except FileNotFoundError:
    print(f"Error: Could not find {fits_file}. Please ensure the path is correct.")
    exit()

# Ensure data is 2D
if image_data.ndim > 2:
    print("Warning: Data is 3D. Extracting the first channel for processing.")
    image_data = image_data[0]

# 1.1 Subtract background (using median)
background = np.median(image_data)
processed_data = image_data - background

# 1.2 Noise Removal
processed_data = median_filter(processed_data, size=3)

# Save processed image (Fixed visualization to handle negative values safely)
plt.figure(figsize=(10, 8))
norm = simple_norm(processed_data, 'asinh', min_cut=0) # 'asinh' for diffuse nebulae
plt.imshow(processed_data, cmap='gray', norm=norm, origin='lower')
plt.colorbar(label="Pixel Value")
plt.title("Processed image Filter:555w(Background Subtracted)")
plt.savefig("imagef775w.png")
plt.close()

# -------------------------
# 2. Center of Mass (CoM)
# -------------------------
# Note: For a diffuse nebula like Orion, this points to the brightest weighted region.
nebula_center_com = center_of_mass(np.maximum(processed_data, 0)) # Prevent negative mass
print(f"Nebula Center (Center of Mass): {nebula_center_com}")

# ---------------------------------------------
# 3. 2D Gaussian Fit 
# ---------------------------------------------
def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y):
    x, y = coords
    return (A * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2)))).ravel()

def fit_gaussian_roi(image, center, roi_size=500):
    if np.isnan(center).any():
        return center, None
        
    y_center, x_center = map(int, center)
    ysize, xsize = image.shape
    y_min, y_max = max(0, y_center - roi_size), min(ysize, y_center + roi_size)
    x_min, x_max = max(0, x_center - roi_size), min(xsize, x_center + roi_size)

    sub_image = image[y_min:y_max, x_min:x_max]
    sub_ysize, sub_xsize = sub_image.shape
    
    x_grid, y_grid = np.meshgrid(np.arange(sub_xsize), np.arange(sub_ysize))
    
    p0 = [sub_image.max(), sub_xsize / 2.0, sub_ysize / 2.0, 10.0, 10.0]

    try:
        popt, _ = curve_fit(gaussian_2d, (x_grid.ravel(), y_grid.ravel()), sub_image.ravel(), p0=p0, maxfev=20000)
        A_fit, x0_sub, y0_sub, sx_fit, sy_fit = popt
        return (y0_sub + y_min, x0_sub + x_min), popt
    except RuntimeError:
        print("Gaussian fitting failed - returning CoM as fallback.")
        return center, None

nebula_center_gauss, popt = fit_gaussian_roi(processed_data, nebula_center_com, roi_size=500)
print(f"Nebula Center (2D Gaussian Fit): {nebula_center_gauss}")

# -------------------------
# 4. Source Detection & Photometry
# -------------------------
# Use a background-subtracted image with negative values clipped for source detection
detect_data = np.maximum(processed_data, 0)
std_val = np.std(detect_data)

# DAOStarFinder on a nebula can also detect dense gas clumps as "stars"
star_finder = DAOStarFinder(fwhm=3.0, threshold=5.0 * std_val)
sources = star_finder(detect_data)

if sources is not None:
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=5)

    plt.figure(figsize=(10, 8))
    plt.imshow(processed_data, cmap='gray', norm=norm, origin='lower')
    apertures.plot(color='red', lw=1.5, alpha=0.5)
    plt.title(f"Detected Sources ({len(sources)} found)")
    plt.savefig("detected_sourcesf775w.png")
    plt.close()

    phot_table = aperture_photometry(processed_data, apertures)
    phot_table['flux_erg_s_cm2_A'] = phot_table['aperture_sum'] * PHOTFLAM

    phot_table.to_pandas().to_csv("photometry_resultsf775w.csv", index=False)
    print(f"Photometry results for {len(sources)} sources saved to photometry_results.csv")
else:
    print("No sources found by DAOStarFinder.")