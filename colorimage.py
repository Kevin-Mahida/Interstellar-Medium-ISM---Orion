import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import make_lupton_rgb

# ----------------------------
# 1. Load the HLSP Composite
# ----------------------------
hlsp_file = "color_hlsp_orion_hst_acs_strip0l_f850lp_f658n_f435w_v1_drz_sci.fits"

try:
    with fits.open(hlsp_file) as hdu:
        # Grab the data array
        image_data = hdu[0].data if hdu[0].data is not None else hdu[1].data
        header = hdu[0].header if hdu[0].data is not None else hdu[1].header
except FileNotFoundError:
    print(f"Error: Could not find {hlsp_file}. Please ensure the path is correct.")
    exit()

# ----------------------------
# 2. Verify and Extract Channels
# ----------------------------
# The image data should be a 3D array
if image_data.ndim == 3 and image_data.shape[0] == 3:
    print("Successfully loaded 3D composite image.")
    
    # Extract the individual bands based on standard RGB ordering
    # For this specific file: R=F850LP, G=F658N, B=F435W
    red_band = image_data[0]   # F850LP
    green_band = image_data[1] # F658N (H-alpha)
    blue_band = image_data[2]  # F435W
    
    # ----------------------------
    # 3. Save as Single-Filter FITS
    # ----------------------------
    fits.writeto("orion_single_f850lp.fits", red_band, header, overwrite=True)
    fits.writeto("orion_single_f658n.fits", green_band, header, overwrite=True)
    fits.writeto("orion_single_f435w.fits", blue_band, header, overwrite=True)
    
    print("Successfully extracted and saved single-filter FITS files:")
    print("- orion_single_f850lp.fits")
    print("- orion_single_f658n.fits")
    print("- orion_single_f435w.fits")
    
    # ----------------------------
    # 4. Generate a Visual RGB Composite
    # ----------------------------
    # Normalize the data for visualization (preventing negative values)
    r = np.maximum(red_band, 0)
    g = np.maximum(green_band, 0)
    b = np.maximum(blue_band, 0)
    
    # Create the RGB image. You may need to tweak 'stretch' and 'Q' for your specific data
    rgb_image = make_lupton_rgb(r, g, b, stretch=0.5, Q=10)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(rgb_image, origin='lower')
    plt.title("Orion Nebula - Extracted RGB Composite")
    plt.axis('off')
    plt.savefig("orion_extracted_rgb.png")
    plt.show()

else:
    print(f"Error: Expected a 3D array with 3 channels, but got shape {image_data.shape}.")