import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. Load the Photometry Data
# ---------------------------------------------------------
# This must be the CSV generated from F658N image
try:
    df = pd.read_csv("photometry_resultsf658n.csv")
except FileNotFoundError:
    print("Error: photometry_results.csv not found. Run photometry script on the F658N file first.")
    exit()

if 'flux_erg_s_cm2_A' not in df.columns:
    print("Error: Calibrated flux column missing. Ensure PHOTFLAM was applied.")
    exit()

# ---------------------------------------------------------
# 2. Define Physical Constants for the Orion Nebula
# ---------------------------------------------------------
# Distance to Orion is roughly 400 parsecs
distance_pc = 400 

# Conversion factors
cm_per_pc = 3.086e18
distance_cm = distance_pc * cm_per_pc

# ---------------------------------------------------------
# 3. Calculate Luminosity
# ---------------------------------------------------------
# Luminosity = 4 * pi * distance^2 * Flux
# Resulting units: ergs/s
df['Luminosity_Halpha_erg_s'] = 4 * np.pi * (distance_cm**2) * df['flux_erg_s_cm2_A']

# ---------------------------------------------------------
# 4. Calculate Star Formation Rate (SFR)
# ---------------------------------------------------------
# Using the Kennicutt (1998) relation for H-alpha
# SFR (Solar Masses / year) = 7.9e-42 * L_Halpha
df['SFR_Msun_yr'] = 7.9e-42 * df['Luminosity_Halpha_erg_s']

# ---------------------------------------------------------
# 5. Save and Display Results
# ---------------------------------------------------------
df.to_csv("orion_physics_results.csv", index=False)

print("Physics calculations complete!")
print(f"Total sources analyzed: {len(df)}")
print(f"Total Nebula SFR: {df['SFR_Msun_yr'].sum():.6e} Solar Masses / Year")
print("Detailed results saved to orion_physics_results.csv")