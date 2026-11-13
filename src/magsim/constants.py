"""Physical constants and copper data (SI units)."""

# Vacuum permeability [H/m]. CODATA 2018 recommended value (Tiesinga et al., Rev. Mod. Phys. 93, 025010, 2021).
MU0 = 1.25663706212e-6

# Copper resistivity at 20 degC [ohm m]: International Annealed Copper Standard (IACS, 100 % conductivity),
# IEC 60028:1925.
RHO_CU_20 = 1.7241e-8

# Temperature coefficient of copper resistivity at 20 degC [1/K]: IEC 60028 / IACS annealed copper.
ALPHA_CU = 3.93e-3

# Copper mass density [kg/m^3]: CRC Handbook of Chemistry and Physics (97th ed.), 8.96 g/cm^3 at 20 degC.
DENSITY_CU = 8960.0

# Copper specific heat capacity at 25 degC [J/(kg K)]: CRC Handbook (97th ed.), 24.44 J/(mol K) / 63.546 g/mol.
CP_CU = 385.0
