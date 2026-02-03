

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
 

# Colors
color_main = '#8e8e8e'

#  Invented counts 
wave_eq_labels = ["Acoustic", "Elastic", "Helmholtz"]
wave_eq_counts = [30, 9, 5]

dim_labels = ["1D", "2D", "3D"]
dim_counts = [4, 32, 8]

data_labels = ["Synthetic", "Experimental"]
data_counts = [36, 8]

#  Figure
fig = plt.figure(figsize=(6.5, 1.6))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1], figure=fig)

fig.subplots_adjust(
    left=0.07,
    right=0.98,
    bottom=0.30,
    top=0.95,
    wspace=0.35
)

# Panel 1: Wave equation 
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(wave_eq_labels, wave_eq_counts, color=color_main)

ax1.set_title("Formulation", fontsize=8)
ax1.set_ylabel("Number of publications", fontsize=8)
ax1.tick_params(axis='x', rotation=25, labelsize=8)
ax1.tick_params(axis='y', labelsize=8)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

#  Panel 2: Dimensionality 
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(dim_labels, dim_counts, color=color_main)

ax2.set_title("Dimensionality", fontsize=8)
#ax2.set_xlabel("Dimensions", fontsize=8)
ax2.tick_params(axis='both', labelsize=8)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Panel 3: Data type 
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(data_labels, data_counts, color=color_main)

ax3.set_title("Data type", fontsize=8)
ax3.tick_params(axis='x', rotation=15, labelsize=8)
ax3.tick_params(axis='y', labelsize=8)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

#  Save
plt.savefig(
    "figs/formulation_dimensionality_data_type.svg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()