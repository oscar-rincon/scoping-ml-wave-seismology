
"""
Created on Mon Oct 11 2025

author: Oscar Rincón-Cardeño
email: os.rinconc@gmail.com
github: https://github.com/orincon

This script analyzes publication data from Scopus related to numerical methods,
machine learning, and wave propagation between 2010 and 2024. It also integrates
Google Trends data to compare the popularity of Python frameworks (TensorFlow,
PyTorch, JAX). The results are visualized in a grid of plots and exported
as PDF and SVG figures.
"""

#%% 
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
 
# Set up paths for utility imports
current_dir = os.getcwd() #os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../utils')

# Change working directory
os.chdir(current_dir)
sys.path.insert(0, utilities_dir)

from plotting import *  # Importar utilidades de trazado personalizadas


 
# Load data 
def read_csv_data(path):
    return pd.read_csv(path, skiprows=6, delimiter=',')

df_ml = read_csv_data('data/ml_scopus_2025_10_11.csv')
df_nm = read_csv_data('data/nm_scopus_2025_10_11.csv')
df_nm_ml = read_csv_data('data/nm_ml_scopus_2025_10_11.csv')
df_waves = read_csv_data('data/wave_scopus_2025_10_11.csv')
df_total = read_csv_data('data/total_scopus_2025_10_11.csv')

# Extract and filter columns 
def extract_columns(df):
    return df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()

years_ml, works_ml = extract_columns(df_ml)
years_nm, works_nm = extract_columns(df_nm)
years_nm_ml, works_nm_ml = extract_columns(df_nm_ml)
years_waves, works_waves = extract_columns(df_waves)
years_total, works_total = extract_columns(df_total)

# Filter for 2010–2024
def filter_range(years, works, start=2010, end=2024):
    mask = (years >= start) & (years <= end)
    return years[mask], works[mask]

years_total, works_total = filter_range(years_total, works_total)
years_ml, works_ml = filter_range(years_ml, works_ml)
years_nm, works_nm = filter_range(years_nm, works_nm)
years_nm_ml, works_nm_ml = filter_range(years_nm_ml, works_nm_ml)
years_waves, works_waves = filter_range(years_waves, works_waves)

# ompute relative fractions 
rel = lambda w: w / works_total

data_relative_ml = {'YEAR': years_ml, 'WORKS': rel(works_ml)}
data_relative_nm = {'YEAR': years_nm, 'WORKS': rel(works_nm)}
data_relative_nm_ml = {'YEAR': years_nm_ml, 'WORKS': rel(works_nm_ml)}
data_relative_waves = {'YEAR': years_waves, 'WORKS': rel(works_waves)}

# Frameworks data 
df = pd.read_csv("data/google_trends_python_frameworks.csv", skiprows=1)

# Clean data
df.replace("<1", 0.0, inplace=True)
df.columns = [c.strip().replace(': (Worldwide)', '') for c in df.columns]
df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
for col in ['TensorFlow', 'PyTorch', 'JAX']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(subset=['Month'], inplace=True)

# Figure 
color_standard = '#000000'
color_ml = '#2255a080'
color_either = "#8e8e8e"
color_wave = '#8e8e8e'
colors_frameworks = [color_standard, color_ml, color_either]
bar_width = 0.25

fig = plt.figure(figsize=(6.7, 4.7), constrained_layout=True)
gs = GridSpec(2, 2, figure=fig)
fig.set_constrained_layout_pads(w_pad=0.15, h_pad=0.15)
  
ax1 = fig.add_subplot(gs[0, 0])

# x positions are the filtered years for each category
pos_std = years_nm - bar_width
pos_ml = years_ml
pos_either = years_nm_ml + bar_width

ax1.bar(pos_std, data_relative_nm['WORKS']*1e2, width=bar_width,
        color=color_standard, label='Standard numerical')
ax1.bar(pos_ml, data_relative_ml['WORKS']*1e2, width=bar_width,
        color=color_ml, label='Machine learning')
ax1.bar(pos_either, data_relative_nm_ml['WORKS']*1e2, width=bar_width,
        color=color_either, label='Either / Both')

ax1.set_xlabel("Year")
ax1.set_ylabel(r"Publications fraction ($\times 10^{-2}$)")
ax1.set_yticks(np.arange(0, max(data_relative_nm_ml['WORKS']*1e2)*1.2, 2))
ax1.set_yticklabels(np.arange(0, max(data_relative_nm_ml['WORKS']*1e2)*1.2, 2))
ax1.set_xticks(np.arange(2010, 2025, 2))
ax1.set_xticklabels(np.arange(2010, 2025, 2))
ax1.set_xlim(2009.5, 2024.5)
ax1.set_ylim(0, max(data_relative_nm_ml['WORKS']*1e2) * 1.2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.tick_params(axis='both', labelsize=8, direction='out', length=3, width=0.8, color='black')
ax1.legend(frameon=False, loc='upper left', facecolor='white')
ax1.grid(False)
 
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(years_waves, data_relative_waves['WORKS']*1e4, width=0.6, color=color_wave,
        linewidth=0.6, label='Wave propagation')
ax2.set_xlabel("Year")
ax2.set_ylabel(r"Publications fraction ($\times 10^{-4}$)")
ticks = np.linspace(0, max(data_relative_waves['WORKS']*1e4)*1.1, 5)
ax2.set_yticks(ticks)
ax2.set_yticklabels([f"{t:.2f}" for t in ticks])
ax2.set_xticks(np.arange(2010, 2025, 2))
ax2.set_xticklabels(np.arange(2010, 2025, 2))
ax2.set_xlim(2009.5, 2024.5)
ax2.set_ylim(0, max(data_relative_waves['WORKS'])*1e4 * 1.1)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color('black')
ax2.spines['left'].set_color('black')
ax2.tick_params(axis='both', labelsize=8, direction='out', length=3, width=0.8, color='black')
ax2.legend(frameon=False, loc='upper left')
ax2.grid(False)


# Average frameworks bar plot  
ax3 = fig.add_subplot(gs[1, 0])
means = df[['TensorFlow', 'PyTorch', 'JAX']].mean()

# Bar plot (normalized)
bars = ax3.bar(means.index, means.values, color=colors_frameworks)

# Add values on top of bars
for bar, value in zip(bars, means.values):
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2,  # x position: center of the bar
        height + 0.02,                     # y position: a bit above the bar
        f"{value:.1f}%",                    # label text
        ha='center', va='bottom', fontsize=8
    )

# Axis labels and styling
ax3.set_ylabel("Average interest (%)", fontsize=8)
ax3.set_xlabel("Python frameworks", fontsize=8)  
ax3.set_ylim(-0.00, 1.01*100)
ax3.set_yticks(np.arange(0, 1.1*100, 20))
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_color('black')
ax3.spines['left'].set_color('black')
ax3.tick_params(axis='both', labelsize=8, direction='out', length=3, width=0.8, color='black')
ax3.grid(False)
 
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(df['Month'], df['TensorFlow'], color=colors_frameworks[0], linewidth=2, label="TensorFlow")
ax4.plot(df['Month'], df['PyTorch'], color=colors_frameworks[1], linewidth=2, label="PyTorch")
ax4.plot(df['Month'], df['JAX'], color=colors_frameworks[2], linewidth=2, label="JAX")

years_line = np.arange(df['Month'].dt.year.min(), 2026, 2)
tick_positions = [pd.Timestamp(f"{y}-01-01") for y in years_line]
ax4.set_xticks(tick_positions)
ax4.set_xticklabels(years_line)
ax4.set_yticks(np.arange(0, 1.1*100, 20))
ax4.set_xlim(df['Month'].min(), pd.Timestamp("2025-03-01"))
ax4.set_ylim(-0.05*100, 1*100)
ax4.set_ylabel("Interest over time (%)", fontsize=8)
ax4.set_xlabel("Year", fontsize=8)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_color('black')
ax4.spines['left'].set_color('black')
ax4.tick_params(axis='both', labelsize=8, direction='out', length=3, width=0.8, color='black')
ax4.legend(frameon=False, fontsize=8)
ax4.grid(False)

plt.savefig("figs/publication_trends_ml_wave_frameworks.pdf", bbox_inches="tight", dpi=300)
plt.savefig("figs/publication_trends_ml_wave_frameworks.svg", bbox_inches="tight", dpi=300)
plt.show()
 
color_standard = '#000000'
color_ml = '#2255a080'
color_either = "#8e8e8e"
color_wave = '#8e8e8e'
colors_frameworks = [color_standard, color_ml, color_either]
bar_width = 0.25

# fig = plt.figure(figsize=(6.5, 4.4), constrained_layout=True)
# gs = GridSpec(2, 2, figure=fig)
# fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1)

# ax1 = fig.add_subplot(gs[0, 0])

# # x positions are the filtered years for each category
# pos_std = years_nm - bar_width
# pos_ml = years_ml
# pos_either = years_nm_ml + bar_width

# ax1.bar(pos_std, data_relative_nm['WORKS']*1e2, width=bar_width,
#         color=color_standard, label='Métodos numéricos estándar')
# ax1.bar(pos_ml, data_relative_ml['WORKS']*1e2, width=bar_width,
#         color=color_ml, label='Aprendizaje automático')
# ax1.bar(pos_either, data_relative_nm_ml['WORKS']*1e2, width=bar_width,
#         color=color_either, label='Ambos / Combinados')

# ax1.set_xlabel("Año")
# ax1.set_ylabel(r"Fracción de publicaciones ($\times 10^{-2}$)")
# ax1.set_yticks(np.arange(0, max(data_relative_nm_ml['WORKS']*1e2)*1.2, 2))
# ax1.set_yticklabels(np.arange(0, max(data_relative_nm_ml['WORKS']*1e2)*1.2, 2))
# ax1.set_xticks(np.arange(2010, 2025, 2))
# ax1.set_xticklabels(np.arange(2010, 2025, 2))
# ax1.set_xlim(2009.5, 2024.5)
# ax1.set_ylim(0, max(data_relative_nm_ml['WORKS']*1e2) * 1.2)
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['bottom'].set_color('black')
# ax1.spines['left'].set_color('black')
# ax1.tick_params(axis='both', labelsize=8, direction='out', length=3, width=0.8, color='black')
# ax1.legend(frameon=False, loc='upper left', facecolor='white')
# ax1.grid(False)

# ax2 = fig.add_subplot(gs[0, 1])
# ax2.bar(years_waves, data_relative_waves['WORKS']*1e4, width=0.6, color=color_wave,
#         linewidth=0.6, label='Propagación de ondas')
# ax2.set_xlabel("Año")
# ax2.set_ylabel(r"Fracción de publicaciones ($\times 10^{-4}$)")
# ticks = np.linspace(0, max(data_relative_waves['WORKS']*1e4)*1.1, 5)
# ax2.set_yticks(ticks)
# ax2.set_yticklabels([f"{t:.2f}" for t in ticks])
# ax2.set_xticks(np.arange(2010, 2025, 2))
# ax2.set_xticklabels(np.arange(2010, 2025, 2))
# ax2.set_xlim(2009.5, 2024.5)
# ax2.set_ylim(0, max(data_relative_waves['WORKS'])*1e4 * 1.1)
# ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['bottom'].set_color('black')
# ax2.spines['left'].set_color('black')
# ax2.tick_params(axis='both', labelsize=8, direction='out', length=3, width=0.8, color='black')
# ax2.legend(frameon=False, loc='upper left')
# ax2.grid(False)


# # === Bottom-left: Average frameworks bar plot ===
# ax3 = fig.add_subplot(gs[1, 0])
# means = df[['TensorFlow', 'PyTorch', 'JAX']].mean()

# # Bar plot (normalized)
# bars = ax3.bar(means.index, means.values, color=colors_frameworks)

# # Add values on top of bars
# for bar, value in zip(bars, means.values):
#     height = bar.get_height()
#     ax3.text(
#         bar.get_x() + bar.get_width() / 2,  # x position: center of the bar
#         height + 0.02,                     # y position: a bit above the bar
#         f"{value:.1f}%",                    # label text
#         ha='center', va='bottom', fontsize=8
#     )

# # Axis labels and styling
# ax3.set_ylabel("Interés promedio (%)", fontsize=8)
# ax3.set_xlabel(r"$Framework$ de Python", fontsize=8)    
# ax3.set_ylim(-0.00, 1.01*100)
# ax3.set_yticks(np.arange(0, 1.1*100, 20))
# ax3.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['bottom'].set_color('black')
# ax3.spines['left'].set_color('black')
# ax3.tick_params(axis='both', labelsize=8, direction='out', length=3, width=0.8, color='black')
# ax3.grid(False)

# # === Bottom-right: Frameworks time series ===
# ax4 = fig.add_subplot(gs[1, 1])
# ax4.plot(df['Month'], df['TensorFlow'], color=colors_frameworks[0], linewidth=2, label="TensorFlow")
# ax4.plot(df['Month'], df['PyTorch'], color=colors_frameworks[1], linewidth=2, label="PyTorch")
# ax4.plot(df['Month'], df['JAX'], color=colors_frameworks[2], linewidth=2, label="JAX")

# years_line = np.arange(df['Month'].dt.year.min(), 2026, 2)
# tick_positions = [pd.Timestamp(f"{y}-01-01") for y in years_line]
# ax4.set_xticks(tick_positions)
# ax4.set_xticklabels(years_line)
# ax4.set_yticks(np.arange(0, 1.1*100, 20))
# ax4.set_xlim(df['Month'].min(), pd.Timestamp("2025-03-01"))
# ax4.set_ylim(-0.05*100, 1*100)
# ax4.set_ylabel("Interés a lo largo del tiempo (%)", fontsize=8)
# ax4.set_xlabel("Año", fontsize=8)
# ax4.spines['top'].set_visible(False)
# ax4.spines['right'].set_visible(False)
# ax4.spines['bottom'].set_color('black')
# ax4.spines['left'].set_color('black')
# ax4.tick_params(axis='both', labelsize=8, direction='out', length=3, width=0.8, color='black')
# ax4.legend(frameon=False, fontsize=8)
# ax4.grid(False)

# plt.savefig("figs/publication_trends_ml_wave_frameworks_esp.pdf", bbox_inches="tight", dpi=300)
# plt.savefig("figs/publication_trends_ml_wave_frameworks_esp.svg", bbox_inches="tight", dpi=300)
# plt.show()