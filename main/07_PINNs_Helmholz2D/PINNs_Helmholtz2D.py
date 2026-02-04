

# imports
import os
import sys
import time
import random
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RegularGridInterpolator
from matplotlib.ticker import FormatStrFormatter

# Set up path for utility imports
current_dir = os.getcwd()
utilities_dir = os.path.join(current_dir, "../../utils")

# Ensure the utilities directory is in the import path
if utilities_dir not in sys.path:
    sys.path.insert(0, utilities_dir)

import plotting
importlib.reload(plotting)  # Recarga útil en notebooks para reflejar cambios locales

# Return to the original working directory
os.chdir(current_dir)

# Torch device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Fix reproducibility
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Exact solution and RHS
def f_exact(x, y):
    return np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y)

def rhs(x, y):
    return -2 * (4 * np.pi)**2 * np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y)

# Sine activation + Hard-constrained PINN
class Sine(nn.Module):
    def forward(self, input):
        return torch.sin(input)

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                layer_list.append(Sine())
        self.model = nn.Sequential(*layer_list)

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        u_hat = self.model(inputs)
        return x * (1 - x) * y * (1 - y) * u_hat  # Hard BCs (Dirichlet)

# Sampling points
def sample_pinn_points(N_int, device="cpu"):
    eps = 1e-6
    x_int = eps + (1 - 2 * eps) * torch.rand((N_int, 1), device=device)
    y_int = eps + (1 - 2 * eps) * torch.rand((N_int, 1), device=device)
    x_int.requires_grad_(True)
    y_int.requires_grad_(True)
    return x_int, y_int

# Loss function
def pinn_loss(model, x_int, y_int, device):
    f_pred = model(x_int, y_int)
    grads = torch.autograd.grad(f_pred, [x_int, y_int],
                                grad_outputs=torch.ones_like(f_pred),
                                create_graph=True)
    f_x, f_y = grads
    f_xx = torch.autograd.grad(f_x, x_int, torch.ones_like(f_x), create_graph=True)[0]
    f_yy = torch.autograd.grad(f_y, y_int, torch.ones_like(f_y), create_graph=True)[0]

    rhs_val = rhs(x_int.detach().cpu().numpy(), y_int.detach().cpu().numpy())
    rhs_torch = torch.tensor(rhs_val, dtype=torch.float32, device=device)

    res = f_xx + f_yy - rhs_torch
    return torch.mean(res ** 2)



# Training + evaluation loop
def run_pinn_experiments(device="cpu", num_repeats=3, save_results=True):
    N_values = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    layers = [2, 275, 275, 1]
    results = []
    os.makedirs("models", exist_ok=True)

    for N in N_values:
        print(f"\n=== Testing with {N} training points ===")
        train_times, eval_times, rel_errors = [], [], []

        for run in range(num_repeats):
            print(f"\n   Run {run + 1}/{num_repeats}")
            set_seed(42 + run)
            x_int, y_int = sample_pinn_points(N, device=device)

            model = PINN(layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            #  Training (Adam)
            print(f"--- Training (Adam) for N={N}, Run={run + 1} ---")
            t0 = time.time()
            for epoch in range(3000):
                optimizer.zero_grad()
                loss = pinn_loss(model, x_int, y_int, device)
                loss.backward()
                optimizer.step()
                if epoch % 500 == 0 or epoch == 2999:
                    print(f"Epoch {epoch:4d} | Loss = {loss.item():.6e}")
            t_adam = time.time() - t0

            #  Refinement (L-BFGS)
            optimizer_lbfgs = torch.optim.LBFGS(model.parameters(),
                                                max_iter=2000,
                                                tolerance_grad=1e-8,
                                                tolerance_change=1e-9,
                                                history_size=100,
                                                line_search_fn="strong_wolfe")

            def closure():
                optimizer_lbfgs.zero_grad()
                loss = pinn_loss(model, x_int, y_int, device)
                loss.backward()
                return loss

            t_lbfgs_start = time.time()
            optimizer_lbfgs.step(closure)
            t_train_total = t_adam + (time.time() - t_lbfgs_start)
            train_times.append(t_train_total)

            #  Evaluation
            nx, ny = 100, 100
            xg = np.linspace(0, 1, nx)
            yg = np.linspace(0, 1, ny)
            X, Y = np.meshgrid(xg, yg)
            XY_torch = torch.tensor(np.column_stack([X.flatten(), Y.flatten()]),
                                    dtype=torch.float32, device=device)

            t_eval_start = time.time()
            with torch.no_grad():
                f_pred = model(XY_torch[:, 0:1], XY_torch[:, 1:2]).cpu().numpy().reshape(ny, nx)
            t_eval = time.time() - t_eval_start
            eval_times.append(t_eval)

            f_true = f_exact(X, Y)
            error_rel = np.linalg.norm(f_pred - f_true, 2) / np.linalg.norm(f_true, 2)
            rel_errors.append(error_rel)
            print(f"      Run {run + 1}: Error = {error_rel:.3e}, Train = {t_train_total:.2f}s, Eval = {t_eval:.3f}s")

            # Save field for plots
            os.makedirs("data/pinn_fields", exist_ok=True)
            field_path = f"data/pinn_fields/pinn_field_N{N}_run{run+1}.npz"
            np.savez_compressed(field_path, X=X, Y=Y, f_pred=f_pred, f_true=f_true)
            print(f"      Saved field data → {field_path}")

            # Save model
            model_name = f"N_{N}_run{run + 1}.pt"
            torch.save(model.state_dict(), os.path.join("models", model_name))

        # Aggregate statistics
        mean_err, std_err = np.mean(rel_errors), np.std(rel_errors)
        mean_train, std_train = np.mean(train_times), np.std(train_times)
        mean_eval, std_eval = np.mean(eval_times), np.std(eval_times)
        best_idx = int(np.argmin(rel_errors))
        best_error = rel_errors[best_idx]

        result = {
            "N_points": N,
            "mean_rel_error": mean_err,
            "std_rel_error": std_err,
            "best_rel_error": best_error,
            "mean_train_time_s": mean_train,
            "std_train_time_s": std_train,
            "mean_eval_time_s": mean_eval,
            "std_eval_time_s": std_eval
        }
        results.append(result)

        print(f"   >> Avg Rel Error: {mean_err:.3e} ± {std_err:.1e}")
        print(f"   >> Avg Train Time: {mean_train:.2f} ± {std_train:.2f}s")

    if save_results:
        os.makedirs("data", exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv("data/pinn_helmholtz_experiment.csv", index=False)
        print("\nResults saved to data/pinn_helmholtz_experiment.csv")

    return results


# Run experiments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = run_pinn_experiments(device=device, num_repeats=10)


# Loading and visualization
df = pd.read_csv("data/pinn_helmholtz_experiment.csv")

# Clean and sort data
idx_sorted = np.argsort(df["N_points"].values)
N_values = df["N_points"].values[idx_sorted]
relative_errors = df["mean_rel_error"].values[idx_sorted]
eval_times = df["mean_eval_time_s"].fillna(0).values[idx_sorted]
train_times = df["mean_train_time_s"].fillna(0).values[idx_sorted]  # ← tiempo de entrenamiento
eval_times = np.maximum(eval_times, 1e-6)
train_times = np.maximum(train_times, 1e-6)

# Load predicted and true fields
results_dict = {}
for N in N_values:
    run_file = f"data/pinn_fields/pinn_field_N{N}_run1.npz"
    if os.path.exists(run_file):
        data = np.load(run_file)
        results_dict[N] = (data["X"], data["Y"], data["f_pred"], data["f_true"])

# Figure configuration
fig = plt.figure(figsize=(7.0, 6.5))
outer_gs = GridSpec(4, 1, height_ratios=[0.6, 0.6, 0.30, 0.40],
                    figure=fig, hspace=0.1)

colors = ['#45A5FF', "#0010A1", "#000000"]
N_examples = [60, 80, 100]

# Sampling points
gs_mesh = outer_gs[0].subgridspec(1, 3, wspace=0.35)
for i, N in enumerate(N_examples):
    ax = fig.add_subplot(gs_mesh[0, i])
    ax.scatter(np.random.rand(N), np.random.rand(N), marker='o', color="#919191", s=15.0, alpha=0.4, edgecolors='none')
    ax.set_aspect("equal")
    ax.axis("off")
    # --- Anotar número de puntos ---
    ax.set_title(f"{N} puntos", fontsize=8, pad=6)
fig.text(0.055, 0.77, r"Puntos de muestreo", fontsize=8, va="center", ha="left", rotation=90)


# Predicted fields (aligned with the cuts)
gs_top = outer_gs[1].subgridspec(1, 3, wspace=0.35)
y_line = 0.375  # common cut line for both rows
for i, (N, color) in enumerate(zip(N_examples, colors)):
    ax = fig.add_subplot(gs_top[0, i])
    X, Y, F_pred, _ = results_dict[N]

    # To ensure exact alignment in the visualization:
    ax.imshow(F_pred.T, extent=(0, 1, 0, 1), cmap="inferno",
              vmin=-1, vmax=1, origin="lower", interpolation="bilinear")
    ax.axhline(y_line, color=color, lw=1.2)
    ax.set_aspect("equal")
    ax.axis("off")
fig.text(0.05, 0.52, r"$\hat{f}(x,y)$", fontsize=8, va="center", ha="left", rotation=90)

# 1D cut comparison (aligned with row 1)
gs_mid = outer_gs[2].subgridspec(1, 3, wspace=0.35)
for i, (N, color) in enumerate(zip(N_examples, colors)):
    ax = fig.add_subplot(gs_mid[0, i])
    X, Y, F_pred, F_true = results_dict[N]

    # Take the cut at the same y value as in row 1
    idx = np.argmin(np.abs(Y[:, 0] - y_line))

    ax.plot(X[idx, :], F_true[idx, :], '-', color="#D4D4D4", lw=5.0, label="Exacta")
    ax.plot(X[idx, :], F_pred[idx, :], '-', color=color, lw=1.2, label="PINN")
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
fig.text(0.05, 0.34, r"$\hat{f}(x,0.375)$", fontsize=8, va="center", ha="left", rotation=90)

# Bottom row: Error and computation times
gs_bottom = outer_gs[3].subgridspec(1, 2, wspace=0.3)
plt.subplots_adjust(hspace=0.15)  # ← agrega separación solo entre la 3ª y 4ª fila

ax_errN = fig.add_subplot(gs_bottom[0, 0])
ax_errT = fig.add_subplot(gs_bottom[0, 1])

# Error relativo vs N
ax_errN.plot(N_values, relative_errors, '-', color='#AFAFAF', marker='s', markersize=4)

# Highlight the three example cases
for N, color in zip(N_examples, colors):
    idx = np.where(N_values == N)[0][0]
    ax_errN.scatter(N_values[idx], relative_errors[idx],
                    color=color, marker='s', s=40, edgecolor='gray', zorder=3)

ax_errN.set_xlabel("Número de puntos")
ax_errN.set_ylabel("Error relativo")
ax_errN.set_yscale("log")
ax_errN.set_ylim(top=1e1, bottom=1e-2)

# --- Tiempos de entrenamiento y evaluación ---
ax_errT.plot(N_values, train_times, '-', color="#AFAFAF", marker='s',
             markersize=4, label="Entrenamiento")
ax_errT.plot(N_values, eval_times, '-', color="#AFAFAF", marker='o',
             markersize=4, label="Evaluación")

# Resaltar los tres casos de ejemplo
for N, color in zip(N_examples, colors):
    idx = np.where(N_values == N)[0][0]
    ax_errT.scatter(N_values[idx], train_times[idx], color=color, marker='s', s=40, edgecolor='gray', zorder=3)
    ax_errT.scatter(N_values[idx], eval_times[idx], color=color, marker='o', s=40, edgecolor='gray', zorder=3)

ax_errT.set_xlabel("Número de puntos")
ax_errT.set_ylabel("Tiempo (s)")
ax_errT.set_yscale("log")
ax_errT.set_ylim(top=1e3, bottom=1e-4)
ax_errT.legend(fontsize=7)

# Save and show figure
os.makedirs("figs", exist_ok=True)
plt.savefig("figs/08_pinn_helmholtz2D_convergence_esp.svg", dpi=150, bbox_inches="tight")
plt.savefig("figs/08_pinn_helmholtz2D_convergence_esp.pdf", dpi=150, bbox_inches="tight")
plt.show()


# Load and prepare data
df = pd.read_csv("data/pinn_helmholtz_experiment.csv")

# Clean and sort
idx_sorted = np.argsort(df["N_points"].values)
N_values = df["N_points"].values[idx_sorted]
relative_errors = df["mean_rel_error"].values[idx_sorted]
eval_times = df["mean_eval_time_s"].fillna(0).values[idx_sorted]
train_times = df["mean_train_time_s"].fillna(0).values[idx_sorted]  # ← training time
eval_times = np.maximum(eval_times, 1e-6)
train_times = np.maximum(train_times, 1e-6)

# Load predicted and true fields
results_dict = {}
for N in N_values:
    run_file = f"data/pinn_fields/pinn_field_N{N}_run1.npz"
    if os.path.exists(run_file):
        data = np.load(run_file)
        results_dict[N] = (data["X"], data["Y"], data["f_pred"], data["f_true"])

# Figure configuration
fig = plt.figure(figsize=(7.5, 7.0))
outer_gs = GridSpec(4, 1, height_ratios=[0.6, 0.6, 0.30, 0.40],
                    figure=fig, hspace=0.1)

colors = ['#45A5FF', "#0010A1", "#000000"]
N_examples = [60, 80, 100]

# Sampling points
gs_mesh = outer_gs[0].subgridspec(1, 3, wspace=0.35)
for i, N in enumerate(N_examples):
    ax = fig.add_subplot(gs_mesh[0, i])
    ax.scatter(np.random.rand(N), np.random.rand(N),
               marker='o', color="#919191", s=12.0, alpha=0.4, edgecolors='none')
    ax.set_aspect("equal")
    ax.axis("off")
    # --- Annotate number of points ---
    ax.set_title(f"{N} points", fontsize=8, pad=6)
fig.text(0.055, 0.77, r"Sampling points", fontsize=8, va="center", ha="left", rotation=90)

# Predicted fields (aligned with cross-sections)
gs_top = outer_gs[1].subgridspec(1, 3, wspace=0.35)
y_line = 0.375  # common cross-section line
for i, (N, color) in enumerate(zip(N_examples, colors)):
    ax = fig.add_subplot(gs_top[0, i])
    X, Y, F_pred, _ = results_dict[N]

    # Ensure exact alignment in visualization:
    ax.imshow(F_pred.T, extent=(0, 1, 0, 1), cmap="inferno",
              vmin=-1, vmax=1, origin="lower", interpolation="bilinear")
    ax.axhline(y_line, color=color, lw=1.2)
    ax.set_aspect("equal")
    ax.axis("off")
fig.text(0.05, 0.52, r"$\hat{f}(x,y)$", fontsize=8, va="center", ha="left", rotation=90)

# 1D cross-section comparison (aligned with row 1)
gs_mid = outer_gs[2].subgridspec(1, 3, wspace=0.35)
for i, (N, color) in enumerate(zip(N_examples, colors)):
    ax = fig.add_subplot(gs_mid[0, i])
    X, Y, F_pred, F_true = results_dict[N]

    # Take the same y-value as in row 1
    idx = np.argmin(np.abs(Y[:, 0] - y_line))

    ax.plot(X[idx, :], F_true[idx, :], '-', color="#D4D4D4", lw=5.0, label="Exact")
    ax.plot(X[idx, :], F_pred[idx, :], '-', color=color, lw=1.2, label="PINN")
    # Format tick labels
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
fig.text(0.05, 0.34, r"$\hat{f}(x,0.375)$", fontsize=8, va="center", ha="left", rotation=90)

# Error and computational times
gs_bottom = outer_gs[3].subgridspec(1, 2, wspace=0.3)
plt.subplots_adjust(hspace=0.15)  # ← add spacing only between 3rd and 4th rows

ax_errN = fig.add_subplot(gs_bottom[0, 0])
ax_errT = fig.add_subplot(gs_bottom[0, 1])

# Relative error vs N 
ax_errN.plot(N_values, relative_errors, '-', color='#AFAFAF', marker='s', markersize=4)

# Highlight the three example cases
for N, color in zip(N_examples, colors):
    idx = np.where(N_values == N)[0][0]
    ax_errN.scatter(N_values[idx], relative_errors[idx],
                    color=color, marker='s', s=40, edgecolor='gray', zorder=3)

ax_errN.set_xlabel("Number of points")
ax_errN.set_ylabel("Relative error")
ax_errN.set_yscale("log")
ax_errN.set_ylim(top=1e1, bottom=1e-2)

# Training and evaluation times
ax_errT.plot(N_values, train_times, '-', color="#AFAFAF", marker='s',
             markersize=4, label="Training")
ax_errT.plot(N_values, eval_times, '-', color="#AFAFAF", marker='o',
             markersize=4, label="Evaluation")

# Highlight the three example cases
for N, color in zip(N_examples, colors):
    idx = np.where(N_values == N)[0][0]
    ax_errT.scatter(N_values[idx], train_times[idx],
                    color=color, marker='s', s=40, edgecolor='gray', zorder=3)
    ax_errT.scatter(N_values[idx], eval_times[idx],
                    color=color, marker='o', s=40, edgecolor='gray', zorder=3)

ax_errT.set_xlabel("Number of points")
ax_errT.set_ylabel("Time (s)")
ax_errT.set_yscale("log")
ax_errT.set_ylim(top=1e3, bottom=1e-4)
ax_errT.legend(fontsize=7)

# Save and display figure
os.makedirs("figs", exist_ok=True)
plt.savefig("figs/08_pinn_helmholtz2D_convergence_en.svg", dpi=150, bbox_inches="tight")
plt.savefig("figs/08_pinn_helmholtz2D_convergence_en.pdf", dpi=150, bbox_inches="tight")
plt.show()
