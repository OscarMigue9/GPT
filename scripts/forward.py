# final_forward_attenuation_lsf.py
# Photonix Team - 2026
# Transmittance and Attenuation Forward Model LBL with LSF

from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

# constants
TREF = 296.0
K_B_erg = 1.380649e-16
C_M_S   = 299792458.0
C_CM_S  = C_M_S * 100.0
C2      = 1.43880285
N_A     = 6.02214086e23
N_L     = 2.47937196e19


from utils import (
    _prepare_forward_variants,
    default_species,
    convert_atm, convert_vmr, txt_to_npy, plot_npy, plot_histogram, downsample,
    load_Q_vals, read_hitran_par_minimal, count_lines_by_index,
    transmittance_for_gas_tile, bin_average,
    apply_lsf_nu, _dedup_by_nu,
)

# ========================= dataclass =========================
@dataclass
class Isotopologue:
    iso: str
    qfile: str
    Wg: float
    Pmol: float | None = None

@dataclass
class Species:
    name: str
    mol: int
    iso: str
    qfile: str
    Wg: float
    Pmol: float  # partial pressure [atm]
    extra_isotopologues: List[Isotopologue] = field(default_factory=list)
    Qref: float = field(init=False, default=np.nan)
    QT: float = field(init=False, default=np.nan)
    idx_all: np.ndarray | None = field(init=False, default=None)

# ========================= MAIN FUNCTION =========================
def run_simulation(
    species: List[Species],
    parfile: str = '../../pars/ALL.par',
    nu_min: float = 666.67,
    nu_max: float = 10000.0,
    dnu: float = 0.01,
    tileW: float = 20.0,
    guard: float = 25.0,
    temp_K: float = 296.0,
    L_m: float = 1.0,
    pres: float = 1.0,
    delta_um: float = 0.020,
    save_csv: bool = False,
    outdir: str = 'out',
    make_plots: bool = True,
    att: bool = True,
    use_all_isotopologues: bool = False,
    simulated_dir: str = 'C:/Users/PC/Documents/GitHub/GPT/simulated',
    transmission_npy_name: str | None = None,
    species_to_use: List[str] | None = None,
    lsf: Dict[str, Any] | None = None,
    plot_each: bool = False,
) -> Dict[str, Any]:
    """
    Compute transmittances/attenuations and return sampled results.

    lsf (optional):
        Example: lsf={"kind":"gaussian","W_cm1":2.0,"domain":"tau"}
        Applied in nu-domain BEFORE lambda binning.
    """
    if not os.path.isfile(parfile):
        raise FileNotFoundError(f"Missing HITRAN .par: {parfile}")

    # Filter species
    if species_to_use is not None:
        species_to_use_upper = [s.upper() for s in species_to_use]
        species = [sp for sp in species if sp.name.upper() in species_to_use_upper]
        if not species:
            raise ValueError(f"No species found matching: {species_to_use}")
        print(f"Using species: {[sp.name for sp in species]}")

    forward_variants, variant_to_parent = _prepare_forward_variants(species, use_all_isotopologues)

    for var in forward_variants:
        if not os.path.isfile(var.qfile):
            raise FileNotFoundError(f"Missing q-file for {var.name} iso {var.iso}: {var.qfile}")

    # Load Q(T)
    for var in forward_variants:
        Qref, QT = load_Q_vals(var.qfile, TREF, temp_K)
        var.Qref, var.QT = Qref, QT

    # Read HITRAN .par
    H = read_hitran_par_minimal(parfile)

    # Line indices per variant
    for var in forward_variants:
        var.idx_all = (H['mol'] == var.mol) & (H['iso'] == var.iso)

    # Spectral tiling
    edges_tiles = np.arange(nu_min, nu_max + 1e-9, tileW)
    nu_all_parts, T_prod_all_parts, T_sum_all_parts = [], [], []
    T_each_acc = [[] for _ in species]

    for a in edges_tiles:
        b = min(a + tileW, nu_max)
        a_ext, b_ext = max(nu_min, a - guard), min(nu_max, b + guard)
        nu_ext = np.arange(a_ext, b_ext + 1e-12, dnu)

        T_ext_each_variants = np.ones((len(forward_variants), nu_ext.size), dtype=np.float64)
        for k, var in enumerate(forward_variants):
            idx_tile = var.idx_all & (H['nu0'] >= a_ext) & (H['nu0'] <= b_ext)
            if np.any(idx_tile):
                T_ext_each_variants[k, :] = transmittance_for_gas_tile(
                    nu_ext, H, var, temp_K, pres, L_m, idx_tile
                )

        T_ext_prod = np.prod(T_ext_each_variants, axis=0)

        T_species_ext = np.ones((len(species), nu_ext.size), dtype=np.float64)
        for variant_idx, parent_idx in enumerate(variant_to_parent):
            T_species_ext[parent_idx] *= T_ext_each_variants[variant_idx, :]

        T_ext_sum = np.sum(T_species_ext, axis=0)

        keep = (nu_ext >= a) & (nu_ext <= b)
        nu_all_parts.append(nu_ext[keep])
        T_prod_all_parts.append(T_ext_prod[keep])
        T_sum_all_parts.append(T_ext_sum[keep])
        for kk in range(len(species)):
            T_each_acc[kk].append(T_species_ext[kk, keep])

    nu_all = np.concatenate(nu_all_parts)
    T_prod = np.concatenate(T_prod_all_parts)
    T_sum  = np.concatenate(T_sum_all_parts)
    T_each = [np.concatenate(T_each_acc[kk]) for kk in range(len(species))]

    # Apply LSF (optional)
    lsf_meta = None
    if lsf is not None:
        arrays = [T_prod, T_sum] + T_each
        nu_all, arrays = _dedup_by_nu(nu_all, arrays)
        T_prod, T_sum = arrays[0], arrays[1]
        T_each = arrays[2:]

        kind = lsf.get("kind", "gaussian")
        W_cm1 = float(lsf.get("W_cm1", 2.0))
        domain = lsf.get("domain", "tau")

        T_each, T_prod, lsf_meta = apply_lsf_nu(
            T_each_species_nu=T_each,
            dnu=dnu,
            kind=kind,
            W_cm1=W_cm1,
            domain=domain,
        )
        T_sum = np.sum(np.stack(T_each, axis=0), axis=0)

    # Convert to wavelength and sort
    lambda_um = 1e4 / nu_all
    ord_idx = np.argsort(lambda_um)
    lambda_sorted = lambda_um[ord_idx]
    T_prod_lambda = T_prod[ord_idx]
    T_sum_lambda  = T_sum[ord_idx]
    T_each_lambda = [arr[ord_idx] for arr in T_each]

    # Bin in wavelength
    lam_min = math.ceil(lambda_sorted.min() / delta_um) * delta_um
    lam_max = math.floor(lambda_sorted.max() / delta_um) * delta_um
    edges = np.arange(lam_min, lam_max + 1e-9, delta_um)

    lambda_centers, T_prod_samp = bin_average(lambda_sorted, T_prod_lambda, edges)
    _, T_sum_samp = bin_average(lambda_sorted, T_sum_lambda, edges)
    T_each_samp = [bin_average(lambda_sorted, arr, edges)[1] for arr in T_each_lambda]

    # Attenuation in dB/m (optional)
    if att:
        invL = 1.0 / max(L_m, 1e-300)
        A_dbm_lam_each = [-(10.0 * invL) * np.log10(np.clip(arr, 1e-300, 1.0)) for arr in T_each_samp]
        A_dbm_lam_sum = np.sum(np.stack(A_dbm_lam_each, axis=0), axis=0)
    else:
        A_dbm_lam_each = None
        A_dbm_lam_sum = None

    # Plotting (same as your current code) — keep as-is if you want,
    # or move it to util.py too. If you want ONLY main method here,
    # tell me and I’ll move the plotting block into util.py as plot_results(...).
    if make_plots:
        os.makedirs(outdir, exist_ok=True)
        FIGSIZE = (14, 7)
        DPI_EXPORT = 600
        EXPORT_FORMATS = ("png", "pdf")

        plt.rcParams.update({
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "legend.fontsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "axes.edgecolor": "#222",
            "axes.linewidth": 1.2,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        })

        def _positivize(a):
            a = np.asarray(a)
            eps = max(1e-12, np.nanmin(a[a > 0]) * 0.1) if np.any(a > 0) else 1e-12
            return np.clip(a, eps, None)

        def _log_axis_limits(arr: np.ndarray) -> Tuple[float, float]:
            arr = np.asarray(arr)
            mask = (arr > 0) & np.isfinite(arr)
            if not np.any(mask):
                return 1e-15, 1e4
            min_val = np.min(arr[mask])
            max_val = np.max(arr[mask])
            lower = max(1e-15, min_val * 0.9)
            upper = max_val * 1.1
            if lower >= upper:
                upper = lower * 1.1
            return lower, upper

        def _lin_axis_limits(_: np.ndarray) -> Tuple[float, float]:
            return 0.0, 1.0

        def save_fig(fig, basename: str):
            for ext in EXPORT_FORMATS:
                dpi = DPI_EXPORT if ext.lower() in ("png", "jpg", "jpeg", "tif", "tiff") else None
                fig.savefig(os.path.join(outdir, f"{basename}.{ext}"),
                            dpi=dpi, bbox_inches="tight", pad_inches=0.05)

        if att:
            A_sum_plot = _positivize(A_dbm_lam_sum)
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.semilogy(lambda_centers, A_sum_plot, lw=2.5, label="Total")
            ax.set_xlabel("Wavelength (µm)")
            ax.set_ylabel("Attenuation (dB/m)")
            ax.set_title("Combined atmospheric attenuation")
            ax.set_ylim(*_log_axis_limits(A_sum_plot))
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
            ax.grid(True, which="major", axis="both", linestyle="-", linewidth=0.8, alpha=0.5)
            ax.grid(True, which="minor", axis="both", linestyle=":", linewidth=0.5, alpha=0.3)
            ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
            plt.tight_layout()
            save_fig(fig, "attenuation_total")
            plt.show()
            plt.close(fig)

            if plot_each:
                fig, ax = plt.subplots(figsize=FIGSIZE)
                for sp, A_arr in zip(species, A_dbm_lam_each):
                    A_plot = _positivize(A_arr)
                    ax.semilogy(lambda_centers, A_plot, lw=2.0, label=sp.name)
                ax.set_xlabel("Wavelength (µm)")
                ax.set_ylabel("Attenuation (dB/m)")
                ax.set_title("Atmospheric attenuation by gas")
                ax.set_ylim(*_log_axis_limits(A_plot))
                ax.yaxis.set_major_locator(LogLocator(base=10.0))
                ax.grid(True, which="major", axis="both", linestyle="-", linewidth=0.8, alpha=0.5)
                ax.grid(True, which="minor", axis="both", linestyle=":", linewidth=0.5, alpha=0.3)
                ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)
                plt.tight_layout()
                save_fig(fig, "attenuation_by_gas")
                plt.show()
                plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.plot(lambda_centers, T_prod_samp, lw=2.5, label="Total")
            ax.set_xlabel("Wavelength (µm)")
            ax.set_ylabel("Transmittance")
            ax.set_title("Combined atmospheric transmittance")
            ax.set_ylim(*_lin_axis_limits(T_prod_samp))
            ax.grid(True, which="both", linestyle="-", linewidth=0.8, alpha=0.5)
            ax.legend(loc="lower left", frameon=True, fancybox=True, shadow=True)
            plt.tight_layout()
            save_fig(fig, "transmittance_total")
            plt.show()
            plt.close(fig)

            if plot_each:
                fig, ax = plt.subplots(figsize=FIGSIZE)
                for sp, T_arr in zip(species, T_each_samp):
                    ax.plot(lambda_centers, T_arr, lw=2.0, label=sp.name)
                ax.set_xlabel("Wavelength (µm)")
                ax.set_ylabel("Transmittance")
                ax.set_title("Atmospheric transmittance by gas")
                ax.set_ylim(*_lin_axis_limits(T_prod_samp))
                ax.grid(True, which="both", linestyle="-", linewidth=0.8, alpha=0.5)
                ax.legend(loc="lower left", frameon=True, fancybox=True, shadow=True)
                plt.tight_layout()
                save_fig(fig, "transmittance_by_gas")
                plt.show()
                plt.close(fig)

    # CSV export (kept)
    if save_csv:
        os.makedirs(outdir, exist_ok=True)
        df_total = {"lambda_um": lambda_centers, "T_total": T_prod_samp}
        if att:
            df_total["A_total_dbm"] = A_dbm_lam_sum
        pd.DataFrame(df_total).to_csv(
            os.path.join(outdir, "transmission_total.csv" if not att else "attenuation_total.csv"),
            index=False
        )

        cols = {"lambda_um": lambda_centers}
        for sp, T_arr in zip(species, T_each_samp):
            cols[f"T_{sp.name}"] = T_arr
        if att:
            for sp, A_arr in zip(species, A_dbm_lam_each):
                cols[f"A_{sp.name}_dbm"] = A_arr
        pd.DataFrame(cols).to_csv(
            os.path.join(outdir, "transmission_by_gas.csv" if not att else "attenuation_by_gas.csv"),
            index=False
        )

    result = dict(
        lambda_centers=lambda_centers,
        T_prod_samp=T_prod_samp,
        T_each_samp=T_each_samp,
        species=species,
        att=att,
        lsf=lsf_meta,
        A_dbm_lam_sum=A_dbm_lam_sum if att else None,
        A_dbm_lam_each=A_dbm_lam_each if att else None,
    )

    if transmission_npy_name:
        sim_root = simulated_dir or "out"
        sim_root = os.path.abspath(sim_root) if os.path.isabs(sim_root) else os.path.abspath(os.path.join(os.getcwd(), sim_root))
        os.makedirs(sim_root, exist_ok=True)
        if not transmission_npy_name.lower().endswith(".npy"):
            transmission_npy_name = f"{transmission_npy_name}.npy"
        transmission_path = os.path.join(sim_root, transmission_npy_name)
        np.save(transmission_path, np.vstack([lambda_centers, T_prod_samp]))
        result["transmission_npy_path"] = transmission_path
    else:
        result["transmission_npy_path"] = None

    return result
