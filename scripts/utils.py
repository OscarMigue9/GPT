# util.py
# Photonix Team - 2026
# Helpers for Transmittance/Attenuation Forward Model (LBL + optional LSF)

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forward import Species

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.special import wofz
from typing import List, Dict, Any, Tuple

# ------------------------- constants -------------------------
TREF = 296.0
K_B_erg = 1.380649e-16
C_M_S   = 299792458.0
C_CM_S  = C_M_S * 100.0
C2      = 1.43880285
N_A     = 6.02214086e23
N_L     = 2.47937196e19



# ========================= isotopologue methods and library =========================
def _prepare_forward_variants(species: List["Species"], use_all: bool) -> Tuple[List["Species"], List[int]]:
    """Expande cada especie en variantes individuales y recuerda su origen."""
    from forward import Species, Isotopologue  # local import to avoid circular

    variants: List[Species] = []
    parent_map: List[int] = []

    for parent_idx, sp in enumerate(species):
        variant_defs = [Isotopologue(sp.iso, sp.qfile, sp.Wg, sp.Pmol)]
        if use_all:
            variant_defs.extend(sp.extra_isotopologues)

        for variant in variant_defs:
            variant_sp = Species(
                name=sp.name,
                mol=sp.mol,
                iso=variant.iso,
                qfile=variant.qfile,
                Wg=variant.Wg,
                Pmol=variant.Pmol if variant.Pmol is not None else sp.Pmol,
            )
            variants.append(variant_sp)
            parent_map.append(parent_idx)

    return variants, parent_map


def _isotopologue_bank(
    base: str = "C:/Users/PC/Documents/GitHub/GPT/tips/"
) -> Dict[str, Dict[str, Any]]:
    """Catálogo de isotopólogos y Q-files."""
    from forward import Isotopologue  # local import

    return {
        "H2O": {"mol": 1, "Pmol": 1.876e+04 / 1e6,
                "variants": [
                    Isotopologue("1", base + "H2O/q1.txt",   18.010565),
                    Isotopologue("2", base + "H2O/q2.txt",   20.014811),
                    Isotopologue("3", base + "H2O/q3.txt",   19.014780),
                    Isotopologue("4", base + "H2O/q4.txt",   19.016740),
                    Isotopologue("5", base + "H2O/q5.txt",   21.020985),
                    Isotopologue("6", base + "H2O/q6.txt",   20.020956),
                    Isotopologue("7", base + "H2O/q129.txt", 20.022915),
                ]},
        "CO2": {"mol": 2, "Pmol": 330 / 1e6,
                "variants": [
                    Isotopologue("1",  base + "CO2/q7.txt",   43.989830),
                    Isotopologue("2",  base + "CO2/q8.txt",   44.993185),
                    Isotopologue("3",  base + "CO2/q9.txt",   45.994076),
                    Isotopologue("4",  base + "CO2/q10.txt",  44.994045),
                    Isotopologue("5",  base + "CO2/q11.txt",  46.997431),
                    Isotopologue("6",  base + "CO2/q12.txt",  45.997400),
                    Isotopologue("7",  base + "CO2/q13.txt",  47.998320),
                    Isotopologue("8",  base + "CO2/q14.txt",  46.998291),
                    Isotopologue("9",  base + "CO2/q15.txt",  45.998262),
                    Isotopologue("10", base + "CO2/q120.txt", 49.001675),
                    Isotopologue("A",  base + "CO2/q121.txt", 48.001646),
                    Isotopologue("B",  base + "CO2/q122.txt", 47.001618),
                ]},
        "O3": {"mol": 3, "Pmol": 0.03017 / 1e6,
               "variants": [
                    Isotopologue("1", base + "O3/q16.txt", 47.984745),
                    Isotopologue("2", base + "O3/q17.txt", 49.988991),
                    Isotopologue("3", base + "O3/q18.txt", 49.988991),
                    Isotopologue("4", base + "O3/q19.txt", 48.988960),
                    Isotopologue("5", base + "O3/q20.txt", 48.988960),
               ]},
        "N2O": {"mol": 4, "Pmol": 0.32 / 1e6,
                "variants": [
                    Isotopologue("1", base + "N2O/q21.txt", 44.001062),
                    Isotopologue("2", base + "N2O/q22.txt", 44.998096),
                    Isotopologue("3", base + "N2O/q23.txt", 44.998096),
                    Isotopologue("4", base + "N2O/q24.txt", 46.005308),
                    Isotopologue("5", base + "N2O/q25.txt", 45.005278),
                ]},
        "CO": {"mol": 5, "Pmol": 0.15 / 1e6,
               "variants": [
                    Isotopologue("1", base + "CO/q26.txt", 27.994915),
                    Isotopologue("2", base + "CO/q27.txt", 28.998270),
                    Isotopologue("3", base + "CO/q28.txt", 29.999161),
                    Isotopologue("4", base + "CO/q29.txt", 28.999130),
                    Isotopologue("5", base + "CO/q30.txt", 31.002516),
                    Isotopologue("6", base + "CO/q31.txt", 30.002485),
               ]},
        "CH4": {"mol": 6, "Pmol": 1.7 / 1e6,
                "variants": [
                    Isotopologue("1", base + "CH4/q32.txt", 16.031300),
                    Isotopologue("2", base + "CH4/q33.txt", 17.034655),
                    Isotopologue("3", base + "CH4/q34.txt", 17.037475),
                    Isotopologue("4", base + "CH4/q35.txt", 18.040830),
                ]},
        "O2": {"mol": 7, "Pmol": 0.20946,
               "variants": [
                    Isotopologue("1", base + "O2/q36.txt", 31.989830),
                    Isotopologue("2", base + "O2/q37.txt", 33.994076),
                    Isotopologue("3", base + "O2/q38.txt", 32.994045),
               ]},
    }


def default_species() -> List["Species"]:
    """Configured Species objects."""
    from forward import Species  # local import

    bank = _isotopologue_bank()
    order = ["H2O", "CO2", "O3", "N2O", "CO", "CH4", "O2"]

    species_list: List[Species] = []
    for name in order:
        entry = bank[name]
        variants = entry["variants"]
        if not variants:
            continue
        main, *others = variants
        species_list.append(Species(
            name=name,
            mol=entry["mol"],
            iso=main.iso,
            qfile=main.qfile,
            Wg=main.Wg,
            Pmol=float(entry["Pmol"]),
            extra_isotopologues=list(others),
        ))
    return species_list


# ========================= basic utilities =========================
def convert_atm(P, u):
    if u == "gcms2":
        return P / 1013.25 * 10**-3
    if u == "mbar":
        return P / 1013.25
    raise ValueError(f"Unknown pressure unit: {u}")


def convert_vmr(ppm):
    return ppm * 1e-6


def txt_to_npy(txt_path, out_path):
    waves, trans = [], []
    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                w, t = map(float, line.split())
                waves.append(w); trans.append(t)
            except Exception:
                continue
    wave = np.array(waves)
    tr   = np.array(trans)
    np.save(out_path, [wave, tr])
    print(f"Guardado como arreglo numpy en: {out_path}.npy")


def plot_npy(wave, tr, name):
    plt.figure(figsize=(16, 7))
    plt.plot(wave, tr, lw=2.5)
    plt.gca().ticklabel_format(useOffset=False)
    e = 1e-4
    plt.ylim(tr.min() - e, tr.max() + e)
    plt.xlabel("Wavelength (microns)")
    plt.ylabel("Transmittance")
    plt.title("Spectral Transmittance of " + name)
    plt.grid(True)
    plt.show()


def plot_histogram(wave, name, bins=60, range=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 7))
    else:
        fig = ax.figure
    ax.hist(wave, bins=bins, range=range, color="#1ae981", edgecolor="white")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram wavelengths of " + name)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def downsample(simulated_dir, gt_dir, output_dir):
    lam_tgt = simulated_dir[0]
    lam_src = gt_dir[0]
    tau_src = gt_dir[1]

    edges = np.empty(lam_tgt.size + 1)
    edges[1:-1] = 0.5 * (lam_tgt[:-1] + lam_tgt[1:])
    edges[0]    = lam_tgt[0]  - 0.5 * (lam_tgt[1] - lam_tgt[0])
    edges[-1]   = lam_tgt[-1] + 0.5 * (lam_tgt[-1] - lam_tgt[-2])

    edges[0]  = max(edges[0],  lam_src[0])
    edges[-1] = min(edges[-1], lam_src[-1])

    cum = np.concatenate([[0.0], np.cumsum(0.5 * (tau_src[:-1] + tau_src[1:]) * (lam_src[1:] - lam_src[:-1]))])

    def cum_at(x):
        return np.interp(x, lam_src, cum)

    area = cum_at(edges[1:]) - cum_at(edges[:-1])
    tau_ds = area / (edges[1:] - edges[:-1])

    valid = (edges[:-1] >= lam_src[0]) & (edges[1:] <= lam_src[-1])
    tau_ds[~valid] = np.nan

    out = np.vstack([lam_tgt, tau_ds])
    np.save(output_dir, out)


# ========================= HITRAN + Q(T) helpers =========================
def load_Q_vals(qfile: str, Tref: float, T: float) -> Tuple[float, float]:
    if not os.path.isfile(qfile):
        raise FileNotFoundError(f"Missing Q(T) file: {qfile}")
    try:
        df = pd.read_csv(qfile, header=None, delim_whitespace=True, comment="#")
        if df.shape[1] < 2:
            raise ValueError
    except Exception:
        df = pd.read_csv(qfile, header=None, sep=r"[\s,;]+", engine="python", comment="#")

    Tcol = df.iloc[:, 0].astype(float).to_numpy()
    Qcol = df.iloc[:, 1].astype(float).to_numpy()

    Qref = np.interp(Tref, Tcol, Qcol)
    QT   = np.interp(T,    Tcol, Qcol)
    return float(Qref), float(QT)


def read_hitran_par_minimal(path: str) -> Dict[str, np.ndarray]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find {path}")

    widths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 15, 6, 12, 1, 7, 7]
    use_up_to = 10
    cum = np.cumsum([0] + widths)
    sl = [(cum[i], cum[i + 1]) for i in range(use_up_to)]

    mol, iso, nu0, Sref, A, g_air, g_self, Elow, n_air, shift = ([] for _ in range(10))

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                s = [line[a:b] for a, b in sl]
                mol.append(int(s[0])); iso.append(s[1].strip())
                nu0.append(float(s[2])); Sref.append(float(s[3]))
                A.append(float(s[4])); g_air.append(float(s[5]))
                g_self.append(float(s[6])); Elow.append(float(s[7]))
                n_air.append(float(s[8])); shift.append(float(s[9]))
            except Exception:
                continue

    return dict(
        mol=np.asarray(mol, dtype=np.int32),
        iso=np.asarray(iso, dtype="<U10"),
        nu0=np.asarray(nu0, dtype=np.float64),
        Sref=np.asarray(Sref, dtype=np.float64),
        A=np.asarray(A, dtype=np.float64),
        g_air=np.asarray(g_air, dtype=np.float64),
        g_self=np.asarray(g_self, dtype=np.float64),
        Elow=np.asarray(Elow, dtype=np.float64),
        n_air=np.asarray(n_air, dtype=np.float64),
        shift=np.asarray(shift, dtype=np.float64),
    )


def count_lines_by_index(parfile: str, index: int) -> int:
    H = read_hitran_par_minimal(parfile)
    return int(np.count_nonzero(H["mol"] == index))


# ========================= line shape + transmittance =========================
def voigt_profile(nu: np.ndarray, nu0_shifted: np.ndarray, alpha: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    s2 = np.sqrt(np.log(2.0))
    x = s2 * (nu[:, None] - nu0_shifted[None, :]) / alpha[None, :]
    y = s2 * (gamma[None, :] / alpha[None, :])
    z = x + 1j * y
    w = wofz(z)
    fV = s2 / np.sqrt(np.pi) / alpha[None, :] * np.real(w)
    return fV


def transmittance_for_gas_tile(
    nu_vec: np.ndarray,
    H: Dict[str, np.ndarray],
    sp: "Species",
    Tgas: float,
    pres: float,
    Lm: float,
    mask_lines: np.ndarray,
) -> np.ndarray:
    """
    IMPORTANT: assumes sp.Pmol is PARTIAL PRESSURE [atm].
    """
    if not np.any(mask_lines):
        return np.ones_like(nu_vec)

    nu0, Sref, g_air, g_self, Elow, n_air, shift = (
        H["nu0"][mask_lines], H["Sref"][mask_lines], H["g_air"][mask_lines],
        H["g_self"][mask_lines], H["Elow"][mask_lines], H["n_air"][mask_lines], H["shift"][mask_lines]
    )

    Pself = float(sp.Pmol)
    Pair  = max(float(pres) - Pself, 0.0)

    nu0s = nu0 + shift * Pair

    # temperature scaling
    S_T = (Sref * (sp.Qref / sp.QT) *
           np.exp(-C2 * Elow / Tgas) / np.exp(-C2 * Elow / TREF) *
           (1.0 - np.exp(-C2 * nu0 / Tgas)) / (1.0 - np.exp(-C2 * nu0 / TREF)))

    Lcm = Lm * 100.0
    line_intensity = S_T * (TREF / Tgas) * N_L * Pself * Lcm

    alpha = nu0 / C_CM_S * np.sqrt(2.0 * N_A * K_B_erg * Tgas * np.log(2.0) / sp.Wg)
    gamma = ((TREF / Tgas) ** n_air) * (g_air * Pair + g_self * Pself)

    fV = voigt_profile(nu_vec, nu0s, alpha, gamma)
    tau = fV @ line_intensity
    return np.exp(-tau)


def bin_average(x_sorted: np.ndarray, y_sorted: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    yb = np.empty(edges.size - 1)

    left  = np.searchsorted(x_sorted, edges[:-1], side="left")
    right = np.searchsorted(x_sorted, edges[1:],  side="left")

    for i, (l, r) in enumerate(zip(left, right)):
        if r > l:
            yb[i] = np.mean(y_sorted[l:r])
        else:
            yb[i] = np.interp(centers[i], x_sorted, y_sorted)
    return centers, yb


# ========================= LSF (nu-domain) =========================
def _gauss_kernel_nu(FWHM_cm1: float, dnu: float, truncate_sigma: float = 4.0) -> np.ndarray:
    sigma = FWHM_cm1 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = max(2, int(truncate_sigma * sigma / max(dnu, 1e-30)))
    x = np.arange(-half, half + 1) * dnu
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


def apply_lsf_nu(
    T_each_species_nu: List[np.ndarray],
    dnu: float,
    kind: str = "gaussian",
    W_cm1: float = 2.0,
    domain: str = "tau",
) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
    kind = (kind or "gaussian").lower()
    domain = (domain or "tau").lower()
    if kind != "gaussian":
        raise ValueError(f"Unknown LSF kind: {kind}")

    k = _gauss_kernel_nu(FWHM_cm1=float(W_cm1), dnu=float(dnu), truncate_sigma=4.0)

    if domain == "tau":
        tau_each = [-np.log(np.clip(Ti, 1e-300, 1.0)) for Ti in T_each_species_nu]
        tau_each_c = [np.convolve(tau, k, mode="same") for tau in tau_each]
        T_each_c = [np.exp(-tau_c) for tau_c in tau_each_c]
    else:
        T_each_c = [np.convolve(Ti, k, mode="same") for Ti in T_each_species_nu]
        T_each_c = [np.clip(Ti, 1e-300, 1.0) for Ti in T_each_c]

    T_total_c = np.ones_like(T_each_c[0])
    for Ti in T_each_c:
        T_total_c *= np.clip(Ti, 1e-300, 1.0)

    meta = dict(kind=kind, W_cm1=float(W_cm1), domain=domain, dnu=float(dnu), kernel_len=int(len(k)))
    return T_each_c, T_total_c, meta


def _dedup_by_nu(nu: np.ndarray, arrays: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    if nu.size == 0:
        return nu, arrays
    if not np.any(np.diff(nu) == 0):
        return nu, arrays

    uniq, inv, counts = np.unique(nu, return_inverse=True, return_counts=True)
    counts = counts.astype(np.float64)

    out_arrays: List[np.ndarray] = []
    for arr in arrays:
        sums = np.bincount(inv, weights=arr.astype(np.float64))
        out_arrays.append(sums / counts)

    return uniq, out_arrays
