# instance.py
from forward import *

def main():
    Ptotal = convert_atm(1013, "mbar")  # total pressure [atm]
    T = 288.2                          # temperature [K]
    L_m = 1.0                           # path length [m]

    sp = default_species()              # already has partial pressures (atm) in s.Pmol

    for s in sp:
        if s.name == 'H2O': s.Pmol = 0.00775 * Ptotal
        if s.name == 'CO2': s.Pmol = 0.00033 * Ptotal
        if s.name == 'O3': s.Pmol = 2.66e-08 * Ptotal
        if s.name == 'N2O': s.Pmol = 3.2e-07 * Ptotal
        if s.name == 'CO': s.Pmol = 1.5e-07 * Ptotal
        if s.name == 'CH4': s.Pmol = 1.7e-06 * Ptotal
    

    res = run_simulation(
        species=sp,
        parfile='C:/Users/PC/Documents/GitHub/GPT/pars/ALL.par',
        nu_min=1900, nu_max=2300, dnu=0.01,
        tileW=20.0, guard=25.0,
        temp_K=T, L_m=L_m, pres=Ptotal, #standard 1 atm
        delta_um= 0.0000575,
        save_csv=True, outdir="C:/Users/PC/Documents/GitHub/GPT/out", make_plots=True,
        att=True,
        transmission_npy_name="Simulated_CH4_lsf.npy",
        use_all_isotopologues=True,
        species_to_use=['CH4'], # IMPORTANT: None => run all gases
        lsf={"kind": "gaussian", "W_cm1": 0.05, "domain": "tau"}, 
    )

    lambda_um = res["lambda_centers"]
    T_total   = res["T_prod_samp"]      # total transmittance (product of all gases)
    T_each    = res["T_each_samp"]      # per-gas transmittance
    names     = [s.name for s in res["species"]]

    print("Bands:", lambda_um.shape, "T_total range:", T_total.min(), T_total.max())
    print("Gases:", names)

if __name__ == "__main__":
    main()
