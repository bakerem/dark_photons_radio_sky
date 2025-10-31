import os
import sys

sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

import py21cmfish as p21fish

output_dir = "/projectnb/darkcosmo/dark_photon_project/21cmfast_cache/cross_power_forecast/"
fid_err_dir = "/projectnb/darkcosmo/dark_photon_project/21cmfish/examples/data/21cmSense_noise/21cmSense_fid_EOS21/"
pess_err_dir = "/projectnb/darkcosmo/dark_photon_project/21cmfish/examples/data/21cmSense_noise/21cmSense_pess_EOS21/"
astro_params_vary, astro_params_fid = p21fish.get_params_fid(
    config_file="/projectnb/darkcosmo/dark_photon_project/21cmfish/21cmFAST_config_files/dark_photon.config"
)

try:
    from palettable.tableau import Tableau_20, ColorBlind_10
    cols = ColorBlind_10.hex_colors

    col_pess  = cols[6]
    col_mod   = cols[0]
    col_alpha = 'k'
    col_mcmc  = cols[3]
    col_P19   = cols[1]

except:
    col_pess  = '0.5'
    col_mod   = 'tab:blue'
    col_alpha = 'k'
    col_mcmc  = '0.7'
    col_P19   = 'tab:orange'


sim_mA_list = np.geomspace(1.5e-14, 1e-13, 15)
halo_mA_list = np.geomspace(1e-13, 1e-11, 20)
mA_list = np.concatenate((sim_mA_list, halo_mA_list))

for i, mA in enumerate(sim_mA_list):
    params = {}
    for param in astro_params_vary[:-1]:
        if os.path.exists(f"{output_dir}power_spectrum_deriv_dict_{param}.npy"):
            new = False
        else:
            new = True
        param_key_with_mA = param + f"{0.0:.3e}"
        params[param_key_with_mA] = p21fish.Parameter(
            param=param,
            output_dir=output_dir,
            HII_DIM=128,
            HII_interp_factor=3,
            BOX_LEN=256,
            min_redshift=5,
            k_HERA=True,
            PS_err_dir=fid_err_dir,
            crosszs=(6,20),
            clobber=False,
            fid_only=False,
            vb=False,
            new=new,
        )
    param = "EPSILON4"
    param_key_with_mA = param + f"{mA:.3e}"
    # if os.path.exists(f"{output_dir}power_spectrum_deriv_dict_EPSILON4_mA={mA:.3e}.npy"):
    #     new = False
    # else:
    #     new = True
    params[param_key_with_mA] = p21fish.Parameter(param=astro_params_vary[-1],
                                        output_dir=output_dir,
                                        HII_DIM=128, HII_interp_factor=3, BOX_LEN=256,
                                        min_redshift=5,
                                        k_HERA=True,
                                        PS_err_dir=fid_err_dir,
                                        mA=mA,
                                        epsilon4step=(1e-8)**4, 
                                        crosszs=(6,20),     
                                        halo_angles=np.geomspace(1e-4, 0.2, 100),      
                                        clobber=False,
                                        vb=False, new=False)
    
    params_no_mass ={}
    for key in params.keys():
        if "EPSILON4" in key:
            params_no_mass[key.split(f"{mA:.3e}")[0]] = params[key]
        else:
            params_no_mass[key.split("0.000e+00")[0]] = params[key]
    Fij_matrix_PS, Finv_PS = p21fish.make_fisher_matrix(params_no_mass, fisher_params=astro_params_vary,
                                                        hpeak=0.0, obs='Cross PS',
                                                        k_min=0.1, k_max=1,
                                                        z_min=5., z_max=35.,
                                                        sigma_mod_frac=0,
                                                        add_sigma_poisson=True)
    print(f"{mA:.3e}, epsilon_limit={2*Finv_PS[-1,-1]**0.125:.3e}")
    np.save(f"{output_dir}Fij_matrix_PS_mA={mA:.3e}.npy", Fij_matrix_PS)
    np.save(f"{output_dir}Finv_PS_mA={mA:.3e}.npy", Finv_PS)
    Finv_PS[:, -1] = Finv_PS[:, -1] * 10**30
    Finv_PS[-1,:] = Finv_PS[-1,:] * 10**30

    astro_params_labels = {'ALPHA_ESC': r'$\alpha_\mathrm{esc}^{II}$',
                        'F_ESC10' : r'$\log_{10}f_\mathrm{esc,10}$',
                        'ALPHA_STAR' : r'$\alpha_\star^{II}$',
                        'F_STAR10' : r'$\log_{10}f_{\star,10}$',
                        'F_STAR7_MINI' : r'$\log_{10}f_\mathrm{\star,7}$',
                        'ALPHA_STAR_MINI' : r'$\alpha_\star^{III}$',
                        'F_ESC7_MINI' : r'$\log_{10}f_\mathrm{esc,7}$',
                        'L_X' : r'$\log_{10}\frac{L_X/{\dot{M}_\star}}{\mathrm{erg}\,\mathrm{s}^{-1}\, M_\odot^{-1}\,\mathrm{yr}}$',
                        'L_X_MINI': r'$\log_{10}\frac{L^{III}_X/{\dot{M}_\star}}{\mathrm{erg}\,\mathrm{s}^{-1}\, M_\odot^{-1}\,\mathrm{yr}}$',
                        'NU_X_THRESH' : r'$E_0$/eV',
                        'A_LW' : r'$A_\mathrm{LW}$',
                        'M_TURN': r'$\log_{10} (M_\mathrm{turn}/M_\odot)$',
                         't_STAR': r'$t_\star$',
                         'EPSILON4': r"$\epsilon^4 10^{30}$"}

    fid_params = np.array([astro_params_fid[param] for param in params_no_mass])
    fid_labels = np.array([astro_params_labels[param] for param in params_no_mass])
    fig, axes = p21fish.plot_triangle(params=astro_params_vary,
                        fiducial=fid_params,
                        labels=fid_labels,
                        cov=Finv_PS,
                        ellipse_color=col_mod,
                        title_fontsize=14,
                        resize_lims=True,
                        xlabel_kwargs={'labelpad': 5, 'fontsize':22},
                        ylabel_kwargs={'labelpad': 5, 'fontsize':22},
                        fig_kwargs={'figsize':(24,24),})
    coeff, exponent = f"{mA:.3e}".split("e")
    fig.suptitle(r"Limits for $m_{A'}=$"+f"{coeff}"+r"$\times$"+f"10$^{{{exponent}}}$", fontsize=24)
    plt.savefig(f'{output_dir}corner_plot_mA{mA:.3e}.pdf', bbox_inches='tight')
