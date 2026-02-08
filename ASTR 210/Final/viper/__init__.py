"""
The inputs (i.e., keys in dictionary inp_dict) required to Fit the VIPER data are as follows
inp_dict["Flux"]          -->> Input flux 1D Array
inp_dict["z_arr_obs"]     -->> Redshift array 1D array
inp_dict["wavelength"]    -->> Wavelength array 1D array
inp_dict["noise"]         -->> Input Noise 1D Array
inp_dict["SNR"]           -->> SNR If there is only one SNR for the spectra (e.g. in case of Simulations) or None -->> Median SNR is calculated from noise
inp_dict["lsf_type"]      -->> Gaussian or None; If None User must give LSF as function of wavelength
inp_dict["sigma_v"]       -->> Width of Gaussian if lsf_type is Gaussian.
inp_dict["sigma_v_csl"]   -->> Width of Gaussian required to smooth the field for peak identification. (sigma_v_csl >= sigma_v)
inp_dict["N_iter"]        -->> Number of Iterations to Perform for fitting
inp_dict["cosmo_param"]   -->> Cosmological Parameters
inp_dict["wave_lb"]       -->> Lower wavelength above which regions need to be fitted, if None -->> wave_lb = wavelength[0]
inp_dict["wave_ub"]       -->> Upper wavelength below which regions need to be fitted, if None -->> wave_ub = wavelength[-1]
inp_dict["log_NXI_lb"]    -->> Lower bound on log NXI Column density, if None -->> log_NXI_lb = 9.0
inp_dict["log_NXI_ub"]    -->> Lower bound on log NXI Column density, if None -->> log_NXI_ub = 18.0
inp_dict["b_lb"]          -->> Lower bound on doppler b parameter, if None -->> b_lb = delta_v (Pixel seperation in velocity)
inp_dict["b_ub"]          -->> Upper bound on doppler b parameter, if None -->> b_ub = 1.5 times the length of region in velocity units.
inp_dict["log_NXI_init"]  -->> Initial guess values for log NXI Column density, if None -->> log NXI is calculated assuming apperent OD approximation (http://iopscience.iop.org/article/10.1086/491735/pdf)
inp_dict["b_init"]        -->> Initial guess values for doppler b parameter, if None -->> b_init is calculated 3 times pixel seperation between velocity
inp_dict["ftol"]          -->> Function tolerence for 1st iteration. With subsequent iteration ftol is decreased by factor of 0.01
inp_dict["xtol"]          -->> Parameter tolerence for 1st iteration. With subsequent iteration xtol is decreased by factor of 0.01
inp_dict["crude_peak_cutoff"] -->> All peaks above this CSL cutoff are detected.
inp_dict["min_csl_cutoff"]    -->> All valleys below this CSL cutoff are detected.
inp_dict["lambda_rest_XI"]   -->> Rest frame Wavelength of Lyman-alpha Transition 1215.6701
inp_dict[crude_valley_cutoff"] = (1.0-0.95)*SNR -->> is set internally by VIPER

Returns out_dict
out_dict = {"param_fit":param_fit,"Flux_fit":Flux_inp_fit_arr,"chi_sq_arr":chi_sq_arr}

"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os,sys
import matplotlib.pyplot as plt
plt.style.use('classic')

sys.dont_write_bytecode = True

# *************** Custom Module Starts *********************
from viper.VIPER_routines import VIPER_routines_main,SAVE_VIPER_DATA
# *************** Custom Module Ends *********************


def fit(vel,flux):
    # flag_specie_trans tells VIPER which transition to fit
    # A list of available transitions that can be fit are given in
    # file ./specie_transition_properties.txt
    # The first column in this file is an alias for specie-transition name.
    # This specie-transition name should be given as input to  VIPER in 
    # the form of variable flag_specie_trans
    # You can add any other transition to this if you like. 
    # (See readme file in lib/VIPER_functions/V3)

    flag_specie_trans = "HI-1215" # Which specie transition to fit
    wave_lb           = None # Minimum wavelength of the region to fit in Angstroms
    wave_ub           = None # Minimum wavelength of the region to fit in Angstroms
    ftol              = 1e-8 # Function tolerence for stopping criteria VIPER fitting
    xtol              = 1e-8 # Variable tolerence for stopping criteria VIPER fitting
    lsf_type          = "Gaussian" # Line spread function type
    sigma_v           = 6.0 / 2.355 # Sigma of LSF Gaussian
    sigma_v_CSL       = 6.0 / 2.355 # Sigma of smoothing window to find peaks in spectra
    N_iter            = 1            # Number of iteration to perform
    crude_peak_cutoff = 1.5          # Above this value all the peaks are identified
    min_csl_cutoff    = 1.5          # Above this value all the peaks are identified
    log_NXI_lb        = 12.0         # Minumum bound on column density of specie
    log_NXI_ub        = 15.0         # Maximum bound on column density of specie
    b_lb              = None         # Minumum bound on line width parameter km/s
    b_ub              = 150.0        # Maximum bound on line width parameter km/s
    log_NXI_init      = 12.0         # Initial guess value of Column Density
    b_init            = 8.0          # Initial guess value of line width parameter km/s
    SNR               = 20.0         # Signal to Noise Ratio of the spectra
    cosmo_param       = cosmological_parameters()  # Cosmological parameters to use
    inp_dict_viper    = vars()

    # Custom Routines to read the spectrum

    Flux = flux
    wavelength = 1216*(1+vel/3e5)*4
    z_arr_obs = wavelength/1216 - 1
    noise     = np.ones(wavelength.shape[0]) / SNR
    spec_indx = 0

    # Custom Routines to read the spectrum

    inp_dict_viper.update({"Flux":Flux,"wavelength":wavelength,\
                           "noise":noise,"z_arr_obs":z_arr_obs})

    grb,out_dict_viper = VIPER_routines_main(inp_dict_viper)
    param_fit          = out_dict_viper["param_fit"]
    print("Number of fitted components =",param_fit.shape[0])
    return param_fit
    

def cosmological_parameters():
    #--------------------- Cosmological Parameters -------------------------
    # Cosmology in Sherwood Simulation Suite is consistent with Planck+2014
    # LCDM Planck + WP + highL + BAO

    omega_m      = 0.3036
    omega_l      = 0.6964
    omega_r      = 0.0
    omega_k      = 0.0
    omega_b      = 0.0479
    hubble_param = 0.6714
    sigma_8      = 0.829
    Y            = 0.24
    omega_bhsq   = omega_b * hubble_param**2.0		# From Planck data
    rho_c_by_hsq = 1.8791e-29 # in gm / cm^3  rho_c (Present day critical density) -->> (3 x H(0)^2)  / (8 x pi x G)
    ns           = 0.961
    cosmo_param  = vars()
    return cosmo_param

#--------------------- Parameters releted to the Simulation  -------------------------

def step_2_post_process_flux(inp_dict_viper):
    cosmo_param            = inp_dict_viper["cosmo_param"]

    lsf_type               = inp_dict_viper["lsf_type"]
    sigma_v                = inp_dict_viper["sigma_v"]
    lsf_dict               = {"lsf_type":lsf_type,"sigma_v":sigma_v}

    flag_specie_trans      = inp_dict_viper["flag_specie_trans"]
    inp_path_specie        = inp_dict_viper["inp_path_specie"]
    line_dict_all          = generate_XI_forest_line_dict(inp_path_specie)
    line_dict              = line_dict_all[flag_specie_trans]
    lambda_rest_XI         = line_dict["lambda_rest"]
    lambda_rest_sim        = lambda_rest_XI
    lambda_rest_obs        = lambda_rest_XI

    SNR_val        = inp_dict_viper["SNR"]
    data_arr       = inp_dict_viper["data_arr"]
    z_arr_sim      = data_arr[:,0]
    z_arr_obs      = z_arr_sim.copy()
    SNR_obs        = np.ones(z_arr_obs.shape[0]) * SNR_val
    Flux_sim       = data_arr[:,1]
    Noise          = np.random.normal(0.0,1.0 / SNR_obs)
    inp_dict_glass = vars()
    out_dict       = MATCH_SIMULATION_SPECTRA_TO_OBSERVATIONS(inp_dict_glass,print_output="y")
    return out_dict


def step_1_read_input_spectra(inp_dict_viper,spec_indx):

    flag_specie_trans      = inp_dict_viper["flag_specie_trans"]
    flag_specie_trans_list = ["HI-1215","CIV-1548","OVI-1031","NeVIII-770"]
    tau_factor_list        = [5.0,25.0,7.0,20.0]

    for loop_indx in xrange(len(flag_specie_trans_list)):
        if flag_specie_trans == flag_specie_trans_list[loop_indx]:
            specie_indx = loop_indx
            col_indx    = loop_indx + 1
            break

    inp_path         = "/data/emergence11/pgaikwad/Collaboration/Sukanya/18_Feb_2022_VIPER_For_Metals/data_files/AGN_WIND_conv_F_OVI_0.4_0.5/"
    inp_file         = str("%i"%spec_indx) + ".txt"
    data_arr_org     = np.loadtxt(inp_path + inp_file)
    data_arr         = np.zeros((data_arr_org.shape[0],2))
    tau_factor       = tau_factor_list[specie_indx]
    data_arr[:,0]    = data_arr_org[:,0]
    data_arr[:,1]    = data_arr_org[:,col_indx]**tau_factor

    inp_dict_viper.update({"data_arr":data_arr})

    return inp_dict_viper


    SNR_val           = 20.0
    lsf_type          = "Gaussian"
    sigma_v           = 17.0
    cosmo_param       = inp_param["cosmo_param"]
    lsf_dict          = {"lsf_type":lsf_type,"sigma_v":sigma_v}
    line_dict_all     = generate_line_dict_main()
    flag_specie_trans = flag_specie_trans_list[specie_indx]
    line_dict         = line_dict_all[flag_specie_trans]
    lambda_rest_XI    = line_dict["lambda_rest"]
    lambda_rest_sim   = lambda_rest_XI
    lambda_rest_obs   = lambda_rest_XI
    inp_dict          = vars()
    return inp_dict

def step_4_viper_fit(inp_dict,out_dict,lib_path):
    inp_dict_viper    = VIPER_DEFAULT_PARAMETER()
    flag_specie_trans = inp_dict["flag_specie_trans"]
    SNR               = inp_dict["SNR_val"]
    sigma_v_CSL       = inp_dict["sigma_v"]
    sigma_v           = inp_dict["sigma_v"]
    data_arr          = inp_dict["data_arr"]
    cosmo_param       = inp_dict["cosmo_param"]
    z_arr_obs         = data_arr[:,0]
    lambda_rest_XI    = inp_dict["lambda_rest_XI"]
    Flux              = out_dict["Flux_sim_mimic"]
    Flux_no_noise     = out_dict["Flux_sim_no_noise"]
    noise             = Flux - Flux_no_noise
    wavelength        = out_dict["wave_arr_obs"]
    inp_dict_viper.update({"sigma_v":sigma_v,"sigma_v_CSL":sigma_v_CSL,\
                           "lambda_rest_XI":lambda_rest_XI,\
                           "Flux":Flux,"wavelength":wavelength,\
                           "noise":noise,"SNR":SNR,"z_arr_obs":z_arr_obs,\
                           "cosmo_param":cosmo_param,\
                           "flag_specie_trans":flag_specie_trans,\
                           "inp_path_specie":lib_path})

    grb,out_dict_viper = __VIPER_MAIN__(inp_dict_viper)
    Flux_fit           = out_dict_viper["Flux_fit"]
    plt.plot(Flux_fit[:,0],Flux_fit[:,1])
    plt.plot(Flux_fit[:,0],Flux_fit[:,2])
    plt.show()
    exit()

def step_2_post_process_flux_old(inp_dict,specie_indx):
    data_arr  = inp_dict["data_arr"]
    SNR_val   = inp_dict["SNR_val"]
    z_arr_sim = data_arr[:,0]
    z_arr_obs = z_arr_sim.copy()
    SNR_arr   = np.ones(z_arr_obs.shape[0]) * SNR_val
    Flux_sim  = data_arr[:,specie_indx+1]
    Noise     = np.random.normal(0.0,1.0 / SNR_arr)
    inp_dict.update({"SNR_obs":SNR_arr,"z_arr_sim":z_arr_sim,\
                     "z_arr_obs":z_arr_obs,"Flux_sim":Flux_sim,\
                     "Noise":Noise})
    out_dict  = MATCH_SIMULATION_SPECTRA_TO_OBSERVATIONS(inp_dict,print_output="y")
    return out_dict

def step_1_default_parameters(spec_indx=0,specie_indx=0):
    inp_path               = "/data/emergence11/pgaikwad/Collaboration/Sukanya/18_Feb_2022_VIPER_For_Metals/data_files/AGN_WIND_conv_F_OVI_0.4_0.5/"
    inp_file               = str("%i"%spec_indx) + ".txt"
    tau_factor_list        = [5.0,25.0,7.0,20.0]
    flag_specie_trans_list = ["HI-1215","CIV-1548","OVI-1031","NeVIII-770"]
    SNR_val                = 20.0
    lsf_type               = "Gaussian"
    sigma_v                = 17.0
    cosmo_param            = inp_param["cosmo_param"]
    lsf_dict               = {"lsf_type":lsf_type,"sigma_v":sigma_v}
    line_dict_all          = generate_line_dict_main()
    flag_specie_trans      = flag_specie_trans_list[specie_indx]
    line_dict              = line_dict_all[flag_specie_trans]
    lambda_rest_XI         = line_dict["lambda_rest"]
    lambda_rest_sim        = lambda_rest_XI
    lambda_rest_obs        = lambda_rest_XI
    inp_dict               = vars()
    return inp_dict

def step_2_read_input_data(inp_dict):
    inp_path        = inp_dict["inp_path"]
    inp_file        = inp_dict["inp_file"]
    tau_factor_list = inp_dict["tau_factor_list"]
    data_arr_org    = np.loadtxt(inp_path + inp_file)
    data_arr        = data_arr_org.copy()

    for indx in xrange(1,data_arr_org.shape[1]):
        tau_factor       = tau_factor_list[indx-1]
        data_arr[:,indx] = data_arr_org[:,indx]**tau_factor

    inp_dict.update({"data_arr":data_arr})
    return inp_dict



