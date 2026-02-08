import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from scipy.ndimage.filters import convolve
from scipy.optimize import curve_fit,leastsq
from math import log10
from scipy.integrate import trapz,quad
sys.dont_write_bytecode = True


def xrange(n):
    return range(n)
#


# *************** Custom Module Starts *********************


# *************** Custom Module Ends *********************


###########################################################################
##### VIPER ROUTINES (DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE) #####
###########################################################################

def VIPER_routines_main(inp_dict):
    inp_dict          = STEP_0_CHECK_INPUT(inp_dict)
    inp_dict          = STEP_1_ORGANIZE_INPUT(inp_dict)
    inp_dict,out_dict = STEP_2_VIPER_FITTING_MULTIPLE_ITERATION(inp_dict)
    return inp_dict,out_dict

#------------------------------------------------------------------------------

def STEP_0_CHECK_INPUT(inp_dict):
    inp_dict_key_list_reqd = ["Flux", "z_arr_obs", "wavelength", "noise", "SNR",\
                              "lsf_type", "sigma_v", "sigma_v_CSL", "N_iter",\
                              "cosmo_param", "wave_lb", "wave_ub", "ftol", "xtol",\
                              "crude_peak_cutoff", "min_csl_cutoff", \
                              "log_NXI_lb","log_NXI_ub","b_lb","b_ub","log_NXI_init",
                              "b_init"]
    CHECK_IF_KEY_PRESENT(inp_dict_key_list_reqd,inp_dict)
    CHECK_AND_INITIALIZE_KEY_VALUE(inp_dict,"flag_divide_region","n")
    CHECK_AND_INITIALIZE_KEY_VALUE(inp_dict,"flag_verbose","y")
    CHECK_AND_INITIALIZE_KEY_VALUE(inp_dict,"flag_specie_trans","HI-1215")
    CHECK_AND_INITIALIZE_KEY_VALUE(inp_dict,"inp_path_specie","./")
    return inp_dict

#------------------------------------------------------------------------------

def STEP_1_ORGANIZE_INPUT(inp_dict):
    inp_path_specie   = inp_dict["inp_path_specie"]
    flag_specie_trans = inp_dict["flag_specie_trans"]
    line_dict_all     = generate_XI_forest_line_dict()
    line_dict         = line_dict_all[flag_specie_trans]
    I_alpha           = line_dict["i_alpha"]
    f_lu              = line_dict["f_lu"]
    damp_gamma        = line_dict["damp_gamma"]
    lambda_rest_XI    = line_dict["lambda_rest"]
    inp_dict.update({"I_alpha":I_alpha,"damp_gamma":damp_gamma,\
                     "lambda_rest_XI":lambda_rest_XI,"f_lu":f_lu})
    inp_dict          = STEP_1A_GET_FLUX_NOISE_IN_WAVELENGTH_RANGE(inp_dict)
    inp_dict          = STEP_1B_GET_INPUT_SPECIFIC_TO_VIPER(inp_dict)
    return inp_dict

#------------------------------------------------------------------------------

def STEP_1A_GET_FLUX_NOISE_IN_WAVELENGTH_RANGE(inp_dict):
    wav_z_arr_vel    = 0  # 0 -->> Wavelength; 1 -->> Redshift, 2 -->> Velocity
    cosmo_param      = inp_dict["cosmo_param"]
    Flux             = inp_dict["Flux"] 
    noise            = inp_dict["noise"]
    SNR              = inp_dict["SNR"]
    z_arr_obs        = inp_dict["z_arr_obs"] 
    wavelength       = inp_dict["wavelength"] 
    wave_lb          = inp_dict["wave_lb"]
    wave_ub          = inp_dict["wave_ub"]
    lambda_rest_XI       = inp_dict["lambda_rest_XI"]
    if wave_lb == None: wave_lb = wavelength[0]
    if wave_ub == None: wave_ub = wavelength[-1]
    bool_arr         = (wave_lb <= wavelength) & (wavelength <= wave_ub)
    wavelength       = wavelength[bool_arr].copy()
    Flux             = Flux[bool_arr].copy()
    noise            = noise[bool_arr].copy()
    z_arr_obs        = z_arr_obs[bool_arr].copy()
    x_axis_data      = w_to_v(wavelength,lambda_rest_XI,cosmo_param)

    if SNR == None:
        SNR_arr = 1.0 / noise
        SNR     = np.median(SNR_arr)

    inp_dict.update({"Flux":Flux,"noise":np.abs(noise),"z_arr_obs":z_arr_obs,\
                     "x_axis_data":x_axis_data,"wavelength":wavelength,"SNR":SNR,\
                     "wav_z_arr_vel":wav_z_arr_vel})
    # Note np.abs(noise) is important for noise
    return inp_dict

#------------------------------------------------------------------------------

def STEP_1B_GET_INPUT_SPECIFIC_TO_VIPER(inp_dict):
    lsf_type         = inp_dict["lsf_type"]
    sigma_v          = inp_dict["sigma_v"]
    sigma_v_CSL      = inp_dict["sigma_v_CSL"]
    z_arr_obs        = inp_dict["z_arr_obs"] 
    cosmo_param      = inp_dict["cosmo_param"] 
    velocity         = REDSHIFT_TO_VELOCITY(z_arr_obs,cosmo_param)

    if lsf_type == "Gaussian":
        lsf = GAUSSIAN_LSF(velocity,sigma_v)
    elif lsf_type == None:
        lsf = None
    else:
        lsf = inp_dict["lsf"]

    gauss_kernel     = GAUSSIAN_LSF(velocity,sigma_v_CSL)

    if np.any(lsf) == None: 
        lsf_gauss_unnorm = gauss_kernel 
    else:
        lsf_gauss_unnorm = convolve(lsf,gauss_kernel,mode = "wrap")

    lsf_final        = lsf_gauss_unnorm / np.sum(lsf_gauss_unnorm)
    inp_dict.update({"velocity":velocity,"lsf":lsf,"lsf_final":lsf_final})
    return inp_dict

#---------------------------------------

def GAUSSIAN_LSF(velocity,sigma_v):
    vdata      = velocity.copy()
    N          = vdata.shape[0]
    vdata     -= vdata[N//2]
    gauss      = np.exp(-0.5*vdata**2.0/sigma_v**2.0)
    gauss_norm = gauss / np.sum(gauss)
    return gauss_norm

#---------------------------------------

def REDSHIFT_TO_VELOCITY(redshift,cosmo_param):
    velocity = np.zeros(redshift.shape[0])
    for indx,z_val in enumerate(redshift):
        velocity[indx] = z_to_v(z_val,cosmo_param)
    return velocity

#------------------------------------------------------------------------------


def STEP_1_READ_SIMULATION_INPUT(inp_dict):
    inp_path     = inp_dict["inp_path"]
    inp_file     = inp_dict["inp_file"]
    hdf_filename = inp_path + inp_file
    blockname    = "Model"
    OD           = hdf_read_data(hdf_filename,blockname + "/Optical_Depth_1")
    redshift     = hdf_read_data(hdf_filename,blockname + "/Redshift")
    inp_dict.update({"OD":OD,"redshift":redshift})
    return inp_dict 

#------------------------------------------------------------------------------

def STEP_2A_VIPER_FITTING_FOR_SINGLE_ITERATION(inp_dict):
    iter_indx           =   inp_dict["iter_indx"]
    wav_z_arr_vel       =   inp_dict["wav_z_arr_vel"]
    min_csl_cutoff      =   inp_dict["min_csl_cutoff"]
    crude_valley_cutoff =   inp_dict["crude_valley_cutoff"]
    flag_divide_region  =   inp_dict["flag_divide_region"]
    flag_verbose        =   inp_dict["flag_verbose"]

    if iter_indx == 0:
        line_dict           =   CRUDE_SIGNIFICANCE_LEVEL(inp_dict)
        line_dict           =   PEAK_VALLEY_FINDER(line_dict)
        peak_indx           =   APPLY_CRUDE_SL_CUTOFF(line_dict,inp_dict)
        peak_lb_ub          =   VALLEY_INDEX_FINDER(line_dict,inp_dict,peak_indx,wav_z_arr_vel)
        W_by_sigma          =   line_dict["W_by_sigma"]

        if peak_lb_ub.size == 0:
            param_fit_save = np.zeros((2,7))
            chi_sq_arr     = np.zeros((2,3))
        else:
            lb_ub_N_component = FIND_NON_OVERLAPPING_REGION(peak_lb_ub,W_by_sigma,crude_valley_cutoff,flag_divide_region)
            inp_dict.update({"lb_ub_N_component":lb_ub_N_component})
        inp_dict.update({"peak_lb_ub":peak_lb_ub})


    peak_lb_ub = inp_dict["peak_lb_ub"]
    param_fit_save = np.zeros((2,7))
    chi_sq_arr     = np.zeros((2,3))
    Flux_fit       = inp_dict["Flux"]

    #if peak_lb_ub.size == 0:
    #    param_fit_save = np.zeros((2,7))
    #    chi_sq_arr     = np.zeros((2,3))
    #    Flux_fit       = np.zeros((2,3))

    if peak_lb_ub.size != 0:   
        lb_ub_N_component = inp_dict["lb_ub_N_component"]

        # ----------- FIT ALL THE IDENTIFIED LINES USING VOIGT PROFILE WITH AICC ------------------

        #out_dict       = voigt_lsq_fit(inp_dict,lb_ub_N_component,"chi_sq",flag_verbose)
        out_dict       = voigt_lsq_fit(inp_dict,lb_ub_N_component,flag_verbose=flag_verbose)
        param_fit      = out_dict["param_fit"]
        param_err      = out_dict["param_err"]
        Flux_fit       = VOIGT_PROFILE_PARAM_FIT(param_fit,inp_dict,wav_z_arr_vel)
        param_fit_SL   = RIGOROUS_SIGNIFICANCE_LEVEL(param_fit,inp_dict,Flux_fit)
        chi_sq_arr     = chi_sq_in_region(inp_dict,lb_ub_N_component,out_dict["chi_sq_arr"])
        param_fit_save = np.column_stack([param_fit_SL,param_err])
    save_dict   =   {"param_fit":param_fit_save,"chi_sq_arr":chi_sq_arr,"Flux_fit":Flux_fit}
    return save_dict



#------------------------------------------------------------------------------

def STEP_2_VIPER_FITTING_MULTIPLE_ITERATION(inp_dict):
    SNR                 = inp_dict["SNR"]
    crude_valley_cutoff = (1.0-0.95)*SNR
    inp_dict.update({"crude_valley_cutoff":crude_valley_cutoff})

    ## ---------------- Do not change these numbers --------------

    ftol_init    = inp_dict["ftol"]
    xtol_init    = inp_dict["xtol"]
    N_iter       = inp_dict["N_iter"]
    flag_verbose = inp_dict["flag_verbose"]

    if flag_verbose == "y":
        print("Total iteration =",N_iter)

    ##------------------------------------------------------------

    for iter_indx in range(N_iter):
        ftol_iter = ftol_init*10**(iter_indx*-2.0)
        xtol_iter = xtol_init*10**(iter_indx*-2.0)
        inp_dict.update({"xtol":xtol_iter,"ftol":ftol_iter,"iter_indx":iter_indx})
        if iter_indx == 0: 
            inp_dict.update({"param_fit":None})
        else:
            inp_dict.update({"param_fit":param_fit})

        save_dict        = STEP_2A_VIPER_FITTING_FOR_SINGLE_ITERATION(inp_dict)
        param_fit        = save_dict["param_fit"]
        chi_sq_arr       = save_dict["chi_sq_arr"]
        Flux_fit         = save_dict["Flux_fit"]
        Flux_inp         = inp_dict["Flux"]
        wavelength       = inp_dict["wavelength"]
        Flux_inp_fit_arr = np.array([wavelength,Flux_inp,Flux_fit]).T
    out_dict = {"param_fit":param_fit,"Flux_fit":Flux_inp_fit_arr,"chi_sq_arr":chi_sq_arr}
    return inp_dict,out_dict

#------------------------------------------------------------------------------


def VIPER_DEFAULT_PARAMETER():
    cosmo_param       = None
    lambda_rest_XI    = 1215.670
    wave_lb           = None
    wave_ub           = None
    ftol              = 1e-10
    xtol              = 1e-10
    lsf_type          = "Gaussian"
    sigma_v           = 2.0
    sigma_v_CSL       = 2.0
    N_iter            = 1
    crude_peak_cutoff = 1.5
    min_csl_cutoff    = 1.5
    log_NXI_lb        = 10.0
    log_NXI_ub        = 18.0
    b_lb              = None
    b_ub              = 150.0
    log_NXI_init      = 12.0
    b_init            = 8.0
    viper_inp_param   = vars()
    return viper_inp_param

################################################################################
# PEAK FINDING ALGORITHM ROUTINES (NO NEED TO CHANGE ANYTHING BELOW THIS LINE) #
################################################################################

def PEAK_VALLEY_FINDER(line_dict):
	W_by_sigma				=	line_dict["W_by_sigma"]
	maxtab, mintab 			=	peakdet(W_by_sigma,0.05)
	line_dict				=	{"maxtab":maxtab,"mintab":mintab,"W_by_sigma":W_by_sigma}
	return line_dict

#-------------------------------------------------

def DELETE_ZERO_N_COMPONENT_ROW(lb_ub_N_component):
	bool_arr	=	lb_ub_N_component[:,2]!=0
	return lb_ub_N_component[bool_arr]

#-------------------------------------------------

def FIND_NON_OVERLAPPING_REGION(peak_lb_ub_arr,W_by_sigma,min_csl_cutoff_inp,flag_divide_region="n"):
    """
    Suppose peak_lb_ub_arr is arranged in following way
    Column 1	-->>	Peak Index
    Column 2	-->>	Lower bound Index
    Column 3	-->>	Upper bound Index

    The uniques array from UNIQUE_ROWS_IN_ARRAY routine is arranged in following way
    a1 b1
    a2 b2
    a3 b3
    ....
    an bn

    The function returns the array with following structure
    Column 1	-->>	Lower bound index of the Region
    Column 2	-->>	Upper bound index of the Region
    Column 3	-->>	Number of components to be fitted in the Region

    min_csl_cutoff          =   max(min_csl_cutoff_inp,np.partition(CSL_uniques[:,1],5)[5])
    Above line ensures that the minimum number of regions in which total spectra divided is ~ Nth_element+1
    regions  This is important for high-z Ly-alpha forest simulation where many lines are blended and saturated
    If this is not used, the code will try to fit entire spectra at once which could be time consuming.
    np.partition(CSL_uniques[:,1],5)[5] ==>> Finds the 5th smallest element in array CSL_uniques[:,1]
    """
    uniques,line_counter	=	UNIQUE_ROWS_IN_ARRAY(peak_lb_ub_arr[:,1:]) # find unique (rows)  regions. Regions are sorted

    CSL_uniques             =   np.zeros(uniques.shape)
    CSL_uniques[:,0]        =   W_by_sigma[uniques[:,0]]
    CSL_uniques[:,1]        =   W_by_sigma[uniques[:,1]]

    # Comment below two lines and uncomment third line below if you do not 
    # want divide spectra in ~ Nth_element+1 regions by default
    if flag_divide_region == "y":
        Nth_element    =  5
        min_csl_cutoff =  max(min_csl_cutoff_inp,np.partition(CSL_uniques[:,1],Nth_element)[Nth_element])
        print("Forcefully dividing the region in 6 parts ...")
    elif flag_divide_region == "n":
        min_csl_cutoff =  min_csl_cutoff_inp
    # ------

    N_uniques				=	uniques.shape[0]	# Number of unique rows
    lb_ub_N_component       =   np.zeros((1,3),dtype=np.int64)
    indx1					=	0
    while indx1 < N_uniques:
        # ---------------------------------------------------------------------
        # Condition for isolated line 
        # ---------------------------------------------------------------------
        if CSL_uniques[indx1,1] <= min_csl_cutoff: 
            add_this_row       = np.array([uniques[indx1,0],uniques[indx1,1],line_counter[indx1]])
            lb_ub_N_component  = np.vstack([lb_ub_N_component,add_this_row])
            indx1	+=	1
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # Condition for blended line 
        # ---------------------------------------------------------------------
        else:                                      # else line may be blended
            if indx1 == N_uniques-1: # Boundary corrections at the end of the spectra for isolated line
                add_this_row       = np.array([uniques[indx1,0],W_by_sigma.shape[0]-1,line_counter[indx1]])
                lb_ub_N_component  = np.vstack([lb_ub_N_component,add_this_row])
                indx1	=	N_uniques
                break

            # ---------------------------------------------------------------------
            # Loop to check how many lines are blended 
            # ---------------------------------------------------------------------
            for indx2 in range(indx1 + 1,N_uniques):	# Loop to check if a_i <= a_j <= b_i
                line_counter[indx1] += line_counter[indx2]
                if (CSL_uniques[indx2,1] <= min_csl_cutoff) | (indx2 == N_uniques-1):
                    add_this_row      = np.array([uniques[indx1,0],uniques[indx2,1],line_counter[indx1]])
                    lb_ub_N_component = np.vstack([lb_ub_N_component,add_this_row])
                    indx1 = indx2 + 1
                    break
    return lb_ub_N_component[1:,:]	

#-------------------------------------------------

def VALLEY_INDEX_FINDER(line_dict,inp_dict,peak_indx,wav_z_arr_vel):
    """
    This routine brackets the peak within valleys.

    Suppose valley_wavelength is arranged in following way
    v1, v2, v3, v4, ...., v_n
    Suppose peak_wavelength is arranged in following way
    p1, p2, p3, p4, ...., p_m

    If v2, v3 encloses p2 then p2-valley_wavelength array gives
    +ve, +ve, -ve, -ve,  ...., -ve

    sub_arr_positive finds least positive value	i.e. v2
    sub_arr_negative finds maximum negative value i.e. v3
    """

    x_axis_data			=	inp_dict["x_axis_data"][:,wav_z_arr_vel] # Wavelength array
    all_valley_indx		=	np.int32(line_dict["mintab"][:,0])		# All minima value index

    # ----------------------------------------------------------------------------------
    # Following two lines are changed by Prakash (4 Jan 2018) to account for edge effects
    # ----------------------------------------------------------------------------------
    all_valley_indx     =   np.insert(all_valley_indx,0,0) # Syntax np.insert(arr,index before which to insert value,value to insert)
    all_valley_indx     =   np.append(all_valley_indx,x_axis_data.shape[0]-1) # Append value at the end 
    # ----------------------------------------------------------------------------------

    peak_lb_ub_arr		=	np.zeros((peak_indx.shape[0],3),dtype=np.int32)
    peak_lb_ub_arr[:,0]	=	np.int32(peak_indx)

    peak_wavelength		=	x_axis_data[peak_lb_ub_arr[:,0]]	# Peak position in wavelength
    valley_wavelength	=	x_axis_data[all_valley_indx]		# Valley position in wavelength

    for indx in range(peak_indx.shape[0]):
        sub_arr				=	peak_wavelength[indx] - valley_wavelength 
        sub_arr_positive	=	sub_arr.copy()	
        sub_arr_negative	=	sub_arr.copy()
        sub_arr_positive[sub_arr_positive < 0.0] 	= 	np.inf	
        sub_arr_negative[sub_arr_negative > 0.0] 	= 	-np.inf
        peak_lb_ub_arr[indx,1]		=	all_valley_indx[np.argmin(sub_arr_positive)]
        peak_lb_ub_arr[indx,2]		=	all_valley_indx[np.argmax(sub_arr_negative)]
        
        if peak_lb_ub_arr[indx,1] == peak_lb_ub_arr[indx,2]: # Boundary condition
            if peak_lb_ub_arr[indx,1] < (x_axis_data.shape[0]/2):
                peak_lb_ub_arr[indx,1]	=	0
            #elif peak_lb_ub_arr[indx,2] > (x_axis_data.shape[0]/2):	
            #	peak_lb_ub_arr[indx,2]	=	x_axis_data.shape[0]

    bool_arr	=	peak_lb_ub_arr[:,1] < peak_lb_ub_arr[:,2]
    return peak_lb_ub_arr[bool_arr]

#-------------------------------------------------

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    from numpy import NaN, Inf, arange, isscalar, asarray, array
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)


###############################################################################################
#### SIGNIFICANCE LEVEL CALCULATIONS ROUTINES (NO NEED TO CHANGE ANYTHING BELOW THIS LINE) ####
###############################################################################################

def CRUDE_SIGNIFICANCE_LEVEL(inp_dict):
    Flux       = inp_dict["Flux"]
    noise      = np.abs(inp_dict["noise"]) # Changed by Prakash 4 Jan 2018
    SNR        = np.median(1.0/noise)
    lsf_final  = inp_dict["lsf_final"]
    W_lambda   = (1.0 - Flux)
    Flux_conv  = convolve(W_lambda,lsf_final,mode = "wrap")
    W_by_sigma = Flux_conv * SNR
    line_dict  = {"W_by_sigma":W_by_sigma}
    return line_dict

#------------------------------------------------------------------------------

def CREATE_SNR_ARRAY_FOR_CSL(noise,N_pix_grp):
    #N_pix_grp  = 512
    #SNR        = CREATE_SNR_ARRAY_FOR_CSL(noise,N_pix_grp)
    N_pix_tot  = noise.shape[0]
    SNR_median = np.median(1.0/noise)
    SNR_arr    = np.ones(N_pix_tot)
    SNR_arr[:] = SNR_median

    if N_pix_tot > N_pix_grp:
        lb,ub = 0,N_pix_grp
        while (ub <= N_pix_tot+1):
            SNR_median     = np.median(1.0/noise[lb:ub])
            SNR_arr[lb:ub] = SNR_median
            lb            += N_pix_grp
            ub            += N_pix_grp
    return SNR_arr

#------------------------------------------------------------------------------

def CRUDE_SIGNIFICANCE_LEVEL_V2(inp_dict):
    Flux       = inp_dict["Flux"]
    noise      = inp_dict["noise"] # Changed by Prakash 4 Jan 2018
    lsf_final  = inp_dict["lsf_final"]
    W_lambda   = (1.0 - Flux)
    Flux_conv  = convolve(W_lambda,lsf_final,mode = "wrap")
    noise_conv = convolve(noise,lsf_final,mode = "wrap")
    W_by_sigma = Flux_conv / noise_conv
    line_dict  = {"W_by_sigma":W_by_sigma}
    return line_dict

#------------------------------------------------------------------------------

def APPLY_CRUDE_SL_CUTOFF(line_dict,inp_dict):
    wav_z_arr_vel   =   inp_dict["wav_z_arr_vel"]
    peak_cutoff     =   inp_dict["crude_peak_cutoff"]
    valley_cutoff   =   inp_dict["crude_valley_cutoff"]
    maxtab			=	line_dict["maxtab"]
    mintab			=	line_dict["mintab"]
    SL_arr			=	line_dict["W_by_sigma"]
    x_axis_data		=	inp_dict["x_axis_data"]
    wavelength		=	x_axis_data[:,wav_z_arr_vel]
    
    peak_idx_sort	=	np.where(peak_cutoff <= maxtab[:,1])[0]
    peak_idx		=	np.asarray(maxtab[peak_idx_sort,0],dtype=np.int64)
    peak_ampl		=	maxtab[peak_idx_sort,1]
    valley_idx_sort	=	np.where(valley_cutoff >= mintab[:,1])[0]
    valley_idx		=	np.asarray(mintab[valley_idx_sort,0],dtype=np.int64)
    valley_ampl		=	mintab[valley_idx_sort,1]
    valley_range	=	np.zeros((peak_idx.shape[0],3))
    for indx in range(peak_idx.shape[0]):
        diff_arr		=	wavelength[valley_idx] - wavelength[peak_idx[indx]]
        diff_negative	=	diff_arr[diff_arr <= 0]
        diff_positive	=	diff_arr[diff_arr >= 0]
        if diff_negative.size == 0:
            valley_range[indx,0]	=	wavelength[0]
        else:
            valley_range[indx,0]	=	wavelength[peak_idx[indx]] + np.amax(diff_negative)
        if diff_positive.size == 0:
            diff_positive 			=	wavelength[-1]
        else:
            valley_range[indx,1]	=	wavelength[peak_idx[indx]] + np.amin(diff_positive)
        valley_range[indx,2]	=	SL_arr[peak_idx[indx]]
    peak_info		=	np.ones((peak_idx.shape[0],3))
    peak_info[:,0]	=	wavelength[peak_idx]
    peak_info[:,1]	=	peak_ampl
    IDX_Flux	=	np.zeros(peak_info.shape[0],dtype=np.int32)
    for indx in range(IDX_Flux.shape[0]):
        IDX_Flux[indx]	=	np.argmin(np.abs(inp_dict["x_axis_data"][:,wav_z_arr_vel] - peak_info[indx,0]))

    return IDX_Flux #peak_info,valley_range

#------------------------------------------------------------------------------

def dv_to_dx_conversion(b,lambda_c,light_speed):
    dx  = b * lambda_c / light_speed
    return dx

#------------------------------------------------------------------------------

def dx_to_dv_conversion(dx,lambda_c,light_speed):
    dv  = dx * light_speed / lambda_c
    return dv

#------------------------------------------------------------------------------

def RIGOROUS_SIGNIFICANCE_LEVEL(param_fit,inp_dict,Flux): 
    #Flux               = inp_dict["Flux"]
    wavelength         = inp_dict["x_axis_data"][:,0]
    noise              = inp_dict["noise"]
    SNR                = 1.0 / noise
    light_speed        = 3e5 # in km/s
    lambda_arr         = param_fit[:,0]
    b_val              = param_fit[:,2]
    param_fit_SL       = np.zeros((param_fit.shape[0],4))
    param_fit_SL[:,:3] = param_fit[:,:]
    for indx in range(param_fit.shape[0]):
        dx                   = dv_to_dx_conversion(b_val[indx],lambda_arr[indx],light_speed)
        lb                   = np.argmin(np.abs(wavelength-(lambda_arr[indx] - dx*2.0)))
        ub                   = np.argmin(np.abs(wavelength-(lambda_arr[indx] + dx*2.0)))
        W_lim                = trapz((1.0 - Flux[lb:ub])*SNR[lb:ub],wavelength[lb:ub]) # Changed by Prakash 4 Jan 2018
        param_fit_SL[indx,3] = W_lim  # np.median((2.0 - Flux[lb:ub])*SNR[lb:ub])

    return param_fit_SL

#------------------------------------------------------------------------------

def RIGOROUS_SIGNIFICANCE_LEVEL_OD_WEIGHTED(param_fit,inp_dict): 
    Flux               = inp_dict["Flux"]
    wav_z_arr_vel      = inp_dict["wav_z_arr_vel"]
    wavelength         = inp_dict["x_axis_data"][:,0]
    #SNR                = inp_dict["SNR"]
    noise              = inp_dict["noise"]
    SNR                = 1.0 / noise
    light_speed        = 3e5 # in km/s
    lambda_arr         = param_fit[:,0]
    log_NXI_arr        = param_fit[:,1]
    b_val              = param_fit[:,2]

    tau_total          = VOIGT_PROFILE_PARAM_FIT_OD(param_fit,inp_dict,wav_z_arr_vel) # Calculate total tau at all pixels
    param_fit_SL       = np.zeros((param_fit.shape[0],4))
    param_fit_SL[:,:3] = param_fit[:,:]
    for indx in range(param_fit.shape[0]):
        param_fit_line       = np.array([[lambda_arr[indx],log_NXI_arr[indx],b_val[indx]]])
        tau_line             = VOIGT_PROFILE_PARAM_FIT_OD(param_fit_line,inp_dict,wav_z_arr_vel) # Calculate tau due to line at all pixels

        dx                   = dv_to_dx_conversion(b_val[indx],lambda_arr[indx],light_speed)
        lb                   = np.argmin(np.abs(wavelength-(lambda_arr[indx] - dx*2.0)))
        ub                   = np.argmin(np.abs(wavelength-(lambda_arr[indx] + dx*2.0)))
        tau_total_cpy        = tau_total.copy()
        tau_total_cpy[tau_total_cpy < 1e-5] = 1e-5 # To avoide division by very small
        tau_weights          = tau_line / tau_total_cpy 
        tau_weights[tau_weights > 1.0] = 0.0 # Weights can not be larger than 1
        SNR_median           = np.median(SNR[lb:ub])
        W_lim                = trapz((1.0 - Flux[lb:ub]*tau_weights[lb:ub])*SNR_median,wavelength[lb:ub]) # Changed by Prakash 4 Jan 2018
        if ub == lb:
            param_fit_SL[indx,3] = 0.0 
        else:
            param_fit_SL[indx,3] = (W_lim) / ((float(ub-lb))**0.5 * dx)

    return param_fit_SL

#------------------------------------------------------------------------------

def RIGOROUS_SIGNIFICANCE_LEVEL_BACKUP_PREVIOUSLY_GIVES_HIGH_RSL_IN_OBSERVATION(param_fit,inp_dict): 
    Flux               = inp_dict["Flux"]
    wavelength         = inp_dict["x_axis_data"][:,0]
    #SNR                = inp_dict["SNR"]
    noise              = inp_dict["noise"]
    SNR                = 1.0 / noise
    light_speed        = 3e5 # in km/s
    lambda_arr         = param_fit[:,0]
    b_val              = param_fit[:,2]
    param_fit_SL       = np.zeros((param_fit.shape[0],4))
    param_fit_SL[:,:3] = param_fit[:,:]
    for indx in range(param_fit.shape[0]):
        dx                   = dv_to_dx_conversion(b_val[indx],lambda_arr[indx],light_speed)
        lb                   = np.argmin(np.abs(wavelength-(lambda_arr[indx] - dx*2.0)))
        ub                   = np.argmin(np.abs(wavelength-(lambda_arr[indx] + dx*2.0)))
        W_lim                = trapz((1.0 - Flux[lb:ub])*SNR[lb:ub],wavelength[lb:ub]) # Changed by Prakash 4 Jan 2018
        if ub == lb:
            param_fit_SL[indx,3] = 0.0
        else:
            param_fit_SL[indx,3] = (W_lim) / ((float(ub-lb))**0.5 * dx)

    return param_fit_SL

#------------------------------------------------------------------------------

def RIGOROUS_SIGNIFICANCE_LEVEL_BACKUP_PREVIOUSLY_USED_FOR_SIMULATIONS(param_fit,inp_dict): 
    Flux               = inp_dict["Flux"]
    wavelength         = inp_dict["x_axis_data"][:,0]
    SNR                = inp_dict["SNR"]
    light_speed        = 3e5 # in km/s
    lambda_arr         = param_fit[:,0]
    b_val              = param_fit[:,2]
    param_fit_SL       = np.zeros((param_fit.shape[0],4))
    param_fit_SL[:,:3] = param_fit[:,:]
    for indx in range(param_fit.shape[0]):
        dx                   = b_val[indx] * lambda_arr[indx] / light_speed
        lb                   = np.argmin(np.abs(wavelength-(lambda_arr[indx] - dx)))
        ub                   = np.argmin(np.abs(wavelength-(lambda_arr[indx] + dx)))
        W_lim                = trapz(1.0 - Flux[lb:ub],wavelength[lb:ub])
        param_fit_SL[indx,3] = (W_lim * SNR) / ((float(ub-lb))**0.5 * dx)

    return param_fit_SL

#####################################################################################
#### VOIGT PROFILE FITTING ROUTINES (NO NEED TO CHANGE ANYTHING BELOW THIS LINE) ####
#####################################################################################


def VOIGT_PROFILE(wavelength,lambda_c_arr,log_N_XI_arr,b_arr,lsf,I_alpha,damp_gamma,lambda_rest_XI):
    """	Assuming lambd_c, N_XI and b are array of same dimension """
    def voigt(x, y):
        from scipy.special import wofz # Faddeeva function
        z = x + 1j*y
        I = wofz(z).real
        return I
    #I_alpha             = 4.45e-18
    #damp_gamma          = 6.265e8 #1.5e9

    #=====================================================
    # Modified by Prakash on 20 June 2018 to avoid overflow warning
    log_N_XI_max        = 30.0
    bool_arr            = log_N_XI_arr < log_N_XI_max 
    N_XI_arr            = np.zeros(log_N_XI_arr.shape)
    N_XI_arr[bool_arr]  = 10.0**log_N_XI_arr[bool_arr]
    N_XI_arr[~bool_arr] = 10.0**log_N_XI_max
    #=====================================================


    light_speed         = 3e5
    tau                 = np.zeros(wavelength.shape[0])
    for line_indx in range(lambda_c_arr.shape[0]):
        B			=	light_speed * (wavelength - lambda_c_arr[line_indx])  / (wavelength * b_arr[line_indx])
        A			=	lambda_rest_XI * damp_gamma * 1e-13 / (4 * np.pi * b_arr[line_indx])	# Changed by Prakash on 25 Feb 2022
        tau_c		=	light_speed * N_XI_arr[line_indx] * I_alpha / (b_arr[line_indx] * np.pi**0.5)
        tau			+=	tau_c  * voigt(B,A)

    if np.any(lsf) == None:
        Flux =	np.exp(-1*tau)
    else:
        Flux =	convolve(np.exp(-1*tau),lsf,mode="wrap") #np.exp(-1*convolve(tau,lsf,mode="wrap"))  #np.exp(-1*tau) 

    return Flux

#------------------------------------------------------------------------------

def VOIGT_PROFILE_OD(wavelength,lambda_c_arr,log_N_XI_arr,b_arr,lsf=None,I_alpha=4.45e-18,damp_gamma=6.265e8,lambda_rest_XI=1215.6701):
    """	Assuming lambd_c, N_XI and b are array of same dimension """
    def voigt(x, y):
        from scipy.special import wofz # Faddeeva function
        z = x + 1j*y
        I = wofz(z).real
        return I
    #I_alpha             = 4.45e-18
    #damp_gamma          = 6.265e8 #1.5e9

    #=====================================================
    # Modified by Prakash on 20 June 2018 to avoid overflow warning
    log_N_XI_max        = 30.0
    bool_arr            = log_N_XI_arr < log_N_XI_max 
    N_XI_arr            = np.zeros(log_N_XI_arr.shape)
    N_XI_arr[bool_arr]  = 10.0**log_N_XI_arr[bool_arr]
    N_XI_arr[~bool_arr] = 10.0**log_N_XI_max
    #=====================================================


    light_speed         = 3e5
    tau                 = np.zeros(wavelength.shape[0])
    for line_indx in range(lambda_c_arr.shape[0]):	
        B			=	light_speed * (wavelength - lambda_c_arr[line_indx])  / (wavelength * b_arr[line_indx])
        A			=	lambda_rest_XI * damp_gamma * 1e-13 / (4 * np.pi * b_arr[line_indx]) # Changed by Prakash on 25 Feb 2022
        tau_c		=	light_speed * N_XI_arr[line_indx] * I_alpha / (b_arr[line_indx] * np.pi**0.5)
        tau			+=	tau_c  * voigt(B,A)

    if np.any(lsf) == None:
        tau_conv =	tau.copy()
    else:
        tau_conv =	convolve(tau,lsf,mode="wrap")
    return tau_conv

#------------------------------------------------------------------------------

def VOIGT_PROFILE_PARAM_FIT(param_fit,inp_dict,wav_z_arr_vel):
    wavelength		=	inp_dict["x_axis_data"][:,wav_z_arr_vel]
    I_alpha         =   inp_dict["I_alpha"]
    damp_gamma      =   inp_dict["damp_gamma"]
    lambda_rest_XI  =   inp_dict["lambda_rest_XI"]
    lsf				=	inp_dict["lsf"]
    lambda_c_arr	=	param_fit[:,0]
    log_N_XI_arr	=	param_fit[:,1]
    b_arr			=	param_fit[:,2]
    Flux_fit		=	VOIGT_PROFILE(wavelength,lambda_c_arr,log_N_XI_arr,b_arr,lsf,I_alpha,damp_gamma,lambda_rest_XI)
    return Flux_fit

#------------------------------------------------------------------------------

def VOIGT_PROFILE_PARAM_FIT_OD(param_fit,inp_dict,wav_z_arr_vel):
	wavelength		=	inp_dict["x_axis_data"][:,wav_z_arr_vel]
	lsf				=	inp_dict["lsf"]
	lambda_c_arr	=	param_fit[:,0]
	log_N_XI_arr	=	param_fit[:,1]
	b_arr			=	param_fit[:,2]
	OD_fit	    	=	VOIGT_PROFILE_OD(wavelength,lambda_c_arr,log_N_XI_arr,b_arr,lsf)
	return OD_fit

#------------------------------------------------------------------------------

def residuals(param_flattened,Flux_actual,wavelength,lsf,I_alpha,damp_gamma,lambda_rest_XI,noise):
	param					= 	np.reshape(param_flattened,(len(param_flattened)//3,3))
	lambda_c_arr			=	np.abs(param[:,0])
	log_N_XI_arr			=	np.abs(param[:,1])
	b_arr					=	np.abs(param[:,2])
	Flux_Fit				=	VOIGT_PROFILE(wavelength,lambda_c_arr,log_N_XI_arr,b_arr,lsf,I_alpha,lambda_rest_XI,damp_gamma)
	return np.abs(Flux_actual - Flux_Fit) / noise

#------------------------------------------------------------------------------

def max_residuals(param_flattened,Flux_actual,wavelength,lsf,I_alpha,damp_gamma,lambda_rest_XI,noise):
	param					= 	np.reshape(param_flattened,(len(param_flattened)//3,3))
	lambda_c_arr			=	np.abs(param[:,0])
	log_N_XI_arr			=	np.abs(param[:,1])
	b_arr					=	np.abs(param[:,2])
	Flux_Fit				=	VOIGT_PROFILE(wavelength,lambda_c_arr,log_N_XI_arr,b_arr,lsf,I_alpha,damp_gamma,lambda_rest_XI)
	return np.amax(np.abs(1.0+Flux_actual - Flux_Fit))

#------------------------------------------------------------------------------

def AICC_CORRECTION(chi_sq_val,Flux,p):
	n			=	Flux.shape[0]
	if n-p-1.0 > 0:
		AICC_corr	=	2.0 * p * (p + 1.0) / (n - p - 1.0)
	else:
		AICC_corr	=	100000.0
	AICC		=	chi_sq_val + 2 * p + AICC_corr
	return AICC

#------------------------------------------------------------------------------

def bounds_array_calc(param,wavelength,b_lb,b_ub,log_NXI_lb,log_NXI_ub):
    bounds_arr	=	np.zeros((param.size,2))
    kounter		=	0
    for row_indx in range(param.shape[0]):
        for col_indx in range(param.shape[1]):

            if col_indx == 0:
                bounds_arr[kounter,0]	=	min(wavelength[0],np.amin(param[:,0]))
                bounds_arr[kounter,1]	=	max(wavelength[-1],np.amax(param[:,0]))
            elif col_indx == 1:
                bounds_arr[kounter,0]	=	min(log_NXI_lb,np.amin(param[:,1])) # Changed by Prakash 4 Jan 2018
                bounds_arr[kounter,1]	=	max(log_NXI_ub,np.amax(param[:,1])) # Changed by Prakash 4 Jan 2018
            else:
                bounds_arr[kounter,0]	=	min(b_lb,np.amin(param[:,2])) # Changed by Prakash 4 Jan 2018
                bounds_arr[kounter,1]	=	max(b_ub,np.amax(param[:,2])) # Changed by Prakash 4 Jan 2018
            kounter	+=	1

    return bounds_arr

#------------------------------------------------------------------------------

def voigt_lsq_fit(inp_dict,lb_ub_N_component,flag_AICC="AICC",flag_verbose="y"):
    Wavelength     = inp_dict["x_axis_data"][:,0]
    lsf            = inp_dict["lsf"]
    I_alpha        = inp_dict["I_alpha"]
    damp_gamma     = inp_dict["damp_gamma"]
    lambda_rest_XI = inp_dict["lambda_rest_XI"]
    chi_sq_arr     = np.zeros(lb_ub_N_component.shape[0])
    param_fit      = np.zeros((1,3))
    param_err      = np.zeros((1,3))
    iter_indx      = inp_dict["iter_indx"]

    if flag_verbose == "y":
        print("Total Region :",lb_ub_N_component.shape[0])

    for N_indx in range(lb_ub_N_component.shape[0]):
        lb_indx         = lb_ub_N_component[N_indx,0]
        ub_indx         = lb_ub_N_component[N_indx,1]
        N_component     = lb_ub_N_component[N_indx,2]

        param_fit_indx,param_err_indx,chi_sq_min 	= 	voigt_lsq_fit_single_region(inp_dict,lb_indx,ub_indx,N_component,None,flag_AICC)

        if flag_verbose == "y":
            print("Region :",N_indx+1,", Estimated component :",N_component,", Fitted componenets :",param_fit_indx.shape[0],", Chi-sq minimum :",chi_sq_min)
        #print "Region :",N_indx+1,", Wavelength_lb :",Wavelength[lb_indx],", Wavelength_ub :",Wavelength[ub_indx],", Chi-sq minimum :",chi_sq_min
        #print "Region :",N_indx+1,", Estimated component :",N_component,", Fitted componenets :",param_fit_indx.shape[0],"For iteration :",iter_indx

        # Following lines are modified on 23 March 2020
        if chi_sq_min > 5.0:

            if flag_verbose == "y":
                print("Reduced chi_sq = ",chi_sq_min," is high. Refitting spectra ...")

            param_fit_indx,param_err_indx,chi_sq_min 	= 	voigt_lsq_fit_single_region(inp_dict,lb_indx,ub_indx,N_component,N_component,"chi_sq")

            if flag_verbose == "y":
                print("Region :",N_indx+1,", Estimated component :",N_component,", Fitted componenets :",param_fit_indx.shape[0],", Chi-sq minimum :",chi_sq_min)

        chi_sq_arr[N_indx]      =       chi_sq_min
        if N_indx == 0:
            param_fit	=	param_fit_indx.copy()
            param_err	=	param_err_indx.copy()
        else:
            param_fit	=	np.vstack([param_fit,param_fit_indx])
            param_err	=	np.vstack([param_err,param_err_indx])
    Flux_fit = VOIGT_PROFILE(Wavelength,param_fit[:,0],param_fit[:,1],param_fit[:,2],lsf,I_alpha,damp_gamma,lambda_rest_XI)
    out_dict = {"Wavelength":Wavelength,"Flux_fit":Flux_fit,"param_fit":param_fit,"param_err":param_err,"chi_sq_arr":chi_sq_arr}
    return out_dict

#------------------------------------------------------------------------------

def log_NXI_guess_value_calculation(Flux,delta_v,N_param,f_lu,lambda_rest_XI):
    """
    log_NXI is calculated using an approximation known as 
    Apperent Optical Depth (AOD, Savage+1991,Fox+2005) 
    http://iopscience.iop.org/article/10.1086/491735/pdf
    """
    N_pixels         = Flux.shape[0]
    tau              = -1.0 * np.log(np.abs(Flux))
    #f_lu             = 0.416
    #lambda_rest_XI   = 1215.6701
    NXI_init_arr     = 3.768e14 * tau / (f_lu * lambda_rest_XI)
    NXI_init_tot     = np.sum(NXI_init_arr * delta_v)
    
    if (NXI_init_tot / N_param) <= 0.0:
        log_NXI_init_tot = 14.0
    else:
        log_NXI_init_tot = log10(NXI_init_tot / N_param)
    return log_NXI_init_tot

#------------------------------------------------------------------------------

def read_log_NXI_b_parameter_bounds_and_guess_values(inp_dict,Flux,delta_v,N_param_tot):
    b_lb           = inp_dict["b_lb"]
    b_ub           = inp_dict["b_ub"]
    log_NXI_lb     = inp_dict["log_NXI_lb"]
    log_NXI_ub     = inp_dict["log_NXI_ub"]
    log_NXI_init   = inp_dict["log_NXI_init"]
    b_init         = inp_dict["b_init"]
    f_lu           = inp_dict["f_lu"]
    lambda_rest_XI = inp_dict["lambda_rest_XI"]
    N_pixels       = Flux.shape[0]

    if b_lb == None:
        b_lb = delta_v / 3.0

    if b_ub == None:
        b_ub = delta_v * N_pixels * 1.5

    if log_NXI_lb == None:
        log_NXI_lb = 9.0

    if log_NXI_ub == None:
        log_NXI_ub = 18.0

    if log_NXI_init == None:
        log_NXI_init = log_NXI_guess_value_calculation(Flux,delta_v,N_param_tot,f_lu,lambda_rest_XI)

    if b_init == None:
        b_init = delta_v * 3.0 #(b_ub - b_lb) / N_param_tot

    return log_NXI_lb,log_NXI_ub,b_lb,b_ub,log_NXI_init,b_init

#------------------------------------------------------------------------------

def voigt_lsq_fit_single_region(inp_dict,lb_indx,ub_indx,N_component,N_AICC_compare=None,flag_AICC="AICC"):
    Flux			=	inp_dict["Flux"]
    Wavelength		=	inp_dict["x_axis_data"][:,0]
    lsf				=	inp_dict["lsf"]
    I_alpha         =   inp_dict["I_alpha"]
    damp_gamma      =   inp_dict["damp_gamma"]
    lambda_rest_XI  =   inp_dict["lambda_rest_XI"]
    noise			=	inp_dict["noise"]
    ftol            =   inp_dict["ftol"]
    xtol            =   inp_dict["xtol"]
    f_lu            =   inp_dict["f_lu"]
    lambda_rest_XI  =   inp_dict["lambda_rest_XI"]
    lb				=	Wavelength[lb_indx] #- 1.5
    ub				=	Wavelength[ub_indx] #+ 1.5
    xdata 			= 	Wavelength[(lb<=Wavelength)&(Wavelength<=ub)]
    ydata 			= 	Flux[(lb<=Wavelength)&(Wavelength<=ub)]
    noise_data		=	noise[(lb<=Wavelength)&(Wavelength<=ub)]
    N_pixels		=	xdata.shape[0]

    # Setting lower bound on b values equal to pixel seperation
    light_speed     =   3e5
    lambda_c        =   np.median(xdata)
    dx              =   np.median(xdata[1:]-xdata[:-1])

    delta_v         =   dx_to_dv_conversion(dx,lambda_c,light_speed)


    #---------------------------------------------------------
    
    iter_indx = inp_dict["iter_indx"]
    if iter_indx == 0:
        N_start              = 1
        N_end                = min(int(N_pixels/3)-1,N_component + 9) # Modified by Prakash 20 June 2018
        if N_end <= N_start: N_end = N_start+1
        N_param_tot          = N_end - N_start + 3
        SNR_median           = np.median(1.0 / noise_data)
        noise_data[:]        = 1.0 / SNR_median
    else:
        param_fit_prev_all = inp_dict["param_fit"]
        bool_arr_prev      = (lb <= param_fit_prev_all[:,0]) & (param_fit_prev_all[:,0] <= ub)
        param_fit_prev     = param_fit_prev_all[bool_arr_prev,:].copy()
        N_component        = param_fit_prev.shape[0]
        N_start            = N_component - 5
        N_end              = min(int(N_pixels/3 -1),N_component + 5) # Modified by Prakash 20 June 2018
        N_param_tot        = N_end - N_start + 3
        if N_start < 0: N_start = 1
        if N_end <= N_start: N_end = N_start+1
        SNR_median         = np.mean(1.0 / noise_data)
        noise_data[:]      = 1.0 / SNR_median

    log_NXI_lb,log_NXI_ub,b_lb,b_ub,log_NXI_init,b_init = read_log_NXI_b_parameter_bounds_and_guess_values(inp_dict,ydata,delta_v,N_param_tot)

    #print "N_start =",N_start,", N_end =",N_end

    AICC        = np.ones(N_end-N_start) * 1e10
    plsq_dict   = {}
    chi_sq_arr  = np.ones(N_end-N_start) * 1e10
    chi_sq_flag = 0
    indx        = 0
    for indx in range(N_end-N_start):
        N_param			=	indx + N_start	#int(raw_input("Enter number of Voigt Components : "))
        wave_guess		=	np.linspace(lb,ub,N_param+2)[1:-1]
        param           =   generate_guess_value_using_gaussian_fit(xdata,ydata,lsf,noise_data,delta_v,N_param,wave_guess,ftol,xtol,f_lu,lambda_rest_XI)
        bounds			=	bounds_array_calc(param,xdata,b_lb,b_ub,log_NXI_lb,log_NXI_ub)
        plsq 			=	leastsq_bounds(residuals,param.flatten(),bounds,boundsweight=10,args=(ydata,xdata,lsf,I_alpha,damp_gamma,lambda_rest_XI,noise_data),ftol=ftol,xtol=xtol,full_output=1)
        
        chi_arr			=	residuals(plsq[0],ydata,xdata,lsf,I_alpha,damp_gamma,lambda_rest_XI,noise_data)
        chi_sq_val		=	np.sum(chi_arr*chi_arr)
        chi_sq_arr[indx]=	chi_sq_val / float(N_pixels - 3.0*N_param)
        
        if np.any(plsq[1]) != None:
            covar_matrix	=	np.asarray(plsq[1]) * chi_sq_arr[indx]
            fit_err_diag	=	np.diag(np.abs(covar_matrix))**0.5

        else:
            # Added by Prakash on 20 June 2018 to Circumvent the problem of zeros
            # The Hessian Matrix = 2*D has at least one eigenvalue as zero.
            # Hence the inverese does not exists.
            # We circumvent this problem by calculating Psuedo-Inverse of Matrix 
            # using Moore-Penrose technique
            try:
                perm             = np.take(np.eye(N_pixels), plsq[2]['ipvt'] - 1, 0)
                r                = np.triu(np.transpose(plsq[2]['fjac'])[:N_pixels, :])
                R                = np.dot(r, perm)
                D                = np.dot(np.transpose(R), R)
                cov_x            = np.dual.pinv(D) # cov_x is analgous to output of leastsq routine i.e., plsq[1]
                covar_matrix     = cov_x * chi_sq_arr[indx]
                fit_err_diag_all = np.diag(np.abs(covar_matrix))**0.5
                fit_err_diag     = fit_err_diag_all[:N_param*3]
            except:
                fit_err_diag    =   np.zeros(param.size)
                print("Can not compute Errors on fitted parameters !!!")

        AICC[indx]		=	AICC_CORRECTION(chi_sq_val,ydata,len(plsq[0]))
        plsq_dict.update({indx:(plsq[0],fit_err_diag,N_param)})
        # Compare if AICC is consistently higher above a given N_voigt points.
        if N_AICC_compare is None:
            N_AICC_compare = 7 #7 Changed by Prakash on 22 Jun 2020 from 7 to 5 to avoid overfitting.
        kounter        = 0
        if indx >= N_AICC_compare:
            for indx2 in range(N_AICC_compare):
                if AICC[indx-N_AICC_compare] < AICC[indx-indx2]: 
                    kounter += 1
        #print indx,AICC[indx],kounter,N_AICC_compare
        if kounter == N_AICC_compare: break


    AICC[AICC<=0.0]	=	np.inf

    if flag_AICC == "AICC":
        indx_min   = np.argmin(AICC)
        chi_sq_min = chi_sq_arr[indx_min]
        if chi_sq_min > 5.0:
            indx_min = np.argmin(chi_sq_arr)

    elif flag_AICC == "chi_sq":
        indx_min	=	np.argmin(chi_sq_arr)

    indx_min        =   min(indx_min,len(plsq_dict)-1)
    plsq_min		=	plsq_dict[indx_min][0]
    plsq_min_error	=	plsq_dict[indx_min][1]
    N_param     	=	plsq_dict[indx_min][2]
    chi_sq_min		=	chi_sq_arr[indx_min]
    AICC_min		=	AICC[indx_min]				
    param_fit 		= 	np.reshape(np.abs(plsq_min),(len(plsq_min)//3,3))
    param_error		= 	np.reshape(np.abs(plsq_min_error),(len(plsq_min_error)//3,3))
    return param_fit,param_error,chi_sq_min


#--------------- CHI SQUARE VALUE IN A REGION ------------------

def chi_sq_in_region(inp_dict,lb_ub_N_component,chi_sq_arr):
    Wavelength     = inp_dict["x_axis_data"][:,0]
    chi_sq_arr_new = np.zeros((lb_ub_N_component.shape[0],3))
    for N_indx in range(lb_ub_N_component.shape[0]):
        lb_indx                  = lb_ub_N_component[N_indx,0]
        ub_indx                  = lb_ub_N_component[N_indx,1]
        chi_sq_arr_new[N_indx,0] = Wavelength[lb_indx]
        chi_sq_arr_new[N_indx,1] = Wavelength[ub_indx]
        chi_sq_arr_new[N_indx,2] = chi_sq_arr[N_indx]
    return chi_sq_arr_new

###################################################################################
#### INITIAL GUESS VALUE ROUTINES (NO NEED TO CHANGE ANYTHING BELOW THIS LINE) ####
###################################################################################

def GAUSSIAN_PROFILE(wavelength,lambda_c_arr,tau_0_arr,b_arr,lsf):
    """	Assuming lambd_c, N_XI and b are array of same dimension """
    light_speed         = 3e5
    tau                 = np.zeros(wavelength.shape[0])
    for line_indx in range(lambda_c_arr.shape[0]):	
        B			=	light_speed * (wavelength - lambda_c_arr[line_indx])  / (wavelength * b_arr[line_indx])
        tau_c		=	tau_0_arr[line_indx] * np.exp(-B**2.0)
        tau			+=	tau_c
    Flux =	np.exp(-1*tau)
    return Flux

#------------------------------------------------------------------------------

def gaussian_residuals(param_flattened,Flux_actual,wavelength,lsf,noise):
	param					= 	np.reshape(param_flattened,(len(param_flattened)//3,3))
	lambda_c_arr			=	np.abs(param[:,0])
	tau_0_arr			    =	np.abs(param[:,1])
	b_arr					=	np.abs(param[:,2])
	Flux_Fit				=	GAUSSIAN_PROFILE(wavelength,lambda_c_arr,tau_0_arr,b_arr,lsf)
	return np.abs(Flux_actual - Flux_Fit) / noise

#------------------------------------------------------------------------------

def bounds_array_calc_guess(param,wavelength,b_lb,b_ub,log_NXI_lb,log_NXI_ub):
	bounds_arr	=	np.zeros((param.size,2))
	kounter		=	0
	for row_indx in range(param.shape[0]):
		for col_indx in range(param.shape[1]):
			if col_indx == 0:
				bounds_arr[kounter,0]	=	wavelength[0]
				bounds_arr[kounter,1]	=	wavelength[-1]
			elif col_indx == 1:
				bounds_arr[kounter,0]	=	log_NXI_lb # Changed by Prakash 4 Jan 2018
				bounds_arr[kounter,1]	=	log_NXI_ub # Changed by Prakash 4 Jan 2018
			else:
				bounds_arr[kounter,0]	=	b_lb # Changed by Prakash 4 Jan 2018
				bounds_arr[kounter,1]	=	b_ub # Changed by Prakash 4 Jan 2018
			kounter	+=	1
	return bounds_arr


#------------------------------------------------------------------------------

def log_NXI_guess_value_calculation(tau_0_arr,b_arr,f_lu,lambda_rest_XI):
    """
    log_NXI is calculated using an approximation known as 
    Apperent Optical Depth (AOD, Savage+1991,Fox+2005) 
    http://iopscience.iop.org/article/10.1086/491735/pdf
    """
    tau_sum          = tau_0_arr * b_arr * 2.0
    #f_lu            = 0.416
    #lambda_rest_XI  = 1215.6701
    NXI_init_arr     = 3.768e14 * tau_sum / (f_lu * lambda_rest_XI)
    log_NXI_arr      = np.log10(NXI_init_arr)
    return log_NXI_arr

#------------------------------------------------------------------------------


def read_tau_b_parameter_bounds_and_guess_values(Flux,delta_v):
    N_pixels   = Flux.shape[0]
    b_init     = delta_v / 2.0 #(b_ub - b_lb) / N_param_tot
    b_lb       = delta_v * 0.1
    b_ub       = delta_v * N_pixels * 1.5
    tau_0_lb   = 0.0
    tau_0_ub   = 1e8
    tau_0_init = 1.0
    return tau_0_lb,tau_0_ub,b_lb,b_ub,tau_0_init,b_init

#------------------------------------------------------------------------------

def generate_guess_value_using_gaussian_fit(xdata,ydata,lsf,noise_data,delta_v,N_param,wave_guess,ftol,xtol,f_lu,lambda_rest_XI):
    #print "Estimating Initial Guess values ... "
    init_val_tup    = read_tau_b_parameter_bounds_and_guess_values(ydata,delta_v)
    tau_0_lb        = init_val_tup[0]
    tau_0_ub        = init_val_tup[1]
    b_lb            = init_val_tup[2]
    b_ub            = init_val_tup[3]
    tau_0_init      = init_val_tup[4]
    b_init          = init_val_tup[5]
    param			=	np.zeros((N_param,3))
    param[:,2]		=	b_init         # Guess values for doppler b parameter
    param[:,1]		=	tau_0_init   # param[:,1] -->> Flux at line center; Get guess N_XI
    param[:,0]		=   wave_guess	
    bounds			=	bounds_array_calc_guess(param,xdata,b_lb,b_ub,tau_0_lb,tau_0_ub)
    plsq 			=	leastsq_bounds(gaussian_residuals,param.flatten(),bounds,boundsweight=10,args=(ydata,xdata,lsf,noise_data),ftol=ftol,xtol=xtol,full_output=1)
    param_fit 		= 	np.reshape(np.abs(plsq[0]),(len(plsq[0])//3,3))
    log_NXI_arr     = log_NXI_guess_value_calculation(param_fit[:,1],param_fit[:,2],f_lu,lambda_rest_XI)
    param_fit[:,1]  = log_NXI_arr.copy()
    #print "Initial Guess value estimated ..."
    return param_fit

##################################@#################################################
#### BOUNDED LEAST SQUARE ROUTINES (NO NEED TO CHANGE ANYTHING BELOW THIS LINE) ####
###################################@################################################

def leastsq_bounds( func, x0, bounds, boundsweight=10, **kwargs ):
    """ leastsq with bound conatraints lo <= p <= hi
    run leastsq with additional constraints to minimize the sum of squares of
        [func(p) ...]
        + boundsweight * [max( lo_i - p_i, 0, p_i - hi_i ) ...]

    Parameters
    ----------
    func() : a list of function of parameters `p`, [err0 err1 ...]
    bounds : an n x 2 list or array `[[lo_0,hi_0], [lo_1, hi_1] ...]`.
        Use e.g. [0, inf]; do not use NaNs.
        A bound e.g. [2,2] pins that x_j == 2.
    boundsweight : weights the bounds constraints
    kwargs : keyword args passed on to leastsq

    Returns
    -------
    exactly as for leastsq,
http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html

    Notes
    -----
    The bounds may not be met if boundsweight is too small;
    check that with e.g. check_bounds( p, bounds ) below.

    To access `x` in `func(p)`, `def func( p, x=xouter )`
    or make it global, or `self.x` in a class.

    There are quite a few methods for box constraints;
    you'll maybe sing a longer song ...
    Comments are welcome, test cases most welcome.

"""
    # Example: test_leastsq_bounds.py

    if bounds is not None  and  boundsweight > 0:
        check_bounds( x0, bounds )
        if "args" in kwargs:  # 8jan 2015
            args = kwargs["args"]
            del kwargs["args"]
        else:
            args = ()
#...............................................................................
        funcbox = lambda p: \
            np.hstack(( func( p, *args ),
                        _inbox( p, bounds, boundsweight ))) 
    else:
        funcbox = func
    return leastsq( funcbox, x0, **kwargs )


def _inbox( X, box, weight=1 ):
    """ -> [tub( Xj, loj, hij ) ... ]
        all 0  <=>  X in box, lo <= X <= hi
    """
    assert len(X) == len(box), \
        "len X %d != len box %d" % (len(X), len(box))
    return weight * np.array([
        np.fmax( lo - x, 0 ) + np.fmax( 0, x - hi )
            for x, (lo,hi) in zip( X, box )])

# def tub( x, lo, hi ):
#     """ \___/  down to lo, 0 lo .. hi, up from hi """
#     return np.fmax( lo - x, 0 ) + np.fmax( 0, x - hi )

#...............................................................................
def check_bounds( X, box ):
    """ print Xj not in box, loj <= Xj <= hij
        return nr not in
    """
    nX, nbox = len(X), len(box)
    assert nX == nbox, \
        "len X %d != len box %d" % (nX, nbox)
    nnotin = 0
    for j, x, (lo,hi) in zip( range(nX), X, box ):
        if not (lo <= x <= hi):
            print("check_bounds: x[%d] %g is not in box %g .. %g" % (j, x, lo, hi))
            nnotin += 1
    return nnotin

#######################################################################################
#### DICTIONARY MANIPULATION ROUTINES (NO NEED TO CHANGE ANYTHING BELOW THIS LINE) ####
#######################################################################################

def CHECK_IF_KEY_PRESENT(key_list,inp_dict):
    inp_dict_key_list = list(inp_dict.keys())
    flag_option       = False
    for key_name_reqd in key_list:
        if key_name_reqd not in inp_dict_key_list:
            flag_option = True
            print("inp_dict should contain key :",key_name_reqd)

    if flag_option:
        print("Exiting the code now !!!")
        exit()

#...............................................................................

def CHECK_AND_INITIALIZE_KEY_VALUE(inp_dict,key_name,default_key_value):
    if key_name not in list(inp_dict.keys()):
        inp_dict.update({key_name:default_key_value})
    return inp_dict

#...............................................................................

def CONVERT_DTYPE_OBJECT_TO_DICTIONARY(data_arr,label_arr):
    """ Structure of data_arr should be as follows
    data_arr[i,0] -->> Key names / Alias
    data_arr[i,1] -->> First element of sub dictionary key_name -->> label_arr[0]
    data_arr[i,2] -->> First element of sub dictionary key_name -->> label_arr[1]

    Very Important data_arr should always be loaded with dtype = None
    data_arr        = np.genfromtxt(specie_filename,dtype=None)

    """

    assert len(label_arr) == (len(data_arr[0])-1)

    N_row     = len(data_arr)
    N_col     = len(data_arr[0])
    data_dict = {}

    for row_indx in range(N_row):
        key_name      = data_arr[row_indx][0]
        data_sub_dict = {}
        for col_indx in range(1,N_col):
            key_name_sub_dict  = label_arr[col_indx-1]
            key_value_sub_dict = data_arr[row_indx][col_indx]
            data_sub_dict.update({key_name_sub_dict:key_value_sub_dict}) 
        data_dict.update({key_name:data_sub_dict})
    return data_dict

###############################################################################
##### SPECIAL NUMPY ROUTINES (NO NEED TO CHANGE ANYTHING BELOW THIS LINE) #####
###############################################################################

def UNIQUE_ROWS_IN_ARRAY(arr):
    """
    Takes arr as input. Finds unique rows in array.
    Return sorted unique rows and number of repeatations of rows in input array.
    """
    N_col			=	arr.shape[1]
    uniques			=	np.vstack({tuple(row) for row in arr}) # Unique rows are not sorted
    uniques         =   SORT_ARRAY_BY_MULTIPLE_COLUMNS(uniques)# Unique rows are sorted now
    line_counter		=	np.zeros(uniques.shape[0],dtype=np.int32)
    for indx_count in xrange(uniques.shape[0]):
        for indx_arr in xrange(arr.shape[0]):
            test_arr	=	np.zeros(N_col,dtype=bool)
            for col_indx in xrange(N_col):
                test_arr[col_indx] = (uniques[indx_count,col_indx] == arr[indx_arr,col_indx])
            if np.all(test_arr):
                line_counter[indx_count] += 1
    return uniques,line_counter

#...............................................................................

def SORT_ARRAY_BY_MULTIPLE_COLUMNS(arr,N_col_sort=None):
    #arr_sorted = arr[np.lexsort(np.transpose(arr)[:,:-1])]
    if N_col_sort == None:
        N_col_sort = arr.shape[1]
    lex_sort_tup = ()
    for col_indx in xrange(N_col_sort):
        lex_sort_tup = lex_sort_tup + (arr[:,N_col_sort-col_indx-1],)
    arr_sorted = arr[np.lexsort(lex_sort_tup)]
    return arr_sorted

###################################################################################
##### ATOMIC TRANSITION ROUTINES (NO NEED TO CHANGE ANYTHING BELOW THIS LINE) #####
###################################################################################

def generate_XI_forest_line_dict():
    """
    --------------------- Dictionary for Line Transition  -------------------------

     Method to calculate i_alpha
     i_alpha   = (pi * e^2 * f_12) / (m_2 * c * nu_12)
     i_alpha_X = 8.83788e-21 * (f_12_X * lambda_rest_X)

     HI   -->> Neutral Hydrogen
     HeI  -->> Neutral Helium
     HeII -->> Single ionized Helium
     1215 -->> Transition Wavelength (first 4 digits)
    """
    inp_path_param    = os.path.dirname(os.path.realpath(__file__)) + "/"
    inp_file_specie   = "specie_transition_properties.txt"
    specie_filename   = inp_path_param + inp_file_specie
    data_arr          = genfromtxt_compatible(inp_path_param,inp_file_specie,None)
    N_data            = data_arr.shape[0]
    key_name_arr      = ["n_proton","n_neutron","lambda_rest","f_lu","damp_gamma"]
    line_dict         = CONVERT_DTYPE_OBJECT_TO_DICTIONARY(data_arr,key_name_arr)
    specie_trans_list = list(line_dict.keys())

    for line_indx in range(len(line_dict)):
        specie_trans  = specie_trans_list[line_indx]
        line_sub_dict = line_dict[specie_trans]
        idx           = specie_trans.find("-")
        specie        = specie_trans[:idx]
        lambda_rest   = line_sub_dict["lambda_rest"]
        f_lu          = line_sub_dict["f_lu"]
        i_alpha       = 8.83788e-21 * (f_lu * lambda_rest)
        line_sub_dict.update({"i_alpha":i_alpha,"specie":specie})
        line_dict.update({specie_trans:line_sub_dict})
    return line_dict

#...............................................................................

def genfromtxt_compatible(inp_path,inp_file,dtype,full_file_path=None):
    if full_file_path is None:
        full_file_path = inp_path + inp_file

    if sys.version[0] == '3':
        data_arr    = np.genfromtxt(full_file_path,dtype=dtype,encoding="latin")
    elif sys.version[0] == '2':
        if np.__version__ < '1.14.0':
            data_arr    = np.genfromtxt(full_file_path,dtype=dtype)
        else:
            data_arr    = np.genfromtxt(full_file_path,dtype=dtype,encoding="latin")
    return data_arr

#######################################################################################
##### COSMOLOGICAL FUNCTION ROUTINES (NO NEED TO CHANGE ANYTHING BELOW THIS LINE) #####
#######################################################################################

def z_to_v(z,cosmo_param): # Redshift to Velocity Conversion
    omega_m 		= 	cosmo_param["omega_m"]
    omega_l 		= 	cosmo_param["omega_l"]
    omega_r 		= 	cosmo_param["omega_r"]
    omega_k 		= 	cosmo_param["omega_k"]
    h  				= 	cosmo_param["hubble_param"]
    c 				= 	299792458 / 1e3 # in km / s
    E_z_inv			= 	lambda z_val: 100.0 * h / hubble_parameter(z_val,cosmo_param)
    integration_value,a = quad(E_z_inv, 0, z)
    velocity = integration_value * c
    return velocity

#...............................................................................

def hubble_parameter(z,cosmo_param):
	omega_m = cosmo_param["omega_m"]
	omega_l = cosmo_param["omega_l"]
	omega_r = cosmo_param["omega_r"]
	omega_k = cosmo_param["omega_k"]
	h  = cosmo_param["hubble_param"]
	E_z = np.sqrt(omega_l + omega_k*(1+z)**2.0 + omega_m*(1+z)**3.0 + omega_r*(1+z)**4.0)
	H_z = E_z * 100.0 * h
	return H_z

#...............................................................................

def w_to_v(Wavelength,rest_wavelength,cosmo_param):
    z_arr           =   Wavelength / rest_wavelength - 1.0
    velocity_arr    =   np.zeros(z_arr.shape[0])
    for indx in range(z_arr.shape[0]):
        velocity_arr[indx]  =   z_to_v(z_arr[indx],cosmo_param)
    velocity_arr[:] -=  velocity_arr[0]
    x_axis_data     =   np.array([Wavelength,z_arr,velocity_arr]).T # Column 0 -->> Wavelength, 1 -->> Redshift, 2 -->> Velocity
    return x_axis_data

#...............................................................................

def SAVE_VIPER_DATA(out_path,out_file,out_dict):
    txt_filename      = out_path + out_file
    param_fit_arr     = out_dict["param_fit"]
    header_str = "Column-00 --> Wavelength (angstroms)\n" + \
                 "Column-01 --> Log Column Density (cm^-2)\n"+\
                 "Column-02 --> Doppler b parameter (km/s)\n"+\
                 "Column-03 --> Wavelength Error (angstroms)\n"+\
                 "Column-04 --> Log Column Density Error (cm^-2)\n"+\
                 "Column-05 --> Doppler b parameter Error (km/s)\n"+\
                 "Column-06 --> Rigrous Siginificance Level\n"
    np.savetxt(txt_filename,param_fit_arr,header=header_str,fmt=["%.4f","%.3f","%.2f","%.4f","%.3f","%.2f","%4f"])
