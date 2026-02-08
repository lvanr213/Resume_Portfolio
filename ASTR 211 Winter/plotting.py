# -*- coding: utf-8 -*-
"""
Auxiliary plotting functions that will be used in the notebooks

Andrey Kravtsov (Spring 2023)
"""

# import relevant packages
import matplotlib.pyplot as plt
import numpy as np 

# the following commands make plots look better
def plot_prettier(dpi=150, fontsize=11, usetex=False): 
    '''
    Make plots look nicer compared to Matplotlib defaults
    Parameters: 
        dpi - int, "dots per inch" - controls resolution of PNG images that are produced
                by Matplotlib
        fontsize - int, font size to use overall
        usetex - bool, whether to use LaTeX to render fonds of axes labels 
                use False if you don't have LaTeX installed on your system
    '''
    plt.rcParams['figure.dpi']= dpi
    plt.rc("savefig", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in') 
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5) 
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5) 
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [2., 2.])
    if usetex:
        plt.rc('text', usetex=usetex)
    else:
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


    

from matplotlib import cm
import scipy.optimize as opt
from matplotlib.colors import LogNorm

def plot_color_map2(x, y, data, xlim=[0.,1], ylim=[0.,1.], 
                   xlabel = ' ', ylabel = ' ', cmap='winter', colorbar=False, 
                   figsize=3.0, cbar_label=None, figsave=None):
    '''
    Visualize f(x,y) as a 2D color map
    
    Parameters
    ----------
    x : 1d numpy array of floats with x variable values
    y : 1d numpy array of floats with x variable values
        
    data : 2d numpy array of shape (x.size, y.size) with f(x,y) values
        
    xlim : list with 2 elements, limits for x-axis, optional
        The default is [0.,1].
    ylim : list with 2 elements, limits for y-axis, optional
        The default is [0.,1]. 
    xlabel : string, optional
        x-axis label. The default is ' '.
    ylabel : string, optional
        y-axis label. The default is ' '.
    cmap : str, optional
        Matplotlib colormap. The default is 'winter'.
    colorbar : bool, optional
        whether to plot colorbar. The default is False.
    figsize : float, optional
        will determine figure size (figsize,figsize). The default is 3.0.
    cbar_label : str, optional
        colorbar label. The default is None.
    figsave : str, optional
        if str is provided figure will be saved in the file given by the str. 
        for example 'plot.pdf' or 'plot.png'. The default is None.

    Returns
    -------
    None.

    '''
    fig, ax = plt.subplots(figsize=(figsize,figsize))
    ax.axis([xlim[0], xlim[1], ylim[0], ylim[1]])

    plt.xlabel(xlabel); plt.ylabel(ylabel)
    cmap = cm.get_cmap(cmap)
    im = ax.pcolormesh(x, y, data, cmap=cmap, rasterized=True)
    if colorbar: 
        cbar = fig.colorbar(im, ax=ax)
        if cbar_label is not None: 
            cbar.ax.set_ylabel(cbar_label)
    if figsave:
        plt.savefig(figsave, bbox_inches='tight')
    plt.show()


from math import factorial
def exp_taylor(x0, N, x):
    '''
    Taylor expansion up to order N for exp(x)
    '''
    dummy = np.zeros_like(x)
    for n in range(N+1):
        dummy += np.exp(x0)*(x-x0)**n/factorial(n)
    return dummy


def taylor_exp_illustration(figsize=3.0):
    '''
    Function to produce an illustration for the Taylor expansion approximation 
    for the exp(x) function
    
    Parameters
    ----------
    figsize : float, Matplotlib figure size 

    Returns
    -------
    None.

    '''
    N = 4; x0 = 1.0

    plt.figure(figsize=(figsize,figsize))
    #plt.title('Taylor expansion of $e^x$ at $x_0=%.1f$'%x0, fontsize=9)
    plt.xlabel('$t$'); plt.ylabel(r'$x(t)$')

    xmin = x0 - 1.0; xmax = x0 + 1.0
    x = np.linspace(xmin, xmax, 100)
    plt.xlim([xmin,xmax]); plt.ylim(0.,8.)

    exptrue = np.exp(x)
    plt.plot(x, exptrue, linewidth=1.5, c='m', label='$x(t)=e^x$')
    colors = ['darkslateblue', 'mediumslateblue', 'slateblue', 'lavender']
    lstyles = [':','--','-.','-','-.']
    for n in range(N):
        expT = exp_taylor(x0, n, x)
        plt.plot(x, expT, linewidth=1.5, c=colors[n], ls=lstyles[n], label='%d Taylor term(s)'%(n+1))

    plt.legend(loc='upper left', frameon=False, fontsize=7)
    plt.show()
    return

def plot_line_points(x=None, y=None, figsize=4, xlabel=' ', ylabel=' ', col= 'darkslateblue', 
                     xp = None, yp = None, eyp=None, points = False, pmarker='.', 
                     psize=1., pcol='slateblue',
                     legend=None, plegend = None, legendloc='lower right', 
                     plot_title = None, grid=None, figsave = None):
    plt.figure(figsize=(figsize,figsize))
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    # Initialize minor ticks
    plt.minorticks_on()
    
    line = not ((x is None) or (y is None))
    points = not ((xp is None) or (yp is None))
    if (not line) and (not points):
        print('no (x, y) or (xp, yp) is supplied ---> nothing to plot')
        return

    if legend:
        if line:
            plt.plot(x, y, lw = 1., c=col, label = legend)
        if points: 
            if plegend:
                plt.scatter(xp, yp, marker=pmarker, s=psize, c=pcol, label=plegend)
            else:
                plt.scatter(xp, yp, marker=pmarker, s=psize, c=pcol)
            if eyp is not None:
                plt.errorbar(xp, yp, eyp, linestyle='none', marker=pmarker, color=pcol, markersize=psize)
        plt.legend(frameon=False, loc=legendloc, fontsize=3.*figsize)
    else:
        if line:
            plt.plot(x, y, lw = 1., c=col)
        if points:
            plt.scatter(xp, yp, marker=pmarker, s=psize, c=pcol)

    if plot_title:
        plt.title(plot_title, fontsize=3.*figsize)
        
    if grid: 
        plt.grid(linestyle='dotted', lw=0.5, color='lightgray')
        
    if figsave:
        plt.savefig(figsave, bbox_inches='tight')

    plt.show()

def colorbar(mappable):
    """
    a hack to make colorbars look good by Joseph Long
    https://joseph-long.com/writing/colorbars/
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def plot_data_fit2d(x, y, z, model, model_name='model'):
    # Plot the data with the best-fit model
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax1.imshow(z, origin='lower', interpolation='nearest', vmin=-1e4, vmax=5e4)
    ax1.set_title("data")

    ax2.imshow(model, origin='lower', interpolation='nearest', vmin=-1e4,
               vmax=5e4)
    ax2.set_title(model_name)
    img3 = ax3.imshow(z - model, origin='lower', interpolation='nearest', vmin=-1e4, 
               vmax=1e4)
    colorbar(img3)
    ax3.set_title("residual")
    plt.show()


def show_image(img, cmap='Greys', auto_aspect=False, figsize=4):
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(img,  cmap='Greys')
    if auto_aspect:
        ax.set_aspect('auto')
    plt.show()

def plot_color_map(x, y, data, figax=None, xlim=[0.,1], ylim=[0.,1.], 
                   xlabel = ' ', ylabel = ' ', cmap='winter', colorbar=None, 
                   contours = False, levels = [], contcmap = 'winter', 
                   cbar_label=None, norm = None,
                   plot_title=None, figsize=3.0, figsave=None):
    """
    Helper function to plot colormaps of a 2d array of values with or without contours
    
    Parameters:
    -----------
    x, y - 1d numpy vectors generated by numpy.meshgrid functions
    data - 2d numpy array, containing data values at the grid represented by x and y
    figux - tuple, with fig and ax object (optional, if not supplied will plot its own plot)
    xlim, ylim - lists of size 2 containing limits for x and y axes
    cmap: string, Matplotlib colormap to use
    colormap: boolean, if True plot colorbar
    contours: boolean, if True plot contours corresponding to data levels specified by levels parameter
    levels: values of data for which to draw contour levels
    contcmap: string, Matplotlib colormap to use for contour lines 
    plot_title: string, if not None will be used to prouce plot title at the top
    figsize: float, size of Matplotlib figure
    figsave: None or string, if string, the string will be used as a path/filename to save PDF of the plot
    
    Returns:
    --------
    Nothing
    
    """
    
    if figax is None: 
        fig, ax = plt.subplots(figsize=(figsize,figsize))
    else:
        fig, ax = figax[0], figax[1]
        
    ax.axis([xlim[0], xlim[1], ylim[0], ylim[1]])

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    cmap = cm.get_cmap(cmap)
    if norm is None: 
        im = ax.pcolormesh(x, y, data, cmap=cmap, rasterized=False)
    else:
        im = ax.pcolormesh(x, y, data, cmap=cmap, norm=norm, rasterized=False)
        
    if contours:
        ax.contour(x, y, data, levels=levels, cmap=contcmap)
    if colorbar: 
       cbar = fig.colorbar(im, ax=ax)
       if cbar_label is not None: 
           cbar.ax.set_ylabel(cbar_label)
    if plot_title:
        ax.set_title(plot_title, fontsize=3.*figsize)

    if figsave:
        plt.savefig(figsave, bbox_inches='tight')
    if ax is None: plt.show()
    else: return ax


def conf_interval(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level

def plot_2d_dist(x,y, xlim, ylim, nxbins, nybins, figsize=(5,5), 
                cmin=1.e-4, cmax=1.0, smooth=None, xpmax=None, ypmax=None, 
                log=False, weights=None, xlabel='x', ylabel='y', 
                clevs=None, fig_setup=None, savefig=None):
    """
    construct and plot a binned, 2d distribution in the x-y plane 
    using nxbins and nybins in x- and y- direction, respectively
    
    log = specifies whether logged quantities are passed to be plotted on log-scale outside this routine
    """
    if fig_setup is None:
        fig, ax = plt.subplots(figsize=figsize)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
    else:
        ax = fig_setup
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim); ax.set_ylim(ylim)

    if xlim[1] < 0.: ax.invert_xaxis()

    if weights is None: weights = np.ones_like(x)
    H, xbins, ybins = np.histogram2d(x, y, weights=weights, bins=(np.linspace(xlim[0], xlim[1], nxbins),np.linspace(ylim[0], ylim[1], nybins)))
    
    H = np.rot90(H); H = np.flipud(H); 
             
    X,Y = np.meshgrid(xbins[:-1],ybins[:-1]) 

    if smooth != None:
        from scipy.signal import wiener
        H = wiener(H, mysize=smooth)
        
    H = H/np.sum(H)        
    Hmask = np.ma.masked_where(H==0,H)
    
    if log:
        X = np.power(10.,X); Y = np.power(10.,Y)

    pcol = ax.pcolormesh(X, Y,(Hmask),  cmap=plt.cm.BuPu, norm = LogNorm(), linewidth=0., rasterized=True)
    pcol.set_edgecolor('face')
    
    # plot contours if contour levels are specified in clevs 
    if clevs is not None:
        lvls = []
        for cld in clevs:  
            sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )   
            lvls.append(sig)
        
        ax.contour(X, Y, H, linewidths=(1.0,0.75, 0.5, 0.25), colors='black', levels = sorted(lvls), 
                norm = LogNorm(), extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]])
    if xpmax is not None:
        ax.scatter(xpmax, ypmax, marker='x', c='orangered', s=20)
    if savefig:
        plt.savefig(savefig,bbox_inches='tight')
    if fig_setup is None:
        plt.show()
    return

def fq(t, y, dt, c):    
    '''Set up dx/dt=f(t), where x = e^(c*t)'''
    return  -dt * c * y 


def plot_solution_derivatives(tsol, xsol, ng=25, ysolution=None, args=None, fsize=3, fontsize=9., label=None):
    '''
    tsol, xsol: np arrays of floats
    ng: int,    the number of grid points to use for plotting derivatives as quivers on a grid of ng x ng
    ysolution:  Python function object, function computing exact solution
    args:       tuple containing arguments to be passed to the function ysolution
    fsize:      float or int, figure size (figure is of size fsize x fsize)
    fontsize:   float or int, fontsize to use
    label:      str, label for numerical solution
    '''
    # parameters of the quiver grid and the function range
    tmin, tmax = min(tsol), max(tsol)
    dt = np.abs(tmax-tmin) / ng

    xmin = np.minimum(1.2*ysolution(tmin,*args), 1.2*ysolution(tmax,*args));
    xmax = np.maximum(1.2*ysolution(tmin,*args), 1.2*ysolution(tmax,*args));
    xmin = -2.; xmax = 2.
    xscale = 2.*(xmax-xmin)**0.75

    # evenly spaced grids of x and y 
    t = np.linspace(tmin, tmax, ng)
    x = np.linspace(xmin, xmax, ng)

    # generated 2d mesh using numpy's meshgrid
    (T,X) = np.meshgrid(t,x)

    # parameters for quivers indicating local slope
    u = dt * np.ones_like(T); v = fq(T, X, dt, *args)

    # vector of x for plotting the function itself
    tp = np.linspace(tmin, tmax, 100)

    # plot
    plt.figure(figsize=(fsize,fsize))
    plt.xlim(tmin, tmax); plt.ylim(xmin, xmax)
    plt.plot(tp, ysolution(tp,*args), lw=3, c='midnightblue', zorder=0, label=r'exact solution $x(t)=e^{-t}$')#, label=r'$y(x)=e^{%.2fx}$'%c)
    plt.plot(tsol, xsol, c='m', zorder=2, label=label)
    plt.quiver(T, X, u, v, angles='xy', headlength=0, headwidth=0, scale=xscale, color='steelblue', zorder=1)

    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.legend(frameon=False, bbox_to_anchor=(1.025,0.75), fontsize=fontsize)
    plt.show()

    
def plot_integral_rectangle(a=0., b=1., nsteps=6, value='average', xlabel='$x$', ylabel='$f(x)$'):
    '''
    Plots rectangle with an area approximating area under integral over e^x in the interval [0,1]
    Used to make an illustration in the integration notebook
    '''
    xg = np.linspace(a, b, 100) # generate a finer grid of evenly spaced values

    plt.figure(figsize=(3,3))
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.plot(xg, np.exp(xg), lw=2., color='indigo') # plot sin(x)
    # fill_between function creates a shaded area
    plt.fill_between(xg, np.zeros_like(xg), np.exp(xg), color='darkslateblue', alpha=0.9) # shade area under sin(x)

    # shade area under <f> over [a, b] interval
    # break interval [a,b] into nsteps subintervals
    xi = np.linspace(a, b, nsteps)
    if value == 'left':
        integral = (b-a) * np.exp(a) 
        plt.fill_between(xi, np.zeros_like(xi), (integral/(b-a))*np.ones_like(xi), color='m', alpha=0.5)
    elif value == 'average':
        integral = (b-a) * np.sum(np.exp(xi)) / nsteps # approximate calculation of the integral
        print('estimated integral = {:.3g}; exact integral = {:.3g}'.format(integral, np.exp(b)-np.exp(a)))
        fracerr = integral / (np.exp(b)-np.exp(a)) - 1.
        print('fractional error = {:.4g}'.format(fracerr))
        plt.fill_between(xi, np.zeros_like(xi), (integral/(b-a))*np.ones_like(xi), color='m', alpha=0.5)
    elif value == 'linear':
        xi = np.linspace(a, b, 2)
        plt.fill_between(xi, np.zeros_like(xi), np.exp(xi), color='purple', alpha=0.5)
        plt.plot(xi, np.exp(xi), lw=1., color='m')
    
    plt.show()
    
    
def plot_integral_pw_const(a=0., b=1., nsteps=6, xlabel='$x$', ylabel='$f(x)$'):
    '''
    Plots  area approximating area under integral over e^x in the interval [0,1] by a piecewise constant representation 
    of the function. 
    Used to make an illustration in the integration notebook
    '''
    #
    # prepare a step-wise function for illustration
    #
    xi = np.linspace(a, b, nsteps)
    xg = np.linspace(a, b, 100)
    xir = np.repeat(xi,2) # repeat each xi value in the array 
    fxir = np.zeros_like(xir)
    fxir[::2] = np.exp(xir[::2])
    fxir[1::2] = fxir[::2]
    dx = (b-a) / (nsteps-1) 
    dxhalf = dx * 0.5
    
    plt.figure(figsize=(3,3)) # define figure
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.plot(xg, np.exp(xg), lw=2., color='indigo')
    plt.fill_between(xg, np.zeros_like(xg), np.exp(xg), color='indigo', alpha=0.4)
    plt.fill_between(xir[1:], np.zeros_like(xir[1:]), fxir[:-1], color='purple', alpha=0.5)
    plt.show()
    
def plot_integral_pw_lin(a=0., b=1., nsteps=6, xlabel='$x$', ylabel='$f(x)$'):
    xi = np.linspace(a, b, nsteps)
    xg = np.linspace(a, b, 100)
    plt.figure(figsize=(3,3))
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.fill_between(xg, np.zeros_like(xg), np.exp(xg), color='indigo', alpha=0.4)
    plt.fill_between(xi, np.zeros_like(xi), np.exp(xi), color='purple', alpha=0.5)
    plt.plot(xg, np.exp(xg), lw=2., color='indigo')
    plt.plot(xi, np.exp(xi), lw=1., color='m')
    plt.scatter(xi, np.exp(xi), s=20., color='m')
    plt.show()
    
def its_demo(figsize=(3,3), xlims=[0,7], cdf=None, cdfi=None, nl=10):
    '''
    Function used to illustrate inverse transform sampling
    '''
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'cdf   $P(x)$')
    ax.set_xlim(xlims)
    ax.set_ylim(0, 1)
    x = np.linspace(xlims[0], xlims[1], 100)
    ax.plot(x, cdf(x), lw=2, c='slateblue', label='cdf')
    yl = np.linspace(0.,1., nl)
    xl = cdfi(yl)
    for i in range(nl):
        ax.plot([0, xl[i]], [yl[i], yl[i]], ls='--', c='gray', lw=0.5)
        ax.plot([xl[i], xl[i]], [0, yl[i]], ls='--', c='gray', lw=0.5)
    #ax.legend(frameon=False, loc='lower right')
    plt.show()
    
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

def its_demo_animation(pdf=None, cdf=None, cdfi=None, nr=1000, xlims=(0,7), figsize=(3,3)):
    '''
    Animation illustrating the inverse transform sampling (ITS) method
    To produce animation this function needs to be run in the
    %matplotlib notebook mode
    
        pdf, cdf, cdfi: Python function objects for the function computing pdf, cdf and inverse cdf
        nr: int, number of random samples to use
        xlims: tuple of floats, containing range of x values to plot
        figsize: tuple of ints, figure size
    
    Returns:
        animation: animation object returned by FuncAnimation
    '''
    d = np.random.uniform(size=nr)
    xlims = xlims # limits of the x-axis
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(8,9, wspace=1, hspace=1)
    ax_l = fig.add_subplot(gs[:6,:2])
    ax = fig.add_subplot(gs[:6,2:])
    ax_b = fig.add_subplot(gs[6:,2:])
    #ax_l.axes.yaxis.set_visible(False)
    ax_l.axes.xaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax_b.axes.yaxis.set_visible(False)

    ax.set_xlim(xlims)
    ax.set_ylim(0, 1)
    ax_b.set_xlim(xlims)
    ax_l.set_ylim(0.,1)
    x = np.linspace(xlims[0], xlims[1],100)
    ax.plot(x, cdf(x), lw=2, c='slateblue', label='cdf')
    ax.legend(frameon=False, loc='lower right')
    ax_b.set_xlabel('$x$')

    def its_anim(i, savefig=False):
        ci = cdfi(d[i])
        di = d[i]

        ax.lines.remove(its_anim.l1)
        ax.lines.remove(its_anim.l2)
        ax.plot(ci, 0., '|k', ms=15, markeredgewidth=0.75)
        its_anim.l1, = ax.plot([0, ci], [di, di], ls='--', c='gray', lw=0.5)
        its_anim.l2, = ax.plot([ci, ci], [0, di], ls='--', c='gray', lw=0.5)
        ax.plot(0.0, di, '_k', ms=15, markeredgewidth=0.75)
        _, bins, _ = ax_b.hist(cdfi(d[:i]), bins=30, color='slateblue');
        xl = np.linspace(xlims[0], xlims[1], 100)
        _, ybins, _ = ax_l.hist(d[:i], bins=20, orientation="horizontal", color='slateblue');
        ax.set_title(f'N={d[:i].size:d}')
        if savefig: fig.savefig(f'its_frames/f{i:04d}.png')

    its_anim.l1, = ax.plot([0, 0], [0, 0], ls='--', c='gray', lw=0.5)
    its_anim.l2, = ax.plot([0, 0], [0, 0], ls='--', c='gray', lw=0.5)

    animation = FuncAnimation(fig, its_anim, interval = 200)

    return animation

def plot_pdf(xr, plot_pdf=True, func=None, args=None,
             xlabel='$x$', ylabel='$p(x)$',  label='samples', 
             bins = 50, xlog=False, ylog=True, 
             xlims=[0.,30.], ylims=[1.e-6, 0.4], figsize=5):
    """
    a utility function to plot samples from a pdf as a histogram and compare
    the histogram to the analytical form of the pdf plotted as a line, if needed
    
    Parameters:
    -----------
    xr:           1d numpy array of float numbers: vector of samples
    plot_pdf:     boolean, determines whether to plot a line of the target pdf
                  if True, func needs to be supplied
    func:         Python function object: function that returns analytic pdf for a given vector xr
    args:         list of possible arguments to func, if any
    xlabel, ylabel: strings, labels for x and y axes
    label:        string, label for the samples histogram for the legend
    bins:         integer or string (e.g., 'auto'), passed on as bins parameter to Pylab's histogram function
    xlog, ylog:   boolean, determine whether x or y axis is to be plotted on logarithmic scale
    xlims, ylims: lists of 2 float elements defining plot limits for x and y axes
    figsize:      float, parameter controlling plot size 
    
    Returns:
    --------
        Nada
    """
    plt.figure(figsize=(figsize,figsize))
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if xlog: plt.xscale('log') # plot y-values on a logarithmic scale
    if ylog: plt.yscale('log') # plot y-values on a logarithmic scale
    plt.xlim(xlims); plt.ylim(ylims) # set axis limits 

    # compute histogram values; 
    # density='True' normalizes histogram properly so it can be compared to pdf
    hist, bins, patches = plt.hist(xr, density='True', color='slateblue', bins=bins, label=label)
    # compute bin centers using numpy slicing 
    binc = 0.5*(bins[1:] + bins[:-1])
    if plot_pdf: 
        plt.plot(binc, func(binc, *args), lw=1.25, c='orangered', label='target pdf')
    plt.legend(loc='best', frameon=False, fontsize=3*figsize)
    plt.show()
    
def fwiggly(x):
    """
    A wiggly function for illustration from Michael Nielsen's 
    "Neural Networks and Deep Learning" book: 
    http://neuralnetworksanddeeplearning.com/chap4.html
    """
    
    return 0.2 + 0.4*x**2 + 0.3*x*np.sin(15.*x) + 0.05*np.cos(50*x)

def shepard_illustrate(ntrain=30, ntest=500):
    '''
    Illustrate Shepard interpolation method
    
    '''
    # produce irregularly spaced training points
    xtrain = np.random.uniform(size=ntrain)
    ftrain = fwiggly(xtrain)

    # test points
    xtest = np.linspace(0.0, 1.0, ntest)
    # compute distances between test and training points
    r = np.abs(xtrain - xtest[:, np.newaxis]) # broadcast

    plt.figure(figsize=(9, 3))
    for i, a in enumerate([1, 2, 4]):
        # compute weights
        weights = 1 / np.power(r, a) 
        sum_weights = np.sum(weights, axis=1, keepdims=True)
        weights /= sum_weights

        # compute function approximations at xtest points
        ftest = weights.dot(ftrain)

        plt.subplot(1, 3, i + 1)
        plt.xlabel('$x$')
        plt.title(r'$\alpha={}$'.format(a))
        plt.plot(xtest, ftest, label='interpolation')
        plt.plot(xtest, fwiggly(xtest), ls='--', c='orangered', label='true $f(x)$')
        plt.scatter(xtrain, ftrain, marker='o',  color='slateblue', ec='indigo', label='training data')
        if i==0: 
            plt.ylabel('$f(x)$') 
            plt.legend(frameon=False, loc='best')

    plt.show()
    
from scipy.interpolate import UnivariateSpline, Rbf, RBFInterpolator

def rbf_illustrate(ntrain=20, ntest=200, kernel='multiquadric', smooth=0., epsilon=0.):
    '''
    function that makes plot illustraing RBF method interpolation in 1d
    '''
    # setup data
    xtrain = np.sort(np.random.uniform(0., 1., ntrain))
    ftrain = fwiggly(xtrain) #np.sin(x)
    xtest = np.linspace(xtrain.min(), xtrain.max(), ntest)

    # spline interpolation
    spl = UnivariateSpline(xtrain, ftrain, s=0.)
    fsptest = spl(xtest)

    # use RBF method
    if kernel == 'thin_plate_spline': kernel = 'thin_plate'
    rbf = Rbf(xtrain, ftrain, function=kernel, smooth=smooth, epsilon=epsilon)
    #rbf = RBFInterpolator(xtrain, ftrain, kernel=kernel, smoothing=smooth, epsilon=epsilon)
    
    frbf_test = rbf(xtest)

    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    ax[0].scatter(xtrain, ftrain, marker='o', color='slateblue', ec='indigo', label='training')
    ax[0].plot(xtest, fsptest, color='navy', label='approx. $f(x)$')
    ax[0].plot(xtest, fwiggly(xtest), c='r', ls='--', label='true $f(x)$')
    ax[0].set_title('Interpolation using univariate spline', fontsize=9)
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$f(x)$')
    ax[0].legend(loc='best', frameon=False)
    ax[1].set_xlabel(r'$x$')
    ax[1].scatter(xtrain, ftrain, marker='o', color='slateblue', ec='indigo', label='training')
    ax[1].plot(xtest, frbf_test, color='navy', label='approx. $f(x)$')
    ax[1].plot(xtest, fwiggly(xtest), c='r', ls='--', label='true $f(x)$')
    ax[1].set_title(f'Interpolation using RBF - {kernel:s}', fontsize=9)
    plt.tight_layout()

    
# You don't need to look through or understand the code in this cell. These are helper functions for plotting
# results of the K-means clustering algorithm

colors = 'brgmck'

def distance_matrix(A, B):
    '''
    Given two sets of data points, computes the Euclidean distances
    between each pair of points.

    *A*: (N, D) array of data points
    *B*: (M, D) array of data points

    Returns: (N, M) array of Euclidean distances between points.
    '''
    Na, D = A.shape
    Nb, Db = B.shape
    assert(Db == D)
    dists = np.zeros((Na,Nb))
    for a in range(Na):
        dists[a,:] = np.sqrt(np.sum((A[a] - B)**2, axis=1))
    return dists

# Copied and very slightly modified from scipy
def voronoi_plot_2d(vor, ax=None):
    #ptp_bound = vor.points.ptp(axis=0)
    ptp_bound = np.array([1000,1000])
    
    center = vor.points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            ax.plot([vor.vertices[i,0], far_point[0]],
                    [vor.vertices[i,1], far_point[1]], 'k--')

def plot_kmeans(i, X, K, centroids, newcentroids, nearest, show=True):
    """
    Helper function to plot K-means clusters produced by the K-means algorithm and boundaries between them
    """
    import pylab as plt
    plt.clf()
    plotsymbol = 'o'
    if nearest is None:
        distances = distance_matrix(X, centroids)
        nearest = np.argmin(distances, axis=1)
        
    for i,c in enumerate(centroids):
        I = np.flatnonzero(nearest == i)
        plt.plot(X[I,0], X[I,1], plotsymbol, mfc=colors[i], mec='k')
        
    ax = plt.axis()
    for i,(oc,nc) in enumerate(zip(centroids, newcentroids)):
        plt.plot(oc[0], oc[1], 'kx', mew=2, ms=10)
        plt.plot([oc[0], nc[0]], [oc[1], nc[1]], '-', color=colors[i])
        plt.plot(nc[0], nc[1], 'x', mew=2, ms=15, color=colors[i])
        
    vor = None
    if K > 2:
        from scipy.spatial import Voronoi #, voronoi_plot_2d
        vor = Voronoi(centroids)
        voronoi_plot_2d(vor, plt.gca())
    else:
        mid = np.mean(centroids, axis=0)
        x0,y0 = centroids[0]
        x1,y1 = centroids[1]
        slope = (y1-y0)/(x1-x0)
        slope = -1./slope
        run = 1000.
        plt.plot([mid[0] - run, mid[0] + run],
                 [mid[1] - run*slope, mid[1] + run*slope], 'k--')
    plt.axis(ax)
    if show:
        plt.show()
        
def plot_lines_points(x=None, y=None, figsize=4, xlabel=' ', ylabel=' ', col= 'darkslateblue', 
                      ls = 'solid', 
                      xp = None, yp = None, eyp=None, points = False, pmarker='.', 
                      psize=1., pcol='slateblue',
                      legend=None, plegend = None, legendloc='best', 
                      plot_title = None, grid=None, figsave = None):
    plt.figure(figsize=(figsize,figsize))
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    # Initialize minor ticks
    plt.minorticks_on()
    
    line = not ((x is None) or (y is None))
    points = not ((xp is None) or (yp is None))
    if (not line) and (not points):
        print('no (x, y) or (xp, yp) is supplied ---> nothing to plot')
        return

    if legend:
        if line:
            if len(np.shape(x)) == 1:
                plt.plot(x, y, lw = 1., c=col, label = legend)
            else: 
                for i, xd in enumerate(x):
                    if len(col) == len(x):
                        if len(ls) == len(x):
                            plt.plot(xd, y[i], lw=1., ls=ls[i], label=legend[i], c=col[i])
                        else:
                            plt.plot(xd, y[i], lw=1., label=legend[i], c=col[i])
        if points: 
            if plegend:
                plt.scatter(xp, yp, marker=pmarker, s=psize, c=pcol, label=plegend)
            else:
                plt.scatter(xp, yp, marker=pmarker, s=psize, c=pcol)
            if eyp is not None:
                plt.errorbar(xp, yp, eyp, linestyle='none', marker=pmarker, color=pcol, markersize=psize)
        plt.legend(frameon=False, loc=legendloc, fontsize=3.*figsize)
    else:
        if line:
            if len(np.shape(x)) == 1:
                plt.plot(x, y, lw = 1., c=col)
            else: 
                for i, xd in enumerate(x):
                    if len(col) == len(x):
                        if len(ls) == len(x):
                            plt.plot(xd, y[i], lw=1., ls=ls[i], c=col[i])
                        else:
                            plt.plot(xd, y[i], lw=1., c=col[i])
        if points:
            plt.scatter(xp, yp, marker=pmarker, s=psize, c=pcol)

    if plot_title:
        plt.title(plot_title, fontsize=3.*figsize)
        
    if grid: 
        plt.grid(linestyle='dotted', lw=0.5, color='lightgray')
        
    if figsave:
        plt.savefig(figsave, bbox_inches='tight')

    plt.show()
    
    
def plot_ellipse(mean, cov, *args, **kwargs):
    import pylab as plt
    u,s,v = np.linalg.svd(cov)
    angle = np.linspace(0., 2.*np.pi, 200)
    u1 = u[0,:]
    u2 = u[1,:]
    s1,s2 = np.sqrt(s)
    xy = (u1[np.newaxis,:] * s1 * np.cos(angle)[:,np.newaxis] +
          u2[np.newaxis,:] * s2 * np.sin(angle)[:,np.newaxis])
    return plt.plot(mean[0] + xy[:,0], mean[1] + xy[:,1], *args, **kwargs)
    
def gaussian_probability(X, mean, cov):
    '''
    Returns the probability of drawing data points from a Gaussian distribution

    *X*: (N,D) array of data points
    *mean*: (D,) vector: mean of the Gaussian
    *cov*: (D,D) array: covariance of the Gaussian

    Returns: (N,) vector of Gaussian probabilities
    '''
    D,d = cov.shape
    assert(D == d)

    # I haven't found a beautiful way of writing this in numpy...
    mahal = np.sum(np.dot(np.linalg.inv(cov), (X - mean).T).T * (X - mean),
                   axis=1)
    return (1./((2.*np.pi)**(D/2.) * np.sqrt(np.linalg.det(cov)))
            * np.exp(-0.5 * mahal))

def plot_em(step, X, K, amps, means, covs, z,
            newamps, newmeans, newcovs, show=True):
    import pylab as plt
    from matplotlib.colors import ColorConverter

    (N,D) = X.shape

    if z is None:
        z = np.zeros((N,K))
        for k,(amp,mean,cov) in enumerate(zip(amps, means, covs)):
            z[:,k] = amp * gaussian_probability(X, mean, cov)
        z /= np.sum(z, axis=1)[:,np.newaxis]
    
    plt.clf()
    # snazzy color coding
    cc = np.zeros((N,3))
    CC = ColorConverter()
    for k in range(K):
        rgb = np.array(CC.to_rgb(colors[k]))
        cc += z[:,k][:,np.newaxis] * rgb[np.newaxis,:]

    plt.scatter(X[:,0], X[:,1], color=cc, s=9, alpha=0.5)

    ax = plt.axis()
    for k,(amp,mean,cov) in enumerate(zip(amps, means, covs)):
        plot_ellipse(mean, cov, lw=4)
        plot_ellipse(mean, cov, color=colors[k], lw=2)

    plt.axis(ax)
    if show:
        plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import os

def visualize_3d_gmm(points, w, mu, stdev, export=True):
    '''
    plots points and their corresponding gmm model in 3D
    Input: 
        points: N X 3, sampled points
        w: n_gaussians, gmm weights
        mu: 3 X n_gaussians, gmm means
        stdev: 3 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''

    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111, projection='3d')
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    axes.set_zlim([-1, 1])
    plt.set_cmap('Set1')
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        axes.scatter(points[idx, 0], points[idx, 1], points[idx, 2], alpha=0.3, c=colors[i])
        plot_sphere(w=w[i], c=mu[:, i], r=stdev[:, i], ax=axes)

    plt.title('3D GMM')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.view_init(35.246, 45)
    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/3D_GMM_demonstration.png', dpi=100, format='png')
    plt.show()


def plot_sphere(w=0, c=[0,0,0], r=[1, 1, 1], subdev=10, ax=None, sigma_multiplier=3):
    '''
        plot a sphere surface
        Input: 
            c: 3 elements list, sphere center
            r: 3 element list, sphere original scale in each axis ( allowing to draw elipsoids)
            subdiv: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
            ax: optional pyplot axis object to plot the sphere in.
            sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
        Output:
            ax: pyplot axis object
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:complex(0,subdev), 0.0:2.0 * pi:complex(0,subdev)]
    x = sigma_multiplier*r[0] * sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier*r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier*r[2] * cos(phi) + c[2]
    cmap = cmx.ScalarMappable()
    cmap.set_cmap('jet')
    c = cmap.to_rgba(w)

    ax.plot_surface(x, y, z, color=c, alpha=0.2, linewidth=1)

    return ax

def visualize_2D_gmm(points, w, mu, stdev, export=True):
    '''
    plots points and their corresponding gmm model in 2D
    Input: 
        points: N X 2, sampled points
        w: n_gaussians, gmm weights
        mu: 2 X n_gaussians, gmm means
        stdev: 2 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''
    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = plt.gca()
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    plt.set_cmap('Set1')
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        plt.scatter(points[idx, 0], points[idx, 1], alpha=0.3, c=colors[i])
        for j in range(8):
            axes.add_patch(
                patches.Ellipse(mu[:, i], width=(j+1) * stdev[0, i], height=(j+1) *  stdev[1, i], fill=False, color=[0.0, 0.0, 1.0, 1.0/(0.5*j+1)]))
        plt.title('GMM')
    plt.xlabel('X')
    plt.ylabel('Y')

    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/2D_GMM_demonstration.png', dpi=100, format='png')

    plt.show()
    
    
def plot_histogram(data, bins=None, tickmarks=False, 
                   xlabel=' ', ylabel=' ', figsize=3., plot_title=None):
    '''
    plot histogram of values in array data
    
    Parameters:
        data: array of floats containing data samples
        bins: int for the number of bins, or any input that bins argument of numpy.hist would understand
        tickmarks: bool, if True tickmarks at sample locations will be plotted along x-axis
        xlabel, ylabel: str, labels for x- and y axes
        figsize: int, figure size will be (figsize,figsize)
        plot_title: str, string to use for plot title, if None there will be no title
        
    Returns:
        nada
    '''
    fig = plt.figure(figsize=(figsize, figsize)) # define figure environment
    plt.xlabel(xlabel); plt.ylabel(ylabel) # define axis labels
    
    # plot histogram of values in data
    plt.hist(data, bins=bins, histtype='stepfilled', 
             facecolor='slateblue', alpha=0.5)
    
    if tickmarks: 
        # this line is not stictly needed for plotting histogram
        # it plots individual values in data as little ticks along x-axis
        plt.plot(data, np.full_like(data, data.max()*1e-6), '|k', 
                 markeredgewidth=1)
    if plot_title is not None: 
        plt.title(plot_title)
    plt.show()