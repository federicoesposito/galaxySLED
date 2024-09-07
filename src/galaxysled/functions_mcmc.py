# galaxySLED: a code to reproduce and fit a galaxy CO SLED
# Copyright (C) 2024  Federico Esposito
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



# IMPORT PACKAGES
import math
import emcee
import corner
from multiprocessing import Pool
from galaxysled.gmcs import *
from galaxysled.functions import *


def SLED_errors(obSLED, thresh=None):
    y = obSLED[0].copy()
    yulim = obSLED[3].copy()
    yerr = []
    for j in range(len(obSLED[0])):
        if not np.isnan(obSLED[0][j]):
            if yulim[j] != 1:
                ob_err = np.mean([obSLED[1][j], obSLED[2][j]])
                if thresh and (ob_err < thresh * obSLED[0][j]):
                    ob_err = thresh * obSLED[0][j]
            else:
                y[j] = obSLED[0][j]/3
                ob_err = obSLED[0][j]/3
            yerr.append(ob_err)
        else:
            yerr.append(np.nan)
    return y, yerr, yulim

def MCMC_model(theta, x):
    alphaCO, logNH = theta
    galdf, gmcs, logLX, FUVparams, G0floor, Jmax, alphaCOin = x
    return baseline_sled(galdf, gmcs, logLX, FUVparams, flatNH=logNH, G0floor=G0floor, Jmax=Jmax)[2] * alphaCO/alphaCOin

def MCMC_lnlike(theta, x, y, yerr, yulim):
    m = MCMC_model(theta, x)
    Js = range(len(y))
    chi_det = np.nansum([((y[j] - m[j])/yerr[j])**2 if yulim[j]==0 else 0 for j in Js])
    chi_ul = -2*np.nansum([np.log(0.5 * (1 + math.erf((y[j] - m[j]) / (y[j]*np.sqrt(2))))) if yulim[j]==1 else 0 for j in Js])
    return - 0.5 * (chi_det + chi_ul)

def MCMC_lnprior(theta):
    alphaCO, logNH = theta
    min_alphaCO, max_alphaCO = 0.043, 43.
    min_logNH, max_logNH = 22., 25.
    if min_alphaCO <= alphaCO <= max_alphaCO and min_logNH <= logNH <= max_logNH:
        return 0.0
    return -np.inf

def MCMC_lnprob(theta, x, y, yerr, yulim):
    lp = MCMC_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + MCMC_lnlike(theta, x, y, yerr, yulim)

def MCMC_main(p0, nwalkers, niter, niter_burnin, ndim, lnprob, data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, niter_burnin, progress=True)
    sampler.reset()
    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
    return sampler, pos, prob, state

def MCMC_main_parallel(p0, nwalkers, niter, niter_burnin, ndim, lnprob, data, backend=None):
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool, backend=backend)
        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, niter_burnin, progress=True)
        sampler.reset()
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
    return sampler, pos, prob, state

def MCMC_plotter(samples, data, savepng=None):
    x, y, yerr, yulim = data
    galdf, gmcs, logLX, FUVparams, G0floor, Jmax, alphaCOin = x
    Jupp = np.arange(1, Jmax+1)
    fig, ax = plt.subplots(1, 1, figsize=(7,4))
    ax.errorbar(Jupp, y, yerr=yerr, uplims=yulim, label='Observed', color='k', capsize=3, lw=3, zorder=10)
    for theta in samples[np.random.randint(len(samples), size=100)]:
        ax.plot(Jupp, MCMC_model(theta, x), color="r", alpha=0.1)
    ax.set_xticks(Jupp)
    ax.set_xlabel(r'$J_{upp}$')
    ax.set_ylabel(r'$L_{CO(J \rightarrow J-1)}$ $[L_{\odot}]$')
    ax.set_yscale('log')
    ax.legend(loc='lower center')
    if savepng:
        plt.savefig('%sMCMC_all.png' % savepng, dpi=300, bbox_inches='tight', facecolor='w')
        plt.close()

def MCMC_sample_walkers(nsamples, flattened_chain, x):
    models = []
    draw = np.floor(np.random.uniform(0, len(flattened_chain), size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for i in thetas:
        mod = MCMC_model(i, x)
        models.append(mod)
    spread = np.std(models, axis=0)
    med_model = np.median(models, axis=0)
    return med_model, spread

def print_results(samples):
    q1, q2, q3 = np.quantile(samples.T[0], [.16, .5, .84])
    alpha_res = r'$\alpha_{CO} = %.2f^{+%.2f}_{-%.2f}$' % (q2, q3-q2, q2-q1)
    q1, q2, q3 = np.quantile(samples.T[1], [.16, .5, .84])
    logNH_res = r'$\log N_H = %.2f^{+%.2f}_{-%.2f}$' % (q2, q3-q2, q2-q1)
    return [alpha_res, logNH_res]

def MCMC_spread_plotter(samples, log_prob, data, nsamples, plot_base=False, savepng=None, plot_residuals=False):
    x, y, yerr, yulim = data
    galdf, gmcs, logLX, FUVparams, G0floor, Jmax, alphaCOin = x
    Jupp = np.arange(1, Jmax+1)
    if plot_residuals:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7,5), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7,4))
    EBkwargs = {'label': 'Observed', 'color': 'k', 'capsize': 4, 'lw': 2}
    ax.errorbar(Jupp, y, yerr=yerr, uplims=yulim, **EBkwargs, zorder=5)
    if plot_base:
        logNHmed = np.log10(galdf['NH_r'].median())
        blSLED = baseline_sled(galdf, gmcs, logLX, FUVparams, G0floor=G0floor, Jmax=Jmax)
        bllab = r'Baseline model ($\alpha_{CO}$ = %.1f, log$N_H^{med}$ = %.1f)' % (alphaCOin, logNHmed)
        ax.plot(Jupp, blSLED[2], label=bllab, color='#ff7f00', lw=2, zorder=4)
    for i, theta in enumerate(samples[np.random.randint(len(samples), size=nsamples)]):
        lab = 'MCMC sampled models (%1d randomly selected)' % nsamples if i == 0 else '_nolabel_'
        ax.plot(Jupp, MCMC_model(theta, x), color='darkred', alpha=.1, label=lab, zorder=1)
    med_model, spread = MCMC_sample_walkers(nsamples, samples, x)
    pslab = r'$1\sigma$ Posterior Spread (%s, %s)' % tuple(print_results(samples))
    FBkwargs = {'fc': '#dede00', 'ec': None, 'alpha': 0.5}
    ax.fill_between(Jupp, med_model-spread, med_model+spread, label=pslab, **FBkwargs, zorder=2)
    theta_max = samples[np.argmax(log_prob)]
    best_fit_model = MCMC_model(theta_max, x)
    bflab = r'Highest likelihood model ($\alpha_{CO}$ = %.2f, log$N_H$ = %.2f)' % tuple(theta_max)
    ax.plot(Jupp, best_fit_model, label=bflab, color='mediumblue', lw=1.5, zorder=3)
    ax.set_ylabel(r'$L_{CO(J \rightarrow J-1)}$ $[L_{\odot}]$')
    ax.set_yscale('log')
    ax.legend(fontsize=7)
    if plot_residuals:
        best_resid = (y - best_fit_model) / best_fit_model
        yerr_resid = yerr/best_fit_model
        mod_lo = ((med_model+spread) - best_fit_model) / best_fit_model
        mod_up = ((med_model-spread) - best_fit_model) / best_fit_model
        ax2.axhline(0, c='mediumblue', lw=.5)
        ax2.fill_between(Jupp, mod_lo, mod_up, **FBkwargs)
        ax2.errorbar(Jupp, best_resid, yerr=yerr_resid, uplims=yulim, capsize=2, lw=1, c='k')
        ax2.set_xticks(Jupp)
        ax2.set_xlabel(r'$J_{upp}$')
        ylo, yup = ax2.get_ylim()
        ax2.set_ylim(-max(abs(ylo), abs(yup)), max(abs(ylo), abs(yup)))
        ax2.set_ylabel('Relative\nresiduals')
        plt.subplots_adjust(hspace=0);
        pngname = '_residuals.png'
    else:
        ax.set_xticks(Jupp)
        ax.set_xlabel(r'$J_{upp}$');
        pngname = '_best.png'
    if savepng:
        plt.savefig('%sMCMC%s' % (savepng, pngname), dpi=300, bbox_inches='tight', facecolor='w')
        plt.close()

def MCMC_cornerplot(samples, ndim, savepng=None):
    labels = ['alphaCO','logNH']
    fig = corner.corner(samples, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[.16,.5,.84])
    ax = np.array(fig.axes).reshape((ndim, ndim))[0,1]
    if savepng:
        plt.savefig('%sMCMC_corner.png' % savepng, dpi=300, bbox_inches='tight', facecolor='w')
        plt.close()

def SLED_residuals(obSLED, bfSLED, loSLED, upSLED):
    dm, dl, du, dulim = obSLED[0], obSLED[0] - obSLED[1], obSLED[0] + obSLED[2], obSLED[3]
    mm, ml, mu = bfSLED, loSLED, upSLED
    resid_bool = [any([dl[j] <= m[j] <= du[j] for m in [mm, ml, mu]]) for j in range(13)]
    resid_ulim = [ml[j] <= dm[j] if dulim[j] else False for j in range(13)]
    resid_zeros = np.invert(np.logical_or(resid_bool, resid_ulim)).astype('int')
    resid = np.min([np.abs(d - m) for d, m in product([dm, dl, du], [mm, ml, mu])], axis=0)
    return resid * resid_zeros

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n
    # Optionally normalize
    if norm:
        acf /= acf[0]
    return acf

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_gw2010(y, c=5.0):  # Goodman & Weare (2010)
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def MCMC_plot_autocorr(sampler, savepng=None):
    names = [r'$\alpha_{CO}$', r'$\log N_H$']
    ndim = len(names)
    fig, axs = plt.subplots(2, ndim, figsize=((ndim*3)+1, 5), gridspec_kw={'height_ratios': [2, 3]})
    TXTkwargs = {'size': 11, 'fontname': 'Manjari', 'ha': 'left'}
    for ax, c, name in zip(axs.T, range(ndim), names):
        chain = sampler.get_chain()[:, :, c].T
        # Histograms
        ax[0].hist(chain.flatten(), 100)
        ax[0].set_yticks([])
        ax[0].set_xlabel(name)
        ax[0].set_ylabel(r'p(%s)' % name)
        ax[0].xaxis.set_label_position('top')
        ax[0].xaxis.tick_top()
        # Compute the estimators for a few different chain lengths
        N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
        gw2010 = np.empty(len(N))
        new = np.empty(len(N))
        for i, n in enumerate(N):
            gw2010[i] = autocorr_gw2010(chain[:, :n])
            new[i] = autocorr_new(chain[:, :n])
        ax[1].loglog(N, gw2010, "o-", label="G&W 2010")
        ax[1].loglog(N, new, "o-", label="new")
        ylim = ax[1].get_ylim()
        ax[1].plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
        ax[1].set_ylim(ylim)
        ax[1].set_xlabel("number of samples, $N$")
        ax[1].set_ylabel(r"$\tau$ estimates")
        ax[1].legend(fontsize=8)
        for axy in axs.T[-1]:
            axy.yaxis.set_label_position('right')
            axy.yaxis.tick_right()
    plt.subplots_adjust(hspace=0.07, wspace=0.05);
    if savepng:
        plt.savefig('%sMCMC_autocorr.png' % savepng, dpi=300, bbox_inches='tight', facecolor='w')
        plt.close()

def MCMC_plot_iterations(sampler, savepng=None):
    names = [r'$\alpha_{CO}$', r'$\log N_H$']
    thetas = sampler.get_chain().T
    ndim = thetas.shape[0]
    fig, axs = plt.subplots(ndim, 1, figsize=(6, ndim*2), sharex=True)
    TXTkwargs = {'size': 13, 'fontname': 'Manjari', 'ha': 'left'}
    for ax, theta, name in zip(axs.flatten(), thetas, names):
        for k, t in enumerate(theta):
            ax.plot(np.arange(len(t)), t, alpha=.8, label=k)
        ax.set_ylabel(name)
    ax.set_xlabel('Iterations')
    ax.legend(title='Walker', bbox_to_anchor=(1, 1.75), loc='upper left')
    plt.subplots_adjust(hspace=0.1);
    if savepng:
        plt.savefig('%sMCMC_iterations.png' % savepng, dpi=300, bbox_inches='tight', facecolor='w')
        plt.close()

