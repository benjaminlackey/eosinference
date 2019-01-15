import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import corner

import equationofstate as eospp
import utilities as util
import runemcee
import postprocess

matplotlib.rcParams['figure.figsize'] = (9.7082039325, 6.0)
matplotlib.rcParams['xtick.labelsize'] = 20.0
matplotlib.rcParams['ytick.labelsize'] = 20.0
matplotlib.rcParams['axes.titlesize'] = 25.0
matplotlib.rcParams['axes.labelsize'] = 25.0
matplotlib.rcParams['legend.fontsize'] = 17.0
matplotlib.rcParams['font.family']= 'Times New Roman'
matplotlib.rcParams['font.sans-serif']= ['Bitstream Vera Sans']
matplotlib.rcParams['text.usetex']= True
matplotlib.rcParams['mathtext.fontset']= 'stixsans'


parser = argparse.ArgumentParser(description="Sample EOS parameters using emcee.")
# Create a group of named arguments that are required.
# Otherwise the parser will call them optional in the help message
required = parser.add_argument_group('Required named arguments')
required.add_argument('--infile', required=True, help='hdf5 input file containing the ns_properties')
required.add_argument('--priorfile', required=True, help='hdf5 output file for prior emcee run.')
required.add_argument('--posteriorfile', required=True, help='hdf5 output file for posterior emcee run.')
required.add_argument('--outdir', required=True, help='Output directory for the plots.')


# Do the argument parsing
args = parser.parse_args()
print('Arguments from command line: {}'.format(args))

# Copy eos_output_page.html to output directory
bin_dir = os.path.dirname(os.path.realpath(__file__))
#cwd = os.getcwd()
#os.mkdir(args.outdir)
cmd = 'cp {}/eos_output_page.html {}/.'.format(bin_dir, args.outdir)
print(cmd)
os.system(cmd)


################################################################################
# Extract contents of ns_properties.hdf5 data file.                            #
################################################################################
print('Reading data file.')

f = h5py.File(args.infile)

# Data from prior run
eos_samples_prior = f['prior/eos_samples'][:]
mmax_prior = f['prior/mmax_samples'][:]
ms = f['prior/mass_grid'][:]
radius_curves_prior = f['prior/radius_curves'][:]
lambda_curves_prior = f['prior/lambda_curves'][:]

# Data from posterior run
eos_samples_post = f['posterior/eos_samples'][:]
mmax_post = f['posterior/mmax_samples'][:]
ms = f['posterior/mass_grid'][:]
radius_curves_post = f['posterior/radius_curves'][:]
lambda_curves_post = f['posterior/lambda_curves'][:]

# Data from posterior run for each BNS
single_bns_properties_list = []
nbns = len(f.keys()) - 2
for i in range(nbns):
    group = f['bns_{}'.format(i)]
    d = {}
    d['mc_mean'] = group.attrs['mc_mean']
    d['q'] = group['q'][:]
    d['lambdat'] = group['lambdat'][:]
    d['m1'] = group['m1'][:]
    d['m2'] = group['m2'][:]
    d['r1'] = group['r1'][:]
    d['r2'] = group['r2'][:]
    d['l1'] = group['l1'][:]
    d['l2'] = group['l2'][:]
    single_bns_properties_list.append(d)

# Print some things
print(f['prior'].keys(), f['prior'].attrs.keys())
print(f['posterior'].keys(), f['posterior'].attrs.keys())
print(f['bns_0'].keys(), f['bns_0'].attrs.keys())

f.close()

print('Number of samples for prior: {}'.format(len(eos_samples_prior)))
print('Number of samples for posterior: {}'.format(len(eos_samples_post)))


################################################################################
# Plot original, uncleaned chains for prior and posterior.                     #
################################################################################
print('Plotting original emcee chains of prior and posterior.')

# Load prior
filename = args.priorfile
mc_mean_prior, lnprob_prior, samples_prior = runemcee.load_emcee_samples(filename)
print mc_mean_prior.shape, lnprob_prior.shape, samples_prior.shape

# Load posterior
filename = args.posteriorfile
mc_mean_post, lnprob_post, samples_post = runemcee.load_emcee_samples(filename)
print mc_mean_post.shape, lnprob_post.shape, samples_post.shape

# TODO: EOS labels should not be hardcoded
# Get parameter labels for plots.
eoslabels = [r'$\log(p_1)$', r'$\Gamma_1$', r'$\Gamma_2$', r'$\Gamma_3$']
qlabels = [r'$q_{}$'.format(i) for i in range(nbns)]
labels = qlabels + eoslabels

fig, axes = postprocess.plot_emcee_chains(lnprob_prior, samples_prior, labels=labels, truths=None)
fig.savefig(args.outdir+'/prior_chains.png', format='png', transparent=True, bbox_inches='tight')

fig, axes = postprocess.plot_emcee_chains(lnprob_post, samples_post, labels=labels, truths=None)
fig.savefig(args.outdir+'/posterior_chains.png', format='png', transparent=True, bbox_inches='tight')

################################################################################
# EOS parameter corner plots for prior and posterior.                          #
# Overlap 1d histograms for prior and posterior.                               #
################################################################################
print('Making corner plots and histograms of prior and posterior.')

fig = corner.corner(eos_samples_prior, labels=eoslabels, truths=None, plot_density=False, plot_contours=False)
fig.savefig(args.outdir+'/prior_eos.png', format='png', transparent=True, bbox_inches='tight')

fig = corner.corner(eos_samples_post, labels=eoslabels, truths=None, plot_density=False, plot_contours=False)
fig.savefig(args.outdir+'/posterior_eos.png', format='png', transparent=True, bbox_inches='tight')

fig, axes = postprocess.compare_2_runs(
    eos_samples_prior, eos_samples_post,
    xlabels=eoslabels, truths=None, label1='Prior', label2='Posterior')
fig.savefig(args.outdir+'/compare_prior_posterior.png', format='png', transparent=True, bbox_inches='tight')


################################################################################
# Plot all curves for Radius(Mass) and Lambda(Mass).                           #
################################################################################
print('Plotting Radius(Mass) curves and Lambda(Mass) curves for prior and posterior.')

def plot_mass_curves(ax, ms, curves, lowerbound=0.0, ncurves=None):
    """Plot all x(mass) curves.
    Mask the curves after they reach the maximum mass,
    by removing and point with value x<=lowerbound.
    """
    # The number of curves to plot
    if ncurves is None:
        n = len(curves)
    else:
        n = min(ncurves, len(curves))

    # plot the x(mass) curves
    for i in range(n):
        xs = curves[i]
        mask = xs>lowerbound
        ax.plot(ms[mask], xs[mask], c='gray', lw=0.3)

    ax.minorticks_on()
    ax.grid(which='major', zorder=0)
    ax.grid(which='minor', ls=':', zorder=0)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

curves = radius_curves_prior
plot_mass_curves(ax1, ms, curves, lowerbound=0.0, ncurves=None)
ax1.set_title('Prior')
ax1.set_xlabel(r'Mass ($M_\odot$)')
ax1.set_ylabel(r'Radius (km)')
ax1.set_xlim(0.5, 3.5)
ax1.set_ylim(8, 18)

curves = radius_curves_post
ax2.set_title('Posterior')
plot_mass_curves(ax2, ms, curves, lowerbound=0.0, ncurves=None)
ax2.set_xlabel(r'Mass ($M_\odot$)')
ax2.set_ylabel(r'Radius (km)')
ax2.set_xlim(0.5, 3.5)
ax2.set_ylim(8, 18)

fig.savefig(args.outdir+'/radius_curves.png', format='png', transparent=True, bbox_inches='tight')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

curves = lambda_curves_prior
plot_mass_curves(ax1, ms, curves, lowerbound=0.0, ncurves=None)
ax1.set_title('Prior')
ax1.set_yscale('log')
ax1.set_xlabel(r'Mass ($M_\odot$)')
ax1.set_ylabel(r'$\Lambda$')
ax1.set_xlim(0.5, 3.5)
ax1.set_ylim(1, 10000)

curves = lambda_curves_post
plot_mass_curves(ax2, ms, curves, lowerbound=0.0, ncurves=None)
ax2.set_title('Posterior')
ax2.set_yscale('log')
ax2.set_xlabel(r'Mass ($M_\odot$)')
ax2.set_ylabel(r'$\Lambda$')
ax2.set_xlim(0.5, 3.5)
ax2.set_ylim(1, 10000)

fig.savefig(args.outdir+'/lambda_curves.png', format='png', transparent=True, bbox_inches='tight')


################################################################################
# Plot confidence intervals for Radius(Mass) and Lambda(Mass).                 #
################################################################################
print('Plotting Radius(Mass) bounds and Lambda(Mass) bounds for prior and posterior.')

def plot_3_bounds_of_mass(ax, bounds):
    """Plot intervals as a function of mass for 3 different percentiles.
    """
    # Largest percentile (should be 100%)
    ax.plot(bounds[2]['ms'], bounds[2]['lows'],
            color='k', ls='--', lw=1, zorder=5, label='{}\%'.format(bounds[2]['p']))
    ax.plot(bounds[2]['ms'], bounds[2]['highs'],
            color='k', ls='--', lw=1, zorder=1)

    # Second largest percentile (e.g. 90%)
    ax.fill_between(bounds[1]['ms'], bounds[1]['lows'], bounds[1]['highs'],
                    color='g', alpha=0.3, zorder=2, label='{}\%'.format(bounds[1]['p']))

    # Smallest percentile (e.g. 50%)
    ax.fill_between(bounds[0]['ms'], bounds[0]['lows'], bounds[0]['highs'],
                    color='b', alpha=0.3, zorder=3, label='{}\%'.format(bounds[0]['p']))

    ax.minorticks_on()
    ax.grid(which='major', zorder=0)
    ax.grid(which='minor', ls=':', zorder=0)
    ax.legend(loc='upper right')


# Calculate the bounds on radius and Lambda for both prior and posterior
radius_bounds_prior = postprocess.bounds_from_curves(ms, radius_curves_prior, percentiles=[50, 90, 100])
radius_bounds_post = postprocess.bounds_from_curves(ms, radius_curves_post, percentiles=[50, 90, 100])
lambda_bounds_prior = postprocess.bounds_from_curves(ms, lambda_curves_prior, percentiles=[50, 90, 100])
lambda_bounds_post = postprocess.bounds_from_curves(ms, lambda_curves_post, percentiles=[50, 90, 100])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

bounds = radius_bounds_prior
plot_3_bounds_of_mass(ax1, bounds)
ax1.set_title('Prior')
ax1.set_xlabel(r'Mass ($M_\odot$)')
ax1.set_ylabel(r'Radius (km)')
ax1.set_xlim(0.5, 3.5)
ax1.set_ylim(8, 18)

bounds = radius_bounds_post
plot_3_bounds_of_mass(ax2, bounds)
ax2.set_title('Posterior')
ax2.set_xlabel(r'Mass ($M_\odot$)')
ax2.set_ylabel(r'Radius (km)')
ax2.set_xlim(0.5, 3.5)
ax2.set_ylim(8, 18)

fig.savefig(args.outdir+'/radius_bounds.png', format='png', transparent=True, bbox_inches='tight')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

bounds = lambda_bounds_prior
plot_3_bounds_of_mass(ax1, bounds)
ax1.set_title('Prior')
ax1.set_yscale('log')
ax1.set_xlabel(r'Mass ($M_\odot$)')
ax1.set_ylabel(r'$\Lambda$')
ax1.set_xlim(0.5, 3.5)
ax1.set_ylim(1, 10000)

bounds = lambda_bounds_post
plot_3_bounds_of_mass(ax2, bounds)
ax2.set_title('Posterior')
ax2.set_yscale('log')
ax2.set_xlabel(r'Mass ($M_\odot$)')
ax2.set_ylabel(r'$\Lambda$')
ax2.set_xlim(0.5, 3.5)
ax2.set_ylim(1, 10000)

fig.savefig(args.outdir+'/lambda_bounds.png', format='png', transparent=True, bbox_inches='tight')


################################################################################
# Plot R1(m1), R2(m2) credible regions for each BNS system, and                #
# Plot Lambda1(m1), Lambda2(m2) credible regions for each BNS system.          #
################################################################################

def plot_single_event_confidence_regions(ax, cd1, cd2, i):
    c = 'r'

    # Plot NS1
    util.plot_posterior_with_contours(
        ax, cd1, cmap=None, levels=[50, 90],
        linewidths=[2, 2], linestyles=['--', '-'],
        colors=[c, c], white_contour_back=True)

    # Plot NS2
    util.plot_posterior_with_contours(
        ax, cd2, cmap=None, levels=[50, 90],
        linewidths=[2, 2], linestyles=['--', '-'],
        colors=[c, c], white_contour_back=True)

    # This is just for making a legend
    ax.scatter([0], [0], marker='o',
               s=200, edgecolor=c, linewidth=2, linestyle='--', facecolor='none',
               label='50\% regions, BNS {}'.format(i))
    ax.scatter([0], [0], marker='o',
               s=200, edgecolor=c, linewidth=2, linestyle='-', facecolor='none',
               label='90\% regions, BNS {}'.format(i))

    ax.minorticks_on()
    ax.grid(which='major', zorder=0)
    ax.grid(which='minor', ls=':', zorder=0)
    ax.legend(loc='upper right', ncol=1, frameon=False)


print('Calculating R1(m1), R2(m2) credible regions for each BNS system:')

# Evaluate Mass-Radius contours for each star of each BNS
radius_cds_list = []
for i in range(len(single_bns_properties_list)):
    print('BNS {} of {}'.format(i, nbns))
    p = single_bns_properties_list[i]
    cds = postprocess.mr_contour_data(p['m1'], p['r1'], p['m2'], p['r2'], gridsize=250)
    radius_cds_list.append(cds)


bounds = radius_bounds_post
nbns = len(single_bns_properties_list)
fig, axes = plt.subplots(nbns, 1, figsize=(8, 6*nbns))

for i in range(nbns):
    if nbns==1:
        ax = axes
    else:
        ax = axes[i]

    plot_3_bounds_of_mass(ax, bounds)

    cd1 = radius_cds_list[i]['cd1']
    cd2 = radius_cds_list[i]['cd2']
    plot_single_event_confidence_regions(ax, cd1, cd2, i)

    ax.set_xlabel(r'Mass ($M_\odot$)')
    ax.set_ylabel(r'Radius (km)')
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(8, 16)

fig.savefig(args.outdir+'/radius_bns.png', format='png', transparent=True, bbox_inches='tight')


print('Calculating Lambda1(m1), Lambda2(m2) credible regions for each BNS system:')

lambda_cds_list = []
for i in range(len(single_bns_properties_list)):
    print('BNS {} of {}'.format(i, nbns))
    p = single_bns_properties_list[i]
    cds = postprocess.mlambda_contour_data(p['m1'], p['l1'], p['m2'], p['l2'], gridsize=250)
    lambda_cds_list.append(cds)


bounds = lambda_bounds_post
nbns = len(single_bns_properties_list)
fig, axes = plt.subplots(nbns, 1, figsize=(8, 6*nbns))

for i in range(nbns):
    if nbns==1:
        ax = axes
    else:
        ax = axes[i]

    plot_3_bounds_of_mass(ax, bounds)

    cd1 = lambda_cds_list[i]['cd1']
    cd2 = lambda_cds_list[i]['cd2']
    plot_single_event_confidence_regions(ax, cd1, cd2, i)

    #ax.set_yscale('log')
    ax.set_xlabel(r'Mass ($M_\odot$)')
    ax.set_ylabel(r'$\Lambda$')
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0, 2000)

fig.savefig(args.outdir+'/lambda_bns.png', format='png', transparent=True, bbox_inches='tight')


################################################################################
# Plot Maximum mass histogram for prior and posterior.                         #
################################################################################
print('Plotting Maximum mass histogram for prior and posterior.')

fig, ax = plt.subplots()

bins = np.linspace(0.5, 3.5, 100)
ax.hist(mmax_prior, bins=bins, density=True, histtype='step', label='Prior')
ax.hist(mmax_post, bins=bins, density=True, histtype='step', label='Posterior')

ax.legend()
ax.minorticks_on()
ax.set_xlim(1.8, 3.5)
ax.set_xlabel(r'$M_{\rm max}$ ($M_\odot$)')
ax.set_ylabel('PDF')
fig.savefig(args.outdir+'/mmax.png', format='png', transparent=True, bbox_inches='tight')
