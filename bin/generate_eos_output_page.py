# TODO: Get rid of the absolute path
import sys
sys.path.insert(0, '/Users/lackey/Research/eosinference/src')
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import corner

import equationofstate as eospp
import utilities as util
import postprocess

matplotlib.rcParams['figure.figsize'] = (9.7082039325, 6.0)
matplotlib.rcParams['xtick.labelsize'] = 20.0
matplotlib.rcParams['ytick.labelsize'] = 20.0
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
required.add_argument('--outdir', required=True, help='Output directory for the plots.')


# Do the argument parsing
args = parser.parse_args()
print('Arguments from command line: {}'.format(args))

# Copy eos_output_page.html to output directory
bin_dir = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()
cmd = 'cp {}/eos_output_page.html {}/.'.format(bin_dir, args.outdir)
print(cmd)
os.system(cmd)


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
print f['prior'].keys(), f['prior'].attrs.keys()
print f['posterior'].keys(), f['posterior'].attrs.keys()
print f['bns_0'].keys(), f['bns_0'].attrs.keys()

f.close()


# TODO: This should not be hardcoded
labels = [r'$\log(p_1)$', r'$\Gamma_1$', r'$\Gamma_2$', r'$\Gamma_3$']

fig = corner.corner(eos_samples_prior, labels=labels, truths=None, plot_density=False, plot_contours=False)
fig.savefig(args.outdir+'/prior_eos.png', format='png', transparent=True, bbox_inches='tight')

fig = corner.corner(eos_samples_post, labels=labels, truths=None, plot_density=False, plot_contours=False)
fig.savefig(args.outdir+'/posterior_eos.png', format='png', transparent=True, bbox_inches='tight')

fig, axes = postprocess.compare_2_runs(
    eos_samples_prior, eos_samples_post,
    xlabels=labels, truths=None, label1='Prior', label2='Posterior')
fig.savefig(args.outdir+'/compare_prior_posterior.png', format='png', transparent=True, bbox_inches='tight')




radius_bounds_prior = postprocess.bounds_from_curves(ms, radius_curves_prior)
radius_bounds_post = postprocess.bounds_from_curves(ms, radius_curves_post)

lambda_bounds_prior = postprocess.bounds_from_curves(ms, lambda_curves_prior)
lambda_bounds_post = postprocess.bounds_from_curves(ms, lambda_curves_post)


fig, ax = plt.subplots()

# MR curves
for i in range(len(radius_curves_prior)):
    rs = radius_curves_prior[i]
    mask = rs>0.0
    ax.plot(ms[mask], rs[mask], c='gray', lw=0.3)

# # True injected value
# eos_params = np.array([34.384, 3.005, 2.988, 2.851])
# eos = eos_class_reference(eos_params)
# mmax = eos.max_mass()
# ms_injected = np.linspace(0.5, mmax, 100)
# rs_injected = np.array([eos.radiusofm(m) for m in ms_injected])
# ax.plot(ms_injected, rs_injected, c='k', lw=2)

ax.minorticks_on()
ax.set_xlabel(r'Mass ($M_\odot$)')
ax.set_ylabel(r'Radius (km)')
ax.set_xlim(0, 3.2)
ax.set_ylim(8, 18)
fig.savefig(args.outdir+'/prior_radius_curves.png', format='png', transparent=True, bbox_inches='tight')

fig, ax = plt.subplots()

# MR curves
for i in range(len(radius_curves_post)):
    rs = radius_curves_post[i]
    mask = rs>0.0
    ax.plot(ms[mask], rs[mask], c='gray', lw=0.3)

# # True injected value
# eos_params = np.array([34.384, 3.005, 2.988, 2.851])
# eos = eos_class_reference(eos_params)
# mmax = eos.max_mass()
# ms_injected = np.linspace(0.5, mmax, 100)
# rs_injected = np.array([eos.radiusofm(m) for m in ms_injected])
# ax.plot(ms_injected, rs_injected, c='k', lw=2)


ax.minorticks_on()
ax.set_xlabel(r'Mass ($M_\odot$)')
ax.set_ylabel(r'Radius (km)')
ax.set_xlim(0, 3.2)
ax.set_ylim(8, 18)
fig.savefig(args.outdir+'/posterior_radius_curves.png', format='png', transparent=True, bbox_inches='tight')


fig, ax = plt.subplots()

# MR curves
for i in range(len(lambda_curves_prior)):
    ls = lambda_curves_prior[i]
    mask = ls>0.0
    ax.plot(ms[mask], ls[mask], c='gray', lw=0.3)

# # True injected value
# eos_params = np.array([34.384, 3.005, 2.988, 2.851])
# eos = eos_class_reference(eos_params)
# mmax = eos.max_mass()
# ms_injected = np.linspace(0.5, mmax, 100)
# ls_injected = np.array([eos.lambdaofm(m) for m in ms_injected])
# ax.plot(ms_injected, ls_injected, c='k', lw=2)

ax.set_yscale('log')
ax.minorticks_on()
ax.grid(which='major')
ax.grid(which='minor', ls=':')
ax.set_xlabel(r'Mass ($M_\odot$)')
ax.set_ylabel(r'$\Lambda$')
ax.set_xlim(0, 3.2)
ax.set_ylim(1, 10000)
fig.savefig(args.outdir+'/prior_lambda_curves.png', format='png', transparent=True, bbox_inches='tight')



fig, ax = plt.subplots()

# MR curves
for i in range(len(lambda_curves_post)):
    ls = lambda_curves_post[i]
    mask = ls>0.0
    ax.plot(ms[mask], ls[mask], c='gray', lw=0.3)

# # True injected value
# eos_params = np.array([34.384, 3.005, 2.988, 2.851])
# eos = eos_class_reference(eos_params)
# mmax = eos.max_mass()
# ms_injected = np.linspace(0.5, mmax, 100)
# ls_injected = np.array([eos.lambdaofm(m) for m in ms_injected])
# ax.plot(ms_injected, ls_injected, c='k', lw=2)

ax.set_yscale('log')
ax.minorticks_on()
ax.grid(which='major')
ax.grid(which='minor', ls=':')
ax.set_xlabel(r'Mass ($M_\odot$)')
ax.set_ylabel(r'$\Lambda$')
ax.set_xlim(0, 3.2)
ax.set_ylim(1, 10000)
fig.savefig(args.outdir+'/posterior_lambda_curves.png', format='png', transparent=True, bbox_inches='tight')




fig, ax = plt.subplots()

bounds = radius_bounds_prior
#ax.fill_between(bounds[:, 0], bounds[:, 1], bounds[:, 2], color='b', alpha=0.1, zorder=2)
ax.fill_between(bounds[:, 0], bounds[:, 3], bounds[:, 4], color='g', alpha=0.1, zorder=1)

bounds = radius_bounds_post
ax.fill_between(bounds[:, 0], bounds[:, 1], bounds[:, 2], color='b', alpha=0.3, zorder=2)
ax.fill_between(bounds[:, 0], bounds[:, 3], bounds[:, 4], color='g', alpha=0.3, zorder=1)

ax.minorticks_on()
ax.grid(which='major')
ax.grid(which='minor', ls=':')
ax.set_xlabel(r'Mass ($M_\odot$)')
ax.set_ylabel(r'Radius (km)')
ax.set_xlim(0.5, 2.5)
ax.set_ylim(9, 16)
fig.savefig(args.outdir+'/prior_radius_bounds.png', format='png', transparent=True, bbox_inches='tight')


fig, ax = plt.subplots()

bounds = lambda_bounds_prior
#ax.fill_between(bounds[:, 0], bounds[:, 1], bounds[:, 2], color='b', alpha=0.1, zorder=2)
ax.fill_between(bounds[:, 0], bounds[:, 3], bounds[:, 4], color='g', alpha=0.1, zorder=1)

bounds = lambda_bounds_post
ax.fill_between(bounds[:, 0], bounds[:, 1], bounds[:, 2], color='b', alpha=0.3, zorder=2)
ax.fill_between(bounds[:, 0], bounds[:, 3], bounds[:, 4], color='g', alpha=0.3, zorder=1)

ax.set_yscale('log')
ax.minorticks_on()
ax.grid(which='major')
ax.grid(which='minor', ls=':')
ax.set_xlabel(r'Mass ($M_\odot$)')
ax.set_ylabel(r'$\Lambda$')
ax.set_xlim(0, 3.2)
ax.set_ylim(1, 10000)
fig.savefig(args.outdir+'/prior_lambda_bounds.png', format='png', transparent=True, bbox_inches='tight')


# Evaluate Mass-Radius contours for each star of each BNS
radius_cds_list = []
for i in range(len(single_bns_properties_list)):
    p = single_bns_properties_list[i]
    cds = postprocess.mr_contour_data(p['m1'], p['r1'], p['m2'], p['r2'], gridsize=250)
    radius_cds_list.append(cds)


def plot_single_event_radii(ax, cd1, cd2, i):
    c = 'r'
    cd1 = cds['cd1']
    cd2 = cds['cd2']

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

    ax.legend(ncol=2)
    ax.minorticks_on()
    ax.grid(which='major')
    ax.grid(which='minor', ls=':')
    ax.set_xlabel(r'Mass ($M_\odot$)')
    ax.set_ylabel(r'Radius (km)')
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(8, 16)


nbns = len(single_bns_properties_list)

fig, axes = plt.subplots(nbns, 1, figsize=(8, 6*nbns))

for i in range(nbns):
    ax = axes[i]
    bounds = radius_bounds_post
    ax.fill_between(bounds[:, 0], bounds[:, 1], bounds[:, 2], color='b', alpha=0.3, zorder=2, label=r'50\% intervals')
    ax.fill_between(bounds[:, 0], bounds[:, 3], bounds[:, 4], color='g', alpha=0.3, zorder=1, label=r'90\% intervals')

    cd1 = radius_cds_list[i]['cd1']
    cd2 = radius_cds_list[i]['cd2']
    plot_single_event_radii(ax, cd1, cd2, i)

fig.savefig(args.outdir+'/radius_bns.png', format='png', transparent=True, bbox_inches='tight')


lambda_cds_list = []
for i in range(len(single_bns_properties_list)):
    p = single_bns_properties_list[i]
    cds = postprocess.mlambda_contour_data(p['m1'], p['l1'], p['m2'], p['l2'], gridsize=250)
    lambda_cds_list.append(cds)


def plot_single_event_lambda(ax, cd1, cd2, i):
    c = 'r'
    cd1 = cds['cd1']
    cd2 = cds['cd2']

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

    ax.legend(ncol=1)
    ax.minorticks_on()
    ax.grid(which='major')
    ax.grid(which='minor', ls=':')
    ax.set_xlabel(r'Mass ($M_\odot$)')
    ax.set_ylabel(r'$\Lambda$')
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(1, 2000)


nbns = len(single_bns_properties_list)

fig, axes = plt.subplots(nbns, 1, figsize=(8, 6*nbns))

for i in range(nbns):
    ax = axes[i]
    bounds = lambda_bounds_post
    ax.fill_between(bounds[:, 0], bounds[:, 1], bounds[:, 2], color='b', alpha=0.3, zorder=2, label=r'50\% intervals')
    ax.fill_between(bounds[:, 0], bounds[:, 3], bounds[:, 4], color='g', alpha=0.3, zorder=1, label=r'90\% intervals')

    cd1 = lambda_cds_list[i]['cd1']
    cd2 = lambda_cds_list[i]['cd2']
    plot_single_event_lambda(ax, cd1, cd2, i)

fig.savefig(args.outdir+'/lambda_bns.png', format='png', transparent=True, bbox_inches='tight')


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
