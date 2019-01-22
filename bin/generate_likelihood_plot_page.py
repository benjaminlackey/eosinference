import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py

import pseudolikelihood as like

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


parser = argparse.ArgumentParser(description="Make plots of the likelihoods for each BNS system.")
required = parser.add_argument_group('Required named arguments')
required.add_argument('--infile', required=True, help='hdf5 file for pseudolikelihoods.')
required.add_argument('--outdir', required=True, help='Output directory for the plots.')

# Do the argument parsing
args = parser.parse_args()
print('Arguments from command line: {}'.format(args))


################################################################################
# Read data file                                                               #
################################################################################

mc_mean_list, mc_q_lambdat_list, lnp_of_ql_grid_list = like.load_pseudolikelihood_data(args.infile)
nbns = len(mc_q_lambdat_list)


################################################################################
# Make html file                                                               #
################################################################################

def write_page(nbns, outdir):
    f = open(outdir+'/pseudolikelihood.html', 'w')

    s0 = r"""<!DOCTYPE html>
<html>

<head>
  <title>Pseudolikelihood plot page</title>
</head>

<body>

<h1 align=center>Plots of ln[Likelihood(q, lambdat)] for each BNS system</h1>
"""
    f.write(s0)

    for i in range(nbns):
        s1 = r'<h1>BNS {}</h1>'.format(i)
        s2 = r'<p><img src="pseudolikelihood_bns_{}.png"></p>'.format(i)
        f.write(s1 + '\n')
        f.write(s2 + '\n')

    s3 = r"""
</body>
</html>
"""
    f.write(s3)

    f.close()

write_page(nbns, args.outdir)


################################################################################
# Make likelihood plots                                                        #
################################################################################
print('Making plots.')


def plot_pseudolikelihood(mc_q_lambdat, lnp_of_ql_grid):
    """Make a bunch of plots that show the ln(likelihood) as a function of q and lambdat.
    Make sure they agree with the samples.
    """
    # Extract samples
    qs = mc_q_lambdat[:, 1]
    lambdats = mc_q_lambdat[:, 2]
    mu_q = np.mean(qs)
    mu_l = np.mean(lambdats)
    sigma_q = np.std(qs)
    sigma_l = np.std(lambdats)

    # Construct interpolating function from grid
    qmin = lnp_of_ql_grid[0, 0, 0]
    qmax = lnp_of_ql_grid[-1, 0, 0]
    lmin = lnp_of_ql_grid[0, 0, 1]
    lmax = lnp_of_ql_grid[0, -1, 1]
    ngrid = lnp_of_ql_grid.shape[0]
    lnp_of_ql = like.interpolate_lnp_of_ql_from_grid(lnp_of_ql_grid)

    # Set up figure grid
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.2], hspace=0, figure=fig)
    # Column 0
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    # Column 1
    ax3 = plt.subplot(gs[0, 1])
    ax4 = plt.subplot(gs[1, 1])
    # Column 2 (span all of the rows)
    ax5 = plt.subplot(gs[:, 2])


    ####### Plot likelihood as function of q #######

    q_grid = np.linspace(qmin, qmax, ngrid)
    l_grid = mu_l + np.linspace(-5, 5, 11)*sigma_l
    l_grid = l_grid[(l_grid>=lmin) & (l_grid<=lmax)]

    ax1.hist(qs, bins=q_grid, label='Samples')
    ax1.set_xticks([])
    ax1.legend()

    for l in l_grid:
        lnps = lnp_of_ql(q_grid, l)
        ax2.plot(q_grid, lnps, label=r'$\tilde\Lambda={:.0f}$'.format(l))
    ax2.set_ylim(-30, 0)
    ax2.legend(loc=(-0.05, 2.0), ncol=3, fontsize=12, frameon=False)
    ax2.set_xlabel(r'$q$')
    ax2.set_ylabel(r'$\ln(p)$')


    ####### Plot likelihood as function of lambdat #######

    l_grid = np.linspace(lmin, lmax, ngrid)
    q_grid = np.linspace(qmin, qmax, 11)

    ax3.hist(lambdats, bins=l_grid, label='Samples')
    ax3.set_xticks([])
    ax3.legend()

    for q in q_grid:
        lnps = lnp_of_ql(q, l_grid)
        ax4.plot(l_grid, lnps, label=r'$q={:.3f}$'.format(q))
    ax4.set_ylim(-30, 0)
    ax4.legend(loc=(-0.05, 2.0), ncol=3, fontsize=12, frameon=False)
    ax4.set_xlabel(r'$\tilde\Lambda$')
    ax4.set_ylabel(r'$\ln(p)$')


    ########### contour plot ##########

    # Plot samples
    ax5.scatter(qs, lambdats, marker='.', s=2, c='gray')

    # Evaluate function on grid
    q_grid = np.linspace(qmin, qmax, ngrid)
    l_grid = np.linspace(lmin, lmax, ngrid)
    qmesh, lmesh = np.meshgrid(q_grid, l_grid, indexing='ij')
    lnpmesh = np.array([[lnp_of_ql(q, l) for l in l_grid] for q in q_grid])

    # Get contour levels
    zmax = np.max(lnpmesh)
    log_levels_flipped = np.logspace(-0, 5, 20)
    levels = zmax - np.flip(log_levels_flipped)

    # Plot contours
    cs = ax5.contour(qmesh, lmesh, lnpmesh, levels)
    plt.clabel(cs, inline=1, colors='k', fmt='%.0f', fontsize=10)
    ax5.minorticks_on()

    ax5.set_xlabel(r'$q$')
    ax5.set_ylabel(r'$\tilde\Lambda$')

    return fig, ax1, ax2, ax3, ax4, ax5


for i in range(nbns):
    print('BNS {}'.format(i))
    mc_q_lambdat = mc_q_lambdat_list[i]
    lnp_of_ql_grid = lnp_of_ql_grid_list[i]
    fig, ax1, ax2, ax3, ax4, ax5 = plot_pseudolikelihood(mc_q_lambdat, lnp_of_ql_grid)
    fig.savefig(args.outdir+'/pseudolikelihood_bns_{}.png'.format(i), format='png', transparent=True, bbox_inches='tight')
