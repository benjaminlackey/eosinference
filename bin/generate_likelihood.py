import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import numpy as np
import pandas as pd

import pseudolikelihood as like


def pseudolikelihood_data_from_pe_samples(
    filename,
    kde_bound_limits=[0.5, 1.0, 0.0, 10000.0],
    grid_limits=[0.5, 1.0, 0.0, 10000.0],
    gridsize=500):
    """Open a CSV data file for a BNS MCMC run and evaluate
    ln(pseudolikelihood(q, lambdat)) on a grid.

    Parameters
    ----------
    filename : CSV file
        MCMC samples with column headers named ['mc', 'q', 'lambdat']

    Returns
    -------
    mc_q_lambdat : 2d array of samples with just ['mc', 'q', 'lambdat'] samples
    lnp_of_ql_grid : 3d array
        [q, lambdat, lnp] for each value of q and lambdat
    """
    # Open csv file as pandas DataFrame
    df = pd.read_csv(filename)
    # Get chirp mass mean
    mcs = df['mc']
    qs = df['q'].values
    lambdats = df['lambdat'].values
    mc_q_lambdat = np.array([mcs, qs, lambdats]).T

    # Check that the KDE bound limits are appropriate.
    # The bounded KDE returns garbage if there are samples beyond the bounds.
    qlow, qhigh, lambdatlow, lambdathigh = kde_bound_limits
    if (qs.min()<qlow or qs.max()>qhigh or
        lambdats.min()<lambdatlow or lambdats.max()>lambdathigh):
        raise ValueError('There are MCMC samples beyond the boundaries of kde_bound_limits.')

    # Check number of grid points per standard deviation.
    # There should be enough samples to accurately interpolate.
    sigma_q = np.std(qs)
    sigma_l = np.std(lambdats)
    qlow, qhigh, lambdatlow, lambdathigh = grid_limits
    q_grid = np.linspace(qlow, qhigh, gridsize)
    l_grid = np.linspace(lambdatlow, lambdathigh, gridsize)
    dq = q_grid[1] - q_grid[0]
    dl = l_grid[1] - l_grid[0]
    points_per_sigma_q = sigma_q/dq
    points_per_sigma_l = sigma_l/dl
    print('q: std {}, grid spacing {}, points per std {}'.format(sigma_q, dq, points_per_sigma_q))
    print('lambdat: std {}, grid spacing {}, points per std {}'.format(sigma_l, dl, points_per_sigma_l))
    spacing_min = 2.0
    if points_per_sigma_q<spacing_min or points_per_sigma_l<spacing_min:
        raise ValueError('There should be >{} grid points per standard deviation.'.format(spacing_min))

    # Construct grid that describes lnp(q, lambdat)
    lnp_of_ql_grid = like.construct_lnp_of_ql_grid(
        qs, lambdats, kde_bound_limits, grid_limits,
        gridsize=gridsize, bw_method=None)

    return mc_q_lambdat, lnp_of_ql_grid


parser = argparse.ArgumentParser(description="Calculate the pseudolikelihood for each BNS system.")
required = parser.add_argument_group('Required named arguments')
required.add_argument(
    '--pefiles', required=True, nargs='+',
    help="""List of MCMC input csv files for each BNS.
    Column headers must contain ['mc', 'q', 'lambdat'].
    The priors in 'q' and 'lambdat' must be uniform.
    """)
required.add_argument('--outfile', required=True, help='hdf5 output file for pseudolikelihoods.')
parser.add_argument('--qmin', type=float, default=0.5, help='Minimum mass ratio.')
parser.add_argument('--lambdatmax', type=float, default=10000, help='Maximum tildeLambda.')
parser.add_argument('--gridsize', type=int, default=500, help='Grid points for q and tildeLambda.')

# Do the argument parsing
args = parser.parse_args()
print('Arguments from command line: {}'.format(args))
print('Results will be saved in {}'.format(args.outfile))


mc_q_lambdat_list = []
lnp_of_ql_grid_list = []
for i in range(len(args.pefiles)):
    print('Generating pseudolikelihood for BNS {}'.format(i))
    filename = args.pefiles[i]
    mc_q_lambdat, lnp_of_ql_grid = pseudolikelihood_data_from_pe_samples(
        filename,
        kde_bound_limits=[args.qmin, 1.0, 0.0, args.lambdatmax],
        grid_limits=[args.qmin, 1.0, 0.0, args.lambdatmax],
        gridsize=args.gridsize)
    mc_q_lambdat_list.append(mc_q_lambdat)
    lnp_of_ql_grid_list.append(lnp_of_ql_grid)

print('Saving output.')
print(len(mc_q_lambdat_list))
like.save_pseudolikelihood_data(args.outfile, mc_q_lambdat_list, lnp_of_ql_grid_list)
