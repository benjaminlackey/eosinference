# TODO: Get rid of the absolute path
import sys
sys.path.insert(0, '/Users/lackey/Research/eosinference/src')

import argparse
import numpy as np
import pandas as pd

import pseudolikelihood as like


def pseudolikelihood_data_from_pe_samples(
    filename,
    kde_bound_limits=[0.5, 1.0, 0.0, 5000.0],
    grid_limits=[0.5, 1.0, 0.0, 5000.0],
    gridsize=250):
    """Open a CSV data file for a BNS MCMC run and evaluate
    ln(pseudolikelihood(q, lambdat)) on a grid.

    Parameters
    ----------
    filename : CSV file
        MCMC samples with column headers named ['mc', 'q', 'lambdat']

    Returns
    -------
    mc_mean : mean value of chirp mass
    lnp_of_ql_grid : 3d array
        [q, lambdat, lnp] for each value of q and lambdat
    """
    # Open csv file as pandas data frame
    #df = pd.read_csv(filename, sep='\s+')
    df = pd.read_csv(filename)

    # Get chirp mass mean
    mc_mean = df['mc'].mean()

    # Construct grid that describes lnp(q, lambdat)
    qs = df['q'].values
    lambdats = df['lam_tilde'].values

    lnp_of_ql_grid = like.construct_lnp_of_ql_grid(
        qs, lambdats, kde_bound_limits, grid_limits,
        gridsize=gridsize, bw_method=None)

    return mc_mean, lnp_of_ql_grid


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
parser.add_argument('--gridsize', type=int, default=250, help='Grid points for q and tildeLambda.')

# Do the argument parsing
args = parser.parse_args()
print('Arguments from command line: {}'.format(args))
print('Results will be saved in {}'.format(args.outfile))


mc_mean_list = []
lnp_of_ql_grid_list = []
for i in range(len(args.pefiles)):
    print('Generating pseudolikelihood for BNS {}'.format(i))
    filename = args.pefiles[i]
    mc_mean, lnp_of_ql_grid = pseudolikelihood_data_from_pe_samples(
    filename,
    kde_bound_limits=[args.qmin, 1.0, 0.0, args.lambdatmax],
    grid_limits=[args.qmin, 1.0, 0.0, args.lambdatmax],
    gridsize=args.gridsize)

    mc_mean_list.append(mc_mean)
    lnp_of_ql_grid_list.append(lnp_of_ql_grid)

print('Saving output.')
like.save_pseudolikelihood_data(args.outfile, mc_mean_list, lnp_of_ql_grid_list)
