import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import timeit
import numpy as np

# Sampler
import emcee

import equationofstate as e
import pseudolikelihood as like
import distributions
import runemcee


parser = argparse.ArgumentParser(description="Sample EOS parameters using emcee.")
# Create a group of named arguments that are required.
# Otherwise the parser will call them optional in the help message
required = parser.add_argument_group('Required named arguments')
required.add_argument('--infile', required=True, help='hdf5 input file containing the pseudolikelihoods')
required.add_argument('--outfile', required=True, help='hdf5 output file for emcee results.')
required.add_argument('--eosname', required=True, help='Name of the EOS model.')
required.add_argument('--distribution', choices=['prior', 'posterior'], required=True, help='The distribution to sample from.')
parser.add_argument('--nwalkers', type=int, default=64, help='Number of walkers for emcee.')
parser.add_argument('--niter', type=int, default=100, help='Number of iterations for emcee.')
parser.add_argument('--nthin', type=int, default=1, help='Only save every nth step.')
parser.add_argument('--qmin', type=float, default=0.125, help='Minimum allowed mass ratio (q=m2/m1<=1).')
parser.add_argument('--lambdatmax', type=float, default=0.125, help='Maxmimum allowed combined tidal parameter tildeLambda.')
parser.add_argument('--mmin', type=float, default=0.5, help='Minimum allowed mass for each NS in the BNS.')
parser.add_argument('--mmax', type=float, default=3.2, help='Maximum allowed mass for each NS in the BNS.')
parser.add_argument('--maxmassmin', type=float, default=1.93,
                    help="""Minimum allowed value for the maximum NS mass (M_sun)
                    calculated for the selected EOS parameters. This should be
                    above 1.93 because there is a pulsar with this mass.
                    """)
parser.add_argument('--maxmassmax', type=float, default=3.2,
                    help="""Maximum allowed value for the maximum NS mass (M_sun)
                    calculated for the selected EOS parameters. This should be
                    less than the causality constraint (~3.2).
                    """)
parser.add_argument('--csmax', type=float, default=1.0, help='Maximum allowed speed of sound (c=1 units).')

# Do the argument parsing
args = parser.parse_args()
print('Arguments from command line: {}'.format(args))
print('Results will be saved in {}'.format(args.outfile))

# Choose the eos model
eos_class_reference = e.choose_eos_model(args.eosname)

# Create the interpolated pseudolikelihood for each BNS event
mc_mean_list, _, lnp_of_ql_grid_list = like.load_pseudolikelihood_data(args.infile)
#mc_mean_list, lnp_of_ql_grid_list = like.load_pseudolikelihood_data(args.infile)
lnp_of_ql_list = [like.interpolate_lnp_of_ql_from_grid(lnp) for lnp in lnp_of_ql_grid_list]
print('The chirp masses from the infile are: {}'.format(mc_mean_list))

# Get the initial parameters for the walkers
# TODO: This currently requires the likelihood data which you shouldn't need
# if you're only sampling a prior.
print('Generating initial walkers.')
walkers0 = runemcee.initial_walker_params(
    args.nwalkers, mc_mean_list, lnp_of_ql_list, args.eosname,
    q_min=args.qmin, lambdat_max=args.lambdatmax, m_min=args.mmin, m_max=args.mmax,
    max_mass_min=args.maxmassmin, max_mass_max=args.maxmassmax, cs_max=args.csmax)
print('Initial walkers shape: {}'.format(walkers0.shape))
print('Walker 0 has parameters {}'.format(walkers0[0]))

# Dimensionality of parameter space.
dim = walkers0.shape[1]

if args.distribution == 'prior':
    # Initialize the sampler
    sampler = emcee.EnsembleSampler(
        args.nwalkers, dim, distributions.log_prior_emcee_wrapper,
        args=[mc_mean_list, eos_class_reference],
        kwargs={
            'q_min':args.qmin, 'lambdat_max':args.lambdatmax,
            'm_min':args.mmin, 'm_max':args.mmax,
            'max_mass_min':args.maxmassmin, 'max_mass_max':args.maxmassmax, 'cs_max':args.csmax},
        threads=1)

    # Do the sampling
    t0 = timeit.time.time()
    chains, lnpost, state = sampler.run_mcmc(walkers0, args.niter)
    t1 = timeit.time.time()

elif args.distribution == 'posterior':
    # Initialize the sampler.
    sampler = emcee.EnsembleSampler(
        args.nwalkers, dim, distributions.log_posterior,
        args=[mc_mean_list, eos_class_reference, lnp_of_ql_list],
        kwargs={
            'q_min':args.qmin, 'lambdat_max':args.lambdatmax,
            'm_min':args.mmin, 'm_max':args.mmax,
            'max_mass_min':args.maxmassmin, 'max_mass_max':args.maxmassmax, 'cs_max':args.csmax},
        threads=1)

    # Do the sampling
    t0 = timeit.time.time()
    chains, lnpost, state = sampler.run_mcmc(walkers0, args.niter)
    t1 = timeit.time.time()

else:
    raise ValueError("distribution must be 'prior' or 'posterior'.")

print('Runtime: {}s'.format(t1-t0))

# Save the emcee run data
print('Saving output.')
runemcee.save_emcee_samples(args.outfile, sampler, mc_mean_list, nthin=args.nthin)
