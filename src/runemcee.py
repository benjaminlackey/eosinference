"""Utility functions for running inference with the emcee sampler. Methods are
provided to save and load emcee samples, and to generate initial guesses for the
emcee walkers.
"""

import numpy as np
import h5py

import distributions
import equationofstate as e

################################################################################
#      Save and load data from emcee run                                       #
################################################################################

def save_emcee_samples(filename, sampler, mc_mean_list, nthin=1):
    """Save the emcee run to an hdf5 fileself.

    Parameters
    ----------
    filename : string
        Name of hdf5 file
    sampler : emcee.EnsembleSampler object
        Contains the data for the emcee run
    mc_mean_list : 1d array
        List of chirp masses for each BNS event
    nthin : int
        Thin each chain using a stride of nthin
    """
    f = h5py.File(filename)
    f['mc_mean'] = np.array(mc_mean_list)
    f['lnprob'] = sampler.lnprobability[:, ::nthin]
    f['samples'] = sampler.chain[:, ::nthin, :]
    f.close()


def load_emcee_samples(filename):
    """Load the emcee run data stored as hdf5 file.

    Parameters
    ----------
    filename : string
        Name of hdf5 file

    Returns
    -------
    mc_mean_list : 1d array
        List of chirp masses for each BNS event
    lnp : 2d array
        ln(posterior) for each walker
    samples : 3d array
        chains for each walker
    """
    f = h5py.File(filename)
    mc_mean_list = f['mc_mean'][:]
    lnp = f['lnprob'][:]
    samples = f['samples'][:]
    f.close()
    return mc_mean_list, lnp, samples


################################################################################
#      Generate initial walkers for the 4-piece piecewise polytrope EOS.       #
################################################################################

# TODO: This is specific to the EOS parameterization. It should go in the specific EOS module.
def single_initial_walker_params(
    mc_mean_list, lnp_of_ql_list, eosname,
    q_min=0.5, lambdat_max=10000,
    m_min=0.5, m_max=3.2,
    max_mass_min=1.93, max_mass_max=3.2, cs_max=1.0):
    """Sample a point for the initial walker parameters.
    """
    n_binaries = len(mc_mean_list)

    # the equivalent of a do while loop:
    n=0
    while True:
        n+=1

        # Draw each of the mass ratios
        qs = np.random.uniform(low=q_min, high=1.0, size=n_binaries)

        # Draw EOS params
        eos_class_reference = e.choose_eos_model(eosname)
        eos_params = e.initialize_walker_eos_params(eosname)

        params = np.concatenate((qs, eos_params))

        # Accept the parameters only if the posterior is nonzero
        try:
            lpost = distributions.log_posterior(
                params, mc_mean_list, eos_class_reference, lnp_of_ql_list,
                q_min=q_min, lambdat_max=lambdat_max,
                m_min=m_min, m_max=m_max,
                max_mass_min=max_mass_min, max_mass_max=max_mass_max, cs_max=cs_max)
            if lpost!=distributions.log_zero:
                print n,
                return params
        except RuntimeError:
            pass


def initial_walker_params(
    nwalkers, mc_mean_list, lnp_of_ql_list, eosname,
    q_min=0.5, lambdat_max=10000,
    m_min=0.5, m_max=3.2,
    max_mass_min=1.93, max_mass_max=3.2, cs_max=1.0):
    """The initial points for the walkers.
    """
    walkers = []
    for i in range(nwalkers):
        p = single_initial_walker_params(
            mc_mean_list, lnp_of_ql_list, eosname,
            q_min=q_min, lambdat_max=lambdat_max, m_min=m_min, m_max=m_max,
            max_mass_min=max_mass_min, max_mass_max=max_mass_max, cs_max=cs_max)
        walkers.append(p)

    return np.array(walkers)
