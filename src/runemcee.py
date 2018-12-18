import numpy as np
import h5py

import posterior
import equationofstate as eospp


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

def single_initial_walker_params_eospp(
    mc_mean_list, lnp_of_ql_list,
    q_min=0.5, m_min=0.5, m_max=3.2, mass_known=1.93, vs_limit=1.0):
    """Sample a point from the prior.
    """
    eos_class_reference = eospp.EOS4ParameterPiecewisePolytropeGammaParams
    n_binaries = len(mc_mean_list)

    # the equivalent of a do while loop:
    n=0
    while True:
        n+=1

        # Draw each of the mass ratios
        qs = np.random.uniform(low=q_min, high=1.0, size=n_binaries)
        # Draw EOS params
        lp = np.random.uniform(34.3, 34.7)
        g1 = np.random.uniform(2.5, 3.5)
        g2 = np.random.uniform(2.5, 3.5)
        g3 = np.random.uniform(2.5, 3.5)

        # Put the sampled parameters in the right order
        eos_params = np.array([lp, g1, g2, g3])
        params = np.concatenate((qs, eos_params))

        # Accept the parameters only if the posterior is nonzero
        try:
            lpost = posterior.log_posterior(
                params, mc_mean_list, eos_class_reference, lnp_of_ql_list,
                q_min=q_min, m_min=m_min, m_max=m_max, mass_known=mass_known, vs_limit=vs_limit)
            if lpost!=posterior.log_zero:
                print n,
                return params
        except RuntimeError:
            pass


def initial_walker_params(
    nwalkers, mc_mean_list, lnp_of_ql_list,
    q_min=0.5, m_min=0.5, m_max=3.2, mass_known=1.93, vs_limit=1.0):
    """The initial points for the walkers.
    """
    walkers = []
    for i in range(nwalkers):
        p = single_initial_walker_params_eospp(
            mc_mean_list, lnp_of_ql_list,
            q_min=q_min, m_min=m_min, m_max=m_max, mass_known=mass_known, vs_limit=vs_limit)
        walkers.append(p)

    return np.array(walkers)
