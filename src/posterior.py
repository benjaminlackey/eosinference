import sys
import numpy as np
import utilities as util

# negative number with large magnitude used to represent the log of zero
log_zero = -sys.float_info.max * sys.float_info.epsilon


################################################################################
# Construct the ln(prior) from the mass bounds, EOS bounds,                    #
# maximum mass, and maximum speed of sound.                                    #
################################################################################

def log_mass_prior(mc, q, q_min=0.5, m_min=0.5, m_max=3.2):
    """Put bounds on the mass ratio and individual masses.
    """
    q_max = 1.0
    eta = util.eta_of_q(q)
    m1 = util.m1_of_mchirp_eta(mc, eta)
    m2 = util.m2_of_mchirp_eta(mc, eta)

    if (q<q_min or q>q_max or
        m1<m_min or m1>m_max or
        m2<m_min or m2>m_max):
        #print "Mass ratio or masses outside bounds:"+str((q, m1, m2))
        return log_zero
    else:
        return 0.0


def log_prior(mc_list, q_list, eos, q_min=0.5, m_min=0.5, m_max=3.2, mass_known=1.93, vs_limit=1.0):
    """Prior for all binaries and the EOS parameters.
    The checks are sorted from least to most computationally expensive,
    and are only performed if the previous checks have passed.
    """
    # Go through each BNS system and check if any masses are out of bounds
    n_binaries = len(mc_list)
    for i in range(n_binaries):
        mc = mc_list[i]
        q = q_list[i]
        lmp = log_mass_prior(mc, q, q_min=q_min, m_min=m_min, m_max=m_max)
        if lmp==log_zero:
            return log_zero

    # Check the EOS bounds
    outside = eos.outside_bounds()
    if outside==True:
        return log_zero

    # Check maxmimum mass
    mmax = eos.max_mass()
    if mmax <= mass_known:
        return log_zero

    # Check that the chosen masses in each BNS are < mmax
    # Note: This requirement causes the q's and EOS parameters to be correlated.
    for i in range(n_binaries):
        mc = mc_list[i]
        q = q_list[i]
        # Get individual masses (with m1>=m2)
        eta = util.eta_of_q(q)
        m1 = util.m1_of_mchirp_eta(mc, eta)
        m2 = util.m2_of_mchirp_eta(mc, eta)
        if (m1 > mmax) or (m2 > mmax):
            return log_zero

    # Check speed of sound requirement
    if eos.max_speed_of_sound() >= vs_limit:
        return log_zero

    # If you get here, the EOS parameters are allowed by the prior
    return 0.0


def log_prior_emcee_wrapper(
    params, mc_mean_list, eos_class_reference,
    q_min=0.5, m_min=0.5, m_max=3.2, mass_known=1.93, vs_limit=1.0):
    """Wrapper for the function log_prior.
    Takes arguments in the form required by emcee.EnsembleSampler()

    Parameters
    ----------
    params : 1d array of length n_bns+n_eos_params
        [q0, ..., qn-1, eosparams]
    mc_mean_list : 1d array of chirp masses for each bns
    eos_class_reference : Name of the EOS class
    """
    n_binaries = len(mc_mean_list)
    q_list = params[:n_binaries]
    eos_params = params[n_binaries:]
    eos = eos_class_reference(eos_params)
    return log_prior(
        mc_mean_list, q_list, eos,
        q_min=q_min, m_min=m_min, m_max=m_max,
        mass_known=mass_known, vs_limit=vs_limit)


################################################################################
# Construct the ln(likelihood) from all BNS observations.                      #
################################################################################

def single_event_log_likelihood(mc_mean, q, eos, lnp_of_ql):
    """Calculate the ln(likelihood) for each BNS event.

    This should only be evaluated after you have evaluated the prior,
    to make sure the parameters are even allowed.
    """
    # Get individual masses (with m1>=m2)
    eta = util.eta_of_q(q)
    m1 = util.m1_of_mchirp_eta(mc_mean, eta)
    m2 = util.m2_of_mchirp_eta(mc_mean, eta)

    # Get lambdat
    lambda1 = eos.lambdaofm(m1)
    lambda2 = eos.lambdaofm(m2)
    lambdat = util.lamtilde_of_eta_lam1_lam2(eta, lambda1, lambda2)

    return lnp_of_ql(q, lambdat)


def log_likelihood(mc_mean_list, q_list, eos, lnp_of_ql_list):
    """Add up the ln(likelihood(q, tildeLambda)) values for each BNS event.
    """
    n_binaries = len(mc_mean_list)
    lnl_array = np.zeros(n_binaries)
    for i in range(n_binaries):
        mc_mean = mc_mean_list[i]
        q = q_list[i]
        lnp_of_ql = lnp_of_ql_list[i]
        lnl_array[i] = single_event_log_likelihood(mc_mean, q, eos, lnp_of_ql)

    # For independent events, likelihood is product of individual likelihoods.
    # (sum of ln(likelihood).)
    return np.sum(lnl_array)


################################################################################
# Construct the ln(posterior) from the prior and likelihood.                   #
################################################################################

def log_posterior(
    params, mc_mean_list, eos_class_reference, lnp_of_ql_list,
    q_min=0.5, m_min=0.5, m_max=3.2, mass_known=1.93, vs_limit=1.0):
    """Evaluate the posterior for all the events.
    """
    n_binaries = len(mc_mean_list)
    q_list = params[:n_binaries]
    eos_params = params[n_binaries:]
    eos = eos_class_reference(eos_params)

    try:
        # First check that the prior is in bounds.
        # Currently the ln(prior) is either log_zero or 0.
        lprior = log_prior(
            mc_mean_list, q_list, eos,
            q_min=q_min, m_min=m_min, m_max=m_max,
            mass_known=mass_known, vs_limit=vs_limit)
        if lprior==log_zero:
            return log_zero

        # Calculate the log_likelihood
        llike = log_likelihood(mc_mean_list, q_list, eos, lnp_of_ql_list)

        # If you get here lprior should be 0, so lpost = llike
        return llike

    except RuntimeError:
        print('LAL had a RuntimeError')
        return log_zero
