"""Definition of the 4-parameter piecewise polytrope EOS model.
"""

import numpy as np

# Matplotlib things
import matplotlib.pyplot as plt

# LAL stuff
import lal
import lalsimulation

################################################################################
#                       NS parameters from EOS                                 #
################################################################################

# class EOS4ParameterPiecewisePolytropeBase(object):
#     """4-piece polytrope equation of state.
#     """
#
#     def __init__(self, params):
#         """Initialize EOS and calculate a family of TOV stars.
#         """
#         # lalsimulation uses SI units.
#         self.lp_cgs = params[0]
#         self.lp_si = self.lp_cgs - 1.
#         self.g1 = params[1]
#         self.g2 = params[2]
#         self.g3 = params[3]
#
#         self.k_low_cgs = np.array([
#             6.11252036792443e12, 9.54352947022931e14, 4.787640050002652e22, 3.593885515256112e13])
#         self.gamma_low_cgs = np.array([1.58424999, 1.28732904, 0.62223344, 1.35692395])
#         self.rho_low_cgs = np.array([0.0, 2.44033979e7, 3.78358138e11, 2.62780487e12])
#
#         # Dividing densities for high-density EOS
#         self.rho1_cgs = 10**14.7
#         self.rho2_cgs = 10**15.0
#
#         # Pressures at dividing densities for low-density EOS
#         self.p_low_cgs = self.k_low_cgs * self.rho_low_cgs**self.gamma_low_cgs
#
#         # EOS parameter bounds
#         #lpmin, lpmax = 33.5, 35.5
#
#         # the lalsimualtion implementation requires the 33.5023 lower bound
#         # self.lpmin, self.lpmax = 33.5023, 35.5
#         # self.g1min, self.g1max = 1.4, 5.0
#         # self.g2min, self.g2max = 1.08, 5.0
#         # self.g3min, self.g3max = 1.08, 5.0
#
#         # These bounds are less likely to cause a GSL interpolation error that causes
#         # the kernel to quit unexpectedly
#         self.lpmin, self.lpmax = 33.6, 35.5
#         self.g1min, self.g1max = 1.5, 5.0
#         self.g2min, self.g2max = 1.1, 5.0
#         self.g3min, self.g3max = 1.1, 5.0
#
#         # Expensive properties to compute. Only calculate them once and store them.
#         self.properties_flag = None
#         self.eos = None
#         self.fam = None
#         self.mmax = None
#         self.vmax = None
#
#     def p1_upper_bound(self, g1):
#         """Upper bound on the pressure p_1 to prevent the
#         high-density EOS joining onto the low density EOS
#         at too low a density.
#         """
#         return self.p_low_cgs[3]*(self.rho1_cgs/self.rho_low_cgs[3])**g1
#
#     def log_p1_upper_bound(self, g1):
#         """log_10 of the p_1 upper bound.
#         """
#         return np.log10(self.p1_upper_bound(g1))
#
#     def calculate_ns_properties(self):
#         # Initialize with piecewise polytrope parameters (logp1 in SI units)
#         self.eos = lalsimulation.SimNeutronStarEOS4ParameterPiecewisePolytrope(
#             self.lp_si, self.g1, self.g2, self.g3)
#
#         # This creates the interpolated functions R(M), k2(M), etc.
#         # after doing many TOV integrations.
#         self.fam = lalsimulation.CreateSimNeutronStarFamily(self.eos)
#
#         # Change flag
#         self.properties_flag = 1
#
#     def max_mass(self):
#         """Calculate the maximum mass.
#         """
#         if self.properties_flag==None:
#             self.calculate_ns_properties()
#
#         if self.mmax==None:
#             mmax = lalsimulation.SimNeutronStarMaximumMass(self.fam)/lal.MSUN_SI
#             # Create a little buffer so you don't interpolate right at the maximum mass
#             # TODO: this is crude and should be fixed
#             self.mmax = mmax - 0.01
#             return self.mmax
#         else:
#             return self.mmax
#
#     def max_speed_of_sound(self, plot=False):
#         """Calculate the maximum speed of sound at any density
#         up to the central density of the maximum mass NS (units of v/c).
#         """
#         mmax = self.max_mass()
#
#         # Value of h at the core of the maximum mass NS.
#         h_max = lalsimulation.SimNeutronStarEOSMaxPseudoEnthalpy(self.eos)
#
#         # Calculate speed of sound at a list of h's up to h_max,
#         # then take the maximum value.
#         hs = np.logspace(np.log10(h_max)-1.0, np.log10(h_max), 100)
#         vs = np.array([lalsimulation.SimNeutronStarEOSSpeedOfSoundGeometerized(h, self.eos) for h in hs])
#         v_max = np.max(vs)
#         if plot:
#             fig, ax = plt.subplots()
#             ax.plot(hs, vs)
#             ax.axhline(1.0, c='k')
#             ax.axvline(h_max)
#             ax.axhline(v_max)
#             ax.set_xlabel(r'$h$')
#             ax.set_ylabel(r'$v/c$')
#             ax.set_xlim(0, 1.1*h_max)
#             ax.set_ylim(0, 1.1*v_max)
#         self.v_max = v_max
#         return self.v_max
#
#     def radiusofm(self, m):
#         """Radius in km.
#         """
#         if self.properties_flag==None:
#             self.calculate_ns_properties()
#
#         r_SI = lalsimulation.SimNeutronStarRadius(m*lal.MSUN_SI, self.fam)
#         return r_SI/1000.0
#
#     def k2ofm(self, m):
#         """Dimensionless Love number.
#         """
#         if self.properties_flag==None:
#             self.calculate_ns_properties()
#
#         return lalsimulation.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, self.fam)
#
#     def lambdaofm(self, m):
#         """Dimensionless tidal deformability.
#         """
#         if self.properties_flag==None:
#             self.calculate_ns_properties()
#
#         r = self.radiusofm(m)
#         k2 = self.k2ofm(m)
#         return (2./3.)*k2*( (lal.C_SI**2*r*1000.0)/(lal.G_SI*m*lal.MSUN_SI) )**5
#
#
# ################################################################################
# #        Derived/child class for a specific parameterization with bounds.      #
# ################################################################################
#
# class EOS4ParameterPiecewisePolytropeGammaParams(EOS4ParameterPiecewisePolytropeBase):
#     def __init__(self, params):
#         super(EOS4ParameterPiecewisePolytropeGammaParams, self).__init__(params)
#
#     def outside_bounds(self):
#         """Boundary for valid EOS parameters.
#
#         Parameters
#         ----------
#         lp : log10(pressure) in cgs units
#         """
#         lp, g1, g2, g3 = self.lp_cgs, self.g1, self.g2, self.g3
#         if (lp<=self.lpmin or lp>=self.lpmax or
#             g1<=self.g1min or g1>=self.g1max or
#             g2<=self.g2min or g2>=self.g2max or
#             g3<=self.g3min or g3>=self.g3max):
#             return True
#
#         if lp >= self.log_p1_upper_bound(g1):
#             return True
#
#         # If you get here, the EOS parameters are in the allowed range
#         return False


class EOS4ParameterPiecewisePolytropeGammaParams(object):
    """4-piece polytrope equation of state.
    """

    def __init__(self, params):
        """Initialize EOS.

        Parameters
        ----------
        params = np.array([lp_cgs, g1, g2, g3])
            lp_cgs : Pressure in dyne/cm^2 at rho1 = 10^14.7g/cm^3
            g1 : Adiabatic index below rho1
            g2 : Adiabatic index between rho1 and rho2 = 10^15g/cm^3
            g3 : Adiabatic index above rho2
        """
        # lalsimulation uses SI units.
        self.lp_cgs = params[0]
        self.lp_si = self.lp_cgs - 1.
        self.g1 = params[1]
        self.g2 = params[2]
        self.g3 = params[3]

        self.k_low_cgs = np.array([
            6.11252036792443e12, 9.54352947022931e14, 4.787640050002652e22, 3.593885515256112e13])
        self.gamma_low_cgs = np.array([1.58424999, 1.28732904, 0.62223344, 1.35692395])
        self.rho_low_cgs = np.array([0.0, 2.44033979e7, 3.78358138e11, 2.62780487e12])

        # Dividing densities for high-density EOS
        self.rho1_cgs = 10**14.7
        self.rho2_cgs = 10**15.0

        # Pressures at dividing densities for low-density EOS
        self.p_low_cgs = self.k_low_cgs * self.rho_low_cgs**self.gamma_low_cgs

        # EOS parameter bounds
        #lpmin, lpmax = 33.5, 35.5

        # the lalsimualtion implementation requires the 33.5023 lower bound
        # self.lpmin, self.lpmax = 33.5023, 35.5
        # self.g1min, self.g1max = 1.4, 5.0
        # self.g2min, self.g2max = 1.08, 5.0
        # self.g3min, self.g3max = 1.08, 5.0

        # These bounds are less likely to cause a GSL interpolation error that causes
        # the kernel to quit unexpectedly
        self.lpmin, self.lpmax = 33.6, 35.5
        self.g1min, self.g1max = 1.5, 5.0
        self.g2min, self.g2max = 1.1, 5.0
        self.g3min, self.g3max = 1.1, 5.0

        # Expensive properties to compute. Only calculate them once and store them.
        self.properties_flag = None
        self.eos = None
        self.fam = None
        self.mmax = None
        self.vmax = None

    def p1_upper_bound(self, g1):
        """Upper bound on the pressure p_1 to prevent the
        high-density EOS joining onto the low density EOS
        at too low a density.
        """
        return self.p_low_cgs[3]*(self.rho1_cgs/self.rho_low_cgs[3])**g1

    def log_p1_upper_bound(self, g1):
        """log_10 of the p_1 upper bound.
        """
        return np.log10(self.p1_upper_bound(g1))

    def calculate_ns_properties(self):
        # Initialize with piecewise polytrope parameters (logp1 in SI units)
        self.eos = lalsimulation.SimNeutronStarEOS4ParameterPiecewisePolytrope(
            self.lp_si, self.g1, self.g2, self.g3)

        # This creates the interpolated functions R(M), k2(M), etc.
        # after doing many TOV integrations.
        self.fam = lalsimulation.CreateSimNeutronStarFamily(self.eos)

        # Change flag
        self.properties_flag = 1

    def max_mass(self):
        """Calculate the maximum mass.
        """
        if self.properties_flag==None:
            self.calculate_ns_properties()

        if self.mmax==None:
            mmax = lalsimulation.SimNeutronStarMaximumMass(self.fam)/lal.MSUN_SI
            # Create a little buffer so you don't interpolate right at the maximum mass
            # TODO: this is crude and should be fixed
            self.mmax = mmax - 0.01
            return self.mmax
        else:
            return self.mmax

    def max_speed_of_sound(self, plot=False):
        """Calculate the maximum speed of sound at any density
        up to the central density of the maximum mass NS (units of v/c).
        """
        mmax = self.max_mass()

        # Value of h at the core of the maximum mass NS.
        h_max = lalsimulation.SimNeutronStarEOSMaxPseudoEnthalpy(self.eos)

        # Calculate speed of sound at a list of h's up to h_max,
        # then take the maximum value.
        hs = np.logspace(np.log10(h_max)-1.0, np.log10(h_max), 100)
        vs = np.array([lalsimulation.SimNeutronStarEOSSpeedOfSoundGeometerized(h, self.eos) for h in hs])
        v_max = np.max(vs)
        if plot:
            fig, ax = plt.subplots()
            ax.plot(hs, vs)
            ax.axhline(1.0, c='k')
            ax.axvline(h_max)
            ax.axhline(v_max)
            ax.set_xlabel(r'$h$')
            ax.set_ylabel(r'$v/c$')
            ax.set_xlim(0, 1.1*h_max)
            ax.set_ylim(0, 1.1*v_max)
        self.v_max = v_max
        return self.v_max

    def radiusofm(self, m):
        """Radius in km.
        """
        if self.properties_flag==None:
            self.calculate_ns_properties()

        r_SI = lalsimulation.SimNeutronStarRadius(m*lal.MSUN_SI, self.fam)
        return r_SI/1000.0

    def k2ofm(self, m):
        """Dimensionless Love number.
        """
        if self.properties_flag==None:
            self.calculate_ns_properties()

        return lalsimulation.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, self.fam)

    def lambdaofm(self, m):
        """Dimensionless tidal deformability.
        """
        if self.properties_flag==None:
            self.calculate_ns_properties()

        r = self.radiusofm(m)
        k2 = self.k2ofm(m)
        return (2./3.)*k2*( (lal.C_SI**2*r*1000.0)/(lal.G_SI*m*lal.MSUN_SI) )**5

    def outside_bounds(self):
        """Boundary for valid EOS parameters.

        Parameters
        ----------
        lp : log10(pressure) in cgs units
        """
        lp, g1, g2, g3 = self.lp_cgs, self.g1, self.g2, self.g3
        if (lp<=self.lpmin or lp>=self.lpmax or
            g1<=self.g1min or g1>=self.g1max or
            g2<=self.g2min or g2>=self.g2max or
            g3<=self.g3min or g3>=self.g3max):
            return True

        if lp >= self.log_p1_upper_bound(g1):
            return True

        # If you get here, the EOS parameters are in the allowed range
        return False


def initialize_walker_piecewise_polytrope_gamma_params():
    """Function for generating initial guess for EOS parameters
    for the emcee sampler.
    """
    lp = np.random.uniform(34.3, 34.7)
    g1 = np.random.uniform(2.5, 3.5)
    g2 = np.random.uniform(2.5, 3.5)
    g3 = np.random.uniform(2.5, 3.5)
    return np.array([lp, g1, g2, g3])


################################################################################
# 2 reparameterizations of the piecewise polytrope EOS                         #
################################################################################

def p_to_gamma_params(p_params):
    """
    """
    rho0, p1, p2, p3 = p_params

    rho1 = 10**14.7
    rho2 = 10**15.0
    rho3 = 2.0 * rho2

    k0 = 3.593885515256112e13
    g0 = 1.35692395
    p0 = k0*rho0**g0

    lp1 = np.log10(p1)
    g1 = np.log10(p1/p0)/np.log10(rho1/rho0)
    g2 = np.log10(p2/p1)/np.log10(rho2/rho1)
    g3 = np.log10(p3/p2)/np.log10(rho3/rho2)

    return np.array([lp1, g1, g2, g3])


def logp_to_gamma_params(logp_params):
    """
    """
    lrho0, lp1, lp2, lp3 = logp_params

    rho0 = 10**lrho0
    p1 = 10**lp1
    p2 = 10**lp2
    p3 = 10**lp3

    gamma_params = p_to_gamma_params(np.array([rho0, p1, p2, p3]))
    return gamma_params


def gamma_to_p_params(gamma_params):
    """
    """
    lp1, g1, g2, g3 = gamma_params

    rho1 = 10**14.7
    rho2 = 10**15.0
    rho3 = 2.0 * rho2

    p1 = 10**lp1
    p2 = p1*(rho2/rho1)**g2
    p3 = p2*(rho3/rho2)**g3

    k0 = 3.593885515256112e13
    g0 = 1.35692395
    k1 = p1/(rho1**g1)
    rho0 = (k1/k0)**(1.0/(g0-g1))

    return np.array([rho0, p1, p2, p3])


def gamma_to_logp_params(gamma_params):
    """
    """
    p_params = gamma_to_p_params(gamma_params)
    lrho0 = np.log10(p_params[0])
    lp1 = np.log10(p_params[1])
    lp2 = np.log10(p_params[2])
    lp3 = np.log10(p_params[3])
    return np.array([lrho0, lp1, lp2, lp3])


class EOS4ParameterPiecewisePolytropePParams(EOS4ParameterPiecewisePolytropeGammaParams):
    """4-parameter piecewise polytrope using the reparameterization from
    F. Ozel, D. Psaltis, PRD 80, 103003 (2009), arXiv:0905.1959.
    """
    def __init__(self, p_params):
        """
        Parameters
        ----------
        p_params = np.array([rho0, p1, p2, p3])
            rho0 : Transition density between fixed low-density EOS and 1st polytrope
            p1 : Pressure at rho1 = 10^14.7g/cm^3
            p2 : Pressure at rho2 = 10^15g/cm^3
            p3 : Pressure at rho3 = 2*rho2
        """
        self.rho0 = p_params[0]
        self.p1 = p_params[1]
        self.p2 = p_params[2]
        self.p3 = p_params[3]
        gamma_params = p_to_gamma_params(p_params)
        EOS4ParameterPiecewisePolytropeGammaParams.__init__(self, gamma_params)
        self.rho3_cgs = 2.0 * self.rho2_cgs


class EOS4ParameterPiecewisePolytropeLogPParams(EOS4ParameterPiecewisePolytropeGammaParams):
    """4-parameter piecewise polytrope using the reparameterization from
    F. Ozel, D. Psaltis, PRD 80, 103003 (2009), arXiv:0905.1959,
    except each parameter is the log10 of the parameters in arXiv:0905.1959
    """
    def __init__(self, logp_params):
        """
        Parameters
        ----------
        logp_params = np.array([log10rho0, log10p1, log10p2, log10p3])
            log10rho0 : Log10 of transition density between fixed low-density EOS and 1st polytrope
            log10p1 : Log10 of pressure at rho1 = 10^14.7g/cm^3
            log10p2 : Log10 of pressure at rho2 = 10^15g/cm^3
            log10p3 : Log10 of pressure at rho3 = 2*rho2
        """
        self.lrho0 = logp_params[0]
        self.lp1 = logp_params[1]
        self.lp2 = logp_params[2]
        self.lp3 = logp_params[3]
        gamma_params = logp_to_gamma_params(logp_params)
        EOS4ParameterPiecewisePolytropeGammaParams.__init__(self, gamma_params)
        self.rho3_cgs = 2.0 * self.rho2_cgs


def initialize_walker_piecewise_polytrope_p_params():
    """Function for generating initial guess for EOS parameters
    for the emcee sampler.
    """
    gamma_params = initialize_walker_piecewise_polytrope_gamma_params()
    p_params = gamma_to_p_params(gamma_params)
    return p_params


def initialize_walker_piecewise_polytrope_log_p_params():
    """Function for generating initial guess for EOS parameters
    for the emcee sampler.
    """
    gamma_params = initialize_walker_piecewise_polytrope_gamma_params()
    logp_params = gamma_to_logp_params(gamma_params)
    return logp_params
