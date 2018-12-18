import numpy as np

# Matplotlib things
import matplotlib.pyplot as plt

# LAL stuff
import lal
import lalsimulation

################################################################################
#                       NS parameters from EOS                                 #
################################################################################

class EOS4ParameterPiecewisePolytropeBase(object):
    """4-piece polytrope equation of state.
    """

    def __init__(self, params):
        """Initialize EOS and calculate a family of TOV stars.
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


################################################################################
#        Derived/child class for a specific parameterization with bounds.      #
################################################################################

class EOS4ParameterPiecewisePolytropeGammaParams(EOS4ParameterPiecewisePolytropeBase):
    def __init__(self, params):
        super(EOS4ParameterPiecewisePolytropeGammaParams, self).__init__(params)

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
            #print "EOS params outside limits."
            return True

        if lp >= self.log_p1_upper_bound(g1):
            return True

        # If you get here, the EOS parameters are in the allowed range
        return False
