# eosinference
This package can be used to estimate the equation of state (EOS) of nuclear matter from observations of BNS inspirals with gravitaional wave detectors. The EOS describes the pressure as a function of density $p(\rho)$.

The methods used here described in this [paper](https://arxiv.org/abs/1410.8866), although I have made some improvements and changes to just take the chirp mass as fixed. The methods were originally inspired by this [paper](https://arxiv.org/abs/1005.0811) on measuring the EOS from neutron-star mass and radius observations.

# Custom parameterized EOS model
You can use your own parameterized EOS by creating a class with the following methods.

```python
class YourEOSClass(object):
    def __init__(self, params):
        """Initialize EOS and calculate things necessary for the other methods.
        Be careful to only calculate things when they are necessary, and store 
        results for the given set of parameters for later use by other methods.
        """

    def max_mass(self):
        """Calculate the maximum mass (M_\odot).
        """
        return self.mmax

    def max_speed_of_sound(self):
        """Calculate the maximum speed of sound at any density
        up to the central density of the maximum mass NS (units of v/c).
        """
        return self.v_max

    def radiusofm(self, m):
        """Radius in km.
        """
        return r

    def lambdaofm(self, m):
        """Dimensionless tidal deformability.
        """
        return Lambda

    def outside_bounds(self):
        """Boundary for valid EOS parameters.
        """
        if outside:
            return True
        else:
            return False
```


