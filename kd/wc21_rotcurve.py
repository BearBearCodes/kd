#!/usr/bin/env python
"""
wc21_rotcurve.py

Utilities involving the Universal Rotation Curve (Persic et al. 1996) from
Wenger & Cheng (2021), henceforth WC21.
Including HMSFR peculiar motion.

Copyright(C) 2017-2021 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

2021-03-DD Isaac Cheng & Trey V. Wenger new in V3.0
"""

import os
import dill
import numpy as np
from kd import kd_utils

#
# WC21 A6 rotation model parameters
#
__R0 = 8.1746
__Usun = 10.879
__Vsun = 10.697
__Wsun = 8.088
__Upec = 4.907
__Vpec = -4.522
__a2 = 0.977
__a3 = 1.626
__Zsun = 5.399
__roll = -0.011
#
# IAU defined LSR
#
__Ustd = 10.27
__Vstd = 15.32
__Wstd = 7.74


def calc_bary_coords(glong, glat, dist):
    """
    Calculate baycentric x & y Cartesian coordinates
    from galactic coordinates.

    Parameters:
      glong, glat :: scalar or array of scalars
        Galactic longitude and latitude (deg)

      dist :: scalar or array of scalars
        line-of-sight distance (kpc)

    Returns: x, y
      x, y :: scalar or array of scalars
        Barycentric Cartesian positions (kpc)
    """
    input_scalar = np.isscalar(glong) and np.isscalar(glat) and np.isscalar(dist)
    glong, glat, dist = np.atleast_1d(glong, glat, dist)
    cos_glat = np.cos(np.deg2rad(glat))
    x = dist * cos_glat * np.cos(np.deg2rad(glong))
    y = dist * cos_glat * np.sin(np.deg2rad(glong))
    # z = dist * np.sin(np.deg2rad(glat))
    if input_scalar:
        return x[0], y[0]
    return x, y


def calc_gcen_coords(glong, glat, dist, R0=__R0,
                     Zsun=__Zsun, roll=__roll, use_Zsunroll=True):
    """
    Calculate galactocentric Cartesian coordinates from
    galactic longitudes, latitudes, and distances from the Sun.

    Parameters:
      glong, glat :: scalar or array of scalars
        Galactic longitude and latitude (deg)

      dist :: scalar or array of scalars
        line-of-sight distance (kpc)

      R0 :: scalar or array of scalars (optional)
        Galactocentric radius of the Sun

      Zsun :: scalar (optional)
        Height of Sun above galactic midplane (pc)

      roll :: scalar (optional)
        Roll angle relative to b=0 (deg)

      use_Zsunroll :: boolean (optional)
        If True, include Zsun and roll into calculation

    Returns: x, y, Rgal, cos_az, sin_az
      x, y :: scalars or arrays of scalars
        Galactocentric Cartesian positions (kpc). Sun is on -x-axis.

      Rgal :: scalar or array of scalars
        Galactocentric cylindrical radius (kpc)

      cos_az, sin_az :: scalar or array of scalars
        Cosine and sine of Galactocentric azimuth (rad)
    """
    glong, glat, dist = np.atleast_1d(glong, glat, dist)
    if np.shape(glong) != np.shape(dist):
        glong = np.array([glong, ] * len(dist))
    Rgal = kd_utils.calc_Rgal(
        glong, glat, dist, R0=R0,
        Zsun=Zsun, roll=roll, use_Zsunroll=use_Zsunroll)
    Rgal[Rgal < 1.0e-6] = 1.0e-6  # Catch small Rgal
    az = kd_utils.calc_az(
        glong, glat, dist, R0=R0,
        Zsun=Zsun, roll=roll, use_Zsunroll=use_Zsunroll)
    cos_az = np.cos(np.deg2rad(az))
    sin_az = np.sin(np.deg2rad(az))

    x = Rgal * -cos_az
    y = Rgal * sin_az

    return x, y, Rgal, cos_az, sin_az


def krige_UpecVpec(x, y, norm=20):
    """
    Estimates the difference between individual peculiar motions
    and their averages at the given (x, y) position(s) using kriging.

    Parameters:
      x, y :: scalars or arrays of scalars
        Barycentric Cartesian positions (kpc)

      norm :: scalar
        Normalization factor that determines slope of kriging to average
        peculiar motion transition. Larger norm is steeper transition

    Returns: Upec, Vpec
      Upec :: scalar or array of scalars
        Peculiar radial velocity of source toward galactic center (km/s)

      Vpec :: scalar or array of scalars
        Peculiar tangential velocity of source in
        direction of galactic rotation (km/s)
    """
    # Check inputs
    input_scalar = np.isscalar(x) and np.isscalar(y)
    x, y  = np.atleast_1d(x, y)
    #
    # Load kriging objects and variance threshold
    #
    infile = os.path.join(os.path.dirname(__file__), "wc21_kde_krige.pkl")
    # infile contains: full KDE + KDEs of each component (e.g. "R0")
    #                  + kriging objects + variance threshold + convex hull object
    with open(infile, "rb") as f:
        file = dill.load(f)
        Upec_krige = file["Upec_krige"]
        Vpec_krige = file["Vpec_krige"]
        # hull = file["hull"]  # convex hull object
        var_threshold = file["var_threshold"]  # sum of Upec/Vpec variances
        file = None  # free up resources
    #
    # Calculate expected Upec and Vpec differences at source location(s)
    #
    interp_pos = np.vstack((x.flatten(), y.flatten())).T
    Upec_diff, Upec_diff_var = Upec_krige.interp(interp_pos, resample=False)
    Vpec_diff, Vpec_diff_var = Vpec_krige.interp(interp_pos, resample=False)
    # #
    # # Use average value if component is outside convex hull
    # # (i.e., difference is zero)
    # #
    # not_in_hull = hull.find_simplex(interp_pos) < 0
    # Upec[not_in_hull] = 0
    # Vpec[not_in_hull] = 0
    #
    # Gaussian-like weighting function
    #
    var_tot = Upec_diff_var + Vpec_diff_var  # total variance (not in quadrature)
    pec_weights = np.exp(norm * (var_threshold / var_tot - 1))
    zero_weights = np.ones_like(pec_weights)
    weights = np.vstack((pec_weights, zero_weights))
    print("pec_weights.shape", np.shape(pec_weights))
    print("weights.shape", np.shape(weights))
    zero_diff = np.zeros_like(Upec_diff)
    Upec_diff = np.average([Upec_diff, zero_diff], weights=weights, axis=0)
    Vpec_diff = np.average([Vpec_diff, zero_diff], weights=weights, axis=0)
    #
    # Multiply kriging by some factor
    #
    Upec_diff = Upec_diff * 3
    Vpec_diff = Vpec_diff * 3
    #
    # Reshape
    #
    Upec_diff = Upec_diff.reshape(np.shape(x))
    Vpec_diff = Vpec_diff.reshape(np.shape(x))
    #
    # Free up resources
    #
    Upec_krige = Vpec_krige = var_threshold = None
    #
    if input_scalar:
        return Upec_diff[0], Vpec_diff[0]
    return Upec_diff, Vpec_diff


def nominal_params(glong=None, glat=None, dist=None,
                   use_kriging=False, norm=20, resample=False):
    """
    Return a dictionary containing the nominal rotation curve
    parameters.

    Parameters:
      glong, glat :: scalars or arrays of scalars (optional, required for kriging)
        Galactic longitude and latitude (deg)

      dist :: scalar or array of scalars (optional, required for kriging)
        Line-of-sight distance (kpc)

      use_kriging :: boolean (optional)
        If True, estimate individual Upec & Vpec from kriging program
        If False, use average Upec & Vpec

      norm :: scalar (optional)
        Normalization factor that determines slope of kriging to average
        peculiar motion transition. Larger norm is steeper transition

      resample :: boolean (optional)
        Boolean representing if kd program will MC sample parameters
        (not in this function, but in resample_params() function)

    Returns: params, Rgal, cos_az, sin_az
      params :: dictionary
        params['R0'], etc. : scalar
          The nominal rotation curve parameter
    """
    krig_args = [glong, glat, dist]
    if use_kriging and all(arg is not None for arg in krig_args):
        # Calculate barycentric Cartesian positions
        x, y = calc_bary_coords(glong, glat, dist)
        # Calculate Upec and Vpec differences from their
        # averages at source location(s)
        Upec, Vpec = krige_UpecVpec(x, y, norm=norm)
        if not resample:
            # Calculate actual Upec and Vpec
            Upec = Upec + __Upec
            Vpec = Vpec + __Vpec
    else:
        # Use average Upec and Vpec
        Upec = __Upec
        Vpec = __Vpec

    params = {
        "R0": __R0,
        "Zsun": __Zsun,
        "Usun": __Usun,
        "Vsun": __Vsun,
        "Wsun": __Wsun,
        "Upec": Upec,
        "Vpec": Vpec,
        "roll": __roll,
        "a2": __a2,
        "a3": __a3,
    }
    return params


def resample_params(kde, size=None, nom_params=None, use_kriging=False):
    """
    Resample the rotation curve parameters within their
    uncertainties using the WC21 kernel density estimator
    to include parameter covariances.

    Parameters:
      kde :: kernel density estimator class (e.g. scipy.stats.gaussian_kde)
        Kernel density estimator containing all the rotation model parameters

      size :: integer
        The number of random samples to generate (per source, if use_kriging).
        If None, generate only one sample and return a scalar

      nom_params :: dictionary (optional, required for kriging)
        Dictionary containing the kriging values of
        the Upec/Vpec differences from the average

      use_kriging :: boolean (optional)
        If True, estimate individual Upec & Vpec from kriging program
        If False, use average Upec & Vpec

    Returns: params
      params :: dictionary
        params['R0'], etc. : scalar or array of scalars
                             The re-sampled parameters
    """
    if size is None:
        samples = kde.resample(1)
        params = {
            "R0": samples[0][0],
            "Zsun": samples[1][0],
            "Usun": samples[2][0],
            "Vsun": samples[3][0],
            "Wsun": samples[4][0],
            "Upec": samples[5][0],
            "Vpec": samples[6][0],
            "roll": samples[7][0],
            "a2": samples[8][0],
            "a3": samples[9][0],
        }
    else:
        samples = kde.resample(size)
        params = {
            "R0": samples[0],
            "Zsun": samples[1],
            "Usun": samples[2],
            "Vsun": samples[3],
            "Wsun": samples[4],
            "Upec": samples[5],
            "Vpec": samples[6],
            "roll": samples[7],
            "a2": samples[8],
            "a3": samples[9],
        }
    kde = None  # free up resources

    if use_kriging and nom_params is not None:
        Upec, Vpec = nom_params["Upec"], nom_params["Vpec"]
        # Calculate actual Upec and Vpec
        Upec = Upec + params["Upec"]
        Vpec = Vpec + params["Vpec"]
        # Save in dictionary
        params_orig = params
        params = {
            "R0": params_orig["R0"],
            "Zsun": params_orig["Zsun"],
            "Usun": params_orig["Usun"],
            "Vsun": params_orig["Vsun"],
            "Wsun": params_orig["Wsun"],
            "Upec": Upec,
            "Vpec": Vpec,
            "roll": params_orig["roll"],
            "a2": params_orig["a2"],
            "a3": params_orig["a3"],
        }
    return params


def calc_theta(R, a2=__a2, a3=__a3, R0=__R0):
    """
    Return circular orbit speed at a given Galactocentric radius.

    Parameters:
      R :: scalar or array of scalars
        Galactocentric radius (kpc)

      a2, a3 :: scalars (optional)
        WC21 rotation curve parameters

      R0 :: scalar (optional)
        Solar Galactocentric radius (kpc)

    Returns: theta
      theta :: scalar or array of scalars
        circular orbit speed at R (km/s)
    """
    input_scalar = np.isscalar(R)
    R = np.atleast_1d(R)
    #
    # Re-production of Reid+2019 FORTRAN code
    #
    rho = R / (a2 * R0)
    lam = (a3 / 1.5) ** 5.0
    loglam = np.log10(lam)
    term1 = 200.0 * lam ** 0.41
    term2 = np.sqrt(
        0.8 + 0.49 * loglam + 0.75 * np.exp(-0.4 * lam) / (0.47 + 2.25 * lam ** 0.4)
    )
    term3 = (0.72 + 0.44 * loglam) * 1.97 * rho ** 1.22 / (rho ** 2.0 + 0.61) ** 1.43
    term4 = 1.6 * np.exp(-0.4 * lam) * rho ** 2.0 / (rho ** 2.0 + 2.25 * lam ** 0.4)
    #
    # Catch non-physical case where term3 + term4 < 0
    #
    term = term3 + term4
    term[term < 0.0] = np.nan
    #
    # Circular velocity
    #
    theta = term1 / term2 * np.sqrt(term)
    if input_scalar:
        return theta[0]
    return theta


def calc_vlsr(glong, glat, dist,
              R0=__R0, Usun=__Usun, Vsun=__Vsun, Wsun=__Wsun,
              Upec=__Upec, Vpec=__Vpec, a2=__a2, a3=__a3,
              Zsun=__Zsun, roll=__roll, peculiar=False):
    """
    Return the IAU-LSR velocity at a given Galactic longitude and
    line-of-sight distance.

    Parameters:
      glong, glat :: scalars or arrays of scalars
        Galactic longitude and latitude (deg).

      dist :: scalar or array of scalars
        line-of-sight distance (kpc).

      R0 :: scalar (optional)
        Solar Galactocentric radius (kpc)

      Usun, Vsun, Wsun, Upec, Vpec, a2, a3 :: scalars (optional)
        WC21 rotation curve parameters

      Zsun :: scalar (optional)
        Height of sun above Galactic midplane (pc)

      roll :: scalar (optional)
        Roll of Galactic midplane relative to b=0 (deg)

      peculiar :: boolean (optional)
        If True, include HMSFR peculiar motion component

    Returns: vlsr
      vlsr :: scalar or array of scalars
        LSR velocity (km/s).
    """
    is_print = False
    if is_print:
        print("glong, glat, dist in calc_vlsr",
              np.shape(glong), np.shape(glat), np.shape(dist))
    input_scalar = np.isscalar(glong) and np.isscalar(glat) and np.isscalar(dist)
    glong, glat, dist = np.atleast_1d(glong, glat, dist)
    cos_glong = np.cos(np.deg2rad(glong))
    sin_glong = np.sin(np.deg2rad(glong))
    cos_glat = np.cos(np.deg2rad(glat))
    sin_glat = np.sin(np.deg2rad(glat))
    #
    # Convert distance to Galactocentric, catch small Rgal
    #
    Rgal = kd_utils.calc_Rgal(glong, glat, dist, R0=R0,
                              Zsun=Zsun, roll=roll, use_Zsunroll=True)
    Rgal[Rgal < 1.0e-6] = 1.0e-6  # Catch small Rgal
    az = kd_utils.calc_az(glong, glat, dist, R0=R0,
                          Zsun=Zsun, roll=roll, use_Zsunroll=True)
    cos_az = np.cos(np.deg2rad(az))
    sin_az = np.sin(np.deg2rad(az))
    if is_print:
        print("Rgal, cos_az, sin_az in calc_vlsr",
              np.shape(Rgal), np.shape(cos_az), np.shape(sin_az))
    #
    # Rotation curve circular velocity
    #
    theta = calc_theta(Rgal, a2=a2, a3=a3, R0=R0)
    theta0 = calc_theta(R0, a2=a2, a3=a3, R0=R0)
    if is_print:
        print("Theta, theta0 in calc_vlsr", np.shape(theta), np.shape(theta0))
        print("Upec, Vpec in calc_vlsr", np.shape(Upec), np.shape(Vpec))
    #
    # Add HMSFR peculiar motion
    #
    if peculiar:
        vR = -Upec
        vAz = theta + Vpec
        vZ = 0.0
    else:
        vR = 0.0
        vAz = theta
        vZ = 0.0
    vXg = -vR * cos_az + vAz * sin_az
    vYg = vR * sin_az + vAz * cos_az
    vZg = vZ
    if is_print:
        print("1st vXg, vYg, vZg in calc_vlsr",
              np.shape(vXg), np.shape(vYg), np.shape(vZg))
    #
    # Convert to barycentric
    #
    X = dist * cos_glat * cos_glong
    Y = dist * cos_glat * sin_glong
    Z = dist * sin_glat
    if is_print:
        print("X, Y, Z in calc_vlsr",
              np.shape(X), np.shape(Y), np.shape(Z))
    # useful constants
    sin_tilt = Zsun / 1000.0 / R0
    cos_tilt = np.cos(np.arcsin(sin_tilt))
    sin_roll = np.sin(np.deg2rad(roll))
    cos_roll = np.cos(np.deg2rad(roll))
    # solar peculiar motion
    vXg = vXg - Usun
    vYg = vYg - theta0 - Vsun
    vZg = vZg - Wsun
    if is_print:
        print("2nd vXg, vYg, vZg in calc_vlsr",
              np.shape(vXg), np.shape(vYg), np.shape(vZg))
    # correct tilt and roll of Galactic midplane
    vXg1 = vXg * cos_tilt - vZg * sin_tilt
    vYg1 = vYg
    vZg1 = vXg * sin_tilt + vZg * cos_tilt
    vXh = vXg1
    vYh = vYg1 * cos_roll + vZg1 * sin_roll
    vZh = -vYg1 * sin_roll + vZg1 * cos_roll
    vbary = (X * vXh + Y * vYh + Z * vZh) / dist
    if is_print: print("vbary in calc_vlsr", np.shape(vbary))
    #
    # Convert to IAU-LSR
    #
    vlsr = (
        vbary + (__Ustd * cos_glong + __Vstd * sin_glong) * cos_glat + __Wstd * sin_glat
    )
    if is_print: print("final vlsr shape", np.shape(vlsr))
    if input_scalar:
        return vlsr[0]
    return vlsr
