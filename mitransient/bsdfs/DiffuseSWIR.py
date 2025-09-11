import sys
import os
import typing as tp 
from collections import OrderedDict
import matplotlib.pyplot as plt
import drjit as dr
import mitsuba as mi
import numpy as np
import csv 
from scipy.optimize import curve_fit

class DiffuseSWIR(mi.BSDF):
    """
        `diffuse_swir_bsdf` plugin
        ==========================

        Diffuse BSDF plugin that includes the specular component seen when measured using SWIR frequencies. 
        The closer the outgoing ray is to the perfect specular reflection of the incident ray,
        the higher the total measured radiance is, similar to a rough BSDF. Said relation depends on 
        the angle between the two rays and can be modelled, using a distribution.

        This class offers the possibility to fit a gaussian or Lorentz distribution to measurement data,
        as well as allowing to manually define the specular peaks, specifying the parameters of the distribution.
        During evaluation, the plugin obtains the angle between the outgoing ray and the perfect specular of 
        the incidence ray, and uses said value to sample the distribution.

        NOTE: the distribution is fit to the specular component given the angle between outgoing ray and
        reflected ray in radians, which should be taken into account when manually setting the parameters
        of a distribution.
        The fitting assumes that the peak will be at the specular reflection angle of the indicent light.

        The `DiffuseSWIR` plugin takes as input the following parameters:
        * `gaussian` (boolean):
            If True, the plugin will use a gaussian distribution for fitting.
            Else, the puglin will choose a Lorentz distribution.
            (default: True)
        * `csv_file` (string):
            Path to the .csv file with the reflectance measurements. It should contain four columns,
            the first one for the angle of the measurement, second one for the measured reflectance,
            third for the diffuse reflectance, fourth for the specular component (measured R - diffuse R).
            If it is set to None, the plugin will try to instantiate a parametric distribution, 
            gaussian if gaussian == True, Lorentz otherwise.
            (default: None)
        * `base_reflectance` (float3):
            Base reflectance of the BSDF.
             Only required when csv_file is None (parametric distribution).
            (default: [0.1, 0.1, 0.1])
        * `amplitude` (float):
            Peak of the parametric distribution (gaussian or Lorentz) used.
            Only required when csv_file is None (parametric distribution). 
            (default: 0.01)
        * `mean` (float):
            Mean (center) of the parametric distribution (gaussian or Lorentz) used.
            Only required when csv_file is None (parametric distribution). 
            (default: 0)
        * `sigma` (float):
            Standard deviation of the gaussian parametric distribution used.
            Only required when csv_file is None (parametric distribution) and gaussian == True
            (default: 0.3, sigma > 0)
        * `width` (float):
            Width (or scale) of the Lorentz parametric distribution used.
            Only required when csv_file is None (parametric distribution) and gaussian == False
            (default: 0.5)
        * `debug` (boolean):
            Plots the read measurements, as well as the distribution fit to the data if set to True.
            (default: False)
    """
    def __init__(self, props=mi.Properties):
        super().__init__(props)
        reflection_flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
        self.m_components = [reflection_flags]
        self.m_flags = reflection_flags
        self.debug = props.get('debug', False)
        self.gaussian_dist = props.get('gaussian', True) # Use gaussian or Lorentz distribution?
        self.measurements_file = props.get('csv_file', None) # Try to load measurements file name

        # --- IF, no file is specified load parametric distribution --- #
        if self.measurements_file == None:
            # Load base reflectance
            self.base_reflectance = props['reflectance']
            try:
                # BSDF loaded from dict as an Array3f
                self.base_reflectance = mi.Color3f(self.base_reflectance)
            except:
                # BSDF loaded from xml as a Texture
                # Evaluate the (constant) color of the texture with a dummy surface interaction
                si = dr.zeros(mi.SurfaceInteraction3f)
                self.base_reflectance = self.base_reflectance.eval(si)
            
            self.amplitude = props.get('amplitude', mi.Float(0.01))
            self.mean = props.get('mean', mi.Float(0.0))

            if self.gaussian_dist:
                self.sigma = props.get('sigma', mi.Float(0.3))
            else:
                self.width = props.get('width', mi.Float(0.5))
            return

        # --- ELSE, Load measured reflectance data from csv file --- #
        thetas_o_rad, self.base_reflectance, self.brdf_thetas_i = self.read_csv(self.measurements_file)
        self.base_reflectance = mi.Float(self.base_reflectance)

        thetas_i = self.brdf_thetas_i.keys()

        # --- Fit a distribution to each measurement, given theta_i --- #
        for theta_i in thetas_i:
            # Recover angles between theta_i, theta_o, measured BRDF, and specular component
            thetas_io_rad = self.brdf_thetas_i[theta_i]["thetas_io_rad"]
            measured_BRDF = self.brdf_thetas_i[theta_i]["measured_BRDF"]
            specular = measured_BRDF - self.base_reflectance

            # Fit a gaussian or lorentz distribution
            if self.gaussian:
                amplitude, sigma = self.fit_gaussian_distribution(theta_i, thetas_io_rad, specular)
                self.brdf_thetas_i[theta_i]["amplitude"] = amplitude
                self.brdf_thetas_i[theta_i]["width_sigma"] = sigma
            else:
                amplitude, width = self.fit_lorentz_distribution(theta_i, thetas_io_rad, specular)
                self.brdf_thetas_i[theta_i]["amplitude"] = amplitude
                self.brdf_thetas_i[theta_i]["width_sigma"] = width


    def fit_gaussian_distribution(self, theta_i, thetas_io_rad, specular):
        # --- Fit a gaussian to the estimated specular component --- #
        popt, pcov = curve_fit(self.gaussian_numpy, thetas_io_rad, specular)
        amplitude, sigma = popt
        sigma = np.abs(sigma)
        amplitude, sigma = mi.Float(amplitude), mi.Float(sigma)

        # --- Debug: plot the gaussian fit to the specular component --- #
        if self.debug:
            print(f'Fit gaussian distribution, amplitude: {amplitude}, sigma: {sigma}')
            print(f'Base reflectance: {self.base_reflectance}')
            specular_fit = self.gaussian_numpy(np.linspace(-np.pi / 2, np.pi / 2, 180), np.array(amplitude), np.array(sigma))

            plt.scatter(thetas_io_rad, specular, label='Measured specular component')
            plt.plot(np.linspace(-np.pi / 2, np.pi / 2, 180), specular_fit, label='Fit specular component')
            plt.xlabel('Angle (rad) between $\omega_o, reflect(\omega_i)$')
            plt.ylabel('Reflectance')
            plt.title(f'Specular component (sr^-1), gaussian fit, $\\theta_i = {theta_i}º$')
            plt.legend()
            plt.show()

        return amplitude, sigma

    def fit_lorentz_distribution(self, theta_i, thetas_io_rad, specular):
        # --- Fit a Lorentz distribution to the estimated specular component --- #
        popt, pcov = curve_fit(self.lorentz_numpy, thetas_io_rad, specular)
        amplitude, width = popt
        amplitude, width = mi.Float(amplitude), mi.Float(width)

        # --- Debug: plot the gaussian fit to the specular component --- #
        if self.debug:
            print(f'Fit lorentz distribution, amplitude: {amplitude}, width: {width}')
            print(f'Base reflectance: {self.base_reflectance}')
            specular_fit = self.lorentz_numpy(np.linspace(-np.pi / 2, np.pi / 2, 180), amplitude, width)

            plt.scatter(thetas_io_rad, specular, label='Measured specular component')
            plt.plot(np.linspace(-np.pi / 2, np.pi / 2, 180), specular_fit, label='Fit specular component')
            plt.xlabel('Angle (rad) between $\omega_o, reflect(\omega_i)$')
            plt.ylabel('Reflectance')
            plt.title(f'Specular component (sr^-1), lorentz fit, $\\theta_i = {theta_i}º$')
            plt.legend()
            plt.show()
        return amplitude, width

    def gaussian(self, x = mi.Float(), amplitude = mi.Float(), sigma = mi.Float()):
        """
            Returns the value for a gaussian distribution given amplitude (peak), mean and std. deviation.
            Uses drtyps types for faster execution
        """
        return self.amplitude * dr.exp(-(x ** 2) / (2 * self.sigma ** 2))

    def gaussian_numpy(self, x, amplitude, sigma):
        """
            Returns the value for a gaussian distribution given amplitude (peak), mean and std. deviation.
            Uses numpty types for compatibility with scipy during curve fitting
        """
        return amplitude * np.exp(-(x ** 2) / (2 * sigma ** 2))

    def lorentz(self, x = mi.Float(), amplitude = mi.Float(), width = mi.Float()):
        """
            Returns the value for a Lorentz distribution given amplitude, mean and width
            Uses drjit types for faster execution
        """
        return (2 * amplitude / dr.pi) * (width / (4 * dr.sqr(x) + dr.sqr(width)))
    
    def lorentz_numpy(self, x, amplitude, width):
        """
            Returns the value for a Lorentz distribution given amplitude, mean and width
            Uses numpty types for compatibility with scipy during curve fitting
        """
        return 2 * amplitude / np.pi * width / (4 * np.square(x) + np.square(width))

    @dr.syntax()
    def interpolate_brdf(self, cos_theta_i):
        thetas_i = list(self.brdf_thetas_i.keys())
        thetas_i.reverse() # Lower to higher cosine

        amplitude = mi.Float(0.0)
        width_sigma = mi.Float(0.0)
        assigned_mask = mi.Bool(False)

        for i, theta_i in enumerate(thetas_i):
            greater_cos = mi.Bool(mi.Bool(cos_theta_i <= self.brdf_thetas_i[theta_i]["cos_theta_i"]) & ~assigned_mask)
            assigned_mask = assigned_mask | greater_cos

            if i == 0:
                amplitude[greater_cos] = self.brdf_thetas_i[theta_i]["amplitude"]
                width_sigma[greater_cos] = self.brdf_thetas_i[theta_i]["width_sigma"]
            else:
                # Base case, interpolate between i-1 and 1
                x0 = mi.Float(thetas_i[i-1])
                x1 = mi.Float(thetas_i[i])

                amplitude_start = mi.Float(self.brdf_thetas_i[thetas_i[i-1]]["amplitude"])
                amplitude_end = mi.Float(self.brdf_thetas_i[theta_i]["amplitude"])

                width_sigma_start = mi.Float(self.brdf_thetas_i[thetas_i[i-1]]["width_sigma"])
                width_sigma_end = mi.Float(self.brdf_thetas_i[theta_i]["width_sigma"])

                x = dr.rad2deg(dr.acos(cos_theta_i))

                amplitude[greater_cos] = (amplitude_start * (x1 - x) + amplitude_end * (x - x0)) / (x1 - x0) 
                width_sigma[greater_cos] = (width_sigma_start * (x1 - x) + width_sigma_end * (x - x0)) / (x1 - x0) 
        
        amplitude[~assigned_mask] = self.brdf_thetas_i[thetas_i[i]]["amplitude"]
        width_sigma[~assigned_mask] = self.brdf_thetas_i[thetas_i[i]]["width_sigma"]
        
        return amplitude, width_sigma
 
    def sample(self, ctx, si, sample1, sample2, active=True):
        """
            Samples an outgoint direction in interaction si, following a diffuse event.
            The total reflectance will have an addition specular component,
            given the angle between the sampled ray and the reflected incident ray.
            Returns the sampled BSDF record, as well as the BSDF value.
        """
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        bs = dr.zeros(mi.BSDFSample3f)
        active &= cos_theta_i > 0.0 # Check backfaces

        bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.DiffuseReflection)
        bs.sampled_component = 0

        # Evaluate the diffuse BSDF at the interaction
        cos_theta_o = mi.Frame3f.cos_theta(bs.wo)
        value = self.eval(ctx, si, bs.wo, active)
        return [bs, dr.select(active, dr.maximum(value, 0.0), 0.0)]

    def eval(self, ctx, si, wo, active=True):
        """
            Given a sampled ray wo, returns the BSDF value taking into account the
            additional specular component, obtained by querying the distribution used
            with the angle between wo and the specular reflection of wi.
        """
        if not ctx.is_enabled(mi.BSDFFlags.DiffuseReflection):
            return 0.0

        # Check none of the directions are back-facing
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        active &= mi.Bool(cos_theta_i > 0.0) & mi.Bool(cos_theta_o > 0.0)

        # Use the angle between reflected wi and wo to recover the estimated specular component
        wi_specular = mi.Vector3f(-si.wi.x, -si.wi.y, si.wi.z)
        theta_io = dr.safe_acos(dr.dot(wo, wi_specular))

        # Recover value of the specular component
        specular_component = 0.0
        if self.measurements_file is None:
            # Parametric BSDF
            specular_component = self.gaussian(theta_io, self.amplitude, self.sigma) if self.gaussian_dist \
                else self.lorentz(theta_io, self.amplitude, self.sigma)
        else:
            # Measured BSDF

            # TODO interpolate between the fit BRDFs to find an amplitude and a width or sigma
            amplitude, width_sigma = self.interpolate_brdf(cos_theta_i)
            specular_component = self.gaussian(theta_io, amplitude, width_sigma) if self.gaussian_dist \
                else self.lorentz(theta_io, amplitude, width_sigma)       

        value = (self.base_reflectance + specular_component) * cos_theta_o * dr.inv_pi

        return dr.select(active, value, 0.0)

    def pdf(self, ctx, si, wo, active=True):
        """
            Returns the pdf of the sampled outgoing direction.
        """
        if not ctx.is_enabled(mi.BSDFFlags.DiffuseReflection):
            return 0.0
        
        # Check that none of the directions (incoming or outgoing) are backfacing
        cos_theta_i = mi.Frame3f.cos_theta(si.wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)

        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wo) 
        return dr.select(mi.Bool(cos_theta_i > 0.0) & mi.Bool(cos_theta_o > 0.0), pdf, 0.0)

    def eval_pdf(self, ctx, si, wo, active=True):
        """
            Returns BSDF value and pdf given an iteraction and an outgoing direction
        """
        return [self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)]

    def traverse(self, callback):
        return

    # Read csv with measurements
    # Returns measured angle (radians), measured reflectance, diffuse component, specular component
    def read_csv(self, csv_file_name):
        """
            Reads csv file with the radiance measurements, stored in path csv_file_name.
            The file should follow the next format:
                * Header row: thetas_o, theta_i_1, theta_i_2, ..., theta_i_n
                * Data rows:  theta_o, BSDF(theta_i_1, theta_o), BSDF(theta_i_2, theta_o), ...
                * thetas_o in range [-90º, 90º], step size = 1º or 2º
                * theta_i columns sorted in the same order as the input file
                * Measured values in sr^-1
                * Measured values padded with zeros, if measured range is smaller or the data has gaps  

            The method returns the following values:
                * thetas_o_rad: numpy float array with the measured observation angles, in radians
                * base_reflectance: lambertian BRDF reflectance value (sr^-1), independent of illumination angle
                * brdf_thetas_i: ordered dictionary containing the different measurements and parameters of the
                    bsdf, given the incidence angle used for the experiment. It contains the following fields
                    * cos_theta_i: precomputed cosine of the incidence angle
                    * thetas_io_rad: numpy float array with the angle between observation and (reflected) incoming light, in radians
                    * measured_BRDF: measured BRDF values, in sr^-1

            If self.debug is set to True, the method will also print the values loaded from the file.
        """
        thetas_o = []
        measured_R = []
        diffuse = []
        specular = []

        # --- Load input csv file to numpy array --- #
        data_np = np.genfromtxt(csv_file_name, delimiter=',', dtype=float)

        # --- Observation and incidence angles --- #
        thetas_o = data_np[1:, 0]                   # Get measured observation angles
        thetas_o_rad = thetas_o * (np.pi / 180.0)   # Observation angles, in radians
        thetas_i = data_np[0, 1:]                   # Get measured incidence angles

        # --- Recover the base diffuse component --- #
        # Obtained by averaging all (non zero) measured values that follow a Lambertian behaviour
        # That is, values in the opposite side to the specular reflection
        mask = np.hstack([False, thetas_i == thetas_i[0]]) # Pick any angle
        measured_BRDF = data_np[1:, mask].flatten()

        lambertian_values = measured_BRDF[0:int(measured_BRDF.size / 2) + 2]
        base_reflectance = np.mean(lambertian_values[~(lambertian_values == 0)])

        # Ordered dict to keep track of the different BRDF measures/parameters given theta_i
        brdf_thetas_i = OrderedDict()

        for theta_i in thetas_i:
            print(f'theta_i = {theta_i}º, cos(theta_i) = {np.cos(np.radians(theta_i))}')

            # Angle between observation and (specularly reflected) incident illumination
            incidence_rad = theta_i * (np.pi / 180.0)
            thetas_io_rad = thetas_o_rad - incidence_rad

            # --- Recover the measured BRDF (sr^-1) --- #
            mask = np.hstack([False, thetas_i == theta_i])
            measured_BRDF = data_np[1:, mask].flatten()

            # Set all zero values measured to the base reflectance, for an easier fitting
            measured_BRDF[measured_BRDF == 0] = base_reflectance

            brdf_thetas_i[theta_i] = {
                "cos_theta_i" : np.cos(np.radians(theta_i)),
                "thetas_io_rad" : thetas_io_rad,
                "measured_BRDF" : measured_BRDF,
            }
        
        # --- Plot the data if Debug is enabled --- # 
        if self.debug:
            for theta_i in thetas_i:
                measured_BRDF = brdf_thetas_i[theta_i]["measured_BRDF"]
                measured_R = measured_BRDF * np.cos(thetas_o_rad)
                diffuse = base_reflectance * np.cos(thetas_o_rad)
                specular = measured_R - diffuse

                plt.plot(thetas_o_rad, specular, label='Measured specular component')
                plt.plot(thetas_o_rad, measured_R, label='Measured reflectance')
                plt.plot(thetas_o_rad, diffuse, label='Lambertian reflectance')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                plt.title(f'{csv_file_name}, %R')
                plt.show()

                plt.scatter(thetas_o_rad, measured_BRDF - base_reflectance, label='Measured specular component')
                plt.scatter(thetas_o_rad, measured_BRDF, label='Measured reflectance')
                plt.axhline(base_reflectance, label='Lambertian reflectance')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                plt.title(f'{csv_file_name}, sr^-1, $\\theta_i = {theta_i}º$')
                plt.show()
        
        return thetas_o_rad, base_reflectance, brdf_thetas_i

mi.register_bsdf('diffuse_swir', lambda props: DiffuseSWIR(props))
