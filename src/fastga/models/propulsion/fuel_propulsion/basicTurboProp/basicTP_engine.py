"""Parametric propeller IC engine."""
# -*- coding: utf-8 -*-
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import math
import numpy as np
import pandas as pd
from typing import Union, Sequence, Tuple, Optional
from scipy.interpolate import interp2d
from scipy.optimize import *
import os.path as pth
import os

from fastoad.model_base import FlightPoint, Atmosphere
from fastoad.constants import EngineSetting
from fastoad.exceptions import FastUnknownEngineSettingError

from .exceptions import FastBasicICEngineInconsistentInputParametersError
from . import resources

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.dict import DynamicAttributeDict, AddKeyAttributes

# Logger for this module
_LOGGER = logging.getLogger(__name__)

PROPELLER_EFFICIENCY = 0.83  # Used to be 0.8 maybe make it an xml parameter

# Set of dictionary keys that are mapped to instance attributes.
ENGINE_LABELS = {
    "power_SL": dict(doc="Power at sea level in watts."),
    "mass": dict(doc="Mass in kilograms."),
    "length": dict(doc="Length in meters."),
    "height": dict(doc="Height in meters."),
    "width": dict(doc="Width in meters."),
}
# Set of dictionary keys that are mapped to instance attributes.
NACELLE_LABELS = {
    "wet_area": dict(doc="Wet area in meters²."),
    "length": dict(doc="Length in meters."),
    "height": dict(doc="Height in meters."),
    "width": dict(doc="Width in meters."),
}


class BasicTPEngine(AbstractFuelPropulsion):
    def __init__(
            self,
            power_design: float,
            t41t_design: float,
            opr_design: float,
            design_altitude: float,
            design_mach: float,
            prop_layout: float,
            bleed_control_design: str,
            speed_SL,
            thrust_SL,
            thrust_limit_SL,
            efficiency_SL,
            speed_CL,
            thrust_CL,
            thrust_limit_CL,
            efficiency_CL,
    ):

        """
        Parametric Internal Combustion engine.

        It computes engine characteristics using fuel type, motor architecture
        and constant propeller efficiency using analytical model from following sources:

        :param max_power: maximum delivered mechanical power of engine (units=W)
        :param design_altitude: design altitude for cruise (units=m)
        :param design_speed: design altitude for cruise (units=m/s)
        :param fuel_type: 1.0 for gasoline and 2.0 for diesel engine and 3.0 for Jet Fuel
        :param strokes_nb: can be either 2-strokes (=2.0) or 4-strokes (=4.0)
        :param prop_layout: propulsion position in nose (=3.0) or wing (=1.0)
        """

        # Definition of the Turboprop design parameters
        self.eta_225 = 0.85  # First compressor stage polytropic efficiency
        self.eta_253 = 0.86  # Second compressor stage polytropic efficiency
        self.eta445 = 0.86  # High preassure turbine  polytropic efficiency
        self.eta455 = 0.86  # Power turbine  polytropic efficiency
        self.etaqL = 43.260e6 * 0.95  # Combustion efficiency [J/kg]
        self.eta_axe = 0.98  # HP axe mechanical efficiency
        self.pi02 = 0.8  # Inlet preassure loss
        self.picc = 0.95  # Combustion chamber preassure loss
        self.c = 0.05  # Percentage of the total aspirated airflow used for turbine cooling
        self.hp_shaft_power_out = 50 * 736  # Power used for electrical generation obtained from the HP shaft (in Watts)
        self.gearbox_efficiency = 0.98  # Power shaft mechanical efficiency
        self.inter_compressor_bleed = 0.04  # Percentage of the total inlet compressor airflow extracted after the first
        # compression stage (in station 25)
        self.exhaust_mach_design = 0.4  # Mach of the exhaust gasses in the design point
        self.opr1_design = 0.25 * opr_design  # Compression ratio of the first stage in the design point
        self.bleed_control_design = bleed_control_design

        # Definition of the Turboprop design parameters
        self.max_power = power_design
        self.t41t_d = t41t_design
        self.opr_d = opr_design

        # Original atributes from the ICE class, modified where convenient
        self.ref = {
            "max_power": 160000,
            "length": 0.0,
            "height": 0.0,
            "width": 0.0,
            "mass": 0.0,
        }
        self.prop_layout = prop_layout
        self.design_altitude = design_altitude
        self.design_mach = design_mach
        self.fuel_type = 3.0  # Turboprops only use JetFuel
        self.idle_thrust_rate = 0.01
        self.speed_SL = speed_SL
        self.thrust_SL = thrust_SL
        self.thrust_limit_SL = thrust_limit_SL
        self.efficiency_SL = efficiency_SL
        self.speed_CL = speed_CL
        self.thrust_CL = thrust_CL
        self.thrust_limit_CL = thrust_limit_CL
        self.efficiency_CL = efficiency_CL
        self.specific_shape = None

        # Evaluate engine volume based on max power @ 0.0m
        self.volume = 1e-6

        # Declare sub-components attribute
        self.engine = Engine(power_SL=power_design)
        self.nacelle = None
        self.propeller = None

        # This dictionary is expected to have a Mixture coefficient for all EngineSetting values
        self.mixture_values = {
            EngineSetting.TAKEOFF: 1.15,
            EngineSetting.CLIMB: 1.15,
            EngineSetting.CRUISE: 1.0,
            EngineSetting.IDLE: 1.0,
        }
        self.rpm_values = {
            EngineSetting.TAKEOFF: 2700.0,
            EngineSetting.CLIMB: 2700.0,
            EngineSetting.CRUISE: 2500.0,
            EngineSetting.IDLE: 2300.0,
        }

        # ... so check that all EngineSetting values are in dict
        unknown_keys = [key for key in EngineSetting if key not in self.mixture_values.keys()]
        if unknown_keys:
            raise FastUnknownEngineSettingError("Unknown flight phases: %s", unknown_keys)

        alfa, alfa_p, a41, a45, a8, eta_compress, mc, t4t, t41t, t45t, opr2_opr1 = \
            self.turboprop_geometry_calculation()

        self.alfa = alfa
        self.alfa_p = alfa
        self.a41 = a41
        self.a45 = a45
        self.a8 = a8
        self.eta_compress_design = eta_compress
        self.mc_dp = mc
        self.t4t_dp = t4t
        self.t45t_dp = t45t
        self.opr2_opr1_dp = opr2_opr1  # Compression ratio relationship between the second and first stages

    @staticmethod
    def air_coefficients_reader():

        """
        This function reads  table with a et of temperatures, Cv and Cp values. It creates two polynomial interpolation
        functions, whose coefficients are returned [one for Cv = f(T) and another for Cp = f(T)]
        """

        parent_file = os.path.dirname(os.getcwd())
        file_dir = (parent_file + "\\fuel_propulsion\\basicTurboProp\\T_Cv_Cp.txt")
        profile = np.loadtxt(file_dir)
        matrix = np.asmatrix(profile)
        temp_n = matrix[:, 0]
        cv_n = matrix[:, 1]
        cp_n = matrix[:, 2]

        temp = np.squeeze(np.asarray(temp_n))
        cv_coefficients = np.squeeze(np.asarray(cv_n))
        cp_coefficients = np.squeeze(np.asarray(cp_n))

        cv_t_coefficients = np.polyfit(temp, cv_coefficients, 15)
        cp_t_coefficients = np.polyfit(temp, cp_coefficients, 15)

        return cv_t_coefficients, cp_t_coefficients

    @staticmethod
    def compute_cp_cv_gamma(cp_coefficient, cv_coefficient, temperature):
        """
        Obtains the Cv and Cp values for a given Temperature

        It computes the polyfcal functions, shortening code:

        :param cp_coefficient: The polynomial interpolation coeficients for the Cp trend [-]
        :param cv_coefficient: The polynomial interpolation coeficients for the Cv trend [-]
        :param temperature: the actual Temperature in Kelnvin [K]

        :return cp_out: The actual Cp value for the given Temperature
        :return cv_out: The actual Cv value for the given Temperature
        :return gamma: The actual gamma value for the given Temperature
        """

        cp_out = np.polyval(cp_coefficient, temperature)
        cv_out = np.polyval(cv_coefficient, temperature)
        gamma = cp_out / cv_out
        return cp_out, cv_out, gamma

    @staticmethod
    def compute_gamma_functions(gamma):
        """
        Computes the three gamma functions for each gamma value
        """
        f1 = gamma / (gamma - 1)
        f2 = 1 / f1
        f_gamma = np.sqrt(gamma) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))

        return f1, f2, f_gamma

    @staticmethod
    def air_renewal_coefficients():
        """
        Creates polynomial coefficients for polynomial interpolation that allow to determine the cabin altitude,
        which is used for cabin air renewal computation
        """
        h = np.array([14500, 20000, 26000, 31000]) * 0.3048
        h_cab = np.array([0, 3500, 6800, 9300]) * 0.3048
        coefficients_2_return = np.polyfit(h, h_cab, 3)
        return coefficients_2_return

    @staticmethod
    def air_renewal(coefficients_air, h, bleed_control="high"):
        """
        Computes the airflow used for cabin air renewal

        :param coefficients_air: The polynomial regression coefficients obtained in air_renewal_coefficients()
        :param h: The flight altitude in meters [m]
        :param bleed_control: The air packs setting "high" or "low"

        :return m_air: The cabin airflow in [kg/s]
        """
        cabin_volume = 5  # in m3

        if bleed_control == "low":
            control = 0.3
        else:
            control = 1

        renovation_time = 2  # in minutes

        if h < 14500 * 0.3048:
            h_cab = 0
        else:
            h_cab = np.polyval(coefficients_air, h)

        t_cab = 20 + 273
        atmosphere = Atmosphere(h_cab, altitude_in_feet=False)
        p_cab = atmosphere.pressure

        rho_cab = p_cab / 287 / t_cab
        m_air = cabin_volume * rho_cab / (renovation_time * 60) * control
        return m_air

    def point_design_solver(self, X, T41t, P0, T2t, T25t, T3t, P3t, P, M_sortie, Cp_c, Cv_c,
                            bleed_control, h0):
        global solution_error
        mc = X[0]
        T4t = X[1]
        T45t = X[2]
        T5t = X[3]
        P45t = X[4]
        P5t = X[5]
        m0 = X[6]

        g = self.air_renewal(self.air_renewal_coefficients(), h0, bleed_control) / m0

        P4t = P3t * self.picc

        Cp2, Cv2, gamma2 = self.compute_cp_cv_gamma(Cp_c, Cv_c, T2t)
        # f1_2, f2_2, Fgamma_2 = self.compute_gammafuns(gamma2)
        Cp25, Cv25, gamma25 = self.compute_cp_cv_gamma(Cp_c, Cv_c, T25t)
        # f1_25, f2_25, Fgamma_25 = self.compute_gammafuns(gamma25)
        Cp3, Cv3, gamma3 = self.compute_cp_cv_gamma(Cp_c, Cv_c, T3t)
        # f1_3, f2_3, Fgamma_3 = self.compute_gammafuns(gamma3)
        Cp4, Cv4, gamma4 = self.compute_cp_cv_gamma(Cp_c, Cv_c, T4t)
        # f1_4, f2_4, Fgamma_4 = self.compute_gammafuns(gamma4)
        Cp41, Cv41, gamma41 = self.compute_cp_cv_gamma(Cp_c, Cv_c, T41t)
        f1_41, f2_41, Fgamma_41 = self.compute_gamma_functions(gamma41)
        Cp45, Cv45, gamma45 = self.compute_cp_cv_gamma(Cp_c, Cv_c, T45t)
        f1_45, f2_45, Fgamma_45 = self.compute_gamma_functions(gamma45)
        Cp5, Cv5, gamma5 = self.compute_cp_cv_gamma(Cp_c, Cv_c, T5t)
        f1_5, f2_5, Fgamma_5 = self.compute_gamma_functions(gamma5)

        fuel_air_ratio = mc / m0
        icb = self.inter_compressor_bleed / m0

        f = np.zeros(7)
        f[0] = (Cp4 * T4t - Cp3 * T3t) * (1 + fuel_air_ratio - g - self.c - icb) - self.etaqL * fuel_air_ratio
        f[1] = T41t - ((T4t * (1 + fuel_air_ratio - g - self.c - icb) + T3t * self.c) / (1 + fuel_air_ratio - g - icb))
        f[2] = (1 + fuel_air_ratio - g - icb) * (
                    Cp41 * T41t - Cp45 * T45t) * self.eta_axe - self.hp_shaft_power_out / m0 - (
                       Cp3 * T3t - Cp25 * T25t) * (1 - icb) - (Cp25 * T25t - Cp2 * T2t)
        f[3] = P45t - (P4t * ((T45t / T41t)) ** (f1_41 / self.eta445))
        f[4] = m0 * 1000 - (((P * 736 / self.gearbox_efficiency) / (Cp45 * T45t - Cp5 * T5t)) / (
                    1 - g + fuel_air_ratio - icb)) * 1000
        f[5] = T5t - T45t * (((P5t / P45t) ** (f2_45 * self.eta455)))
        f[6] = P5t - (P0 * (1 + (gamma5 - 1) / 2 * M_sortie ** 2) ** (f1_5))

        solution_error = np.array(
            [f[0] / self.etaqL * fuel_air_ratio, f[1] / T41t, f[2] / (Cp3 * T3t - Cp2 * T2t), f[3] / P45t, \
             f[5] / T5t, f[6] / P5t])

        return f

    def turboprop_geometry_calculation(self):
        Rg = 287

        M0 = self.design_mach
        h0 = self.design_altitude
        P = self.max_power
        OPR = self.opr_d
        T41t = self.t41t_d
        M_sortie = self.exhaust_mach_design
        bleed_control = self.bleed_control_design
        cab_bleed = self.air_renewal(self.air_renewal_coefficients(), h0, bleed_control)
        OPR1 = self.opr1_design
        OPR2 = OPR / self.opr1_design

        Cv_c, Cp_c = self.air_coefficients_reader()

        atmosphere_0 = Atmosphere(h0, altitude_in_feet=False)
        P0 = atmosphere_0.pressure
        T0 = atmosphere_0.temperature

        P0t = P0 * (1 + (1.4 - 1) / 2 * M0 ** 2) ** 3.5
        T0t = T0 * (1 + (1.4 - 1) / 2 * M0 ** 2)

        P2t = P0t * self.pi02
        T2t = T0t

        Cp2, Cv2, gamma2 = self.compute_cp_cv_gamma(Cp_c, Cv_c, T2t)
        f1_2, f2_2, Fgamma_2 = self.compute_gamma_functions(gamma2)

        T25t = T2t * OPR1 ** (f2_2 / self.eta_225)
        P25t = P2t * OPR1
        P3t = P25t * OPR2

        Cp25, Cv25, gamma25 = self.compute_cp_cv_gamma(Cp_c, Cv_c, T25t)
        f1_25, f2_25, Fgamma_25 = self.compute_gamma_functions(gamma25)

        T3t = T25t * OPR2 ** (f2_25 / self.eta_253)
        eta_compress = math.log(OPR) / math.log(T3t / T2t) * f2_2

        P4t = P3t * self.picc
        P41t = P4t

        global solution_error
        solution_error = np.zeros(7)

        X0 = np.array([0.06, 1350, 1000, 800, 400000, 110000, 3.5])
        # z = np.zeros(len(X0))
        solution_vector = fsolve(self.point_design_solver, X0,
                                 (T41t, P0, T2t, T25t, T3t, P3t, P, M_sortie, Cp_c, Cv_c, bleed_control, h0), xtol=1e-4)
        mc = solution_vector[0]
        T4t = solution_vector[1]
        T45t = solution_vector[2]
        T5t = solution_vector[3]
        P45t = solution_vector[4]
        P5t = solution_vector[5]
        m0 = solution_vector[6]

        f = mc / m0
        g = cab_bleed / m0
        icb = self.inter_compressor_bleed / m0

        Cp3, Cv3, gamma3 = self.compute_cp_cv_gamma(Cp_c, Cv_c, T3t)
        f1_3, f2_3, Fgamma_3 = self.compute_gamma_functions(gamma3)
        Cp4, Cv4, gamma4 = self.compute_cp_cv_gamma(Cp_c, Cv_c, T4t)
        f1_4, f2_4, Fgamma_4 = self.compute_gamma_functions(gamma4)
        Cp41, Cv41, gamma41 = self.compute_cp_cv_gamma(Cp_c, Cv_c, T41t)
        f1_41, f2_41, Fgamma_41 = self.compute_gamma_functions(gamma41)
        Cp45, Cv45, gamma45 = self.compute_cp_cv_gamma(Cp_c, Cv_c, T45t)
        f1_45, f2_45, Fgamma_45 = self.compute_gamma_functions(gamma45)
        Cp5, Cv5, gamma5 = self.compute_cp_cv_gamma(Cp_c, Cv_c, T5t)
        f1_5, f2_5, Fgamma_5 = self.compute_gamma_functions(gamma5)

        alfa = T45t / T41t
        alfa_p = P45t / P41t

        OPR_check = (Cp2 / Cp3 / (1 - icb) + self.eta_axe * (1 + f - g - icb) / (1 - icb) * (
                Cp41 - Cp45 * alfa) / Cp3 * T41t / T2t \
                     - self.hp_shaft_power_out / (Cp3 * m0 * (1 - icb) * T2t) - T25t / T2t * Cp25 / Cp3 * (
                                 1 / (1 - icb) - 1)) ** (
                            f1_2 * eta_compress)

        A41 = m0 * (1 + f - g - icb) * np.sqrt(T41t * Rg) / P4t / Fgamma_41
        A45 = m0 * (1 + f - g - icb) * np.sqrt(T45t * Rg) / P45t / Fgamma_45
        A8_1 = m0 * (1 + f - g - icb) * np.sqrt(T5t * Rg) / P5t
        A8_2 = np.sqrt(gamma5) * M_sortie * (1 + (gamma5 - 1) / 2 * M_sortie ** 2) ** (
                (gamma5 + 1) / (2 * (1 - gamma5)))
        A8 = A8_1 / A8_2

        T8 = T5t / (1 + (gamma5 - 1) / 2 * M_sortie ** 2)
        V8 = M_sortie * np.sqrt(gamma5 * Rg * T8)

        Exhaust_Thrust = m0 * (1 + f - icb - g) * (V8 - M0 * np.sqrt(T0 * 287 * 1.4))

        #
        Power_check = (Cp45 * T45t - Cp5 * T5t) * m0 / 736 * (1 - g + f - icb) * self.gearbox_efficiency
        print("--")
        print("DESSIGN CP variable  |||    eta23 =", round(eta_compress, 5), "   eta445 =", round(self.eta445, 5),
              "   eta455 =",
              round(self.eta455, 5), "   eta_axe", round(self.eta_axe, 5))
        print("Temperatures [K]  :     ", "T2t=", round(T2t), "T25t=", round(T25t), "T3t=", round(T3t), "T4t=",
              round(T4t),
              " T41t=", round(T41t), "T45t=", round(T45t), "T5t=", round(T5t))
        print("Pressures  [Pa]   :     ", "P2t=", round(P2t), "P25t=", round(P25t), "P3t=", round(P3t), "P4t=",
              round(P4t),
              "P45t=", round(P45t), "P5t=", round(P5t))
        print("Airflows [kg/s]   :       m_air = ", round(m0, 5), "    m_fuel = ", round(mc, 8), "   m_bleed=",
              round(g * m0, 5), "   m_cooling=", round(self.c * m0, 5), "   m_inter-compressor=", round(icb * m0, 5))
        print("A41=", round(A41, 5), "A45=", round(A45, 4), "A8=", round(A8, 3), "alfa=", round(alfa, 4))
        print(" OPR_check=", round(OPR_check, 2), "P =", round(Power_check), "Thrust =", round(Exhaust_Thrust))
        print("----------------------------------------------------")
        print("CONVERGENCE ERROR=", solution_error)
        # print("CP2", round(Cp2), "CP3=", round(Cp3), "CP41=", round(Cp41), "CP45=", round(Cp45), "CP5=", round(Cp5))

        return alfa, alfa_p, A41, A45, A8, eta_compress, mc, T4t, T41t, T45t, OPR2 / OPR1

    @staticmethod
    def read_map(map_file_path):

        data = pd.read_csv(map_file_path)
        values = data.to_numpy()[:, 1:].tolist()
        labels = data.to_numpy()[:, 0].tolist()
        data = pd.DataFrame(values, index=labels)
        rpm = data.loc["rpm", 0][1:-2].replace("\n", "").replace("\r", "")
        for idx in range(10):
            rpm = rpm.replace("  ", " ")
        rpm_vect = np.array([float(i) for i in rpm.split(" ") if i != ""])
        pme = data.loc["pme", 0][1:-2].replace("\n", "").replace("\r", "")
        for idx in range(10):
            pme = pme.replace("  ", " ")
        pme_vect = np.array([float(i) for i in pme.split(" ") if i != ""])
        pme_limit = data.loc["pme_limit", 0][1:-2].replace("\n", "").replace("\r", "")
        for idx in range(10):
            pme_limit = pme_limit.replace("  ", " ")
        pme_limit_vect = np.array([float(i) for i in pme_limit.split(" ") if i != ""])
        sfc = data.loc["sfc", 0][1:-2].replace("\n", "").replace("\r", "")
        sfc_lines = sfc[1:-2].split("] [")
        sfc_matrix = np.zeros(
            (len(np.array([i for i in sfc_lines[0].split(" ") if i != ""])), len(sfc_lines))
        )
        for idx in range(len(sfc_lines)):
            sfc_matrix[:, idx] = np.array([i for i in sfc_lines[idx].split(" ") if i != ""])

        return rpm_vect, pme_vect, pme_limit_vect, sfc_matrix

    def compute_flight_points(self, flight_points: FlightPoint):
        # pylint: disable=too-many-arguments  # they define the trajectory
        self.specific_shape = np.shape(flight_points.mach)
        if isinstance(flight_points.mach, float):
            sfc, thrust_rate, thrust = self._compute_flight_points(
                flight_points.mach,
                flight_points.altitude,
                flight_points.engine_setting,
                flight_points.thrust_is_regulated,
                flight_points.thrust_rate,
                flight_points.thrust,
            )
            flight_points.sfc = sfc
            flight_points.thrust_rate = thrust_rate
            flight_points.thrust = thrust
        else:
            mach = np.asarray(flight_points.mach)
            altitude = np.asarray(flight_points.altitude).flatten()
            engine_setting = np.asarray(flight_points.engine_setting).flatten()
            if flight_points.thrust_is_regulated is None:
                thrust_is_regulated = None
            else:
                thrust_is_regulated = np.asarray(flight_points.thrust_is_regulated).flatten()
            if flight_points.thrust_rate is None:
                thrust_rate = None
            else:
                thrust_rate = np.asarray(flight_points.thrust_rate).flatten()
            if flight_points.thrust is None:
                thrust = None
            else:
                thrust = np.asarray(flight_points.thrust).flatten()
            self.specific_shape = np.shape(mach)
            sfc, thrust_rate, thrust = self._compute_flight_points(
                mach.flatten(), altitude, engine_setting, thrust_is_regulated, thrust_rate, thrust,
            )
            if len(self.specific_shape) != 1:  # reshape data that is not array form
                # noinspection PyUnresolvedReferences
                flight_points.sfc = sfc.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.thrust_rate = thrust_rate.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.thrust = thrust.reshape(self.specific_shape)
            else:
                flight_points.sfc = sfc
                flight_points.thrust_rate = thrust_rate
                flight_points.thrust = thrust

    def _compute_flight_points(
            self,
            mach: Union[float, Sequence],
            altitude: Union[float, Sequence],
            engine_setting: Union[EngineSetting, Sequence],
            thrust_is_regulated: Optional[Union[bool, Sequence]] = None,
            thrust_rate: Optional[Union[float, Sequence]] = None,
            thrust: Optional[Union[float, Sequence]] = None,
    ) -> Tuple[Union[float, Sequence], Union[float, Sequence], Union[float, Sequence]]:
        """
        Same as :meth:`compute_flight_points`.

        :param mach: Mach number
        :param altitude: (unit=m) altitude w.r.t. to sea level
        :param engine_setting: define engine settings
        :param thrust_is_regulated: tells if thrust_rate or thrust should be used (works element-wise)
        :param thrust_rate: thrust rate (unit=none)
        :param thrust: required thrust (unit=N)
        :return: SFC (in kg/s/N), thrust rate, thrust (in N)
        """
        """
        Computes the Specific Fuel Consumption based on aircraft trajectory conditions.
        
        :param flight_points.mach: Mach number
        :param flight_points.altitude: (unit=m) altitude w.r.t. to sea level
        :param flight_points.engine_setting: define
        :param flight_points.thrust_is_regulated: tells if thrust_rate or thrust should be used (works element-wise)
        :param flight_points.thrust_rate: thrust rate (unit=none)
        :param flight_points.thrust: required thrust (unit=N)
        :return: SFC (in kg/s/N), thrust rate, thrust (in N)
        """

        # Treat inputs (with check on thrust rate <=1.0)
        if thrust_is_regulated is not None:
            thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        thrust_is_regulated, thrust_rate, thrust = self._check_thrust_inputs(
            thrust_is_regulated, thrust_rate, thrust
        )
        thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        thrust_rate = np.asarray(thrust_rate)
        thrust = np.asarray(thrust)

        # Get maximum thrust @ given altitude & mach
        atmosphere = Atmosphere(np.asarray(altitude), altitude_in_feet=False)
        mach = np.asarray(mach) + (np.asarray(mach) == 0) * 1e-12
        atmosphere.mach = mach
        max_thrust = self.max_thrust(np.asarray(engine_setting), atmosphere)

        # We compute thrust values from thrust rates when needed
        idx = np.logical_not(thrust_is_regulated)
        if np.size(max_thrust) == 1:
            maximum_thrust = max_thrust
            out_thrust_rate = thrust_rate
            out_thrust = thrust
        else:
            out_thrust_rate = (
                np.full(np.shape(max_thrust), thrust_rate.item())
                if np.size(thrust_rate) == 1
                else thrust_rate
            )
            out_thrust = (
                np.full(np.shape(max_thrust), thrust.item()) if np.size(thrust) == 1 else thrust
            )
            maximum_thrust = max_thrust[idx]
        if np.any(idx):
            out_thrust[idx] = out_thrust_rate[idx] * maximum_thrust
        if np.any(thrust_is_regulated):
            out_thrust[thrust_is_regulated] = np.minimum(
                out_thrust[thrust_is_regulated], max_thrust[thrust_is_regulated]
            )

        # thrust_rate is obtained from entire thrust vector (could be optimized if needed,
        # as some thrust rates that are computed may have been provided as input)
        out_thrust_rate = out_thrust / max_thrust

        # Now SFC (g/kwh) can be computed and converted to sfc_thrust (kg/N) to match computation from turboshaft
        sfc, mech_power = self.sfc(out_thrust, engine_setting, atmosphere)
        sfc_time = (mech_power * 1e-3) * sfc / 3.6e6  # sfc in kg/s
        sfc_thrust = sfc_time / np.maximum(out_thrust, 1e-6)  # avoid 0 division

        return sfc_thrust, out_thrust_rate, out_thrust

    @staticmethod
    def _check_thrust_inputs(
            thrust_is_regulated: Optional[Union[float, Sequence]],
            thrust_rate: Optional[Union[float, Sequence]],
            thrust: Optional[Union[float, Sequence]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Checks that inputs are consistent and return them in proper shape.
        Some of the inputs can be None, but outputs will be proper numpy arrays.
        :param thrust_is_regulated:
        :param thrust_rate:
        :param thrust:
        :return: the inputs, but transformed in numpy arrays.
        """
        # Ensure they are numpy array
        if thrust_is_regulated is not None:
            # As OpenMDAO may provide floats that could be slightly different
            # from 0. or 1., a rounding operation is needed before converting
            # to booleans
            thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        if thrust_rate is not None:
            thrust_rate = np.asarray(thrust_rate)
        if thrust is not None:
            thrust = np.asarray(thrust)

        # Check inputs: if use_thrust_rate is None, we will use the provided input between
        # thrust_rate and thrust
        if thrust_is_regulated is None:
            if thrust_rate is not None:
                thrust_is_regulated = False
                thrust = np.empty_like(thrust_rate)
            elif thrust is not None:
                thrust_is_regulated = True
                thrust_rate = np.empty_like(thrust)
            else:
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When use_thrust_rate is None, either thrust_rate or thrust should be provided."
                )

        elif np.size(thrust_is_regulated) == 1:
            # Check inputs: if use_thrust_rate is a scalar, the matching input(thrust_rate or
            # thrust) must be provided.
            if thrust_is_regulated:
                if thrust is None:
                    raise FastBasicICEngineInconsistentInputParametersError(
                        "When thrust_is_regulated is True, thrust should be provided."
                    )
                thrust_rate = np.empty_like(thrust)
            else:
                if thrust_rate is None:
                    raise FastBasicICEngineInconsistentInputParametersError(
                        "When thrust_is_regulated is False, thrust_rate should be provided."
                    )
                thrust = np.empty_like(thrust_rate)

        else:
            # Check inputs: if use_thrust_rate is not a scalar, both thrust_rate and thrust must be
            # provided and have the same shape as use_thrust_rate
            if thrust_rate is None or thrust is None:
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When thrust_is_regulated is a sequence, both thrust_rate and thrust should be "
                    "provided."
                )
            if np.shape(thrust_rate) != np.shape(thrust_is_regulated) or np.shape(
                    thrust
            ) != np.shape(thrust_is_regulated):
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When use_thrust_rate is a sequence, both thrust_rate and thrust should have "
                    "same shape as use_thrust_rate"
                )

        return thrust_is_regulated, thrust_rate, thrust

    def propeller_efficiency(
            self, thrust: Union[float, Sequence[float]], atmosphere: Atmosphere
    ) -> Union[float, Sequence]:
        """
        Compute the propeller efficiency.

        :param thrust: Thrust (in N)
        :param atmosphere: Atmosphere instance at intended altitude
        :return: efficiency
        """

        propeller_efficiency_SL = interp2d(
            self.thrust_SL, self.speed_SL, self.efficiency_SL, kind="cubic"
        )
        propeller_efficiency_CL = interp2d(
            self.thrust_CL, self.speed_CL, self.efficiency_CL, kind="cubic"
        )
        if isinstance(atmosphere.true_airspeed, float):
            thrust_interp_SL = np.minimum(
                np.maximum(np.min(self.thrust_SL), thrust),
                np.interp(atmosphere.true_airspeed, self.speed_SL, self.thrust_limit_SL),
            )
            thrust_interp_CL = np.minimum(
                np.maximum(np.min(self.thrust_CL), thrust),
                np.interp(atmosphere.true_airspeed, self.speed_CL, self.thrust_limit_CL),
            )
        else:
            thrust_interp_SL = np.minimum(
                np.maximum(np.min(self.thrust_SL), thrust),
                np.interp(list(atmosphere.true_airspeed), self.speed_SL, self.thrust_limit_SL),
            )
            thrust_interp_CL = np.minimum(
                np.maximum(np.min(self.thrust_CL), thrust),
                np.interp(list(atmosphere.true_airspeed), self.speed_CL, self.thrust_limit_CL),
            )
        if np.size(thrust) == 1:  # calculate for float
            lower_bound = float(propeller_efficiency_SL(thrust_interp_SL, atmosphere.true_airspeed))
            upper_bound = float(propeller_efficiency_CL(thrust_interp_CL, atmosphere.true_airspeed))
            altitude = atmosphere.get_altitude(altitude_in_feet=False)
            propeller_efficiency = np.interp(
                altitude, [0, self.design_altitude], [lower_bound, upper_bound]
            )
        else:  # calculate for array
            propeller_efficiency = np.zeros(np.size(thrust))
            for idx in range(np.size(thrust)):
                lower_bound = propeller_efficiency_SL(
                    thrust_interp_SL[idx], atmosphere.true_airspeed[idx]
                )
                upper_bound = propeller_efficiency_CL(
                    thrust_interp_CL[idx], atmosphere.true_airspeed[idx]
                )
                altitude = atmosphere.get_altitude(altitude_in_feet=False)[idx]
                propeller_efficiency[idx] = (
                        lower_bound
                        + (upper_bound - lower_bound)
                        * np.minimum(altitude, self.design_altitude)
                        / self.design_altitude
                )

        return propeller_efficiency

    def compute_max_power(self, flight_points: FlightPoint) -> Union[float, Sequence]:
        """
        Compute the ICE maximum power @ given flight-point.

        :param flight_points: current flight point(s)
        :return: maximum power in kW
        """

        atmosphere = Atmosphere(np.asarray(flight_points.altitude), altitude_in_feet=False)
        sigma = atmosphere.density / Atmosphere(0.0).density
        max_power = (self.max_power / 1e3) * (sigma - (1 - sigma) / 7.55)  # max power in kW

        return max_power

    def sfc(
            self,
            thrust: Union[float, Sequence[float]],
            engine_setting: Union[float, Sequence[float]],
            atmosphere: Atmosphere,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computation of the SFC.

        :param thrust: Thrust (in N)
        :param engine_setting: Engine settings (climb, cruise,... )
        :param atmosphere: Atmosphere instance at intended altitude
        :return: SFC (in g/kw) and Power (in W)
        """

        # Load engine map and save interpolation formula
        rpm_vect, pme_vect, _, sfc_matrix = self.read_map(self.map_file_path)
        torque_vect = pme_vect * 1e5 * self.volume / (8.0 * np.pi)
        ICE_sfc = interp2d(torque_vect, rpm_vect, sfc_matrix, kind="cubic")

        # Define RPM & mixture using engine settings
        if np.size(engine_setting) == 1:
            rpm_values = self.rpm_values[int(engine_setting)]
            mixture_values = self.mixture_values[int(engine_setting)]
        else:
            rpm_values = np.array(
                [self.rpm_values[engine_setting[idx]] for idx in range(np.size(engine_setting))]
            )
            mixture_values = np.array(
                [self.mixture_values[engine_setting[idx]] for idx in range(np.size(engine_setting))]
            )

        # Compute sfc @ 2500RPM
        real_power = np.zeros(np.size(thrust))
        torque = np.zeros(np.size(thrust))
        sfc = np.zeros(np.size(thrust))
        if np.size(thrust) == 1:
            real_power = (
                    thrust * atmosphere.true_airspeed / self.propeller_efficiency(thrust, atmosphere)
            )
            torque = real_power / (rpm_values * np.pi / 30.0)
            sfc = ICE_sfc(torque, rpm_values) * mixture_values
        else:
            for idx in range(np.size(thrust)):
                local_atmosphere = Atmosphere(
                    atmosphere.get_altitude()[idx], altitude_in_feet=False
                )
                local_atmosphere.mach = atmosphere.mach[idx]
                real_power[idx] = (
                        thrust[idx]
                        * atmosphere.true_airspeed[idx]
                        / self.propeller_efficiency(thrust[idx], local_atmosphere)
                )
                torque[idx] = real_power[idx] / (rpm_values[idx] * np.pi / 30.0)
                sfc = ICE_sfc(torque[idx], rpm_values[idx]) * mixture_values[idx]
        return sfc, real_power

    def max_thrust(
            self, engine_setting: Union[float, Sequence[float]], atmosphere: Atmosphere,
    ) -> np.ndarray:
        """
        Computation of maximum thrust either due to propeller thrust limit or ICE max power.

        :param engine_setting: Engine settings (climb, cruise,... )
        :param atmosphere: Atmosphere instance at intended altitude (should be <=20km)
        :return: maximum thrust (in N)
        """

        # Calculate maximum propeller thrust @ given altitude and speed
        if isinstance(atmosphere.true_airspeed, float):
            lower_bound = np.interp(atmosphere.true_airspeed, self.speed_SL, self.thrust_limit_SL)
            upper_bound = np.interp(atmosphere.true_airspeed, self.speed_CL, self.thrust_limit_CL)
        else:
            lower_bound = np.interp(
                list(atmosphere.true_airspeed), self.speed_SL, self.thrust_limit_SL
            )
            upper_bound = np.interp(
                list(atmosphere.true_airspeed), self.speed_CL, self.thrust_limit_CL
            )
        altitude = atmosphere.get_altitude(altitude_in_feet=False)
        thrust_max_propeller = (
                lower_bound
                + (upper_bound - lower_bound)
                * np.minimum(altitude, self.design_altitude)
                / self.design_altitude
        )

        # Calculate engine max power @ given RPM & altitude
        rpm_vect, _, pme_limit_vect, _ = self.read_map(self.map_file_path)
        torque_vect = pme_limit_vect * 1e5 * self.volume / (8.0 * np.pi)
        power_max_vect = torque_vect * rpm_vect * (np.pi / 30.0)
        if np.size(engine_setting) == 1:
            rpm_values = np.array(self.rpm_values[int(engine_setting)])
            max_power_SL = np.interp(rpm_values, rpm_vect, power_max_vect)
        else:
            rpm_values = np.array(
                [self.rpm_values[engine_setting[idx]] for idx in range(np.size(engine_setting))]
            )
            max_power_SL = np.interp(list(rpm_values), rpm_vect, power_max_vect)
        sigma = atmosphere.density / Atmosphere(0.0).density
        max_power = max_power_SL * (sigma - (1 - sigma) / 7.55)

        # Found thrust relative to ICE maximum power @ given altitude and speed:
        # calculates first thrust interpolation vector (between min and max of propeller table) and associated
        # efficiency, then calculates power and found thrust (interpolation limits to max propeller thrust)
        thrust_interp = np.linspace(
            np.min(self.thrust_SL) * np.ones(np.size(thrust_max_propeller)),
            thrust_max_propeller,
            10,
        ).transpose()
        if np.size(altitude) == 1:  # Calculate for float
            thrust_max_global = 0.0
            local_atmosphere = Atmosphere(
                altitude * np.ones(np.size(thrust_interp)), altitude_in_feet=False
            )
            local_atmosphere.mach = atmosphere.mach * np.ones(np.size(thrust_interp))
            propeller_efficiency = self.propeller_efficiency(thrust_interp[0], local_atmosphere)
            mechanical_power = thrust_interp[0] * atmosphere.true_airspeed / propeller_efficiency
            if np.min(mechanical_power) > max_power:
                efficiency_relative_error = 1
                propeller_efficiency = propeller_efficiency[0]
                while efficiency_relative_error > 1e-2:
                    thrust_max_global = max_power * propeller_efficiency / atmosphere.true_airspeed
                    propeller_efficiency_new = self.propeller_efficiency(
                        thrust_max_global, atmosphere
                    )
                    efficiency_relative_error = np.abs(
                        (propeller_efficiency_new - propeller_efficiency)
                        / efficiency_relative_error
                    )
                    propeller_efficiency = propeller_efficiency_new
            else:
                thrust_max_global = np.interp(max_power, mechanical_power, thrust_interp[0])
        else:  # Calculate for array
            thrust_max_global = np.zeros(np.size(altitude))
            for idx in range(np.size(altitude)):
                local_atmosphere = Atmosphere(
                    altitude[idx] * np.ones(np.size(thrust_interp[idx])), altitude_in_feet=False
                )
                local_atmosphere.mach = atmosphere.mach[idx] * np.ones(np.size(thrust_interp[idx]))
                propeller_efficiency = self.propeller_efficiency(
                    thrust_interp[idx], local_atmosphere
                )
                mechanical_power = (
                        thrust_interp[idx] * atmosphere.true_airspeed[idx] / propeller_efficiency
                )
                if (
                        np.min(mechanical_power) > max_power[idx]
                ):  # take the lower bound efficiency for calculation
                    efficiency_relative_error = 1
                    local_atmosphere = Atmosphere(altitude[idx], altitude_in_feet=False)
                    local_atmosphere.mach = atmosphere.mach[idx]
                    propeller_efficiency = propeller_efficiency[0]
                    while efficiency_relative_error > 1e-2:
                        thrust_max_global[idx] = (
                                max_power[idx] * propeller_efficiency / atmosphere.true_airspeed[idx]
                        )
                        propeller_efficiency_new = self.propeller_efficiency(
                            thrust_max_global[idx], local_atmosphere
                        )
                        efficiency_relative_error = np.abs(
                            (propeller_efficiency_new - propeller_efficiency)
                            / efficiency_relative_error
                        )
                        propeller_efficiency = propeller_efficiency_new
                else:
                    thrust_max_global[idx] = np.interp(
                        max_power[idx], mechanical_power, thrust_interp[idx]
                    )

        return thrust_max_global

    def compute_weight(self) -> float:
        """
        Computes weight of installed propulsion (engine, nacelle and propeller) depending on maximum power.
        Uses model described in : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and Procedures.
        Butterworth-Heinemann, 2013. Equation (6-44)

        """

        power_sl = self.max_power / 745.7  # conversion to european hp
        uninstalled_weight = (power_sl - 21.55) / 0.5515
        self.engine.mass = uninstalled_weight

        return uninstalled_weight

    def compute_dimensions(self) -> (float, float, float, float):
        """
        Computes propulsion dimensions (engine/nacelle) from maximum power.
        Model from :...

        """

        # Compute engine dimensions
        self.engine.length = self.ref["length"] * (self.max_power / self.ref["max_power"]) ** (
                1 / 3
        )
        self.engine.height = self.ref["height"] * (self.max_power / self.ref["max_power"]) ** (
                1 / 3
        )
        self.engine.width = self.ref["width"] * (self.max_power / self.ref["max_power"]) ** (1 / 3)

        if self.prop_layout == 3.0:
            nacelle_length = 1.15 * self.engine.length
            # Based on the length between nose and firewall for TB20 and SR22
        else:
            nacelle_length = 2.0 * self.engine.length

        # Compute nacelle dimensions
        self.nacelle = Nacelle(
            height=self.engine.height * 1.1, width=self.engine.width * 1.1, length=nacelle_length,
        )
        self.nacelle.wet_area = 2 * (self.nacelle.height + self.nacelle.width) * self.nacelle.length

        return (
            self.nacelle["height"],
            self.nacelle["width"],
            self.nacelle["length"],
            self.nacelle["wet_area"],
        )

    def compute_drag(self, mach, unit_reynolds, wing_mac):
        """
        Compute nacelle drag coefficient cd0.

        """

        # Compute dimensions
        _, _, _, _ = self.compute_dimensions()
        # Local Reynolds:
        reynolds = unit_reynolds * self.nacelle.length
        # Roskam method for wing-nacelle interaction factor (vol 6 page 3.62)
        cf_nac = 0.455 / (
                (1 + 0.144 * mach ** 2) ** 0.65 * (math.log10(reynolds)) ** 2.58
        )  # 100% turbulent
        f = self.nacelle.length / math.sqrt(4 * self.nacelle.height * self.nacelle.width / math.pi)
        ff_nac = 1 + 0.35 / f  # Raymer (seen in Gudmunsson)
        if_nac = 1.2  # Jenkinson (seen in Gudmundsson)
        drag_force = cf_nac * ff_nac * self.nacelle.wet_area * if_nac

        return drag_force


@AddKeyAttributes(ENGINE_LABELS)
class Engine(DynamicAttributeDict):
    """
    Class for storing data for engine.

    An instance is a simple dict, but for convenience, each item can be accessed
    as an attribute (inspired by pandas DataFrames). Hence, one can write::

        >>> engine = Engine(power_SL=10000.)
        >>> engine["power_SL"]
        10000.0
        >>> engine["mass"] = 70000.
        >>> engine.mass
        70000.0
        >>> engine.mass = 50000.
        >>> engine["mass"]
        50000.0

    Note: constructor will forbid usage of unknown keys as keyword argument, but
    other methods will allow them, while not making the matching between dict
    keys and attributes, hence::

        >>> engine["foo"] = 42  # Ok
        >>> bar = engine.foo  # raises exception !!!!
        >>> engine.foo = 50  # allowed by Python
        >>> # But inner dict is not affected:
        >>> engine.foo
        50
        >>> engine["foo"]
        42

    This class is especially useful for generating pandas DataFrame: a pandas
    DataFrame can be generated from a list of dict... or a list of FlightPoint
    instances.

    The set of dictionary keys that are mapped to instance attributes is given by
    the :meth:`get_attribute_keys`.
    """


@AddKeyAttributes(NACELLE_LABELS)
class Nacelle(DynamicAttributeDict):
    """
    Class for storing data for nacelle.

    Similar to :class:`Engine`.
    """
