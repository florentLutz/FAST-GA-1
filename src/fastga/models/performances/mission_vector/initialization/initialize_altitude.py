"""FAST - Copyright (c) 2021 ONERA ISAE."""

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

import numpy as np
import openmdao.api as om

from fastga.models.performances.mission.mission_components import (
    POINTS_NB_CLIMB,
    POINTS_NB_CRUISE,
    POINTS_NB_DESCENT,
)


class InitializeAltitude(om.ExplicitComponent):
    """Computes the fuel consumed at each time step."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

        self.add_output("altitude", shape=n, units="m")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]

        altitude_climb = np.linspace(0, cruise_altitude, POINTS_NB_CLIMB)[:, 0]
        altitude_cruise = np.full(POINTS_NB_CRUISE, cruise_altitude)
        altitude_descent = np.linspace(cruise_altitude, 0.0, POINTS_NB_DESCENT)[:, 0]

        altitude = np.concatenate((altitude_climb, altitude_cruise, altitude_descent))

        outputs["altitude"] = altitude
