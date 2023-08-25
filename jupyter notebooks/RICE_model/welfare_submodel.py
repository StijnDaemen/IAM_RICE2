import numpy as np
import pandas as pd
from RICE_model.model_limits import ModelLimits
import os

package_directory = os.path.dirname(os.path.abspath(__file__))

class WelfareSubmodel:
    def __init__(self, years, regions):
        self.years = years
        self.simulation_horizon = len(self.years)
        self.delta_t = self.years[1] - self.years[0]  # Assuming equally spaced intervals between the years
        self.regions = regions
        self.n_regions = len(regions)
        self.limits = ModelLimits()
        input_path = os.path.join(package_directory)
        # self.Alpha_data = pd.read_excel("RICE_model/input_data/RICE_input_data.xlsx", sheet_name="Stijn_MUST_change_input", usecols=range(67, 98), nrows=12).to_numpy()
        self.Alpha_data = pd.read_excel(input_path+"/input_data/RICE_input_data.xlsx",
                                        sheet_name="Stijn_MUST_change_input", usecols=range(67, 98),
                                        nrows=12).to_numpy()
        self.Utility_Input = pd.read_excel(input_path+"/input_data/RICE_input_data.xlsx", sheet_name="Stijn_RICE_input", usecols=range(0, 36), nrows=12)

        # Make sure irstp is set to the same value as in the economic submodel
        # self.irstp = 0.015  # Initial rate of social time preference  (RICE2010 OPT))
        # Elasticity of marginal utility of consumption (1.45) # CHECKED
        self.elasmu = 1.50

        self.multiplutacive_scaling_weights = self.Utility_Input['multiplutacive_scaling_weights for utility calculation']
        self.additative_scaling_weights1 = self.Utility_Input['additative_scaling_weights1']
        self.additative_scaling_weights2 = self.Utility_Input['additative_scaling_weights2']

        self.util_sdr = np.zeros((self.n_regions, self.simulation_horizon))
        self.inst_util = np.zeros((self.n_regions, self.simulation_horizon))
        self.period_util = np.zeros((self.n_regions, self.simulation_horizon))
        self.cum_period_util = np.zeros((self.n_regions, self.simulation_horizon))
        self.inst_util_ww = np.zeros((self.n_regions, self.simulation_horizon))
        self.period_util_ww = np.zeros((self.n_regions, self.simulation_horizon))
        self.reg_cum_util = np.zeros((self.n_regions, self.simulation_horizon))
        self.regional_cum_util = np.zeros(self.simulation_horizon)

        self.region_util = np.zeros((self.n_regions,))
        self.global_damages = np.zeros(self.simulation_horizon)
        self.global_output = np.zeros(self.simulation_horizon)
        self.global_period_util_ww = np.zeros(self.simulation_horizon)
        self.utility = 0

        # Temp -------------------
        self.temp_overshoots = np.zeros(self.simulation_horizon)
        # A lower bound of 0 for temperature overshoots, so if the temp_atm is 1.4 degrees and the threshold is 1.5, the overshoot is equal to 0 instead of -0.1
        self.temp_overshoots_lower_bound = 0
        self.temp_threshold = 2.0  # 1.5 Degrees Celsius, in accordance with the IPCC and COP21 Paris. -> changed to 2.0

    def run_utilitarian(self, t, year, CPC, labour_force, damages, net_output, temp_atm, irstp):
        # irstp: Initial rate of social time preference per year
        self.util_sdr[:, t] = 1 / ((1 + irstp) ** (self.delta_t * (t)))

        # instantaneous welfare without ww
        self.inst_util[:, t] = ((1 / (1 - self.elasmu)) * (CPC[:, t]) ** (1 - self.elasmu) + 1)

        # period utility
        self.period_util[:, t] = self.inst_util[:, t] * labour_force[:, t] * self.util_sdr[:, t]

        # cummulativie period utilty without WW
        self.cum_period_util[:, 0] = self.cum_period_util[:, t - 1] + self.period_util[:, t]

        # Instantaneous utility function with welfare weights
        self.inst_util_ww[:, t] = self.inst_util[:, t] * self.Alpha_data[:, t]

        # period utility with welfare weights
        self.period_util_ww[:, t] = self.inst_util_ww[:, t] * labour_force[:, t] * self.util_sdr[:, t]

        # cummulative utility with ww
        self.reg_cum_util[:, t] = self.reg_cum_util[:, t - 1] + self.period_util_ww[:, t]

        self.regional_cum_util[t] = self.reg_cum_util[:, t].sum()


        # scale utility with weights derived from the excel
        if year == self.years[-1]:
            self.region_util[:] = 10 * self.multiplutacive_scaling_weights * self.reg_cum_util[:, t] + self.additative_scaling_weights1 - self.additative_scaling_weights2

        # calculate worldwide utility
        self.utility = self.region_util.sum()

        # additional per time step aggregated objectives utilitarian case
        self.global_damages[t] = damages[:, t].sum(axis=0)
        self.global_output[t] = net_output[:, t].sum(axis=0)
        self.global_period_util_ww[t] = self.period_util_ww[:, t].sum(axis=0)

        ##########

        self.temp_overshoots[t] = self.temp_threshold - temp_atm[t]
        self.temp_overshoots = np.where(self.temp_overshoots >= self.temp_overshoots_lower_bound, self.temp_overshoots, self.temp_overshoots_lower_bound)
        return


# class WelfareSubmodel:
#     def __init__(self, years, regions):
#         self.years = years
#         self.simulation_horizon = len(self.years)
#         self.delta_t = self.years[1] - self.years[0]  # Assuming equally spaced intervals between the years
#         self.regions = regions
#         self.n_regions = len(regions)
#         self.limits = ModelLimits()
#         self.Alpha_data = pd.read_excel("RICE_input_data.xlsx", sheet_name="Stijn_MUST_change_input", usecols=range(67, 98), nrows=11).to_numpy()
#         # print(self.Alpha_data)
#
#         # Make sure irstp is set to the same value as in the economic submodel
#         self.irstp = 0.015  # Initial rate of social time preference  (RICE2010 OPT))
#         # Elasticity of marginal utility of consumption (1.45) # CHECKED
#         self.elasmu = 0.45  # 1.50
#
#         self.R = np.zeros((self.n_regions, self.simulation_horizon))
#         self.U = np.zeros((self.n_regions, self.simulation_horizon))
#         self.W_t = np.zeros((self.n_regions, self.simulation_horizon))
#         self.W = np.zeros((self.n_regions, ))
#
#     def run_utilitarian(self, t, year, CPC, labour_force):
#         # irstp: Initial rate of social time preference per year
#         def calculate_discount_factor(t):
#             self.R[:, t] = (1 + self.irstp)**(-t)
#             return self.R
#
#         def calculate_utility(t, CPC, labour_force):
#             self.U[:, t] = labour_force[:, t] * (CPC[:, t]**(1-self.elasmu)/(1-self.elasmu))
#             return self.U
#
#         def calculate_welfare(t):
#             self.W_t[:, t] = self.U[:, t] * self.R[:, t]
#             return self.W_t
#
#         def welfare_function():
#             self.W[:] = self.W_t[:].sum(axis=1)
#             return self.W
#
#         self.R = calculate_discount_factor(t)
#         self.U = calculate_utility(t, CPC, labour_force)
#         self.W_t = calculate_welfare(t)
#         if year == self.years[-1]:
#             self.W = welfare_function()
#         return