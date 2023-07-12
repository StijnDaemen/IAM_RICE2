import pandas as pd
import numpy as np

from RICE_model import *


class RICE:
    def __init__(self, years, regions):
        self.years = years
        self.regions = regions

        self.economic_submodel = RICE_economic_submodel.EconomicSubmodel(self.years, self.regions)
        self.carbon_submodel = RICE_carboncycle_submodel.CarbonSubmodel(self.years)
        self.climate_submodel = RICE_climate_submodel.ClimateSubmodel(self.years, self.regions)
        self.welfare_submodel = welfare_submodel.WelfareSubmodel(self.years, self.regions)

    def run(self, write_to_excel=False):
        t = 0
        for year in self.years:
            self.economic_submodel.run_gross(t, year, mu_target=2135, sr=0.248)
            self.carbon_submodel.run(t, year, E=self.economic_submodel.E)
            self.climate_submodel.run(t, year, forc=self.carbon_submodel.forc, gross_output=self.economic_submodel.gross_output)
            self.economic_submodel.run_net(t, year, temp_atm=self.climate_submodel.temp_atm, SLRDAMAGES=self.climate_submodel.SLRDAMAGES)
            self.welfare_submodel.run_utilitarian(t, year, CPC=self.economic_submodel.CPC, labour_force=self.economic_submodel.labour_force, damages=self.economic_submodel.damages, net_output=self.economic_submodel.net_output)
            t += 1
        if write_to_excel:
            self.write_to_excel(collection='executive variables', file_name='record_executive_variables_for_verification_w_pyRICE2022_4')
        # print(self.welfare_submodel.global_period_util_ww)

    def write_to_excel(self, collection='all variables', file_name='record_all_dynamic_variables_test'):
        def collect_all_variables():
            # Collect all class variables from the different submodels
            model_variables = vars(self.economic_submodel)
            model_variables.update(vars(self.carbon_submodel))
            model_variables.update(vars(self.climate_submodel))
            model_variables.update(vars(self.welfare_submodel))
            model_variables_names = list(model_variables.keys())

            # Create a dictionary with all of the region variables (so the matrix variables/ ndarray) expanded into unique keys holding an array
            model_variables_region = []
            for name in model_variables_names:
                if type(model_variables[name]) == np.ndarray:
                    if model_variables[name].shape == (len(self.regions), len(self.years)):
                        model_variables_region.append(name)

            sub_dict = {}
            for name in model_variables_region:
                for index, key in enumerate(self.regions):
                    sub_dict[f'{name}_{key}'] = model_variables[name][index]

            model_variables_general = []
            for name in model_variables_names:
                if type(model_variables[name]) == np.ndarray:
                    if model_variables[name].shape == (len(self.years),):
                        model_variables_general.append(name)

            model_variables_dynamic = {}
            for key in self.regions:
                model_variables_dynamic[key] = {}
                for name in model_variables_region:
                    model_variables_dynamic[key][name] = sub_dict[f"{name}_{key}"]
                for name in model_variables_general:
                    model_variables_dynamic[key][name] = model_variables[name]
            return model_variables_dynamic

        def collect_executive_variables():
            executive_variables_dict = {'mu': self.economic_submodel.mu,
                                        'S': self.economic_submodel.S,
                                        'E': self.economic_submodel.E,
                                        'damages': self.economic_submodel.damages,
                                        'abatement_cost': self.economic_submodel.abatement_cost,
                                        'SLRDAMAGES': self.climate_submodel.SLRDAMAGES,
                                        'gross_output': self.economic_submodel.gross_output,
                                        'net_output': self.economic_submodel.net_output,
                                        'I': self.economic_submodel.I,
                                        'CPC': self.economic_submodel.CPC,
                                        'forc': self.carbon_submodel.forc,
                                        'temp_atm': self.climate_submodel.temp_atm,
                                        'global_damages': self.welfare_submodel.global_damages,
                                        'global_output': self.welfare_submodel.global_output,
                                        'global_period_util_ww': self.welfare_submodel.global_period_util_ww} #  ,
                                        # 'mat': self.carbon_submodel.mat,
                                        # 'forcoth': self.carbon_submodel.forcoth,
                                        # 'E_worldwide_per_year': self.carbon_submodel.E_worldwide_per_year,
                                        # 'labour_force': self.economic_submodel.labour_force,
                                        # 'total_factor_productivity': self.economic_submodel.total_factor_productivity,
                                        # 'capital_stock': self.economic_submodel.capital_stock,
                                        # 'sigma_ratio': self.economic_submodel.sigma_ratio,
                                        # 'Eind': self.economic_submodel.Eind,
                                        # 'sigma_gr': self.economic_submodel.sigma_gr,
                                        # 'damage_frac': self.economic_submodel.damage_fraction}

            exec_var_dict = {}
            for idx, region in enumerate(self.regions):
                exec_var_dict[region] = {}
                for key, item in executive_variables_dict.items():
                    if item.shape == (len(self.regions), len(self.years)):
                        exec_var_dict[region][key] = item[idx]
                    else:
                        exec_var_dict[region][key] = item[:]
            return exec_var_dict

        model_variables_to_excel = {}
        if collection == 'executive variables':
            model_variables_to_excel = collect_executive_variables()
        elif collection == 'all variables':
            model_variables_to_excel = collect_all_variables()

        # Write dictionaries to an excel file
        writer = pd.ExcelWriter(f'output_data/{file_name}.xlsx')
        for region_key in model_variables_to_excel:
            df = pd.DataFrame.from_dict(model_variables_to_excel[region_key])
            df.index = self.years
            df.to_excel(writer, sheet_name=region_key)
        writer.close()
        return

    def POT_control(self, P):
        t = 0
        for year in self.years:
            # Determine policy based on indicator variables
            policy, rules = P.evaluate([self.carbon_submodel.mat[t], self.economic_submodel.net_output[:, t].sum(axis=0), year])

            # Initialize levers
            mu_target = 2135
            sr = 0.248
            # Match policy to RICE variable lever
            if policy == 'miu_2100':
                mu_target = 2100
            if policy == 'miu_2150':
                mu_target = 2150
            if policy == 'miu_2200':
                mu_target = 2200
            if policy == 'miu_2125':
                mu_target = 2125
            if policy == 'sr_02':
                sr = 0.2
            if policy == 'sr_03':
                sr = 0.3
            if policy == 'sr_04':
                sr = 0.4
            if policy == 'sr_05':
                sr = 0.5

            # Run one timestep of RICE
            self.economic_submodel.run_gross(t, year, mu_target=mu_target, sr=sr)
            self.carbon_submodel.run(t, year, E=self.economic_submodel.E)
            self.climate_submodel.run(t, year, forc=self.carbon_submodel.forc,
                                      gross_output=self.economic_submodel.gross_output)
            self.economic_submodel.run_net(t, year, temp_atm=self.climate_submodel.temp_atm,
                                           SLRDAMAGES=self.climate_submodel.SLRDAMAGES)
            self.welfare_submodel.run_utilitarian(t, year, CPC=self.economic_submodel.CPC,
                                                  labour_force=self.economic_submodel.labour_force,
                                                  damages=self.economic_submodel.damages,
                                                  net_output=self.economic_submodel.net_output,
                                                  temp_atm=self.climate_submodel.temp_atm)
            t += 1

        # objective_function_value assumes minimization
        utilitarian_objective_function_value1 = -self.welfare_submodel.global_period_util_ww.sum()
        utilitarian_objective_function_value2 = self.welfare_submodel.temp_overshoots.sum()
        return utilitarian_objective_function_value1, utilitarian_objective_function_value2  # objective_function_value
