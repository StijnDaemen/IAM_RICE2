# Import the generative model by SD
from RICE_model.IAM_RICE import RICE
# Import the policy tree optimizer by Herman
from POT.ptreeopt import PTreeOpt
import logging
# Import the policy tree optimizer with borg
from POT.borg_optimization import PolicyTreeOptimizer
# Import the ema workbench by professor Kwakkel
from ema_workbench import RealParameter, ScalarOutcome, Constant, Model, IntegerParameter
from ema_workbench import SequentialEvaluator, ema_logging
from ema_workbench import save_results
# Import the homemade POT optimizer
from POT.homemade_optimization import Cluster

import pandas as pd
import numpy as np
import sqlite3
import time
import os

package_directory = os.path.dirname(os.path.abspath(__file__))
path_to_dir = os.path.join(package_directory)


def view_sqlite_database(database, table_name):
    # df = pd.DataFrame()
    conn = sqlite3.connect(database)
    # c = conn.cursor()

    # c.execute(f"""SELECT count(*) FROM sqlite_master WHERE type='table' AND name={table_name}""")
    df = pd.read_sql_query(f'''SELECT * FROM {table_name}''', conn)

    conn.commit()
    conn.close()

    return df


if __name__ == '__main__':
    years_10 = []
    for i in range(2005, 2315, 10):
        years_10.append(i)

    regions = [
        "US",
        "OECD-Europe",
        "Japan",
        "Russia",
        "Non-Russia Eurasia",
        "China",
        "India",
        "Middle East",
        "Africa",
        "Latin America",
        "OHI",
        "Other non-OECD Asia",
    ]

    save_location = path_to_dir + '/output_data'

    # BASIC RUN ----------------------------------------
    def basic_run_RICE(years_10, regions, save_location):
        # RICE(years_10, regions).run(write_to_excel=False, file_name='Basic RICE - Nordhaus Policy - 2')
        title_of_run = ''
        start = time.time()
        RICE(years_10, regions, save_location=save_location, file_name=title_of_run).run(write_to_excel=False, write_to_sqlite=False)
        end = time.time()
        return print(f'Total elapsed time: {(end - start)/60} minutes.')

    def basic_run_RICE_with_scenarios(years_10, regions, save_location):
        title_of_run = ''
        start = time.time()
        levers = {'mu_target': 2135,
                  'sr': 0.248,
                  'irstp': 0.015}
        scenario1 = {'SSP_scenario': 1,                                        # 1, 2, 3, 4, 5
                    'fosslim': 11720,                                          # range(4000, 13650), depending on SSP scenario
                    'climate_sensitivity_distribution': 'lognormal',          # 'log', 'lognormal', 'Cauchy'
                    'elasticity_climate_impact': 0,                          # -1, 0, 1
                    'price_backstop_tech': 1.260,                             # [1.260, 1.470, 1.680, 1.890]
                    'negative_emissions_possible': 'no'}                     # 'yes' or 'no'
        RICE(years_10, regions, scenario=scenario1, levers=levers, save_location=save_location, file_name=title_of_run).run(write_to_excel=False, write_to_sqlite=False)

        # scenario2 = {'SSP_scenario': 2,  # 1, 2, 3, 4, 5
        #             'fosslim': 9790,  # range(4000, 13650), depending on SSP scenario
        #             'climate_sensitivity_distribution': 'lognormal',  # 'log', 'lognormal', 'Cauchy'
        #             'elasticity_climate_impact': 0,  # -1, 0, 1
        #             'price_backstop_tech': 1.260,  # [1.260, 1.470, 1.680, 1.890]
        #             'negative_emissions_possible': 'no'}  # 'yes' or 'no'
        # RICE(years_10, regions, scenario=scenario2, levers=levers).run(write_to_excel=True, write_to_sqlite=False,
        #                                                               file_name='SSP2_Emissions.xlsx')
        #
        # scenario3 = {'SSP_scenario': 3,  # 1, 2, 3, 4, 5
        #             'fosslim': 7860,  # range(4000, 13650), depending on SSP scenario
        #             'climate_sensitivity_distribution': 'lognormal',  # 'log', 'lognormal', 'Cauchy'
        #             'elasticity_climate_impact': 0,  # -1, 0, 1
        #             'price_backstop_tech': 1.260,  # [1.260, 1.470, 1.680, 1.890]
        #             'negative_emissions_possible': 'no'}  # 'yes' or 'no'
        # RICE(years_10, regions, scenario=scenario3, levers=levers).run(write_to_excel=True, write_to_sqlite=False,
        #                                                               file_name='SSP3_Emissions.xlsx')
        #
        # scenario4 = {'SSP_scenario': 4,  # 1, 2, 3, 4, 5
        #             'fosslim': 5930,  # range(4000, 13650), depending on SSP scenario
        #             'climate_sensitivity_distribution': 'lognormal',  # 'log', 'lognormal', 'Cauchy'
        #             'elasticity_climate_impact': 0,  # -1, 0, 1
        #             'price_backstop_tech': 1.260,  # [1.260, 1.470, 1.680, 1.890]
        #             'negative_emissions_possible': 'no'}  # 'yes' or 'no'
        # RICE(years_10, regions, scenario=scenario4, levers=levers).run(write_to_excel=True, write_to_sqlite=False,
        #                                                               file_name='SSP4_Emissions.xlsx')
        #
        # scenario5 = {'SSP_scenario': 5,  # 1, 2, 3, 4, 5
        #             'fosslim': 4000,  # range(4000, 13650), depending on SSP scenario
        #             'climate_sensitivity_distribution': 'lognormal',  # 'log', 'lognormal', 'Cauchy'
        #             'elasticity_climate_impact': 0,  # -1, 0, 1
        #             'price_backstop_tech': 1.260,  # [1.260, 1.470, 1.680, 1.890]
        #             'negative_emissions_possible': 'no'}  # 'yes' or 'no'
        # RICE(years_10, regions, scenario=scenario5, levers=levers).run(write_to_excel=True, write_to_sqlite=False,
        #                                                               file_name='SSP5_Emissions.xlsx')
        end = time.time()
        return print(f'Total elapsed time: {(end - start)/60} minutes.')

    # CONNECT TO EMA -----------------------------------
    def connect_to_EMA(years_10, regions, save_location):
        # # Now all parameters must be given in the experiments.py file, this function simply calls it. Must fix later.
        # ConnectToEMA()
        title_of_run = ''
        start = time.time()
        def RICE_wrapper_ema_workbench(years_10,
                                       regions,
                                       SSP_scenario=5,
                                       fosslim=4000,
                                       climate_sensitivity_distribution='lognormal',
                                       elasticity_climate_impact=0,
                                       price_backstop_tech=1.260,
                                       negative_emissions_possible='no',  # 1 = 'no'; 1.2 = 'yes'
                                       t2xco2_index=0,
                                       mu_target=2135,
                                       sr=0.248,
                                       irstp=0.015):
            '''
            This wrapper connects the RICE model to the ema workbench. The ema workbench requires that the uncertainties and
            levers of the model are given as direct inputs to the model. The RICE model instead accepts dictionaries of the
            uncertainties and levers. This wrapper takes the input uncertainties and levers and puts them in a dictionary
            that serves as the input to RICE.
            '''
            # years_10 = []
            # for i in range(2005, 2315, 10):
            #     years_10.append(i)
            #
            # regions = [
            #     "US",
            #     "OECD-Europe",
            #     "Japan",
            #     "Russia",
            #     "Non-Russia Eurasia",
            #     "China",
            #     "India",
            #     "Middle East",
            #     "Africa",
            #     "Latin America",
            #     "OHI",
            #     "Other non-OECD Asia",
            # ]
            if negative_emissions_possible == 0:
                negative_emissions_possible = 'no'
            elif negative_emissions_possible == 1:
                negative_emissions_possible = 'yes'
            else:
                print('incorrect input for negative_emissions_possible variable')

            if climate_sensitivity_distribution == 0:
                climate_sensitivity_distribution = 'log'
            elif climate_sensitivity_distribution == 1:
                climate_sensitivity_distribution = 'lognormal'
            elif climate_sensitivity_distribution == 2:
                climate_sensitivity_distribution = 'Cauchy'
            else:
                print('incorrect input for climate_sensitivity_distribution variable')

            scenario = {'SSP_scenario': SSP_scenario,
                        'fosslim': fosslim,
                        'climate_sensitivity_distribution': climate_sensitivity_distribution,
                        'elasticity_climate_impact': elasticity_climate_impact,
                        'price_backstop_tech': price_backstop_tech,
                        'negative_emissions_possible': negative_emissions_possible,
                        't2xco2_index': t2xco2_index}
            levers = {'mu_target': mu_target,
                      'sr': sr,
                      'irstp': irstp}
            utilitarian_objective_function_value1, utilitarian_objective_function_value2, utilitarian_objective_function_value3 = RICE(
                years_10, regions, scenario=scenario, levers=levers).ema_workbench_control()
            return utilitarian_objective_function_value1, utilitarian_objective_function_value2, utilitarian_objective_function_value3

        # Set up RICE model -----------------------------------------------------------------------------------------------
        model = Model("WrapperRICE", function=RICE_wrapper_ema_workbench)

        # specify model constants
        model.constants = [
            Constant("years_10", years_10),
            Constant("regions", regions)
        ]

        # specify uncertainties
        model.uncertainties = [
            IntegerParameter("SSP_scenario", 1, 5),
            RealParameter("fosslim", 4000, 13650),
            IntegerParameter("climate_sensitivity_distribution", 0, 2),
            IntegerParameter("elasticity_climate_impact", -1, 1),
            RealParameter("price_backstop_tech", 1.260, 1.890),
            IntegerParameter("negative_emissions_possible", 0, 1),
            IntegerParameter("t2xco2_index", 0, 999),
        ]

        # set levers
        model.levers = [
            IntegerParameter("mu_target", 2065, 2305),
            RealParameter("sr", 0.1, 0.5),
            RealParameter("irstp", 0.001, 0.015),
        ]

        # specify outcomes
        model.outcomes = [
            ScalarOutcome("utilitarian_objective_function_value1", ScalarOutcome.MINIMIZE),
            ScalarOutcome("utilitarian_objective_function_value2", ScalarOutcome.MINIMIZE),
            ScalarOutcome("utilitarian_objective_function_value3", ScalarOutcome.MINIMIZE),
        ]

        # Set up experiments ---------------------------------------------------------------------------------------------
        ema_logging.log_to_stderr(ema_logging.INFO)

        with SequentialEvaluator(model) as evaluator:
            results = evaluator.perform_experiments(scenarios=10, policies=5)

        # Save results ----------------------------------------------------------------------------------------------------
        save_results(results, f'{save_location}/{title_of_run}.tar.gz')

        end = time.time()
        return print(f'Total elapsed time: {(end - start)/60} minutes.')

    # POLICY TREE OPTIMIZATION -------------------------
    def optimization_RICE_POT_Herman(years_10, regions, save_location):
        title_of_run = ''
        start = time.time()
        # input_path = os.path.join(package_directory)
        # model = RICE(years_10, regions, database_POT=input_path+'/ptreeopt/output_data/POT_Experiments.db', table_name_POT='indicator_groupsize_3_bin_tournament_1')
        model = RICE(years_10, regions, save_location=save_location, file_name=title_of_run)
        algorithm = PTreeOpt(model.POT_control_Herman,
                             # feature_bounds=[[0.8, 2.8], [700, 900], [2005, 2305]],
                             # feature_names=['temp_atm', 'mat', 'year'],
                             # feature_bounds=[[2005, 2305]],
                             # feature_names=['year'],
                             # feature_bounds=[[0.8, 2.8]],
                             # feature_names=['temp_atm'],
                             feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
                             feature_names=['mat', 'net_output', 'year'],
                             discrete_actions=False,
                             # action_names=['miu_2100_sr_low', 'miu_2125_sr_low', 'miu_2150_sr_low',
                             #               'miu_2100_sr_high', 'miu_2125_sr_high', 'miu_2150_sr_high'],
                             # action_names=['miu_2100_sr_low', 'miu_2150_sr_high'],
                             # action_names=['miu_2100_sr_low', 'miu_2125_sr_low', 'miu_2150_sr_low'],
                             # action_names=['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05'],
                             action_names=['miu', 'sr', 'irstp'],
                             action_bounds=[[2065, 2305], [0.1, 0.5], [0.001, 0.015]],
                             mu=20,  # number of parents per generation, 20
                             cx_prob=0.70,  # crossover probability
                             population_size=100,  # 100
                             max_depth=4,
                             multiobj=True
                             )

        logging.basicConfig(level=logging.INFO,
                            format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')

        # With only 1000 function evaluations this will not be very good
        best_solution, best_score, snapshots = algorithm.run(max_nfe=10000,
                                                             log_frequency=100,
                                                             snapshot_frequency=100)
        print(best_solution)
        print(best_score)
        print(snapshots)

        ## View POT data ---------------------------------------------------------------
        # df = view_sqlite_database(database=input_path + '/ptreeopt/output_data/POT_Experiments.db',
        #                           table_name='indicator_groupsize_3_bin_tournament_1')
        df = view_sqlite_database(database=save_location + '/Experiments.db', table_name=title_of_run)
        print(df.head())
        print(df.info())
        df.to_excel(f'{save_location}/{title_of_run}.xlsx')

        end = time.time()
        return print(f'Total elapsed time: {(end - start)/60} minutes.')

    def optimization_RICE_POT_Borg(years_10, regions, save_location):
        # from POT.optimization import PolicyTreeOptimizer
        #
        # model = RICE(years_10, regions)
        # feature_bounds = [[780, 1300], [55, 2300], [2005, 2305]]
        # feature_names = ['mat', 'net_output', 'year']
        # action_names = ['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05']
        # PolicyTreeOptimizer(model.POT_control, feature_bounds=feature_bounds,
        #                     feature_names=feature_names,
        #                     action_names=action_names,
        #                     discrete_actions=True,
        #                     population_size=4,
        #                     mu=2).run(max_nfe=4)

        # np.random.seed(1)

        title_of_run = ''
        start = time.time()
        # Model variables

        # Tree variables
        # action_names = ['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05']
        action_names = ['miu', 'sr', 'irstp']
        action_bounds = [[2065, 2305], [0.1, 0.5], [0.001, 0.015]]
        feature_names = ['mat', 'net_output', 'year']
        feature_bounds = [[780, 1300], [55, 2300], [2005, 2305]]
        # Save variables
        # database_POT = 'C:/Users/Stijn Daemen/Documents/master thesis TU Delft/code/IAM_RICE2/jupyter notebooks/Tests_Borg.db'
        # table_name_POT = 'Test3_couplingborg_not_edited_borg'

        df_optimized_metrics = PolicyTreeOptimizer(model=RICE(years_10, regions, save_location=save_location, file_name=title_of_run),
                            # model=RICE(years_10, regions, database_POT=database_POT, table_name_POT=table_name_POT),
                            action_names=action_names,
                            action_bounds=action_bounds,
                            discrete_actions=False,
                            feature_names=feature_names,
                            feature_bounds=feature_bounds,
                            discrete_features=False,
                            epsilon=0.1,
                            max_nfe=10000,
                            max_depth=4,
                            population_size=100
                            ).run()
        df_optimized_metrics.to_excel(f'{save_location}/{title_of_run}.xlsx')
        end = time.time()
        return print(f'Total elapsed time: {(end - start)/60} minutes.')

    def optimization_RICE_POT_Homemade(years_10, regions, save_location):
        title_of_run = ''
        start = time.time()
        master_rng = np.random.default_rng(42)  # Master RNG
        run = Cluster(20, 80, master_rng=master_rng,
                years_10=years_10,
                regions=regions,
                metrics=['period_utility', 'utility', 'temp_overshoots'],
                # Tree variables
                action_names=['miu', 'sr', 'irstp'],
                action_bounds=[[2100, 2250], [0.2, 0.5], [0.01, 0.1]],
                feature_names=['mat', 'net_output', 'year'],
                feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
                max_depth=4,
                discrete_actions=False,
                discrete_features=False,
                # Optimization variables
                mutation_prob=0.5,
                max_nfe=30000,
                epsilons=np.array([0.05, 0.05]),
                ).run()

        with pd.ExcelWriter(f'{save_location}/{title_of_run}.xlsx', engine='xlsxwriter') as writer:
            run[0].to_excel(writer, sheet_name='graveyard')
            run[1].to_excel(writer, sheet_name='VIPs')
            run[2].to_excel(writer, sheet_name='pareto front')
            run[3].to_excel(writer, sheet_name='convergence')

        end = time.time()
        return print(f'Total elapsed time: {(end - start) / 60} minutes.')

    # basic_run_RICE(years_10, regions, save_location)
    # optimization_RICE_POT_Borg(years_10, regions, save_location)
