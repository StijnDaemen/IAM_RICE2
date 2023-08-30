from RICE_model.IAM_RICE import RICE
from ptreeopt import PTreeOpt, MPIExecutor
import logging
from POT.optimization import PolicyTreeOptimizer

import pandas as pd
import numpy as np
import sqlite3
import time
import os

package_directory = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(package_directory)


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

    ### Test 1
    # scenario1 = [2, 10000]
    # scenario2 = [5, 4200]
    # levers = {'mu_target': 2135,
    #           'sr': 0.248,
    #           'irstp': 0.015}
    #
    # RICE(years_10, regions, scenario=scenario1, levers=levers).run(write_to_excel=False, write_to_sqlite=True)
    # RICE(years_10, regions, scenario=scenario2, levers=levers).run(write_to_excel=False, write_to_sqlite=True)

    ### Test 2
    # start_time = time.time()
    # levers = {'mu_target': 2135,
    #           'sr': 0.248,
    #           'irstp': 0.015}
    #
    # SSP_scenario = 5
    # counter = 0
    # for fosslim in range(4000, 5930):  # range(4000, 13650)
    #     if counter >= 1930:
    #         SSP_scenario -= 1
    #         counter = 0
    #     scenario = [SSP_scenario, fosslim]
    #     counter += 1
    #     print(scenario)
    #     RICE(years_10, regions, scenario=scenario, levers=levers).run(write_to_excel=False, write_to_sqlite=True)
    # print("--- %s seconds ---" % (time.time() - start_time))
    # # 1930 in 1091 seconds = 18,183 minutes

    ### Test 3
    # example scenario:
    # scenario = {'SSP_scenario': 5,                                        # 1, 2, 3, 4, 5
    #             'fosslim': 4200,                                          # range(4000, 13650), depending on SSP scenario
    #             'climate_sensitivity_distribution': 'lognormal',          # 'log', 'lognormal', 'Cauchy'
    #             'elasticity_climate_impact': -1,                          # -1, 0, 1
    #             'price_backstop_tech': 1260,                              # [1.260, 1.470, 1.680, 1.890]
    #             'negative_emissions_possible': 'yes'}                     # 'yes' or 'no'

    # levers = {'mu_target': 2135,
    #           'sr': 0.248,
    #           'irstp': 0.015}
    # scenario = {'SSP_scenario': 5,                                        # 1, 2, 3, 4, 5
    #             'fosslim': 4200,                                          # range(4000, 13650), depending on SSP scenario
    #             'climate_sensitivity_distribution': 'lognormal',          # 'log', 'lognormal', 'Cauchy'
    #             'elasticity_climate_impact': -1,                          # -1, 0, 1
    #             'price_backstop_tech': 1.260,                             # [1.260, 1.470, 1.680, 1.890]
    #             'negative_emissions_possible': 'yes'}                     # 'yes' or 'no'
    # RICE(years_10, regions, scenario=scenario, levers=levers).run(write_to_excel=False, write_to_sqlite=True)

    # ### Test 4
    # # scenario = {'SSP_scenario': 5,                                        # 1, 2, 3, 4, 5
    # #             'fosslim': 4200,                                          # range(4000, 13650), depending on SSP scenario
    # #             'climate_sensitivity_distribution': 'lognormal',          # 'log', 'lognormal', 'Cauchy'
    # #             'elasticity_climate_impact': -1,                          # -1, 0, 1
    # #             'price_backstop_tech': 1260,                              # range(1.260, 1.890) [1.260, 1.470, 1.680, 1.890]
    # #             'negative_emissions_possible': 'yes',                     # 'yes' or 'no'
    # #             't2xco2_index': 500}                                      # [0, 999]
    #
    # start_time = time.time()
    # levers = {'mu_target': 2135,
    #           'sr': 0.248,
    #           'irstp': 0.015}
    #
    # scenarios_econ = []
    # SSP_scenario = 5
    # counter = 0
    # for fosslim in range(4000, 13650, 386):  # range(4000, 13650) -- 1930
    #     if counter >= 5:
    #         SSP_scenario -= 1
    #         counter = 0
    #     scenarios_econ.append([SSP_scenario, fosslim])
    #     counter += 1
    # # print(scenarios_econ)
    #
    # scenarios_climate = []
    # climate_sensitivity = ['log', 'lognormal', 'Cauchy']
    # climate_elasticity = [-1, 0, 1]
    # climate_index = [0, 999]
    # for i in climate_sensitivity:
    #     for j in climate_elasticity:
    #         for k in range(climate_index[0], climate_index[1], 200):
    #             scenarios_climate.append([i, j, k])
    # # print(scenarios_climate)
    #
    # scenarios_backstop = []
    # backstop_price = [1.260, 1.470, 1.680, 1.890]
    # backstop_possible = ['yes', 'no']
    # for i in backstop_price:
    #     for j in backstop_possible:
    #         scenarios_backstop.append([i, j])
    # # print(scenarios_backstop)
    #
    # scenarios_list = []
    # for i in scenarios_econ:
    #     for j in scenarios_climate:
    #         for k in scenarios_backstop:
    #             scenarios_list.append([i, k, j])
    # # print(scenarios_list)
    #
    # # Flatten the list of lists into a list
    # for idx in range(len(scenarios_list)):
    #     scenarios_list[idx] = [item for sublist in scenarios_list[idx] for item in sublist]
    #
    # # Turn all scenarios into scenario input for the RICE model
    # for idx, scenario_ in enumerate(scenarios_list):
    #     # print(scenario_)
    #     scenario = {'SSP_scenario': scenario_[0],
    #                 'fosslim': scenario_[1],
    #                 'climate_sensitivity_distribution': scenario_[4],
    #                 'elasticity_climate_impact': scenario_[5],
    #                 't2xco2_index': scenario_[6],
    #                 'price_backstop_tech': scenario_[2],
    #                 'negative_emissions_possible': scenario_[3]}
    #     print(f'scenario {idx} / {len(scenarios_list)}')
    #
    #     RICE(years_10, regions, scenario=scenario, levers=levers).run(write_to_excel=False, write_to_sqlite=True, file_name='test4_8')
    #
    # print(len(scenarios_list))
    # # 694800 different scenarios -> running this many scenarios would (see test 2) take 4.546 days of non-stop running
    # # if you take 100 sized interval for availability of fossil fuels you have 6984 scenarios
    # # Note that the above did not consider the uncertainty of climate_index
    # # current run: 5654 scenarios in 57.79 minutes
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    # #
    # ### View sqlite database
    # df = view_sqlite_database(database='C:/Users/Stijn Daemen/Documents/master thesis TU Delft/code/IAM_RICE2/RICE_model/output_data/Experiment1.db', table_name='test4_8')  # 'RICE_model/output_data/Experiment1.db'
    # print(df.head(50))
    # print(df.info())
    # df.to_excel("F:/Thesis RICE/IAM_RICE2/Nordhaus policy - 5x SSP - 9000 scenarios_8.xlsx")
    # # Run "Nordhaus policy - 5x SSP - 9000 scenarios_5.xlsx" -> Test4_5 was done with a temp overshoot threshold of 1.5. Later it was changed to 2.0!

    # BASIC RUN ----------------------------------------
    # RICE(years_10, regions).run(write_to_excel=True, file_name='Basic RICE - Nordhaus Policy - 2')

    # levers = {'mu_target': 2135,
    #           'sr': 0.248,
    #           'irstp': 0.015}
    # scenario1 = {'SSP_scenario': 1,                                        # 1, 2, 3, 4, 5
    #             'fosslim': 11720,                                          # range(4000, 13650), depending on SSP scenario
    #             'climate_sensitivity_distribution': 'lognormal',          # 'log', 'lognormal', 'Cauchy'
    #             'elasticity_climate_impact': 0,                          # -1, 0, 1
    #             'price_backstop_tech': 1.260,                             # [1.260, 1.470, 1.680, 1.890]
    #             'negative_emissions_possible': 'no'}                     # 'yes' or 'no'
    # RICE(years_10, regions, scenario=scenario1, levers=levers).run(write_to_excel=True, write_to_sqlite=False, file_name='SSP1_Emissions.xlsx')
    #
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
    ### ------------------------------------------------

    ### POT tests -----------------------------------------------------------------------------------------------------

    # input_path = os.path.join(package_directory)
    # # model = RICE(years_10, regions, database_POT=input_path+'/ptreeopt/output_data/POT_Experiments.db', table_name_POT='indicator_groupsize_3_bin_tournament_1')
    # model = RICE(years_10, regions)
    # algorithm = PTreeOpt(model.POT_control,
    #                      # feature_bounds=[[0.8, 2.8], [700, 900], [2005, 2305]],
    #                      # feature_names=['temp_atm', 'mat', 'year'],
    #                      # feature_bounds=[[2005, 2305]],
    #                      # feature_names=['year'],
    #                      # feature_bounds=[[0.8, 2.8]],
    #                      # feature_names=['temp_atm'],
    #                      feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
    #                      feature_names=['mat', 'net_output', 'year'],
    #                      discrete_actions=True,
    #                      # action_names=['miu_2100_sr_low', 'miu_2125_sr_low', 'miu_2150_sr_low',
    #                      #               'miu_2100_sr_high', 'miu_2125_sr_high', 'miu_2150_sr_high'],
    #                      # action_names=['miu_2100_sr_low', 'miu_2150_sr_high'],
    #                      # action_names=['miu_2100_sr_low', 'miu_2125_sr_low', 'miu_2150_sr_low'],
    #                      action_names=['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05'],
    #                      mu=3,  # number of parents per generation, 20
    #                      cx_prob=0.70,  # crossover probability
    #                      population_size=5,  # 100
    #                      max_depth=7,
    #                      multiobj=True
    #                      )
    #
    # logging.basicConfig(level=logging.INFO,
    #                     format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')
    #
    # # With only 1000 function evaluations this will not be very good
    # best_solution, best_score, snapshots = algorithm.run(max_nfe=10,
    #                                                      log_frequency=100,
    #                                                      snapshot_frequency=100)
    # print(best_solution)
    # print(best_score)
    # print(snapshots)

    # ## View POT data ---------------------------------------------------------------
    # df = view_sqlite_database(database=input_path + '/ptreeopt/output_data/POT_Experiments.db',
    #                           table_name='indicator_groupsize_3_bin_tournament_1')
    # df.head()
    # df.info()

    # Test BORG POT ----------------------------------------------------------------------------------------------
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

    np.random.seed(1)
    # Model variables
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
    # Tree variables
    # action_names = ['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05']
    action_names = ['miu', 'sr', 'irstp']
    action_bounds = [[2100, 2250], [0.2, 0.5], [0.01, 0.1]]
    feature_names = ['mat', 'net_output', 'year']
    feature_bounds = [[780, 1300], [55, 2300], [2005, 2305]]
    # Save variables
    database_POT = 'C:/Users/Stijn Daemen/Documents/master thesis TU Delft/code/IAM_RICE2/jupyter notebooks/Tests_Borg.db'
    table_name_POT = 'Test1_couplingborg_not_edited_borg'

    df_optimized_metrics = PolicyTreeOptimizer(model=RICE(years_10, regions, database_POT=database_POT, table_name_POT=table_name_POT),
                        # model=RICE(years_10, regions, database_POT=database_POT, table_name_POT=table_name_POT),
                        action_names=action_names,
                        action_bounds=action_bounds,
                        discrete_actions=False,
                        feature_names=feature_names,
                        feature_bounds=feature_bounds,
                        discrete_features=False,
                        epsilon=0.1,
                        max_nfe=6,
                        max_depth=4,
                        population_size=3
                        ).run()
    # df_optimized_metrics.to_excel('optimized_metrics.xlsx')







## POT coupling ------------------------------------------------------------------------------------------------
# algorithm = PTreeOpt(model.f,
#                      # feature_bounds=[[0.8, 2.8], [700, 900], [2005, 2305]],
#                      # feature_names=['temp_atm', 'mat', 'year'],
#                      # feature_bounds=[[2005, 2305]],
#                      # feature_names=['year'],
#                      # feature_bounds=[[0.8, 2.8]],
#                      # feature_names=['temp_atm'],
#                      feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#                      feature_names=['mat', 'Y', 'year'],
#                      discrete_actions=True,
#                      # action_names=['miu_2100_sr_low', 'miu_2125_sr_low', 'miu_2150_sr_low',
#                      #               'miu_2100_sr_high', 'miu_2125_sr_high', 'miu_2150_sr_high'],
#                      # action_names=['miu_2100_sr_low', 'miu_2150_sr_high'],
#                      # action_names=['miu_2100_sr_low', 'miu_2125_sr_low', 'miu_2150_sr_low'],
#                      action_names=['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05'],
#                      mu=20,  # number of parents per generation, 20
#                      cx_prob=0.70,  # crossover probability
#                      population_size=100,  # 100
#                      max_depth=7,
#                      multiobj=True
#                      )
#
# logging.basicConfig(level=logging.INFO,
#                     format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')
#
# # With only 1000 function evaluations this will not be very good
# best_solution, best_score, snapshots = algorithm.run(max_nfe=1000,
#                                                      log_frequency=100,
#                                                      snapshot_frequency=100)
# import pickle
#
# pickle.dump(snapshots, open('snapshots.pkl', 'wb'))
#
# import matplotlib.pyplot as plt
# from ptreeopt.plotting import *
#
# # snapshots = pickle.load(open('test-results.pkl', 'rb'), encoding='latin1')
# snapshots = pickle.load(open('snapshots.pkl', 'rb'), encoding='latin1')
# P = snapshots['best_P'][-1]
# print(P)
#
# print(snapshots['best_P'])
#
# colors = {'miu_2100': 'cornsilk',
#           'miu_2150': 'cornsilk',
#           'miu_2200': 'indianred',
#           'miu_2125': 'cornsilk',
#           'sr_02': 'lightsteelblue',
#           'sr_03': 'lightsteelblue',
#           'sr_04': 'lightsteelblue',
#           'sr_05': 'indianred'
#           }
#
# # colors = {'miu_2100_sr_low': 'cornsilk',
# #           # 'Hedge_90': 'cornsilk',
# #           # 'Hedge_80': 'indianred',
# #           # 'Hedge_70': 'indianred',
# #           # 'Hedge_60': 'indianred',
# #           # 'Hedge_50': 'indianred',
# #           'miu_2150_sr_high': 'lightsteelblue'}
#
# graphviz_export(P, 'test.svg', colordict=colors)  # creates one SVG
