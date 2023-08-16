from RICE_model.IAM_RICE import RICE
from ptreeopt import PTreeOpt, MPIExecutor
import logging


def main():
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

    # RICE(years_10, regions).run(write_to_excel=False)
    model = RICE(years_10, regions)
    algorithm = PTreeOpt(model.POT_control,
                         # feature_bounds=[[0.8, 2.8], [700, 900], [2005, 2305]],
                         # feature_names=['temp_atm', 'mat', 'year'],
                         # feature_bounds=[[2005, 2305]],
                         # feature_names=['year'],
                         # feature_bounds=[[0.8, 2.8]],
                         # feature_names=['temp_atm'],
                         feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
                         feature_names=['mat', 'net_output', 'year'],
                         discrete_actions=True,
                         # action_names=['miu_2100_sr_low', 'miu_2125_sr_low', 'miu_2150_sr_low',
                         #               'miu_2100_sr_high', 'miu_2125_sr_high', 'miu_2150_sr_high'],
                         # action_names=['miu_2100_sr_low', 'miu_2150_sr_high'],
                         # action_names=['miu_2100_sr_low', 'miu_2125_sr_low', 'miu_2150_sr_low'],
                         action_names=['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05'],
                         mu=20,  # number of parents per generation, 20
                         cx_prob=0.70,  # crossover probability
                         population_size=100,  # 100
                         max_depth=7,
                         multiobj=True
                         )

    logging.basicConfig(level=logging.INFO,
                        format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')

    # With only 1000 function evaluations this will not be very good
    best_solution, best_score, snapshots = algorithm.run(max_nfe=1000,
                                                         log_frequency=100,
                                                         snapshot_frequency=100)
    pass


if __name__ == '__main__':
    main()












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