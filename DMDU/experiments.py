from RICE_model import IAM_RICE
from ema_workbench import RealParameter, ScalarOutcome, Constant, Model, IntegerParameter
from ema_workbench import MultiprocessingEvaluator, ema_logging, perform_experiments
from ema_workbench import SequentialEvaluator, ema_logging, perform_experiments
from ema_workbench import save_results

import pandas as pd


def main():
    # scenario5 = {'SSP_scenario': 5,  # 1, 2, 3, 4, 5
    #             'fosslim': 4000,  # range(4000, 13650), depending on SSP scenario
    #             'climate_sensitivity_distribution': 'lognormal',  # 'log', 'lognormal', 'Cauchy'
    #             'elasticity_climate_impact': 0,  # -1, 0, 1
    #             'price_backstop_tech': 1.260,  # [1.260, 1.470, 1.680, 1.890]
    #             'negative_emissions_possible': 'no'}  # 'yes' or 'no'
    # levers = {'mu_target': 2135,
    #           'sr': 0.248,
    #           'irstp': 0.015}

    def RICE_wrapper_ema_workbench(SSP_scenario=5,
                                   fosslim=4000,
                                   climate_sensitivity_distribution='lognormal',
                                   elasticity_climate_impact=0,
                                   price_backstop_tech=1.260,
                                   negative_emissions_possible=1, # 1 = 'no'; 1.2 = 'yes'
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
            print('incorrect input for negative_emissions_possible variable')

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
        RICE_wrapper = IAM_RICE.RICE(years_10, regions, scenario=scenario, levers=levers)
        return RICE_wrapper

    # Set up RICE model -----------------------------------------------------------------------------------------------

    # model = RICE_wrapper_ema_workbench()
    model = Model("RICE", function=IAM_RICE.RICE_for_DMDU)

    # specify uncertainties
    model.uncertainties = [
        IntegerParameter("SSP_scenario", 1, 5),
        RealParameter("fosslim", 4000, 13650),
        IntegerParameter("climate_sensitivity_distribution", 0, 2),
        IntegerParameter("elasticity_climate_impact", -1, 1),
        RealParameter("price_backstop_tech", 1.260, 1.890),
        IntegerParameter("negative_emissions_possible", 0, 1),
        RealParameter("t2xco2_index", 0, 999),
    ]

    # set levers
    model.levers = [
        IntegerParameter("mu_target", 2065, 2305),
        RealParameter("sr", 0.1, 0.1),
        RealParameter("irstp", 0.001, 0.015),
    ]

    # specify outcomes
    model.outcomes = [
        ScalarOutcome("utilitarian_objective_function_value1"),
        ScalarOutcome("utilitarian_objective_function_value2"),
        ScalarOutcome("utilitarian_objective_function_value3"),
    ]

    # override some of the defaults of the model
    model.constants = [
        Constant("t_steps", 10),
    ]

    # Set up experiments ---------------------------------------------------------------------------------------------
    ema_logging.log_to_stderr(ema_logging.INFO)

    # The n_processes=-1 ensures that all cores except 1 are used, which is kept free to keep using the computer
    with MultiprocessingEvaluator(model, n_processes=-1) as evaluator:
    # with SequentialEvaluator(model) as evaluator:
        # Run 1000 scenarios for 5 policies
        # experiments, outcomes = evaluator.perform_experiments(scenarios=1000, policies=5)
        results = evaluator.perform_experiments(scenarios=1000, policies=5)

    # experiments, outcomes = results
    # print(experiments.shape)
    # print(list(outcomes.keys()))
    #
    # data = pd.DataFrame(outcomes)
    # data["policy"] = experiments["policy"]

    # Save results ----------------------------------------------------------------------------------------------------
    save_results(results, 'output_data/1000 scenarios 5 policies.tar.gz')
    pass


if __name__ == '__main__':
    main()
