'''
Run multiple simulations of the game and store the results in a database.

Author: Jaime Ruiz Serra
Date:   2024-09-23

Extended by: Hanchen Wang
Date: 2025-11
'''

import argparse
import logging
import numpy as np
import torch.multiprocessing
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

import sys

sys.path.append('../')

from games import *
import utils.simulation
from generator2 import *

import os

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num-repeats', type=int, default=4)
    argparser.add_argument('--db-path', type=str, default='dummy_c.db')

    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Game configuration ----------------------------------------------------------
    GAMES = {
        "2x2": load_csv_to_games(file_path="../game/GAMES_2_2.csv", game_form="2x2"),
        "2x3": load_csv_to_games(file_path="../game/GAMES_2_3.csv", game_form="2x3"),
        "3x2": load_csv_to_games(file_path="../game/GAMES_3_2.csv", game_form="3x2"),
    }

    GENERATORS = [
        generate_dynamic_transitions(dict_games=GAMES,
                                     game_form="2x2",
                                     game_ids=["1_2_2_Chicken", "1_1_1_Prisoners", "1_2_2_Chicken"],
                                     game_to_change=[],
                                     num_steps=[400, 200, 400],
                                     num_changes=0),
        # generate_dynamic_transitions(dict_games=GAMES,
        #                              game_form="2x2",
        #                              game_ids=["1_2_2_Chicken", "3_6_6_Harmony", "1_2_2_Chicken"],
        #                              game_to_change=[],
        #                              num_steps=[400, 200, 400],
        #                              num_changes=0),
        generate_dynamic_transitions(dict_games=GAMES,
                                     game_form="2x2",
                                     game_ids=["3_2_2_StagHunt", "1_1_1_Prisoners", "3_2_2_StagHunt"],
                                     game_to_change=[],
                                     num_steps=[400, 200, 400],
                                     num_changes=0),
        # generate_dynamic_transitions(dict_games=GAMES,
        #                              game_form="2x2",
        #                              game_ids=["3_2_2_StagHunt", "3_6_6_Harmony", "3_2_2_StagHunt"],
        #                              game_to_change=[],
        #                              num_steps=[400, 200, 400],
        #                              num_changes=0),
        #
        # generate_dynamic_transitions(dict_games=GAMES,
        #                              game_form="2x2",
        #                              game_ids=[
        #                                  "1_2_2_Chicken",
        #                                  "1_1_1_Prisoners",
        #                                  "3_2_2_StagHunt",
        #                                  "3_6_6_Harmony",
        #                                  "1_2_2_Chicken"
        #                              ],
        #                              game_to_change=[],
        #                              num_steps=[500, 250, 500, 250, 500],
        #                              num_changes=0
        #                              )
    ]

    META_GAME_TRANSITIONS = [g[0] for g in GENERATORS]

    META_GAME_STRATEGIES = [g[1] for g in GENERATORS]

    # Run simulations --------------------------------------------------------------
    for idx, game_transitions in enumerate(META_GAME_TRANSITIONS):
        # Agent configuration ---------------------------------------------------------
        num_players = game_transitions[0][1].ndim
        nash_strategy = META_GAME_STRATEGIES[idx]

        META_AGENT_KWARGS = [
            [
                dict(
                    beta_1=30,  # Default Rationality
                    # beta_1=5,                 # Encourages exploration over sharp action preferences
                    # interoception=True,
                    A_prior=99,
                    A_learning=False,
                    B_prior=0,
                    B_learning=True,
                    B_BMR="epsilon",  # Bayesian Model Reduction. One of ['epsilon', 'softmax', None]
                    B_candidates="Full TFT Grim Pavlov",  # "Full TFT Grim Pavlov"
                    B_learning_rate=1,  # Update heavily reflect observed data
                    alpha_r=0.5,
                    gamma_r=1.0,
                    compute_novelty=True,
                    # D_prior=[torch.tensor([0.4, 0.6]), torch.tensor([0.6, 0.4])],  # Prior beliefs about hidden states
                    # E_prior=torch.tensor([0.3, 0.7])  # Behaviour prior more likely to defect
                ),
                dict(
                    strategy="TFT",  # "TFT" or "Grim" or "Pavlov"
                    action=[1, 0],   # Limited to cooperate and defect
                )
            ]
        ]

        logging.info(f'Running simulation for game transitions: {game_transitions}')
        description = '-'.join([game[0] for game in game_transitions])

        for agent_kwargs in META_AGENT_KWARGS:

            logging.debug(f'Agent configuration: {agent_kwargs}')

            # If only one set of agent_kwargs is provided, use it for all players
            if len(agent_kwargs) == 1:
                agent_kwargs = agent_kwargs * num_players
            elif len(agent_kwargs) != num_players:
                raise ValueError('"agent_kwargs" must have length equal to the number of players, or length 1.')

            # Run simulation --------------------------------------------------------------
            utils.simulation.simulate_parallel(
                game_transitions,
                nash_strategy,
                agent_kwargs=agent_kwargs,
                num_repeats=args.num_repeats,
                results_db_path=args.db_path,
                description=description
            )

    logging.info('\n😎 Done!')
    exit()
