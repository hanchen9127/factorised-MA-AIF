'''
Test file checking for parsing games

Auther: Hanchen Wang
Date: 2025-05
'''

import unittest
import generator2 as gen
from games import *
import torch
import numpy as np
import sys
import random

GAME_DURATION = random.randint(500, 5000)

GAMES = {
    "2x2": gen.load_csv_to_games(file_path="../game/GAMES_2_2.csv", game_form="2x2"),
    "2x3": gen.load_csv_to_games(file_path="../game/GAMES_2_3.csv", game_form="2x3"),
    "3x2": gen.load_csv_to_games(file_path="../game/GAMES_3_2.csv", game_form="3x2"),
}


def print_mismatches(mismatches):
    """Helper method to format mismatches for readable output."""
    return "\n".join(
        f"Game: {game_id}\n"
        f"  Expected: {data['expected']}\n"
        f"  Actual:   {data['result']}\n"
        for game_id, data in mismatches.items()
    )


def check_tensor_equality(game1, game2):
    """Helper method to check the equality of two transition lists."""
    return game1[0] == game2[0] and torch.equal(game1[1], game2[1]) and game1[2] == game2[2]


def compare_nested_dicts(d1, d2, tol=1e-5):
    """Helper method to check the equality of two dictionaries."""
    if d1.keys() != d2.keys():
        return False
    for key in d1:
        if d1[key].keys() != d2[key].keys():
            return False
        for subkey in d1[key]:
            arr1 = np.array(d1[key][subkey])
            arr2 = np.array(d2[key][subkey])
            if not np.allclose(arr1, arr2, atol=tol):
                return False
    return True


class TestGenerator(unittest.TestCase):

    def test_3x2_payoff_tensor(self):
        """
        Test if the payoff matrix of all 3x2 games stored in CSV matches the hardcoded tensors in games.py.
        """
        expected = [prisoners_dilemma_3player, harmony_3player, chicken_3player, coordination_3player, leader_3player,
                    stag_hunt_3player_green, stag_hunt_3player_red, stag_hunt_3player_penalty]
        result = [game.get_payoff_tensor() for game in GAMES.get("3x2")]

        for idx, (exp, res) in enumerate(zip(expected, result)):
            if not torch.equal(exp, res):
                self.fail(f"Mismatch at index {idx}:\nExpected:\n{exp}\nGot:\n{res}")

    def test_2x3_payoff_tensor(self):
        """
        Test if the payoff matrix of three 2x3 games stored in CSV matches the hardcoded tensors in games.py.
        """
        expected = [climbing_game_2player, rockpaperscissors_2player, test_game_2player]
        result = [game.get_payoff_tensor() for game in GAMES.get("2x3")]

        for idx, (exp, res) in enumerate(zip(expected, result)):
            if not torch.equal(exp, res):
                self.fail(f"Mismatch at index {idx}:\nExpected:\n{exp}\nGot:\n{res}")

    def test_2x2_random_transitions(self):
        """
        Test if the result transition list matches the form of META_GAME_TRANSITIONS in INFG-simulate.py.
        """
        result = gen.generate_random_transitions(dict_games=GAMES,
                                                 game_form="2x2",
                                                 num_selections=3,
                                                 num_steps=GAME_DURATION,
                                                 seed=1)
        expected = [
            [('1_4_4_Hero', torch.tensor([[1., 3.], [4., 2.]]), GAME_DURATION),
             ('3_5_5_Peace', torch.tensor([[4., 3.], [1., 2.]]), GAME_DURATION),
             ('1_3_3_BoS', torch.tensor([[2., 3.], [4., 1.]]), GAME_DURATION)
             ],

            {'1_4_4_Hero': {'Row_NBS': [[np.float64(0.38), np.float64(0.62)]],
                            'Col_NBS': [[np.float64(0.38), np.float64(0.62)]],
                            'Row_NE': [[1., 0.], [0., 1.], [0.25, 0.75]],
                            'Col_NE': [[0., 1.], [1., 0.], [0.25, 0.75]]},
             '3_5_5_Peace': {'Row_NBS': [[np.float64(1.0), np.float64(0.0)]],
                             'Col_NBS': [[np.float64(1.0), np.float64(0.0)]], 'Row_NE': [[1., 0.]],
                             'Col_NE': [[1., 0.]]},
             '1_3_3_BoS': {'Row_NBS': [[np.float64(0.62), np.float64(0.38)]],
                           'Col_NBS': [[np.float64(0.62), np.float64(0.38)]],
                           'Row_NE': [[1., 0.], [0., 1.], [0.5, 0.5]],
                           'Col_NE': [[0., 1.], [1., 0.], [0.5, 0.5]]}
             }
        ]
        result_transitions, result_data = result
        expected_transitions, expected_data = expected
        if_same = (all(check_tensor_equality(r, e) for r, e in zip(result_transitions, expected_transitions))
                   and compare_nested_dicts(result_data, expected_data)
                   )
        self.assertTrue(if_same,
                        f"Unmatched game transition list. Get:{str(result)}")

    def test_2x2_NBS(self):
        result = {}
        for g in GAMES.get("2x2"):
            if not g.game_id.endswith("NA"):
                result[g.game_id] = g.NBS
        expected = {'1_2_2_Chicken': {'Row_NBS': [[np.float64(1.0), np.float64(0.0)]],
                                      'Col_NBS': [[np.float64(1.0), np.float64(0.0)]]},
                    '1_3_3_BoS': {'Row_NBS': [[np.float64(0.62), np.float64(0.38)]],
                                  'Col_NBS': [[np.float64(0.62), np.float64(0.38)]]},
                    '1_4_4_Hero': {'Row_NBS': [[np.float64(0.38), np.float64(0.62)]],
                                   'Col_NBS': [[np.float64(0.38), np.float64(0.62)]]},
                    '1_5_5_Compro': {'Row_NBS': [[np.float64(0.0), np.float64(1.0)]],
                                     'Col_NBS': [[np.float64(0.0), np.float64(1.0)]]},
                    '1_6_6_Deadlock': {'Row_NBS': [[np.float64(0.0), np.float64(1.0)]],
                                       'Col_NBS': [[np.float64(0.0), np.float64(1.0)]]},
                    '1_1_1_Prisoners': {'Row_NBS': [[np.float64(1.0), np.float64(0.0)]],
                                        'Col_NBS': [[np.float64(1.0), np.float64(0.0)]]},
                    '3_2_2_StagHunt': {'Row_NBS': [[np.float64(1.0), np.float64(0.0)]],
                                       'Col_NBS': [[np.float64(1.0), np.float64(0.0)]]},
                    '3_3_3_Assurance': {'Row_NBS': [[np.float64(1.0), np.float64(0.0)]],
                                        'Col_NBS': [[np.float64(1.0), np.float64(0.0)]]},
                    '3_4_4_Coord': {'Row_NBS': [[np.float64(1.0), np.float64(0.0)]],
                                    'Col_NBS': [[np.float64(1.0), np.float64(0.0)]]},
                    '3_5_5_Peace': {'Row_NBS': [[np.float64(1.0), np.float64(0.0)]],
                                    'Col_NBS': [[np.float64(1.0), np.float64(0.0)]]},
                    '3_6_6_Harmony': {'Row_NBS': [[np.float64(1.0), np.float64(0.0)]],
                                      'Col_NBS': [[np.float64(1.0), np.float64(0.0)]]},
                    '3_1_1_NoConflinct': {'Row_NBS': [[np.float64(1.0), np.float64(0.0)]],
                                          'Col_NBS': [[np.float64(1.0), np.float64(0.0)]]},
                    }
        mismatches = {}
        for key in expected:
            if key not in result or result[key] != expected[key]:
                mismatches[key] = {
                    "expected": expected[key],
                    "result": result.get(key, "Key not found in result")
                }
        self.assertTrue(not mismatches,
                        f"Mismatched NBS strategies:\n"
                        f"{print_mismatches(mismatches)}")

    def test_2x2_NE(self):
        '''
        Validate the computed NE with the following matrix solver
        https://cgi.csc.liv.ac.uk/~rahul/bimatrix_solver/
        '''
        result = {}
        for g in GAMES.get("2x2"):
            if not g.game_id.endswith("NA"):
                result[g.game_id] = g.get_nash()
        expected = {'1_2_2_Chicken': [(([1., 0.]), ([0., 1.])), (([0., 1.]), ([1., 0.])),
                                      (([0.5, 0.5]), ([0.5, 0.5]))],
                    '1_3_3_BoS': [(([1., 0.]), ([0., 1.])), (([0., 1.]), ([1., 0.])),
                                  (([0.5, 0.5]), ([0.5, 0.5]))],
                    '1_4_4_Hero': [(([1., 0.]), ([0., 1.])), (([0., 1.]), ([1., 0.])),
                                   (([0.25, 0.75]), ([0.25, 0.75]))],
                    '1_5_5_Compro': [(([0., 1.]), ([0., 1.]))],
                    '1_6_6_Deadlock': [(([0., 1.]), ([0., 1.]))],
                    '1_1_1_Prisoners': [(([0., 1.]), ([0., 1.]))],
                    '3_2_2_StagHunt': [(([1., 0.]), ([1., 0.])), (([0., 1.]), ([0., 1.])),
                                       (([0.5, 0.5]), ([0.5, 0.5]))],
                    '3_3_3_Assurance': [(([1., 0.]), ([1., 0.])), (([0., 1.]), ([0., 1.])),
                                        (([0.5, 0.5]), ([0.5, 0.5]))],
                    '3_4_4_Coord': [(([1., 0.]), ([1., 0.])), (([0., 1.]), ([0., 1.])),
                                    (([0.25, 0.75]), ([0.25, 0.75]))],
                    '3_5_5_Peace': [(([1., 0.]), ([1., 0.]))],
                    '3_6_6_Harmony': [(([1., 0.]), ([1., 0.]))],
                    '3_1_1_NoConflinct': [(([1., 0.]), ([1., 0.]))],
                    }
        self.assertTrue(
            all(
                game_id in expected and
                all(any(np.array_equal(actual, exp) for exp in expected[game_id]) for actual in result[game_id])
                for game_id in result
            ),
            f"Mismatched pure strategies:\n{result}"
        )


if __name__ == '__main__':
    unittest.main()
