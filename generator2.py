'''
A game transition sequence generator.
1. Support game selection for various type of games.
2. Run specified statistical tests for hypothesis testing.
3. Animate B-matrix heatmaps over entire simulation duration.

Author: Hanchen Wang
Date:   2025-11
'''
import os
import logging
import torch
import random
import csv
import re
import nashpy
import copy
import sys
from scipy.optimize import minimize
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from tqdm import tqdm

sys.path.append('../')
import utils.database

# Add a small CONSTANT to avoid log(0) issues
CONSTANT = 1
NEG_CONSTANT = -5
EPSILON = torch.finfo(torch.float32).eps
EPSILON = 1


class Game(ABC):
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.NE = []
        self.utilities = []
        self.NBS = {}

    @abstractmethod
    def get_payoff_tensor(self):
        """
        Return the payoff structure:
        - (row_payoffs, col_payoffs) for 2-player games
        - full_tensor (shape: 2x2x2x3) for 3-player games
        """
        pass

    def get_nash(self):
        """
        Compute and return a list of Nash equilibria:
        Each element is a tuple of strategy vectors
        """
        return None

    def get_utility(self):
        """
        Return expected payoffs for each NE.
        Fills `self.utilities` if not already filled.
        """
        return None

    def get_nash_strategy_prob(self):
        """
        Return NE strategies as a dict: {"Row_NE": [...], "Col_NE": [...]}. Optional to override in child class.
        If not implemented, return empty.
        """
        return {}

    def get_nbs_strategy_prob(self):
        """
        Return NBS strategy as a dict. Optional to override in child class.
        If not implemented, return empty.
        """
        return {}

    def __str__(self):
        result = f"Game ID: {self.game_id}\n"

        # Print row and col payoffs if defined
        if hasattr(self, "row_payoffs"):
            result += f"Row Payoff:\n{self.row_payoffs}\n"
        if hasattr(self, "col_payoffs"):
            result += f"Col Payoff:\n{self.col_payoffs}\n"
        if hasattr(self, "player_payoffs"):
            result += f"3-Player Payoff Tensor:\n{self.player_payoffs}\n"

        # NBS info
        result += f"NBS:\n"
        stored_nbs = getattr(self, "NBS_stored", {})
        result += f"\tStored: {stored_nbs}\n"
        result += f"\tProbability Of Cooperation: {self.NBS}\n"

        # Compute NE and utilities if needed
        try:
            if not self.NE:
                self.get_nash()
            if not self.utilities:
                self.get_utility()

            for i, (eq, utility) in enumerate(zip(self.NE, self.utilities), 1):
                if isinstance(eq, tuple) and len(eq) == 2 and len(utility) == 2:
                    row_stg, col_stg = eq
                    u_row, u_col = utility
                    result += f"NE {i}:\n"
                    result += f"\tRow Strategy & Utility: {row_stg} -> {u_row}\n"
                    result += f"\tCol Strategy & Utility: {col_stg} -> {u_col}\n"
                else:
                    result += f"NE {i}:\n"
                    result += f"\tStrategy: {eq}\n"
                    result += f"\tUtility: {utility}\n"

        except NotImplementedError:
            result += "NE or utility computation not implemented for this game.\n"
        except Exception as e:
            result += f"Error computing NE/utilities: {e}\n"

        return result


class Game3P2A(Game):
    def __init__(self, game_id, payoff_cells, tensor_cell):
        super().__init__(game_id)
        reward, temptation, sucker, penalty = map(float, payoff_cells)
        tensor_structure = tensor_cell

        # Construct the tensor using cell data
        values = []
        for symbol in tensor_structure:
            if symbol == "R":
                values.append(reward)
            elif symbol == "T":
                values.append(temptation)
            elif symbol == "S":
                values.append(sucker)
            elif symbol == "P":
                values.append(penalty)
            else:
                raise ValueError(f"Invalid payoff symbol '{symbol}'. Only 'R', 'T', 'S', 'P' are allowed.")

        self.row_payoffs = torch.tensor([
            [[values[0], values[1]],  # Player 1: C, Player 2: C, Player 3: (C / D)
             [values[2], values[3]]],  # Player 1: C, Player 2: D, Player 3: (C / D)

            [[values[4], values[5]],  # Player 1: D, Player 2: C, Player 3: (C / D)
             [values[6], values[7]]],  # Player 1: D, Player 2: D, Player 3: (C / D)
        ], dtype=torch.float)

    def get_payoff_tensor(self):
        """Return the payoff matrix in a tensor form"""
        return self.row_payoffs


class Game2P3A(Game):
    def __init__(self, game_id: str, payoff_cells):
        super().__init__(game_id)
        i11, i12, i13, i21, i22, i23, i31, i32, i33 = map(float, payoff_cells)

        # Payoff of row player
        self.row_payoffs = torch.tensor([
            [i11, i12, i13],
            [i21, i22, i23],
            [i31, i32, i33]
        ], dtype=torch.float)

        # Payoff of column player (Assumption: The game is always symmetric.)
        self.col_payoffs = self.row_payoffs.T

    def get_payoff_tensor(self, row_player: bool = True):
        """Return the payoff matrix in a tensor form"""
        if row_player:
            return self.row_payoffs
        else:
            return self.col_payoffs

    def get_nash(self):
        """Return all NE using support enumeration."""
        if not self.NE:
            game = nashpy.Game(self.row_payoffs, self.col_payoffs)
            self.NE = list(game.support_enumeration())
        return self.NE

    def get_utility(self):
        """Return expected payoffs of players for all NE in form (row_player, col_player)."""
        if not self.NE:
            self.get_nash()

        if not self.utilities:
            for eq in self.NE:
                row_stg, col_stg = eq
                utility_row = row_stg @ self.row_payoffs.numpy() @ col_stg.T
                utility_col = row_stg @ self.col_payoffs.numpy() @ col_stg.T
                self.utilities.append((utility_row, utility_col))
        return self.utilities

    def get_nash_strategy_prob(self):
        """Return the Nash strategy probabilities in a particular form {"Row": [], "Col": []}."""
        if not self.NE:
            self.get_nash()

        result = {"Row_NE": [], "Col_NE": []}
        for eq in self.NE:
            row_stg, col_stg = eq
            result["Row_NE"].append(row_stg)
            result["Col_NE"].append(col_stg)
        return result


class Game2P2A(Game):
    def __init__(self, game_id: str, payoff_cells, nbs_cell):
        super().__init__(game_id)

        # Unpack payoff_cells entries (Order in CSV dataset: CD, CC, DD, DC)
        cooperate_defect, cooperate_cooperate, defect_defect, defect_cooperate = payoff_cells

        # Payoff of row player
        self.row_payoffs = torch.tensor([
            [cooperate_cooperate[0], cooperate_defect[0]],  # Reward, Sucker
            [defect_cooperate[0], defect_defect[0]]  # Temptation, Penalty
        ], dtype=torch.float)

        # Payoff of column player
        self.col_payoffs = torch.tensor([
            [cooperate_cooperate[1], cooperate_defect[1]],  # Reward, Temptation
            [defect_cooperate[1], defect_defect[1]]  # Sucker, Penalty
        ], dtype=torch.float)

        # # Empty lists for Nash equilibrium initialization
        # self.NE = []
        # self.utilities = []

        # NBS strategy Computation (optimal cooperation probabilities)
        self.NBS = self.get_nbs_strategy_prob()

        # Interpret NBS stored in CSV
        self.NBS_stored = {}
        if "/" in nbs_cell:
            nbs_cell = nbs_cell.split("/")
        else:
            nbs_cell = [nbs_cell]
        for nbs in nbs_cell:
            if nbs == "CC":
                self.NBS_stored[nbs] = cooperate_cooperate
            elif nbs == "CD":
                self.NBS_stored[nbs] = cooperate_defect
            elif nbs == "DD":
                self.NBS_stored[nbs] = defect_defect
            elif nbs == "DC":
                self.NBS_stored[nbs] = defect_cooperate

    def get_payoff_tensor(self, row_player: bool = True):
        """Return the payoff matrix in a tensor form"""
        if row_player:
            return self.row_payoffs
        else:
            return self.col_payoffs

    def get_nash(self):
        """Return all NE using support enumeration."""
        if not self.NE:
            game = nashpy.Game(self.row_payoffs, self.col_payoffs)
            self.NE = list(game.support_enumeration())
        return self.NE

    def get_utility(self):
        """Return expected payoffs of players for all NE in form (row_player, col_player)."""
        if not self.NE:
            self.get_nash()

        if not self.utilities:
            for eq in self.NE:
                row_stg, col_stg = eq
                utility_row = row_stg @ self.row_payoffs.numpy() @ col_stg.T
                utility_col = row_stg @ self.col_payoffs.numpy() @ col_stg.T
                self.utilities.append((utility_row, utility_col))
        return self.utilities

    def get_nash_strategy_prob(self):
        """Return the Nash strategy probabilities in a particular form {"Row": [], "Col": []}."""
        if not self.NE:
            self.get_nash()

        result = {"Row_NE": [], "Col_NE": []}
        for eq in self.NE:
            row_stg, col_stg = eq
            result["Row_NE"].append(row_stg)
            result["Col_NE"].append(col_stg)
        return result

    def get_nbs_strategy_prob(self):
        """Return the NBS strategy, an optimal cooperation strategy"""

        def find_threat_point():
            self.get_utility()  # Ensure NEs with their utilities are computed
            pareto_optimal_payoff = None
            max_payoff_sum = -float('inf')

            # Pair each NE with its utility
            for eq, (u_row, u_col) in zip(self.NE, self.utilities):
                current_sum = u_row + u_col
                if current_sum > max_payoff_sum:
                    max_payoff_sum = current_sum
                    pareto_optimal_payoff = (u_row, u_col)  # Store payoffs

            # Return the selected NE's payoffs
            return pareto_optimal_payoff

        def compute_expected_row(p, q):
            return (
                    p * q * self.row_payoffs[0, 0].item() +  # CC
                    p * (1 - q) * self.row_payoffs[0, 1].item() +  # CD
                    (1 - p) * q * self.row_payoffs[1, 0].item() +  # DC
                    (1 - p) * (1 - q) * self.row_payoffs[1, 1].item()  # DD
            )

        def compute_expected_col(p, q):
            return (
                    p * q * self.col_payoffs[0, 0].item() +  # CC
                    p * (1 - q) * self.col_payoffs[0, 1].item() +  # CD
                    (1 - p) * q * self.col_payoffs[1, 0].item() +  # DC
                    (1 - p) * (1 - q) * self.col_payoffs[1, 1].item()  # DD
            )

        def neg_nash_product(params):
            p, q = params
            surplus_row = compute_expected_row(p, q) - disagree_row
            surplus_col = compute_expected_col(p, q) - disagree_col
            return -surplus_row * surplus_col

        # Disagreement payoffs
        disagree_row = find_threat_point()[0]
        disagree_col = find_threat_point()[1]
        disagree_row = 0
        disagree_col = 0

        # Minimize the negative of this product to maximize the Nash product
        initial_guess = [0.5, 0.5]  # Start with 50% cooperation
        res = minimize(neg_nash_product, initial_guess, bounds=[(0, 1), (0, 1)], method='L-BFGS-B')

        # Extract the result
        optimal_p = round(res.x[0], 2)
        optimal_q = round(res.x[1], 2)

        # Construct the strategy as [p, 1-p] for both players.
        result = {
            "Row_NBS": [[optimal_p, round(1 - optimal_p, 2)]],
            "Col_NBS": [[optimal_q, round(1 - optimal_q, 2)]]
        }

        return result

    def duplicate_self(self, num_changes, change_type="penalty"):
        """
        Return duplicates of the game object while modifying the payoff matrix.
        For each duplicate, DD payoff is increased gradually, reducing the punishment for defect-defect.
        Changes are applied to both row and col players while keeping the overall RTSP structure.
        """
        result = [self]
        num_duplicates = num_changes - 1  # Include self

        if change_type == "enlarge":
            for count in range(1, num_duplicates + 1):
                # Create a deep copy of self with new game_id
                new_game = copy.deepcopy(self)
                new_game.game_id = f"{self.name[0:2]}{count}"

                # Enlarge the size of payoff matrix
                new_game.row_payoffs *= count
                new_game.col_payoffs *= count

                # Reset cached Nash equilibria and utilities since we are modifying payoffs
                new_game.NE = []
                new_game.utilities = []

                result.append(new_game)
        else:
            # Retrieve original DD payoff for both players
            original_dd_row = self.row_payoffs[1, 1].item()
            original_dd_col = self.col_payoffs[1, 1].item()

            # Compute the maximum allowable increment.
            # We assume that the maximum increment is 90% of the gap between the cooperate-cooperate payoff (CC)
            # and the original defect-defect payoff (DD) so that the DD payoff does not exceed the CC payoff.
            cc_payoff_row = self.row_payoffs[0, 0].item()
            cc_payoff_col = self.col_payoffs[0, 0].item()

            # Compute gaps (for row and column separately; they might differ)
            gap_row = (cc_payoff_row - original_dd_row) * 1
            gap_col = (cc_payoff_col - original_dd_col) * 1

            for count in range(1, num_duplicates + 1):
                # Create a deep copy of self with new game_id
                new_game = copy.deepcopy(self)
                new_game.game_id = f"{self.game_id}{count}"

                # Reset cached Nash equilibria and utilities since we are modifying payoffs
                new_game.NE = []
                new_game.utilities = []

                # Calculate an increment that increases linearly with the duplicate index
                increment_row = count / num_duplicates * gap_row
                increment_col = count / num_duplicates * gap_col

                # Update the defect-defect payoffs
                new_game.row_payoffs[1, 1] = round(original_dd_row + increment_row, 4)
                new_game.col_payoffs[1, 1] = round(original_dd_col + increment_col, 4)

                result.append(new_game)

        return result


def load_csv_to_games(file_path: str, game_form: str):
    # Initialize an empty list storing Game objects
    game_lst = []

    # Open and read the CSV file
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        # Skip header
        next(reader)
        for row in reader:
            # Skip empty lines
            if not row or all(cell.strip() == "" for cell in row):
                continue

            if game_form == "2x2":
                # Extract 4 pairs of (x,y)
                payoff_cells = [re.findall(r"\((\d+),(\d+)\)", row[1]), re.findall(r"\((\d+),(\d+)\)", row[2]),
                                re.findall(r"\((\d+),(\d+)\)", row[3]), re.findall(r"\((\d+),(\d+)\)", row[4])]
                payoff_cells = [(int(x), int(y)) for sublist in payoff_cells for x, y in sublist]

                # Create Game objects
                new_game = Game2P2A(game_id=row[0],
                                    payoff_cells=[payoff_cells[0], payoff_cells[1],  # CD CC
                                                  payoff_cells[2], payoff_cells[3]],  # DC DD
                                    nbs_cell=row[5])
                game_lst.append(new_game)

            elif game_form == "2x3":
                new_game = Game2P3A(game_id=row[0],
                                    payoff_cells=row[1:10])
                game_lst.append(new_game)

            elif game_form == "3x2":
                # print(row[0], row[1:5], row[5])
                new_game = Game3P2A(game_id=row[0],
                                    payoff_cells=row[1:5],
                                    tensor_cell=row[5])
                game_lst.append(new_game)

            else:
                raise ValueError(f"Unsupported game form '{game_form}'. Only '2x2', '2x3', '3x2' are available.")

    return game_lst


def generate_random_transitions(dict_games,
                                game_form: str = "2x2",
                                num_selections: int = 2,
                                num_steps: int = 500,
                                game_has_name: bool = True,
                                seed: int = None):
    """
    Randomly (seed available) generate a form game entry for simulation.

    Args:
        dict_games: The dictionary of all games (GAMES)
        game_form (str): The form of game ('2x2', '2x3', etc.).
        num_selections (int): The number of random selected unique games in the transition list.
        num_steps (int): A list of The number of time steps. AKA GAME_DURATION.
        game_has_name: A boolean constraint if the game has to have a name
        seed: For reproducibility of randomness.

    Returns: (Plugged into META_GAME_TRANSITIONS & META_GAME_STRATEGIES in scripts/INFG-simulate)
        transition_lst: Arrays of a game object's (game_id, row_payoffs, num_steps).
        strategy_lst: Key-value pairs of a game object's NE and NBS strategy in form of probability.
    """

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Get all games of required form from GAMES list
    game_objs = dict_games.get(game_form)
    if not game_objs:
        raise ValueError(f"Unsupported game form '{game_form}'. Only '2x2', '2x3', '3x2' are available.")

    # Check constraint of games need have names
    temp = []
    if game_has_name:
        for g in game_objs:
            if not g.game_id.endswith("NA"):
                temp.append(g)
    game_objs = temp

    # Ensure num_selections is not greater than available games
    num_selections = min(num_selections, len(game_objs))

    # Randomly select unique games
    selected_games = random.sample(game_objs, num_selections)

    # Append game obj properties into return list
    transition_lst = []  # Game transition list
    strategy_lst = {}  # NE strategy list, NBS strategy list
    for game_obj in selected_games:
        transition_lst.append((game_obj.game_id,
                               game_obj.row_payoffs,
                               num_steps))
        strategy_lst[game_obj.game_id] = game_obj.get_nbs_strategy_prob() | game_obj.get_nash_strategy_prob()

    return transition_lst, strategy_lst


def generate_dynamic_transitions(dict_games,
                                 game_ids,
                                 game_to_change,
                                 game_form: str = "2x2",
                                 num_steps: list = [500, 1000],
                                 num_changes: int = 10,
                                 payoff_multiplier: int = 1,
                                 change_type="penalty"):
    """
    Randomly (seed available) generate a form game entry for simulation.

    Args:
        dict_games: The dictionary of all games. E.g. GAMES
        game_ids: A list of game_id in the dict_games. E.g. ["1_2_2_Chicken", "1_1_1_Prisoners"]
        game_to_change: A list of index to select the dynamic payoff-changing game. E.g. [0, 1]
        game_form (str): The form of game ('2x2', '2x3', etc.).
        num_steps (int): A list of number of time steps for the corresponding game in in game_ids.
        num_changes (int): The number of payoff changes for the selected game.
        payoff_multipler: A ratio for increasing the size of payoff matrix.
        change_type: A string identifies if changing penalty or all payoffs.

    Returns: (Plugged into META_GAME_TRANSITIONS & META_GAME_STRATEGIES in scripts/INFG-simulate)
        transition_lst: Arrays of a game object's (game_id, row_payoffs, num_steps).
        strategy_lst: Key-value pairs of a game object's NE and NBS strategy in form of probability.
    """

    # Get all games of required form from GAMES list
    game_objs = dict_games.get(game_form)
    if not game_objs:
        raise ValueError(f"Unsupported game form '{game_form}'. Only '2x2', '2x3', '3x2' are available.")

    selected_games = []
    for game_id in game_ids:
        for game_obj in game_objs:
            if game_obj.game_id == game_id:
                selected_games.append(game_obj)

    if len(selected_games) != len(game_ids):
        raise ValueError(f"Not all 'game_ids' found in 'dict_games'. Please make sure all 'game_ids' are valid.")

    if game_to_change and game_form != "2x2":
        print("\nHyperparameter 'game_to_change' not used! It is a feature only for 2x2 games.\n")

    transition_lst = []  # Game transition list
    strategy_lst = {}  # NE strategy list, NBS strategy list
    for idx, game_obj in enumerate(selected_games):
        if idx in game_to_change and game_form == "2x2":
            # Dynamic payoff is an only-for-2x2 feature
            duplicates = game_obj.duplicate_self(num_changes, change_type=change_type)
            for duplicate in duplicates:
                transition_lst.append((duplicate.game_id,
                                       duplicate.row_payoffs * payoff_multiplier,
                                       int(num_steps[idx] / num_changes)))
                strategy_lst[duplicate.game_id] = duplicate.get_nbs_strategy_prob() | duplicate.get_nash_strategy_prob()
        else:
            transition_lst.append((game_obj.game_id,
                                   game_obj.row_payoffs * payoff_multiplier,
                                   num_steps[idx]))

            strategy_lst[game_obj.game_id] = game_obj.get_nbs_strategy_prob() | game_obj.get_nash_strategy_prob()

    return transition_lst, strategy_lst


def spike_checking_test(experiments,
                        game_transitions,
                        defection_threshold: float = 0.5,
                        spike_threshold: float = 0.15,
                        save_path: str = None):
    """
       In INFG-plot.py:
       Run a Chi-square test for studying the association between defection behavior and blip in prob of cooperation.

       Args:
           experiments: The result from utils.database.retrieve_timeseries_matching.
           game_transitions: The elements in META_GAME_TRANSITIONS
           defection_threshold (float): The threshold for detecting defection behavior. The average of prob of
                                        cooperation in the initial game < this value is seen as defection behavior.
           spike_threshold (float): The threshold for detecting spike/blip. Prob of cooperation in any step > this
                                    value is seen as a spike.
       """

    inital_game = game_transitions[0][0]  # Name of initial game (Ch)
    next_game = game_transitions[1][0]  # Name of next game (PD)
    inital_steps = game_transitions[0][-1]  # Total number of initial game duration
    inertia_steps = 5  # Number of steps in the later game that could be highly affected by the end of former one
    alpha = 0.05  # The significance level for this statistical test

    result = ""
    result += f"First game is {inital_game} of {inital_steps} steps."

    # Collect boolean observations for both players
    defection_bools = []
    spike_bools = []

    for i in range(len(experiments)):
        loaded_vars = utils.database.load_single_timeseries(experiments, i)
        q_pi_history = np.round(loaded_vars['q_u'], 2)
        seed = loaded_vars['seed']

        prob_cooperate_i = np.round(q_pi_history[:, 0, 0], 2)
        prob_cooperate_j = np.round(q_pi_history[:, 1, 0], 2)

        result += f"\nExp{i} (Seed{seed}) :\n"

        for player_name, prob_cooperate in zip(["i", "j"], [prob_cooperate_i, prob_cooperate_j]):
            inital_mean = prob_cooperate[:inital_steps].mean()
            next_mean = prob_cooperate[inital_steps:].mean()
            next_arr = prob_cooperate[inital_steps + inertia_steps:]
            # blip = np.any(prob_cooperate[inital_steps+1:] > spike_threshold)

            blip = False
            first_blip_step = None
            for idx in range(1, len(next_arr)):
                if next_arr[idx] > spike_threshold:
                    first_blip_step = inital_steps + idx
                    blip = True
                    break

            defected = inital_mean < defection_threshold

            result += (f"  Agent {player_name}: {inital_game} = {inital_mean:.2f} → {next_game} = {next_mean:.2f}"
                       f"| Defected: {defected}, Blip: {blip}"
                       f"{f' (at step {first_blip_step})' if first_blip_step is not None else ''}\n")

            defection_bools.append(defected)
            spike_bools.append(blip)

    # Build contingency table
    table = np.zeros((2, 2))  # [[no-defect no-blip, no-defect blip], [defect no-blip, defect blip]]
    for defected, blip in zip(defection_bools, spike_bools):
        table[int(defected)][int(blip)] += 1

    result += "\nContingency Table (rows=Defected, cols=Blip):\n"
    result += str(table)

    # Perform Chi-squared test and Fisher's test
    chi2, p, dof, expected = chi2_contingency(table)
    odds_ratio, p = fisher_exact(table)

    result += f"\nChi-squared test result:\n"
    result += f"  chi2 = {chi2:.2f}, p = {p}, dof = {dof}\n"
    result += f"  expected frequencies:\n{expected}\n"

    if p < alpha:
        result += "=> Significant association found (reject null hypothesis).\n"
    else:
        result += "=> No significant association (fail to reject null).\n"

    result += f"\nFisher's exact test result:\n"
    result += f"  Odds ratio: {odds_ratio:.2f}\n"
    result += f"  p-value: {p:.4f}\n"

    if p < alpha:
        result += "=> Significant association found (reject null hypothesis).\n"
    else:
        result += "=> No significant association (fail to reject null).\n"

    print(result)

    # Save result locally if path provided
    if save_path:
        save_path += "/spike_test.txt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(result)


def strategic_consistency_test(experiments,
                               game_transitions,
                               defection_threshold: float = 0.5,
                               save_path: str = None):
    """
    Run Chi-square & Fisher test to study the association between strategic behavior in the initial game
    and strategic behavior in the last game.

    Args:
        experiments: The result from utils.database.retrieve_timeseries_matching.
        game_transitions: The elements in META_GAME_TRANSITIONS, a list of tuples [(game_name, steps), ...].
        defection_threshold (float): The threshold for detecting defection behavior.
                                    Average probability of cooperation < this value is considered defection.
    """
    print(game_transitions)
    initial_game = game_transitions[0][0]  # Name of initial game (Ch)
    last_game = game_transitions[-1][0]  # Name of last game (Ch)
    initial_steps = game_transitions[0][-1]  # Total number of initial game steps

    last_game_steps = game_transitions[-1][-1]  # Steps in last game
    last_game_start = sum(g[2] for g in game_transitions) - last_game_steps
    alpha = 0.05  # Significance level for the statistical test

    result = ""
    result += f"Initial game is {initial_game} of {initial_steps} steps. "
    result += f"Last game is {last_game} of {last_game_steps} steps, start from {last_game_start}.\n"

    # Collect boolean observations for both players
    defection_initial_bools = []
    defection_last_bools = []

    from collections import defaultdict
    # Collect a dictionary in form of seed:bool for return
    consistency = defaultdict(list)
    for i in range(len(experiments)):
        loaded_vars = utils.database.load_single_timeseries(experiments, i)
        q_pi_history = np.round(loaded_vars['q_u'], 2)
        seed = loaded_vars['seed']

        prob_cooperate_i = np.round(q_pi_history[:, 0, 0], 2)
        prob_cooperate_j = np.round(q_pi_history[:, 1, 0], 2)

        result += f"\nExp{i} (Seed{seed}) :\n"

        for player_name, prob_cooperate in zip(["i", "j"], [prob_cooperate_i, prob_cooperate_j]):
            initial_mean = prob_cooperate[:initial_steps].mean()
            last_mean = prob_cooperate[last_game_start:last_game_start + last_game_steps].mean()

            defected_initial = initial_mean < defection_threshold
            defected_last = last_mean < defection_threshold

            result += (f"  Agent {player_name}: {initial_game} = {initial_mean:.2f} → "
                       f"{last_game} = {last_mean:.2f} | Defected Initial: {defected_initial}, "
                       f"Defected Last: {defected_last} | Consistency: {defected_initial == defected_last}\n")
            consistency[seed].append(defected_initial == defected_last)

            defection_initial_bools.append(defected_initial)
            defection_last_bools.append(defected_last)

    # Build contingency table
    table = np.zeros((2, 2))  # [[no-defect_initial no-defect_last, no-defect_initial defect_last],
    #  [defect_initial no-defect_last, defect_initial defect_last]]
    for defect_initial, defect_last in zip(defection_initial_bools, defection_last_bools):
        table[int(defect_initial)][int(defect_last)] += 1

    result += "\nContingency Table (rows=Defected Initial, cols=Defected Last):\n"
    result += str(table)

    # Perform Chi-squared test
    chi2, p, dof, expected = chi2_contingency(table + EPSILON)
    min_expected = expected.min()

    result += f"\nChi-squared test result:\n"
    result += f"  Minimum expected frequency: {min_expected:.2f}\n"

    if min_expected < 5:
        result += "  Warning: Chi-squared test may be unreliable due to expected frequencies < 5.\n"
        result += "  Consider using Fisher's exact test or increasing sample size.\n"

    result += f"  Chi-squared test: chi2 = {chi2:.2f}, p = {p:.4f}, dof = {dof}\n"

    if p < alpha:
        result += "=> Significant association found (reject null hypothesis).\n"
    else:
        result += "=> No significant association (fail to reject null).\n"

    # Perform Fisher's test
    odds_ratio, p = fisher_exact(table)
    result += f"\nFisher's exact test result:\n"
    result += f"  Odds ratio: {odds_ratio:.2f}\n"
    result += f"  p-value: {p:.4f}\n"

    if p < alpha:
        result += "=> Significant association found (reject null hypothesis).\n"
    else:
        result += "=> No significant association (fail to reject null).\n"

    print(result)

    # Save result locally if path provided
    if save_path:
        save_path += "/consistency_test.txt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(result)

    return consistency


def animate_transition_matrix_B(experiments,
                                consistency,
                                save_path,
                                fps=60,
                                dpi=100,
                                selected_seed=[1],
                                q_s_computation=False):
    """
    Create a video showing the evolution of B-matrix heatmaps over time for each experiment.

    Parameters:
    - experiments: List of experiment data
    - consistency: Dictionary with consistency results for each seed
    - save_path: Directory to save the video files
    - fps: Frames per second for the output video
    - dpi: Dots per inch for video quality
    - selected_seed: Select the seed num for producing the
    - q_s_computation: A boolean controlling if computing P(s′=0∣a) and P(s′=1∣a)
    """

    # Create output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    for i in range(len(experiments)):
        loaded_vars = utils.database.load_single_timeseries(experiments, i)
        seed = loaded_vars['seed']
        if seed not in selected_seed:
            continue

        B_all = np.array(loaded_vars['B'])
        q_s_history = loaded_vars.get('q_s', None)

        T, num_agents, num_factors, num_actions, _, _ = B_all.shape

        cmap_action = {
            0: mcolors.LinearSegmentedColormap.from_list('blue_map', ['#ffffff', '#03dffc']),
            1: mcolors.LinearSegmentedColormap.from_list('pink_map', ['#ffffff', '#f70ce4'])
        }

        # Layout: depends on whether q_s_computation is used
        if not q_s_computation:
            fig, axes = plt.subplots(
                2, num_factors * num_actions,
                figsize=(12, 6), squeeze=False, dpi=dpi
            )
        else:
            fig, axes = plt.subplots(
                2, num_actions,
                figsize=(8, 6), squeeze=False, dpi=dpi
            )
            bar_axes = [[None] * num_actions for _ in range(num_agents)]

        heatmaps = [[None] * (num_factors * num_actions) for _ in range(num_agents)]
        texts = [[None] * (num_factors * num_actions) for _ in range(num_agents)]

        # Init plots once
        for agent in range(num_agents):
            if not q_s_computation:
                factors_to_show = range(num_factors)
            else:
                factors_to_show = [0]

            for f_idx, factor in enumerate(factors_to_show):
                for action in range(num_actions):
                    col_idx = f_idx * num_actions + action
                    ax = axes[agent, col_idx]

                    B_plot = B_all[0, agent, factor, action, :, :]
                    im = ax.imshow(B_plot, vmin=0, vmax=1, cmap=cmap_action[action])
                    heatmaps[agent][col_idx] = im

                    txts = []
                    for ii in range(2):
                        for jj in range(2):
                            txts.append(ax.text(jj, ii, f'{B_plot[ii, jj]:.2f}',
                                                ha='center', va='center',
                                                color='black', fontsize=8))
                    texts[agent][col_idx] = txts

                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])

                    if agent == 0:
                        ax.set_title(f'Factor={factor}, Action={action}')
                    if col_idx == 0:
                        test_result = "Consistent" if consistency[seed][agent] else "Inconsistent"
                        text_color = "green" if consistency[seed][agent] else "red"
                        ax.set_ylabel(f'Agent {agent}\n({test_result})',
                                      color=text_color, fontweight='bold')

                    # Pre-allocate bar axes if q_s
                    if q_s_computation:
                        bar_ax = fig.add_axes([
                            0.45 + col_idx * 0.42,  # x position
                            0.6 - agent * 0.41,  # y position
                            0.03,  # width
                            0.2  # height
                        ])
                        bar_ax.set_xlim(0, 1)
                        bar_ax.set_ylim(0, 1)
                        bar_ax.set_xticks([])
                        bar_ax.set_yticks([0, 1])
                        bar_ax.set_yticklabels([f"P(s′=0|a={action})", f"P(s′=1|a={action})"], fontsize=7)
                        bar_ax.yaxis.tick_right()
                        bar_ax.yaxis.set_label_position("right")

                        bar = bar_ax.barh([0, 1], [0, 0], color=[cmap_action[action](0.8)] * 2)
                        bar_texts = [
                            bar_ax.text(0, 0, "0.00", va="center", ha="left",
                                        fontsize=7, color="black", fontweight="bold"),
                            bar_ax.text(0, 1, "0.00", va="center", ha="left",
                                        fontsize=7, color="black", fontweight="bold")
                        ]
                        bar_axes[agent][action] = (bar_ax, bar, bar_texts)

        # Update function
        def update(frame):
            fig.suptitle(f'B-matrix Evolution - Seed: {seed} - Time: {frame}', fontsize=16)

            for agent in range(num_agents):
                if not q_s_computation:
                    factors_to_show = range(num_factors)
                else:
                    factors_to_show = [0]

                for f_idx, factor in enumerate(factors_to_show):
                    for action in range(num_actions):
                        col_idx = f_idx * num_actions + action
                        B_plot = B_all[frame, agent, factor, action, :, :]

                        # update heatmap
                        heatmaps[agent][col_idx].set_data(B_plot)

                        # update texts
                        for k, (ii, jj) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                            texts[agent][col_idx][k].set_text(f'{B_plot[ii, jj]:.2f}')

                        # update q_s bars
                        if q_s_computation:
                            q_s_vals = q_s_history[frame, agent, 0, :]
                            p_future = B_plot @ q_s_vals  # effective transition
                            bar_ax, bar, bar_texts = bar_axes[agent][action]
                            bar[0].set_width(p_future[0])
                            bar[1].set_width(p_future[1])

                            # update text labels with actual values
                            bar_texts[0].set_text(f"{p_future[0]:.2f}")
                            bar_texts[0].set_x(p_future[0] + 0.02)  # offset to the right of the bar
                            bar_texts[0].set_y(0.2)  # align with lower bar

                            bar_texts[1].set_text(f"{p_future[1]:.2f}")
                            bar_texts[1].set_x(p_future[1] + 0.02)
                            bar_texts[1].set_y(0.8)  # align with upper bar

            return [*sum(heatmaps, []), *sum(texts, [])]

        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=T,
            interval=1000 / fps, blit=False
        )

        # Save the animation
        video_path = os.path.join(save_path, f'B_evolution_seed{seed}.mp4')
        ani.save(video_path, writer='ffmpeg', fps=fps, dpi=dpi)

        plt.close(fig)
        logging.info(f'✅ Saved video: {video_path}')

    logging.info('✅ All videos created successfully!')



if __name__ == '__main__':
    # Load csv for needed game forms
    GAMES = {
        "2x2": load_csv_to_games(file_path="game/GAMES_2_2.csv", game_form="2x2"),
        "2x3": load_csv_to_games(file_path="game/GAMES_2_3.csv", game_form="2x3"),
        "3x2": load_csv_to_games(file_path="game/GAMES_3_2.csv", game_form="3x2"),
    }

    # Select the game form to test
    GAME_FORM = "2x2"

    # Observe all named game objects
    for game in GAMES.get(GAME_FORM):
        if not game.game_id.endswith("NA"):
            print(game)

    # Sample usage of generate_random_transitions

    # print(generate_random_transitions(dict_games=GAMES,
    #                                   game_form=GAME_FORM,
    #                                   num_steps=2000,
    #                                   num_selections=2,
    #                                   game_has_name=True,
    #                                   seed=2)
    #       )

    # Sample usage of generate_dynamic_transitions

    # print(generate_dynamic_transitions(dict_games=GAMES,
    #                                    game_form=GAME_FORM,
    #                                    game_ids=["Chicken", "Prisoners"],
    #                                    game_to_change=[0],
    #                                    num_steps=[1000, 1000],
    #                                    num_changes=0)
    #       )
