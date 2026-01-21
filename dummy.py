'''
Canonical Dummy Opponent (TFT, Grim, Pavlov) that does not learn or infer

Authors: Hanchen Wang
Date: 2026-01
'''
import torch
import torch.nn.functional as F

class DummyAgent:
    def __init__(self, id, strategy, action, game_matrix=None, num_states=2, **kwargs):
        self.id = id
        self.is_dummy = True

        # --- Fixed action policy ---
        self.num_actions = num_actions = game_matrix.shape[0]
        self.t = 0
        self.u = None

        self.log_C = game_matrix  # will be set by simulation via set_log_C
        self.game_matrix = game_matrix.to(torch.float)  # Rewards from row player's perspective (force to float)
        self.num_agents = num_agents = game_matrix.ndim  # Number of players (rank of game tensor)

        # Dummy placeholders for logging
        self.VFE = [0.0] * num_agents
        self.energy = [0.0] * num_agents
        self.entropy = [0.0] * num_agents
        self.accuracy = [0.0] * num_agents
        self.complexity = [0.0] * num_agents
        self.o_pred_record = torch.zeros((self.num_agents, self.num_actions))

        # --- Expected Free Energy ---
        self.EFE = torch.zeros(num_actions)
        self.EFE_terms = torch.zeros((1, 5))  # match Agent shape

        # --- Scalars ---
        self.gamma = 0.0
        self.learn_record = 0

        # --- Beliefs ---
        self.q_u = torch.tensor(action)

        self.q_s = torch.ones((num_agents, num_states)) / num_states

        # --- Generative model placeholders ---
        self.A = torch.zeros((num_states, num_actions))
        self.B = torch.zeros((num_agents, num_actions, num_actions, num_actions))

        # --- BMR-related ---
        self.B_candidates = "None"
        self.B_model_weights = torch.zeros(self.num_agents, len(self.B_candidates.split()))
        self.delta_F = torch.zeros(1)


    def set_log_C(self, game_matrix):
        # Dummy agent does not care about preferences
        self.log_C = game_matrix

    def select_action(self):
        self.u = torch.argmax(self.q_u).item()

        #print("Dummy:", self.q_u, self.u)

        return self.u

    def infer_state(self, o):
        # No inference
        return self.q_s

    def learn(self):
        # No learning
        self.learn_record = 0
