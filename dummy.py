'''
Canonical Dummy Opponent with fixed strategy:
TFT, Grim, Pavlov

Authors: Hanchen Wang
Date: 2026-01
'''
import torch

class DummyAgent:
    def __init__(self, id, strategy, game_matrix=None, num_states=2, **kwargs):
        self.id = id
        self.is_dummy = True
        self.strategy = strategy

        # --- Game structure ---
        self.log_C = game_matrix
        self.game_matrix = game_matrix.to(torch.float)
        self.num_agents = game_matrix.ndim
        self.num_actions = game_matrix.shape[0]
        self.num_states = num_states

        # --- Time & memory ---
        self.t = 0
        self.u = None

        self.last_opponent_action = None
        self.last_own_action = None
        self.last_payoff = None

        # --- Grim ---
        self.defect_streak = 0
        self.grim_threshold = 50
        self.grim_triggered = False

        # --- Logging placeholders ---
        self.VFE = [0.0] * self.num_agents
        self.energy = [0.0] * self.num_agents
        self.entropy = [0.0] * self.num_agents
        self.accuracy = [0.0] * self.num_agents
        self.complexity = [0.0] * self.num_agents
        self.o_pred_record = torch.zeros((self.num_agents, self.num_actions))

        # --- Expected Free Energy ---
        self.EFE = torch.zeros(self.num_actions)
        self.EFE_terms = torch.zeros((1, 5))

        # --- Scalars ---
        self.gamma = 0.0
        self.learn_record = 0

        # --- Beliefs ---
        self.q_u = torch.zeros(self.num_actions)
        self.q_s = torch.ones((self.num_agents, self.num_states)) / self.num_states

        # --- Generative model placeholders ---
        self.A = torch.zeros((num_states, self.num_actions))
        self.B = torch.zeros((self.num_agents, self.num_actions, self.num_actions, self.num_actions))

        # --- BMR-related ---
        self.B_candidates = "None"
        self.B_model_weights = torch.zeros(self.num_agents, 1)
        self.delta_F = torch.zeros(1)

    # ------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------

    def set_log_C(self, game_matrix):
        self.log_C = game_matrix

    def select_action(self):
        """
        Fixed strategy action selection.
        """
        if self.strategy == "Cooperator":
            self.u = 0  # Always cooperate

        elif self.strategy == "Defector":
            self.u = 1  # Always defect

        elif self.strategy == "TFT":
            if self.t == 0 or self.last_opponent_action is None:
                self.u = 0  # Inital cooperator
            else:
                self.u = self.last_opponent_action

        elif self.strategy == "Grim":
            if self.grim_triggered:
                self.u = 1  # Defect forever
            else:
                self.u = 0  # Cooperate

        elif self.strategy == "Pavlov":
            if self.t == 0 or self.last_payoff is None:
                self.u = 0  # Cooperate initially
            else:
                win = self.last_payoff >= self.game_matrix.mean()
                if win:
                    self.u = self.last_own_action
                else:
                    self.u = 1 - self.last_own_action

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Deterministic policy posterior
        self.q_u.zero_()
        self.q_u[self.u] = 1.0

        self.t += 1
        self.last_own_action = self.u

        #print(f"dummy {self.id} q_u:", self.q_u)

        return self.u

    def infer_state(self, o, u_all):
        """
        Observe opponent action and payoff from last round.
        """
        opponent_id = 1 - self.id
        if self.t > 0:
            self.last_opponent_action = u_all[opponent_id]


        # Compute payoff for Pavlov
        joint_action = tuple(torch.argmax(o_i).item() for o_i in o)
        self.last_payoff = self.log_C[joint_action].item()

        # --- Update opponent defection streak ---
        if self.last_opponent_action == 1:
            self.defect_streak += 1
        else:
            self.defect_streak = 0

        # --- Trigger Grim only after persistent defection ---
        if self.strategy == "Grim" and self.defect_streak >= self.grim_threshold:
            self.grim_triggered = True

        return self.q_s

    def learn(self):
        # No learning
        self.learn_record = 0
