'''
Factorised Active Inference Agent

Authors: Jaime Ruiz Serra, Patrick Sweeney, Mike Harré
Date: 2024-07

Extended by: Hanchen Wang
Date: 2026-03
'''

import torch
import random
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from typing import Union

# import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# PYTORCH_ENABLE_MPS_FALLBACK = 1
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# torch.set_default_device(device)
torch.set_printoptions(precision=2)

EPSILON = torch.finfo().eps
TOLERANCE = 1e-5


class Agent:

    def __init__(
            self,
            id=None,
            game_matrix=None,
            # Perception
            interoception: bool = False,
            inference_num_iterations: int = 10,
            inference_num_samples: int = 100,
            inference_learning_rate: float = 1e-2,
            # Planning/action selection
            policy_length: int = 1,
            compute_novelty: bool = False,
            deterministic_actions: bool = False,
            # Precision
            dynamic_precision: bool = True,
            beta_0: float = 10.,
            beta_1: float = 10.,
            # Learning
            A_prior: Union[torch.Tensor, float] = 99,  # identity
            B_prior: Union[torch.Tensor, float] = 0,  # uniform
            D_prior: Union[torch.Tensor, None] = None,
            E_prior: Union[torch.Tensor, None] = None,
            A_learning: bool = False,
            B_learning: bool = True,
            B_learning_rate: Union[float, None] = 0.05,
            learn_every_t_steps: int = 24,
            learning_offset: int = 6,
            # decay: float = 0.5,
            # threshold: float = 0.25,
            A_BMR: Union[str, None] = 'identity',
            B_BMR: Union[str, None] = 'epsilon',
            alpha_r: float = 0.25,
            gamma_r: float = 1,
            B_candidates="Full",
            epistemic_gain: float = 1.0,
    ):
        """Initialise an agent with the following parameters

        Args:
            - id (int): The identifier of the agent
            - game_matrix (torch.Tensor): The game matrix (payoffs) of the game
            - interoception (bool): Whether the agent can perceive its own hidden state
            - inference_num_iterations (int): The number of variational optimisation iterations
            - inference_num_samples (int): The number of variational optimisation samples
            - inference_learning_rate (float): The learning rate for variational optimisation
            - compute_novelty (bool): Whether to compute novelty
            - deterministic_actions (bool): Whether the agent takes deterministic actions
            - dynamic_precision (bool): Whether the agent updates the precision
            - beta_0 (float): The beta_0 hyperparameter
            - beta_1 (float): The beta_1 hyperparameter
            - decay (float): The decay rate determines how peaky or flat the softmax-reduced prior becomes.
            - A_prior (Union[torch.Tensor, float]): The prior for the observation model. If float, the strength of softmax (0 -> uniform, inf -> identity)
            - B_prior (Union[torch.Tensor, float]): The prior for the transition model. If float, the strength of softmax (0 -> uniform, inf -> identity)
            - A_learning (bool): Whether the agent learns the observation model
            - B_learning (bool): Whether the agent learns the transition model
            - B_learning_rate (Union[float, NoneType]): The learning rate for the transition model
            - learn_every_t_steps (int): The length of the interval for learning
            - A_BMR (Union[str, NoneType]): The Bayesian Model Reduction method for the observation model. One of ['identity', 'uniform', None]
            - B_BMR (Union[str, NoneType]): The Bayesian Model Reduction method for the transition model. One of ['identity', 'uniform', None]
            - E_prior (Union[torch.Tensor, NoneType]): The prior for the habits
            - theta_prior (Union[torch.Tensor, NoneType]): The prior for the initial state
        """
        self.id = id
        self.is_dummy = False

        # Generative model hyperparameters -------------------------------------
        self.game_matrix = game_matrix.to(torch.float)  # Rewards from row player's perspective (force to float)
        self.num_actions = num_actions = game_matrix.shape[0]  # Number of actions (assuming symmetrical actions)
        self.num_agents = num_agents = game_matrix.ndim  # Number of players (rank of game tensor)

        # Generative model parameters ------------------------------------------

        # A matrix encodes the likelihood: A[o,s]=P(o∣s).
        # The probability of observing o given that the true hidden state is s.
        # Is used to create a default prior where each agent strongly expects observation o when in state s = o
        if isinstance(A_prior, torch.Tensor):
            # If the user passes in a specific tensor (a custom observation model), use that directly as A_params.
            self.A_params = A_prior
        elif isinstance(A_prior, (int, float)):
            # If a scalar value (e.g. 1.0) is passed in instead of a matrix: It assumes an identity-like prior—each
            # observation maps mostly to the corresponding hidden state. A_prior * torch.eye(num_actions) creates a
            # diagonal matrix scaled by the value (so higher values → stronger identity belief). softmax makes each row
            # a probability distribution over observations. This is repeated for each agent.
            self.A_params = torch.stack([
                torch.softmax(A_prior * torch.eye(num_actions) + EPSILON, dim=-1)
                for _ in range(num_agents)])  # Identity observation model prior
        else:
            raise ValueError(f"Invalid A_prior: {A_prior}")

        # Normalize A_params over observations to ensure that each column (per state) sums to 1. This makes sure A is
        # a valid probability distribution over observations for each state and agent.
        self.A = self.A_params / self.A_params.sum(dim=1, keepdim=True)

        # B defines the transition probability of the hidden state given an action: B[u,s′,s]=P(s′∣s,u).
        # It models how the hidden state evolves when the agent takes action u.
        if isinstance(B_prior, torch.Tensor):
            self.B_params = B_prior
        elif isinstance(B_prior, (int, float)):
            # If a scalar B_prior is given: Create a soft identity transition matrix: strong prior that s' ≈ s when
            # action u is taken. softmax(B_prior * eye + EPSILON) builds a transition model where action u tends to
            # leave the state unchanged. This is repeated for each action and each agent.
            self.B_params = torch.stack([
                torch.softmax(B_prior * torch.eye(num_actions) + EPSILON, dim=-1)
                for _ in range(num_actions)
                for _ in range(num_agents)
            ]).reshape(num_agents, num_actions, num_actions,
                       num_actions)  # shape: (funk): factor, action (u), next state, kurrent state
        else:
            raise ValueError(f"Invalid B_prior: {B_prior}")
        self.B = self.B_params / self.B_params.sum(dim=2, keepdim=True)

        self.set_log_C(game_matrix)  # Log preference over observations (payoffs)

        # theta is the Dirichlet prior over hidden states for each agent.
        # If none is provided, it defaults to a uniform prior ([1, 1, ..., 1]).
        self.theta = [torch.ones(num_actions) for _ in
                      range(num_agents)] if D_prior is None else D_prior  # Dirichlet state prior

        # D (and q_s) are initialized as the expected categorical distributions from the Dirichlet prior.
        # This gives the initial belief over states.
        self.D = self.q_s = torch.stack([Dirichlet(theta).mean for theta in self.theta])  # Categorical state prior (D)

        # E is the prior over policies, i.e., habits or learned tendencies.
        # If not set, it's initialized as uniform over all possible action sequences of length policy_length.
        self.E = torch.ones(num_actions ** policy_length) / num_actions if E_prior is None else E_prior  # Habits

        # Learning parameters --------------------------------------------------
        # Precision: How confident agent is about its beliefs and how strongly it updates them with new observations.
        self.dynamic_precision = dynamic_precision  # Enables precision updating
        self.beta_0 = beta_0  # beta in Gamma distribution (stays fixed). Scales how fast the agent forgets or weighs new evidence.
        self.beta_1 = beta_1  # alpha in Gamma distribution (sets the prior precision mean). Related to how much data you’ve seen (confidence).
        self.gamma = self.beta_1 / self.beta_0  # Current precision. Higher gamma, more confident less exploratory. Lower gamma, more uncertainty and more exploration.
        # Perception: How it forms beliefs about hidden states and outcomes based on observations.
        self.interoception = interoception  # Ego can perceive its own hidden state
        self.inference_num_iterations = inference_num_iterations  # Number of updates for variational inference
        self.inference_num_samples = inference_num_samples  # Samples for approximating expectations (Monte Carlo)
        self.inference_learning_rate = inference_learning_rate  # Step size for inference optimization
        # Planning
        self.compute_novelty = compute_novelty
        self.deterministic_actions = deterministic_actions
        self.policy_length = policy_length  # Length of the policy (number of actions to consider)
        # Learning
        self.learn_record = 0  # Store step indices where learning happened (0 if not a learning step)
        # self.A_model_change = torch.zeros(self.num_agents,
        #                                   dtype=torch.float,
        #                                   device=self.B_params.device)  # Track the per-factor change of observation model
        # self.B_model_change = torch.zeros(self.num_agents,
        #                                   dtype=torch.float,
        #                                   device=self.B_params.device)  # Track the per-factor change of transition model
        self.learn_every_t_steps = learn_every_t_steps  # Learn every t steps
        self.learning_offset = learning_offset  # Random offset for learning
        self.current_random_offset_learning = random.randint(-self.learning_offset,
                                                             self.learning_offset)  # Random offset for learning
        self.A_learning = A_learning  # Learn the observation model
        self.B_learning = B_learning  # Learn the transition model
        self.B_learning_rate = B_learning_rate
        self.A_BMR = A_BMR
        self.B_BMR = B_BMR
        # self.threshold = threshold  # Value lower than this quantile will be set to epsilon (hard-pruning)
        # self.decay = decay  # Forgetting rate for learning
        self.alpha_r = alpha_r  # Strength parameter for BMR
        self.gamma_r = gamma_r  # Softmax strength for BMA
        self.B_candidates = B_candidates
        self.delta_F = torch.zeros(self.num_agents, len(B_candidates.split()),  # num of B models in BMA
                                   dtype=torch.float32,
                                   device=self.B_params.device)  # log p(y|M_red) - log p(y|M_full)
        self.B_model_weights = torch.zeros(self.num_agents, len(B_candidates.split()),  # num of B models in BMA
                                           dtype=torch.float32, device=self.B_params.device)

        self.delta_F_A = torch.zeros(self.num_agents,
                                     dtype=torch.float32,
                                     device=self.A_params.device)  # log p(y|M_red) - log p(y|M_full)
        # self.F_red = torch.zeros((self.num_agents, self.num_actions),
        #                          dtype=torch.float32,
        #                          device=self.B_params.device)  # F(M_red) ≈ log p(y|M_red)
        # self.F_full = torch.zeros((self.num_agents, self.num_actions),
        #                           dtype=torch.float32,
        #                           device=self.B_params.device)  # F(M_full) ≈ log p(y|M_full)
        # self.weight_full = torch.zeros((self.num_agents, self.num_actions),
        #                                dtype=torch.float32,
        #                                device=self.B_params.device)  # Weight for full model
        # self.pruning_method = pruning_method  # Define the type to prune the full model (softmax / epsilon)

        # Store blanket states history for learning ----------------------------
        self.o_history = []  # Observations (opponent actions) history
        self.q_s_history = []  # Beliefs (opponent policies) history
        self.u_history = []  # Actions (ego) history

        # Agents values (change at each time step) -----------------------------
        self.u = torch.multinomial(self.E, 1).item()  # Starting action randomly sampled according to habits

        self.VFE = [None] * num_agents  # Variational Free Energy for each agent (state factor)
        self.accuracy = [None] * num_agents
        self.complexity = [None] * num_agents
        self.energy = [None] * num_agents
        self.entropy = [None] * num_agents
        # q(s_j|u_i) Posterior predictive state distribution (for each factor) conditional on action u_i
        self.q_s_u = torch.empty((num_agents, num_actions, num_actions))  # shape (n_agents, n_actions, n_actions)

        self.EFE = torch.zeros(num_actions)  # Expected Free Energy (for each possible action)
        self.EFE_terms = torch.zeros((num_actions ** policy_length, policy_length,
                                      5))  # Store ambiguity, risk, salience, pragmatic value, novelty for the current time step
        self.q_u = torch.zeros(num_actions)  # q(u_i) Policy of ego (self) agent

        self.expected_EFE = None  # Expected EFE (under current policy q(u)): <G> = E_q(u)[ G[u] ]

        self.o_pred_record = torch.zeros((self.num_agents, self.num_actions))  # Shape: (n_agents, n_actions)

        self.epistemic_gain = epistemic_gain

    def set_log_C(self, game_matrix):
        '''Set the log preference over observations (payoffs)

        Args:
            game_matrix (torch.Tensor): The game matrix (payoffs) of the game, of shape (n_actions, ) * n_agents
        '''
        # Normalise log_C into a proper probability distribution
        # This is not strictly necessary, but it ensures the EFE values are positive
        flattened_C = torch.softmax(game_matrix.to(torch.float).flatten(), dim=0)
        C = flattened_C.view_as(game_matrix)
        self.log_C = torch.log(C + EPSILON)
        # assert torch.isclose(torch.exp(self.log_C).sum(), torch.tensor(1.0), atol=TOLERANCE), (
        #     "The sum of exponentiated values of log C is not 1."
        # )

    # ==========================================================================
    # Summaries
    # ==========================================================================

    def __repr__(self):
        return (f"Agent(id={self.id!r}, beta_1={self.beta_1},"
                f"gamma={self.gamma:.3f}, game_matrix_shape={self.game_matrix.shape}, "
                f"action={self.u}, VFE={self.VFE}, expected_EFE={self.expected_EFE})")

    def __str__(self):
        # Function to format a tensor for readable output
        def format_tensor(tensor):
            # Recursively formats the tensor depending on its number of dimensions
            if tensor.dim() == 1:
                # If 1D, format the elements directly
                return '\t'.join([f'{item:.2f}' for item in tensor])
            else:
                # If multidimensional, apply the formatting to each slice along the first dimension
                return '\n'.join([format_tensor(tensor_slice) for tensor_slice in tensor])

        # Format the game matrix and opponent model parameters
        formatted_game_matrix = format_tensor(self.log_C)

        # Format the state priors as percentages, with labels for each state factor
        formatted_state_priors = [
            f"State Factor {i + 1} Estimate: {', '.join([f'{prob * 100:.2f}%' for prob in state])}"
            for i, state in enumerate(self.q_s)
        ]

        total_vfe = sum(vfe for vfe in self.VFE if vfe is not None)
        total_accuracy = sum(acc for acc in self.accuracy if acc is not None)
        total_complexity = sum(comp for comp in self.complexity if comp is not None)
        total_energy = sum(en for en in self.energy if en is not None)
        total_entropy = sum(ent for ent in self.entropy if ent is not None)

        # Format the VFE values to 2 decimal places
        formatted_VFE = [f'{vfe:.2f}' if vfe is not None else 'None' for vfe in self.VFE]

        # Format the expected EFE values to 2 decimal places
        formatted_expected_EFE = [f'{efe:.2f}' if efe is not None else 'None' for efe in self.expected_EFE]

        # Format the EFE values per action to 2 decimal places
        formatted_EFE = [f'{efe:.2f}' for efe in self.EFE]

        # Format the additional per action metrics (ambiguity, risk, salience, pragmatic value)
        formatted_ambiguity = [f'{amb:.2f}' for amb in self.ambiguity]
        formatted_risk = [f'{risk:.2f}' for risk in self.risk]
        formatted_salience = [f'{sal:.2f}' for sal in self.salience]
        formatted_pragmatic_value = [f'{pv:.2f}' for pv in self.pragmatic_value]

        summary = (f"Agent ID: {self.id}\n"
                   f"Gamma: {self.gamma:.3f}\n"
                   f"Gamma Hyperparameters: beta=1.0, alpha={self.beta_1}\n"
                   f"Current Action: {self.u}\n"
                   f"Log C (Payoffs):\n{formatted_game_matrix}\n"
                   f"{', '.join(formatted_state_priors)}\n"  # Join the state estimates on a single line
                   f"Habits E: {', '.join([f'{prob * 100:.2f}%' for prob in self.E])}\n"

                   f"Total Variational Free Energy (VFE): {total_vfe:.2f}, Per Factor: {', '.join(formatted_VFE)}\n"
                   f"Accuracy: {total_accuracy:.2f}\n"
                   f"Complexity: {total_complexity:.2f}\n"
                   f"Energy: {total_energy:.2f}\n"
                   f"Entropy: {total_entropy:.2f}\n"

                   f"Expected EFE: {', '.join(formatted_expected_EFE)} (Per Action: {', '.join(formatted_EFE)})\n"
                   f"Ambiguity: {', '.join(formatted_ambiguity)}\n"
                   f"Risk: {', '.join(formatted_risk)}\n"
                   f"Salience: {', '.join(formatted_salience)}\n"
                   f"Pragmatic Value: {', '.join(formatted_pragmatic_value)}")

        return summary

    # ==========================================================================
    # Perception
    # ==========================================================================

    def infer_state(self, o):
        '''
        Infer the hidden state of each agent (factor_idx) (i.e. the probability
        distribution over actions of each agent) given the observation `o`
        (i.e. the action taken by each agent).

        Employs self.A (observation model) and self.B (transition model),
        and updates the variational parameters self.theta for each agent
        through a Monte Carlo approximation of the variational free energy.

        Args:
            o (torch.Tensor): Observation tensor of shape (n_agents, n_actions),
                i.e. one-hot encoding of actions for each agent

        Returns:
            s (torch.Tensor): Hidden state tensor of shape (n_agents, n_actions)
        '''
        if self.interoception:
            '''
            If proprioception is enabled, ego can perceive the true hidden state for the ego factor, 
            i.e. q(s_i) = q(u_i)
            and will have to infer the hidden states of the other agents ("theory of mind")
            '''
            # Ego factor
            factor_idx = 0
            self.q_s[factor_idx] = self.q_u.clone().detach()  # Ego factor is the previous timestep's policy
            self.theta[factor_idx] = torch.zeros_like(self.theta[factor_idx])  # PLACEHOLDER
            self.VFE[factor_idx] = 0  # PLACEHOLDER
            self.entropy[factor_idx] = 0  # PLACEHOLDER
            self.energy[factor_idx] = 0  # PLACEHOLDER
            self.accuracy[factor_idx] = 0  # PLACEHOLDER
            self.complexity[factor_idx] = 0  # PLACEHOLDER
            # Alter factors
            factors = range(1, len(self.q_s))
        else:
            '''
            Otherwise, ego has to infer all hidden states including its own, 
            i.e. "introspection" (towards self) and "theory of mind" (towards others)
            '''
            factors = range(len(self.q_s))

        # Iterate over (remaining) factors
        for factor_idx in factors:
            s_prev = self.q_s[factor_idx].clone().detach()  # State t-1
            # assert torch.allclose(s_prev.sum(), torch.tensor(1.0)), "s_prev tensor does not sum to 1."
            log_prior = torch.log(self.B[factor_idx, self.u] @ s_prev + EPSILON)  # New prior is old posterior
            log_likelihood = torch.log(
                self.A[factor_idx].T @ o[factor_idx] + EPSILON)  # Likelihood of hidden states for a given observation

            variational_params = self.theta[factor_idx].clone().detach().requires_grad_(
                True)  # Variational Dirichlet distribution for each factor (agent)
            optimizer = torch.optim.Adam([variational_params], lr=self.inference_learning_rate)

            for _ in range(self.inference_num_iterations):
                optimizer.zero_grad()

                s_samples = Dirichlet(variational_params).rsample(
                    (self.inference_num_samples,))  # Variational dist samples
                log_s = torch.log(s_samples + EPSILON)  # Log variational dist samples
                vfe_samples = torch.sum(s_samples * (log_s - log_likelihood - log_prior), dim=-1)
                VFE = vfe_samples.mean()

                VFE.backward()
                optimizer.step()
                variational_params.data.clamp_(min=1e-3)

            # Results of variational inference: variational posterior, variational parameters, VFE
            self.q_s[factor_idx] = Dirichlet(variational_params).mean.detach()  # Variational posterior
            self.theta[factor_idx] = variational_params.detach()  # Store variational parameters (for next timestep)
            self.VFE[factor_idx] = VFE.detach()

            # Compute additional metrics (for validation and plotting)
            self.entropy[factor_idx] = entropy = -torch.sum(s_samples * log_s, dim=-1).mean().detach()
            self.energy[factor_idx] = energy = -torch.sum(s_samples * (log_prior + log_likelihood),
                                                          dim=-1).mean().detach()
            self.accuracy[factor_idx] = accuracy = torch.sum(s_samples * log_likelihood, dim=-1).mean().detach()
            self.complexity[factor_idx] = complexity = -torch.sum(s_samples * (log_prior - log_s),
                                                                  dim=-1).mean().detach()
            # assert torch.allclose(VFE, energy - entropy, atol=TOLERANCE), "VFE != energy + entropy"
            # assert torch.allclose(VFE, complexity - accuracy, atol=TOLERANCE), "VFE != complexity - accuracy"

        # Data collection (for learning and plotting)
        self.q_s_history.append(self.q_s)
        self.o_history.append(o)


        return self.q_s

    # ==========================================================================
    # Action
    # ==========================================================================

    def compute_efe(self, u, q_s_u, A, log_C, depth):
        '''
        Compute the Expected Free Energy (EFE) of a given action, respecting causal depth.
        Args:
            u (int): action
            q_s_u (torch.Tensor): variational posterior over states given action is u
            A (torch.Tensor): observation likelihood model
            log_C (torch.Tensor): log preference over observations

        Returns:
            EFE (torch.Tensor): Expected Free Energy for a given action
        '''
        EFE = 0
        ambiguity = torch.tensor(0.0)
        risk = torch.tensor(0.0)
        salience = torch.tensor(0.0)
        pragmatic_value = torch.tensor(0.0)
        ppo_entropy = torch.tensor(0.0)
        novelty = torch.tensor(0.0)

        # Predictive observation posterior -------------------------------------
        q_o_u = torch.einsum('fso,fs->fo', A, q_s_u)
        # Ego's guaranteed observation for proposed action
        q_o_u[0] = F.one_hot(u, self.num_actions).to(torch.float)

        # Joint predictive observation posterior ---------------------------
        einsum_str = (
                ','.join([chr(105 + i) for i in range(self.num_agents)])
                + '->'
                + ''.join([chr(105 + i) for i in range(self.num_agents)])
        )
        q_o_joint_u = torch.einsum(einsum_str, *[q_o_u[i] for i in range(self.num_agents)])

        # Pragmatic value term (Expected Reward)
        pragmatic_value = torch.tensordot(q_o_joint_u, log_C, dims=self.num_agents)

        # ------------------------------------------------------------------
        # CAUSAL ALIGNMENT: Depth 1 vs Depth > 1
        # ------------------------------------------------------------------
        if depth == 1:
            # Step 1 (Respond to the past): G_2(a_2)
            # The opponent's state is a product of our PAST action (a_1).
            # Our current proposal (u) cannot reduce uncertainty about it.
            # Thus, epistemic terms are constant w.r.t the policy and are dropped.
            risk = -pragmatic_value
            salience = torch.tensor(0.0)
        else:
            # Step 2+ (Shape the future): G_3(a_2, a_3)
            # We evaluate how our proposed actions reduce future ambiguity.
            for factor_idx in range(self.num_agents):
                s_pred = q_s_u[factor_idx]

                H = -torch.diag(A[factor_idx].T @ torch.log(A[factor_idx] + EPSILON))
                factor_ambiguity = (H @ s_pred)
                ambiguity += torch.clip(factor_ambiguity, min=0.0, max=None)

                o_pred = q_o_u[factor_idx]
                o_pred = o_pred[o_pred > 0]
                ppo_entropy += -torch.sum(o_pred * torch.log(o_pred))

            # Joint risk
            risk = torch.tensordot(
                q_o_joint_u,
                (torch.log(q_o_joint_u + EPSILON) - log_C),
                dims=self.num_agents
            )
            salience = ppo_entropy - ambiguity

        # Novelty ----------------------------------------------------------
        # For causal alignment: at depth==1, epistemic effects should not
        # depend on the candidate action, so we skip novelty there.
        if self.compute_novelty and depth > 1:
            if self.A_learning:
                novelty += self.compute_A_novelty(q_s_u, q_o_u)
            if self.B_learning:
                # Assuming u is passed as an integer/tensor index
                action_idx = u.item() if isinstance(u, torch.Tensor) else u
                novelty += self.compute_B_novelty(self.q_s, q_s_u, action_idx)

        # Final EFE summation
        # Epistemic gain scales information-seeking terms (salience + novelty).
        EFE = -pragmatic_value - self.epistemic_gain * (salience + novelty)

        return EFE.unsqueeze(0), torch.tensor((ambiguity, risk, salience, pragmatic_value, novelty)), q_o_u

    def compute_A_novelty_ALGEBRAIC(self, q_s_u, u):
        '''
        Compute the novelty of the likelihood model A for action u_i
        '''
        novelty = 0
        # Add a small constant to A_params to avoid W blowing up
        A_params = self.A_params.clone().detach() + 0.5  # FIXME: softmax??
        # A_params = self.A.clone().detach()
        # Da Costa et al. (2020; Eq. D.17)
        W = 0.5 * (1 / A_params - 1 / A_params.sum(dim=1, keepdim=True))
        for factor_idx in range(self.num_agents):
            s_pred = q_s_u[factor_idx]
            novelty += torch.dot(
                self.A[factor_idx] @ s_pred,
                W[factor_idx] @ s_pred)
        return novelty

    def compute_A_novelty(self, q_s_u, q_o_u):

        # A' = A + q_o_u x q_s_u
        A_prime_params = self.A_params + torch.einsum(
            'fs,fo->fos',  # (f, o, s): factor, observation, state
            q_s_u,  # (f, s)
            q_o_u  # (f, o)
        )
        A_prime = A_prime_params / A_prime_params.sum(dim=1, keepdim=True)
        # TODO: account for BMR?

        # KL divergence D[ A' || A ] for each factor
        kl_div = (
                A_prime * torch.log(A_prime / (self.A + EPSILON) + EPSILON)
        ).sum(dim=(1, 2))

        return kl_div.sum()  # over factors

    def compute_B_novelty(self, q_s, q_s_u, u):

        # B'[u] = B[u] + q_s_u x q_s
        outer_product = torch.einsum(
            'fn,fk->fnk',  # (f, n, k): factor, next (state), kurrent (state)
            q_s_u,  # (f, n)
            q_s  # (f, k)
        )
        B_prime_params = self.B_params[:, u].squeeze() + self.B_learning_rate * outer_product
        B_prime = B_prime_params / B_prime_params.sum(dim=1, keepdim=True)

        # KL divergence D[ B'[u] || B[u] ] for each factor
        kl_div = (
                B_prime * torch.log(B_prime / (self.B[:, u].squeeze() + EPSILON) + EPSILON)
        ).sum(dim=(1, 2))

        return kl_div.sum()  # over factors

    def collect_policies(
            self,
            node,
            q_s,
            policy_EFEs=None,
            policy_EFE_terms=None,
            current_policy=None,
    ):
        '''Function to traverse the tree and collect policies (as tensors)'''

        # Root node case
        if node.u is None:
            node.q_s_u = q_s
            current_policy = []
            policy_EFEs = []
            new_policy_EFEs = policy_EFEs
        # Other nodes
        else:
            # FIX 1: The incoming q_s has ALREADY been transitioned using the causally
            # correct action from the previous step. We assign it directly.
            node.q_s_u = q_s
            # pass node.depth into the EFE computation.
            node.EFE_u, node.EFE_terms, node.q_o_u = self.compute_efe(node.u, node.q_s_u, self.A, self.log_C,
                                                                      node.depth)
            new_policy_EFEs = policy_EFEs + [node.EFE_u]  # EFEs collected top-down

            # EFE terms collected top-down
            if policy_EFE_terms is None:
                policy_EFE_terms = node.EFE_terms.unsqueeze(0)
            else:
                policy_EFE_terms = torch.vstack((policy_EFE_terms, node.EFE_terms.unsqueeze(0)))

        # Base case (leaf node)
        if not node.children:
            return [torch.cat(new_policy_EFEs)], policy_EFE_terms, [torch.cat(current_policy)]

        # Recursive case
        EFEs = []
        EFE_terms = []
        policies = []

        # FIX 2: Determine which action causes the state transition for the NEXT step.
        # Root node: the real previous action (self.u) causes the next state.
        # Deeper nodes: the action proposed at this current node (node.u) causes the next state.
        causal_action = self.u if node.u is None else node.u.item()

        # Pre-compute the transitioned state for the children
        # Slicing with an integer removes the action dimension seamlessly, yielding shape (f, n)
        next_q_s = torch.einsum(
            'funk,fk->fun',
            self.B,
            node.q_s_u
        )[:, causal_action]

        for child in node.children:
            new_policy = current_policy + [child.u]  # Policies collected bottom-up

            # Pass the causally correct next state down the tree
            subtree_EFEs, subtree_EFE_terms, sub_policy = self.collect_policies(
                child, next_q_s, new_policy_EFEs, policy_EFE_terms, new_policy)

            EFEs.extend(subtree_EFEs)
            EFE_terms.extend(subtree_EFE_terms)
            policies.extend(sub_policy)

        return torch.vstack(EFEs), torch.vstack(EFE_terms), torch.vstack(policies)


    def select_action(self):

        # Build the policy tree and "collect" policies
        root = build_policy_tree(torch.arange(self.num_actions), self.policy_length)
        EFEs, EFE_terms, policies = self.collect_policies(
            root,
            q_s=self.q_s
        )

        EFE_policies = EFEs.sum(dim=1)
        q_u = torch.softmax(
            torch.log(self.E) - self.gamma * EFE_policies,
            dim=0
        )

        # EFE = self.compute_efe()
        # q_u = torch.softmax(torch.log(self.E) - self.gamma * EFE, dim=0)
        # assert torch.allclose(q_u.sum(), torch.tensor(1.0)), (
        #     "q_u policy tensor does not sum to 1.",
        #     f"q_u: {q_u}",
        #     f"q_u.sum(): {q_u.sum()}",
        #     f"EFE: {EFE_policies}"
        # )
        # Data collection ------------------------------------------------------
        self.EFE = EFE_policies
        self.EFE_terms = EFE_terms.reshape(*EFEs.shape, -1)
        self.q_u = q_u

        # Select action
        policy_idx = torch.multinomial(q_u, 1).item() if not self.deterministic_actions else torch.argmax(q_u).item()
        self.u = policies[policy_idx][0].item()
        # self.u = torch.multinomial(q_u, 1).item() if not self.deterministic_actions else torch.argmax(q_u).item()
        self.u_history.append(self.u)

        # Retrieve q_o_u for the selected action from the policy tree
        for child in root.children:
            if child.u == self.u:
                # Store the current predicted observation (shape: n_agents x n_actions)
                self.o_pred_record = child.q_o_u.detach().clone()
                break

        if self.dynamic_precision:
            self.update_precision(EFE_policies, q_u)

        #print(f"Agent {self.id}:", "q_u", self.q_u, "u", self.u)

        return self.u

    def update_precision(self, EFE, q_u):
        # Compute the expected EFE as a scalar value
        self.expected_EFE = torch.dot(q_u, EFE).item()

        # Update gamma (the precision) based on the expected EFE
        self.gamma = self.beta_1 / (self.beta_0 - self.expected_EFE)

        return self.gamma

    # ==========================================================================
    # Learning
    # ==========================================================================

    def learn(self):

        # Learn every t steps
        if len(self.u_history) == (self.learn_every_t_steps + self.current_random_offset_learning):
            # Record the step index of this learning
            self.learn_record = self.learn_every_t_steps + self.current_random_offset_learning

            # Convert history to tensors
            self.q_s_history = torch.stack(self.q_s_history)  # Shape: (T, n_agents, n_actions)
            self.o_history = torch.stack(self.o_history)  # Shape: (T, n_agents, n_actions)
            self.u_history = torch.tensor(self.u_history)  # Shape: (T, )

            # Learn
            if self.A_learning:
                self.learn_A()
            if self.B_learning:
                self.learn_B()

            # Reset history
            self.q_s_history = []
            self.o_history = []
            self.u_history = []
            self.current_random_offset_learning = random.randint(-self.learning_offset,
                                                                 self.learning_offset)  # Renew random offset

        else:
            # Clear the record at this step
            self.learn_record = 0

    def learn_A(self):

        # Perform the row-wise outer product
        outer_products = torch.einsum(  # Compute outer products
            'tfs,tfo->tfos',  # t (time), f (factor), s (state), o (observation)
            self.q_s_history,
            self.o_history
        )  # Shape: (T, n_agents, n_actions, n_actions)

        # Posterior parameters
        delta_params = outer_products.mean(dim=0)  # Shape: (n_agents, n_actions, n_actions)
        A_posterior_params = self.A_params + delta_params  # Shape: (n_agents, n_actions, n_actions)

        # Bayesian Model Reduction ---------------------------------------------
        if self.A_BMR:
            for factor_idx in range(self.num_agents):
                # Compute reduced posterior (BMR identity)
                a_prior = self.A_params[factor_idx].flatten()
                a_post_full = A_posterior_params[factor_idx].flatten()

                # Choose method to produce candidate reduced model
                a_red = make_reduced_prior(
                    a_prior,
                    method=self.A_BMR,  # "epsilon" or "softmax"
                    alpha=self.alpha_r,
                    preserve_total=False  # Friston-style: do NOT preserve total by default
                )
                # print(a_prior, a_red)
                a_red_candidates = [
                    a_prior,  # First candidate is always the full model
                    # a_red,
                    # torch.ones_like(a_prior) * a_prior.mean(),  # Uniform baseline
                    # torch.tensor([1., 0., 0., 1.]) + EPSILON,
                    # torch.tensor([0., 1., 1., 0.]) + EPSILON,
                    # torch.tensor([1., 1., 0., 0.]) + EPSILON,
                    torch.tensor([1., 0., 1., 0.]) + EPSILON,
                    # torch.tensor([0., 0., 1., 1.]) + EPSILON,
                ]

                # Compute difference in log evidence F(M_red) - F(M_full)
                # Friston et al. (2016, Bayesian model reduction, Equation 12)
                delta_F_vector = torch.tensor([
                    delta_free_energy(a_post_full, a_prior, a_red_i)
                    for a_red_i in a_red_candidates
                ])
                assert delta_F_vector[0].abs() < 1e-4, f"Delta F for full model ({delta_F_vector[0]}) should be zero."
                delta_E = 0  # torch.log(prior_red[factor_idx, action_idx] / prior_full[factor_idx, action_idx])

                # Compute weight for two models, Can be derived from:
                # Bayes' Theorem / Friston et al. (2016, Active Inference and learning, Equation 1.e)
                # weight_full = torch.sigmoid(-(self.gamma_r * delta_F + delta_E))
                # weight_red = 1 - weight_full
                weight_vector = torch.softmax(self.gamma_r * delta_F_vector, dim=0)
                a_post_candidates = torch.stack([
                    (a_post_full + a_red_i - a_prior).clamp_min(EPSILON)
                    for a_red_i in a_red_candidates
                ], dim=0)

                # assert torch.isclose(weight_full + weight_red, torch.tensor(1.0), atol=TOLERANCE), (
                #     "The weights of full and reduced model should sum to 1!")

                # Track updates
                # self.B_model_change[factor_idx, action_idx] = weight_red.detach().item()
                self.delta_F_A[factor_idx] = delta_F_vector[1].detach().item()
                # self.F_full[factor_idx, action_idx] = F_full.detach().item()
                # self.F_red[factor_idx, action_idx] = F_red.detach().item()
                # self.weight_full[factor_idx, action_idx] = weight_full.detach().item()
                # print("Weight_red:", self.B_model_change[factor_idx, action_idx],
                #       "\tDeltaF:", delta_F, "=", F_red, "-", F_full,
                #       )
                # print(factor_idx, action_idx, self.delta_F[factor_idx, action_idx])

                # BMA: Weighted average of concentration params
                self.A_params[factor_idx] = (
                        weight_vector @ a_post_candidates
                ).view_as(self.A_params[factor_idx])

        else:
            self.A_params = A_posterior_params

        self.A = self.A_params / self.A_params.sum(dim=1, keepdim=True)  # Shape: (n_agents, n_actions, n_actions)

    def learn_B(self):

        # Shift arrays for prev and next
        s_prev = self.q_s_history[:-1]  # Shape: (T-1, n_agents, n_actions)
        s_next = self.q_s_history[1:]  # Shape: (T-1, n_agents, n_actions)

        outer_products = torch.einsum(  # Compute outer products
            'tfn,tfk->tfnk',  # t (time), f (factor), n (next), k (kurrent)
            s_next,
            s_prev
        )  # Shape: (T-1, n_agents, n_actions, n_actions)
        T = outer_products.shape[0]

        # Update parameters for every transition (s, u, s') in the history
        B_posterior_params = self.B_params.clone()
        LEARNING_RATE = self.B_learning_rate if self.B_learning_rate is not None else 1 / T

        # Note: len(outer_products) = self.learn_every_t_steps + self.current_random_offset_learning
        for t in range(outer_products.shape[0]):
            # Likelihood parameters update
            delta_params = outer_products[t]  # Shape: (n_agents, n_actions, n_actions)
            u_it = self.u_history[t].item()  # Action u_i at time t
            B_posterior_params[:, u_it] = self.B_params[:, u_it] + LEARNING_RATE * delta_params
            # print("-------------")
            # print("t:", t)
            # print("B_posterior_params[:, u_it]", B_posterior_params[:, u_it])
            # print("delta_params:", delta_params)
            # print("u_it:", u_it)

        # Bayesian Model Reduction ---------------------------------------------
        if self.B_BMR:
            for factor_idx in range(self.num_agents):
                # Compute reduced posterior (BMR identity)
                a_prior = self.B_params[factor_idx].flatten()
                a_post_full = B_posterior_params[factor_idx].flatten()

                # Choose method to produce candidate reduced model
                a_red = make_reduced_prior(
                    a_prior,
                    method=self.B_BMR,  # "epsilon" or "softmax"
                    alpha=self.alpha_r,
                    preserve_total=False
                )

                a_red_candidates = [
                    a_prior,  # First candidate is always the full model
                    *[
                        candidate
                        for name, candidate in [
                            ("TFT", torch.tensor([[[0.99, 0.99],
                                                   [0.01, 0.01]],
                                                  [[0.01, 0.01],
                                                   [0.99, 0.99]]]).flatten()),
                            ("Grim", torch.tensor([[[0.99, 0.01],
                                                    [0.01, 0.99]],
                                                   [[0.01, 0.01],
                                                    [0.99, 0.99]]]).flatten()),
                            ("Pavlov", torch.tensor([[[0.99, 0.01],
                                                      [0.01, 0.99]],
                                                     [[0.01, 0.99],
                                                      [0.99, 0.01]]]).flatten()),
                            ("Reduce", a_red),
                        ]
                        if name in self.B_candidates
                    ]
                    # Other candidates can be added here:
                    # torch.ones_like(a_prior) * a_prior.mean(),  # Uniform model
                    # torch.tensor([1., 0., 0., 1.]) + EPSILON,
                    # torch.tensor([0., 1., 1., 0.]) + EPSILON,
                    # torch.tensor([1., 1., 0., 0.]) + EPSILON,
                    # torch.tensor([1., 0., 1., 0.]) + EPSILON,
                    # torch.tensor([0., 0., 1., 1.]) + EPSILON,
                ]

                # Compute difference in log evidence F(M_red) - F(M_full)
                # Friston et al. (2016, Bayesian model reduction, Equation 12)
                delta_F_vector = torch.tensor([
                    delta_free_energy(a_post_full, a_prior, a_red_i)
                    for a_red_i in a_red_candidates
                ])

                assert delta_F_vector[0].abs() < 1e-3, f"Delta F for full model ({delta_F_vector[0]}) should be zero."
                delta_E = 0  # torch.log(prior_red[factor_idx, action_idx] / prior_full[factor_idx, action_idx])

                # Compute weight for models
                weight_vector = torch.softmax(self.gamma_r * delta_F_vector, dim=0)
                a_post_candidates = torch.stack([
                    (a_post_full + a_red_i - a_prior).clamp_min(EPSILON)
                    for a_red_i in a_red_candidates
                ], dim=0)

                # Track updates
                self.delta_F[factor_idx, :] = delta_F_vector.detach()
                self.B_model_weights[factor_idx, :] = weight_vector.detach()

                # BMA: Weighted average of concentration params
                self.B_params[factor_idx] = (
                        weight_vector @ a_post_candidates
                ).view_as(self.B_params[factor_idx])

        else:
            self.B_params = B_posterior_params

        self.B = self.B_params / self.B_params.sum(dim=2, keepdim=True)


# ==============================================================================
# Helper functions
# ==============================================================================

def make_reduced_prior(
        a_prior: torch.Tensor,
        method: str = "epsilon",
        alpha: float = 0.1,
        preserve_total: bool = False,
) -> torch.Tensor:
    """
    Construct a reduced Dirichlet prior for a single conditional (1D Dirichlet vector).

    Args:
        a_prior (torch.Tensor): 1D tensor of prior concentration parameters (shape: K).
        method (str): Reduction method: "epsilon" (hard prune) or "softmax" (smooth sharpening).
        threshold (float): For "epsilon" method: quantile cutoff in [0,1] to prune small probs.
        strength (float): For "softmax" method: sharpening factor (STRENGTH > 1 -> sparser).
        preserve_total (bool): If True, rescale retained entries to preserve the original total
                               concentration (NOT Friston default). Default: False.

    Returns:
        a_red (torch.Tensor): 1D tensor of reduced Dirichlet concentrations (clamped >= EPSILON).
    Notes:
        - This function assumes `a_post_full` is 1D (one conditional row).
        - By default both methods produce a reduced total concentration (Friston-style).
          Set preserve_total=True to keep the original sum (Heuristic behaviour).
    """
    # Ensure 1D
    if a_prior.ndim != 1:
        raise ValueError("a_post_full must be a 1D tensor representing one Dirichlet vector.")
    assert alpha is not None, "Threshold must be specified."

    # Numerical safety
    a = a_prior.clone().detach().float().clamp_min(EPSILON)
    total = a.sum()

    # Normalize to probabilities (posterior -> p_post)
    p_prior = a / total.clamp_min(EPSILON)

    if method == "epsilon":
        # Hard-prune entries below quantile cutoff -> set to EPSILON
        cutoff = torch.quantile(p_prior, alpha)
        mask = p_prior < cutoff

        a_red = a.clone()
        a_red[mask] = EPSILON

        if preserve_total:
            # Rescale retained entries to preserve the original total concentration
            retained_mask = ~mask
            retained_sum = a_red[retained_mask].sum()
            if retained_sum > 0:
                a_red[retained_mask] = a_red[retained_mask] * (
                        total - EPSILON * mask.sum()
                ) / retained_sum

    elif method == "softmax":
        # Smooth sharpening in log-space
        p_red = F.softmax(alpha * torch.log(p_prior + EPSILON), dim=-1)

        # Friston-style: reduce concentration by multiplying with original concentrations,
        # so total concentration typically decreases when sharpening.
        a_red = (p_red * a).clone()

        if preserve_total:
            # If user requests preserve_total, rescale a_red to sum to original total
            sum_a_red = a_red.sum()
            if sum_a_red > 0:
                a_red = a_red * (total / sum_a_red)

    elif method == "flat":
        a_red = torch.ones_like(a_prior)

    elif method == "identity":
        a_red = torch.eye(a_prior.shape[0] // 2).view_as(a_prior)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'epsilon' or 'softmax'.")

    return a_red.clamp_min(EPSILON)


def log_MBF(alpha):
    """
    Compute the log of the multivariate Beta function (MBF), log B(alpha).
    Args:
    - alpha (torch.Tensor): 1D tensor of concentration parameters alpha (shape: K)
    Returns:
    - log B(alpha) (torch.Tensor): The computed log multivariate Beta function value
    """
    a = alpha.clamp_min(EPSILON)
    return torch.lgamma(a).sum() - torch.lgamma(a.sum())

    # gamma_sum = torch.lgamma(alpha.sum())  # log(Gamma(sum(alpha)))
    # gamma_individual = torch.lgamma(alpha).sum()  # sum(log(Gamma(alpha_i)))
    # return torch.exp(gamma_individual - gamma_sum)


def delta_free_energy(a_posterior, a_prior, a_reduced):
    """
    Compute the change in free energy (ΔF) using Bayesian Model Reduction.
    ΔF = log p(y|M_red) - log p(y|M_full), using Dirichlet-Categorical BMR identity:
    ΔF = log B(a_post) + log B(a_red) - log B(a_prior) - log B(a_post + a_red - a_prior)
    If positive, we prefer reduced model.

    Args:
    - a_prior (torch.Tensor): 1D tensor of prior concentration parameters alpha (shape: K)
    - a_posterior (torch.Tensor): 1D tensor of posterior concentration parameters alpha (shape: K)
    - a_reduced (torch.Tensor): 1D tensor of reduced concentration parameters alpha (shape: K)
    Returns:
    - delta_F (torch.Tensor): The change in free energy ΔF
    """

    a_post = a_posterior.clamp_min(EPSILON)
    a_prior = a_prior.clamp_min(EPSILON)
    a_reduced = a_reduced.clamp_min(EPSILON)

    # Combined posterior under reduced model
    a_comb = (a_post + a_reduced - a_prior).clamp_min(EPSILON)

    # Difference
    delta_F = (
            (log_MBF(a_comb) - log_MBF(a_reduced))
            - (log_MBF(a_post) - log_MBF(a_prior))
    )

    return delta_F


# ==============================================================================
# Policy tree
# ==============================================================================

class TreeNode:
    def __init__(self, action=None, depth=0):
        self.u = action  # Tensor or action at this node
        self.EFE_u = torch.tensor(0)
        self.children = []  # List to hold child nodes
        self.depth = depth  # Depth of the node

    def add_child(self, action):
        # Add a child node with the given action (as a tensor)
        child = TreeNode(action, self.depth + 1)
        self.children.append(child)
        return child

    def __repr__(self):
        return f"TreeNode(action={self.u}, depth={self.depth}, children={len(self.children)})"


def build_policy_tree(action_space, max_depth, node=None):
    '''Recursive function to build the tree'''

    if node is None:
        node = TreeNode()

    # Base case
    if node.depth == max_depth:
        return node

    # Add a child for each action in the action space
    # (recursively until the max depth is reached)
    for action in action_space:
        child = node.add_child(torch.tensor([action]))
        build_policy_tree(action_space, max_depth, node=child)

    return node
