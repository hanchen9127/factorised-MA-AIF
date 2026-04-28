'''
Factorised Active Inference Agent

Authors: Jaime Ruiz Serra, Patrick Sweeney, Mike Harré
Date: 2024-07

Extended by: Hanchen Wang
Date: 2026-04
'''

import torch
import random
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from typing import Union

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

        # A matrix (n_agents, n_actions, n_actions) encodes (factors, observations, states)
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
        self.E = torch.ones(num_actions ** policy_length) / (num_actions ** policy_length) if E_prior is None else E_prior  # Habits

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
        self.learn_every_t_steps = learn_every_t_steps  # Learn every t steps
        self.learning_offset = learning_offset  # Random offset for learning
        self.current_random_offset_learning = random.randint(-self.learning_offset,
                                                             self.learning_offset)  # Random offset for learning
        self.A_learning = A_learning  # Learn the observation model
        self.B_learning = B_learning  # Learn the transition model
        self.B_learning_rate = B_learning_rate
        self.A_BMR = A_BMR
        self.B_BMR = B_BMR
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

        # Store blanket states history for learning ----------------------------
        self.o_history = []  # Observations (opponent actions) history
        self.q_s_history = []  # Beliefs (opponent policies) history
        self.u_history = []  # Actions (ego) history

        # Agents values (change at each time step) -----------------------------
        # self.E has num_actions**policy_length entries (one per full policy sequence),
        # so torch.multinomial(self.E, 1) returns a *policy index* in [0, num_actions**policy_length),
        # which can exceed the valid action range [0, num_actions) when policy_length > 1.
        # self.u is used directly to index into B of shape (f, num_actions, num_actions, num_actions),
        # so it must be a valid action index. Sample uniformly over actions instead.
        self.u = torch.randint(0, num_actions, (1,)).item()
        self.u_prev = self.u  # Action taken at the *previous* timestep (a₁ when inferring at t+1)

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
        '''Set log preference over observations from game payoffs.'''
        flattened_C = torch.softmax(game_matrix.to(torch.float).flatten(), dim=0)
        C = flattened_C.view_as(game_matrix)
        self.log_C = torch.log(C + EPSILON)

    # ==========================================================================
    # Summaries
    # ==========================================================================

    def __repr__(self):
        return (f"Agent(id={self.id!r}, beta_1={self.beta_1},"
                f"gamma={self.gamma:.3f}, game_matrix_shape={self.game_matrix.shape}, "
                f"action={self.u}, VFE={self.VFE}, expected_EFE={self.expected_EFE})")

    def __str__(self):
        def format_tensor(tensor):
            if tensor.dim() == 1:
                return '\t'.join([f'{item:.2f}' for item in tensor])
            else:
                return '\n'.join([format_tensor(tensor_slice) for tensor_slice in tensor])

        formatted_game_matrix = format_tensor(self.log_C)
        formatted_state_priors = [
            f"State Factor {i + 1} Estimate: {', '.join([f'{prob * 100:.2f}%' for prob in state])}"
            for i, state in enumerate(self.q_s)
        ]

        total_vfe = sum(vfe for vfe in self.VFE if vfe is not None)
        total_accuracy = sum(acc for acc in self.accuracy if acc is not None)
        total_complexity = sum(comp for comp in self.complexity if comp is not None)
        total_energy = sum(en for en in self.energy if en is not None)
        total_entropy = sum(ent for ent in self.entropy if ent is not None)

        formatted_VFE = [f'{vfe:.2f}' if vfe is not None else 'None' for vfe in self.VFE]
        formatted_expected_EFE = [f'{efe:.2f}' if efe is not None else 'None' for efe in self.expected_EFE]
        formatted_EFE = [f'{efe:.2f}' for efe in self.EFE]
        formatted_ambiguity = [f'{amb:.2f}' for amb in self.ambiguity] if hasattr(self, 'ambiguity') else []
        formatted_risk = [f'{risk:.2f}' for risk in self.risk] if hasattr(self, 'risk') else []
        formatted_salience = [f'{sal:.2f}' for sal in self.salience] if hasattr(self, 'salience') else []
        formatted_pragmatic_value = [f'{pv:.2f}' for pv in self.pragmatic_value] if hasattr(self, 'pragmatic_value') else []

        return (f"Agent ID: {self.id}\n"
                f"Gamma: {self.gamma:.3f}\n"
                f"Gamma Hyperparameters: beta=1.0, alpha={self.beta_1}\n"
                f"Current Action: {self.u}\n"
                f"Log C (Payoffs):\n{formatted_game_matrix}\n"
                f"{', '.join(formatted_state_priors)}\n"
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

    # ==========================================================================
    # Perception
    # ==========================================================================

    def infer_state(self, o):
        '''Infer the hidden state of each factor via variational free energy minimisation.

        Args:
            o (torch.Tensor): One-hot observations, shape (n_agents, n_actions)
        Returns:
            q_s (torch.Tensor): Posterior beliefs, shape (n_agents, n_actions)
        '''
        if self.interoception:
            # Ego knows its own past action with certainty; only infer alter factors
            factor_idx = 0
            self.q_s[factor_idx] = F.one_hot(torch.tensor(self.u_prev), num_classes=self.num_actions).to(torch.float)
            self.theta[factor_idx] = torch.zeros_like(self.theta[factor_idx])
            self.VFE[factor_idx] = 0
            self.entropy[factor_idx] = 0
            self.energy[factor_idx] = 0
            self.accuracy[factor_idx] = 0
            self.complexity[factor_idx] = 0
            factors = range(1, len(self.q_s))
        else:
            factors = range(len(self.q_s))

        for factor_idx in factors:
            s_prev = self.q_s[factor_idx].clone().detach()
            # Prior conditioned on a₁ (u_prev), not â₂ (u) — ō₂ is causally downstream of a₁ only
            log_prior = torch.log(self.B[factor_idx, self.u_prev] @ s_prev + EPSILON)
            log_likelihood = torch.log(self.A[factor_idx].T @ o[factor_idx] + EPSILON)

            variational_params = self.theta[factor_idx].clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([variational_params], lr=self.inference_learning_rate)

            for _ in range(self.inference_num_iterations):
                optimizer.zero_grad()
                s_samples = Dirichlet(variational_params).rsample((self.inference_num_samples,))
                log_s = torch.log(s_samples + EPSILON)
                VFE = torch.sum(s_samples * (log_s - log_likelihood - log_prior), dim=-1).mean()
                VFE.backward()
                optimizer.step()
                variational_params.data.clamp_(min=1e-3)

            self.q_s[factor_idx] = Dirichlet(variational_params).mean.detach()
            self.theta[factor_idx] = variational_params.detach()
            self.VFE[factor_idx] = VFE.detach()
            self.entropy[factor_idx] = -torch.sum(s_samples * log_s, dim=-1).mean().detach()
            self.energy[factor_idx] = -torch.sum(s_samples * (log_prior + log_likelihood), dim=-1).mean().detach()
            self.accuracy[factor_idx] = torch.sum(s_samples * log_likelihood, dim=-1).mean().detach()
            self.complexity[factor_idx] = torch.sum(s_samples * (log_s - log_prior), dim=-1).mean().detach()

        self.q_s_history.append(self.q_s.detach().clone())
        self.o_history.append(o)
        return self.q_s

    # ==========================================================================
    # Action
    # ==========================================================================

    def compute_efe(self, u, q_s_u, A, log_C, depth, q_s_parent=None):
        '''Compute EFE for a given action at a given planning depth.

        Args:
            u: candidate action
            q_s_u: predicted next state s̄_n, shape (f, s)
            A: observation likelihood model
            log_C: log preference over observations
            depth: 1 = G₂ (respond to past), >1 = G₃ (shape future)
            q_s_parent: state s_n from which this transition originates (Eq. 19).
                        None falls back to self.q_s (safe for depth-1, which skips novelty).
        '''
        EFE = 0
        ambiguity = torch.tensor(0.0)
        risk = torch.tensor(0.0)
        salience = torch.tensor(0.0)
        pragmatic_value = torch.tensor(0.0)
        ppo_entropy = torch.tensor(0.0)
        novelty = torch.tensor(0.0)

        q_o_u = torch.einsum('fos,fs->fo', A, q_s_u)
        q_o_u[0] = F.one_hot(u.squeeze(), self.num_actions).to(torch.float)

        einsum_str = (
                ','.join([chr(105 + i) for i in range(self.num_agents)])
                + '->'
                + ''.join([chr(105 + i) for i in range(self.num_agents)])
        )
        q_o_joint_u = torch.einsum(einsum_str, *[q_o_u[i] for i in range(self.num_agents)])

        pragmatic_value = torch.tensordot(q_o_joint_u, log_C, dims=self.num_agents)

        if depth == 1:
            # G₂: ō₂ is causally fixed by a₁; epistemic terms are constant w.r.t. policy → drop
            risk = -pragmatic_value
            salience = torch.tensor(0.0)
        else:
            # G₃: â₂ causally affects s̄₃ and ō₃ → include full epistemic terms
            for factor_idx in range(self.num_agents):
                s_pred = q_s_u[factor_idx]
                H = -torch.diag(A[factor_idx].T @ torch.log(A[factor_idx] + EPSILON))
                ambiguity += torch.clip(H @ s_pred, min=0.0, max=None)
                o_pred = q_o_u[factor_idx]
                o_pred = o_pred[o_pred > 0]
                ppo_entropy += -torch.sum(o_pred * torch.log(o_pred))
            risk = torch.tensordot(
                q_o_joint_u,
                (torch.log(q_o_joint_u + EPSILON) - log_C),
                dims=self.num_agents
            )
            salience = ppo_entropy - ambiguity

        if self.compute_novelty and depth > 1:
            if self.A_learning:
                novelty += self.compute_A_novelty(q_s_u, q_o_u)
            if self.B_learning:
                action_idx = u.item() if isinstance(u, torch.Tensor) else u
                prior_state = q_s_parent if q_s_parent is not None else self.q_s
                novelty += self.compute_B_novelty(prior_state, q_s_u, action_idx)

        EFE = -pragmatic_value - self.epistemic_gain * (salience + novelty)
        return EFE.unsqueeze(0), torch.tensor((ambiguity, risk, salience, pragmatic_value, novelty)), q_o_u

    def compute_A_novelty(self, q_s_u, q_o_u):
        A_prime_params = self.A_params + torch.einsum(
            'fs,fo->fos',
            q_s_u,
            q_o_u
        )
        A_prime = A_prime_params / A_prime_params.sum(dim=1, keepdim=True)
        kl_div = (
                A_prime * torch.log(A_prime / (self.A + EPSILON) + EPSILON)
        ).sum(dim=(1, 2))
        return kl_div.sum()

    def compute_B_novelty(self, q_s, q_s_u, u):
        # B'[u] = B[u] + α_l (s̄_n ⊗ s_n)  — Eq. 19
        outer_product = torch.einsum('fn,fk->fnk', q_s_u, q_s)
        B_prime_params = self.B_params[:, u] + self.B_learning_rate * outer_product
        B_prime = B_prime_params / B_prime_params.sum(dim=1, keepdim=True)
        kl_div = (
                B_prime * torch.log(B_prime / (self.B[:, u] + EPSILON) + EPSILON)
        ).sum(dim=(1, 2))
        return kl_div.sum()

    def collect_policies(
            self,
            node,
            q_s,
            policy_EFEs=None,
            policy_EFE_terms=None,
            current_policy=None,
            parent_q_s_u=None,
    ):
        '''Traverse the policy tree, compute EFE at each node, and collect results.

        q_s: predicted state arriving at this node (already transitioned by parent)
        parent_q_s_u: state from which q_s was transitioned (s_n in Eq. 19 novelty term)
        '''
        if node.u is None:
            node.q_s_u = q_s
            current_policy = []
            policy_EFEs = []
            new_policy_EFEs = policy_EFEs
        else:
            node.q_s_u = q_s
            node.EFE_u, node.EFE_terms, node.q_o_u = self.compute_efe(
                node.u, node.q_s_u, self.A, self.log_C, node.depth,
                q_s_parent=parent_q_s_u
            )
            new_policy_EFEs = policy_EFEs + [node.EFE_u]
            if policy_EFE_terms is None:
                policy_EFE_terms = node.EFE_terms.unsqueeze(0)
            else:
                policy_EFE_terms = torch.vstack((policy_EFE_terms, node.EFE_terms.unsqueeze(0)))

        if not node.children:
            return [torch.cat(new_policy_EFEs)], policy_EFE_terms, [torch.cat(current_policy)]

        EFEs = []
        EFE_terms = []
        policies = []

        # Root uses self.u (committed a_t); deeper nodes use their own action
        causal_action = self.u if node.u is None else node.u.item()
        next_q_s = torch.einsum('funk,fk->fun', self.B, node.q_s_u)[:, causal_action]

        for child in node.children:
            new_policy = current_policy + [child.u]
            subtree_EFEs, subtree_EFE_terms, sub_policy = self.collect_policies(
                child, next_q_s, new_policy_EFEs, policy_EFE_terms, new_policy,
                parent_q_s_u=node.q_s_u
            )
            EFEs.extend(subtree_EFEs)
            EFE_terms.extend(subtree_EFE_terms)
            policies.extend(sub_policy)

        return torch.vstack(EFEs), torch.vstack(EFE_terms), torch.vstack(policies)


    def select_action(self):
        root = build_policy_tree(torch.arange(self.num_actions), self.policy_length)
        EFEs, EFE_terms, policies = self.collect_policies(root, q_s=self.q_s)

        EFE_policies = EFEs.sum(dim=1)
        q_u = torch.softmax(torch.log(self.E) - self.gamma * EFE_policies, dim=0)

        self.EFE = EFE_policies
        self.EFE_terms = EFE_terms.reshape(*EFEs.shape, -1)
        self.q_u = q_u

        policy_idx = torch.multinomial(q_u, 1).item() if not self.deterministic_actions else torch.argmax(q_u).item()
        self.u_prev = self.u  # capture a₁ before overwrite — used in infer_state()
        self.u = policies[policy_idx][0].item()
        self.u_history.append(self.u)

        for child in root.children:
            if child.u.item() == self.u:
                self.o_pred_record = child.q_o_u.detach().clone()
                break

        if self.dynamic_precision:
            self.update_precision(EFE_policies, q_u)

        return self.u

    def update_precision(self, EFE, q_u):
        self.expected_EFE = torch.dot(q_u, EFE).item()
        denom = max(self.beta_0 - self.expected_EFE, 1e-6)
        self.gamma = self.beta_1 / denom
        return self.gamma

    # ==========================================================================
    # Learning
    # ==========================================================================

    def learn(self):
        if len(self.u_history) == (self.learn_every_t_steps + self.current_random_offset_learning):
            self.learn_record = self.learn_every_t_steps + self.current_random_offset_learning
            self.q_s_history = torch.stack(self.q_s_history)
            self.o_history = torch.stack(self.o_history)
            self.u_history = torch.tensor(self.u_history)
            if self.A_learning:
                self.learn_A()
            if self.B_learning:
                self.learn_B()
            self.q_s_history = []
            self.o_history = []
            self.u_history = []
            self.current_random_offset_learning = random.randint(-self.learning_offset, self.learning_offset)
        else:
            self.learn_record = 0

    def learn_A(self):
        outer_products = torch.einsum('tfo,tfs->tfos', self.q_s_history, self.o_history)
        delta_params = outer_products.mean(dim=0)
        A_posterior_params = self.A_params + delta_params

        if self.A_BMR:
            for factor_idx in range(self.num_agents):
                a_prior = self.A_params[factor_idx].flatten()
                a_post_full = A_posterior_params[factor_idx].flatten()
                a_red = make_reduced_prior(a_prior, method=self.A_BMR, alpha=self.alpha_r, preserve_total=False)
                a_red_candidates = [
                    a_prior,
                    torch.tensor([1., 0., 1., 0.], device=self.A_params.device) + EPSILON,
                ]
                delta_F_vector = torch.stack([
                    delta_free_energy(a_post_full, a_prior, a_red_i) for a_red_i in a_red_candidates
                ])
                assert delta_F_vector[0].abs() < 1e-4, f"Delta F for full model ({delta_F_vector[0]}) should be zero."
                weight_vector = torch.softmax(self.gamma_r * delta_F_vector, dim=0).to(a_post_full.dtype)
                a_post_candidates = torch.stack([
                    (a_post_full + a_red_i - a_prior).clamp_min(EPSILON) for a_red_i in a_red_candidates
                ], dim=0)
                self.delta_F_A[factor_idx] = delta_F_vector[1].detach().item()
                self.A_params[factor_idx] = (weight_vector @ a_post_candidates).view_as(self.A_params[factor_idx])
        else:
            self.A_params = A_posterior_params

        self.A = self.A_params / self.A_params.sum(dim=1, keepdim=True)

    def learn_B(self):
        s_prev = self.q_s_history[:-1]
        s_next = self.q_s_history[1:]
        u_seq = self.u_history[:-1]  # u_history[t] caused transition q_s[t] -> q_s[t+1]

        outer_products = torch.einsum('tfn,tfk->tfnk', s_next, s_prev)
        T = outer_products.shape[0]
        B_posterior_params = self.B_params.clone()
        LEARNING_RATE = self.B_learning_rate if self.B_learning_rate is not None else 1 / T

        for t in range(T):
            u_it = u_seq[t].item()
            B_posterior_params[:, u_it] += LEARNING_RATE * outer_products[t]

        if self.B_BMR:
            for factor_idx in range(self.num_agents):
                a_prior = self.B_params[factor_idx].flatten()
                a_post_full = B_posterior_params[factor_idx].flatten()
                a_red = make_reduced_prior(a_prior, method=self.B_BMR, alpha=self.alpha_r, preserve_total=False)
                a_red_candidates = [
                    a_prior,
                    *[
                        candidate
                        for name, candidate in [
                            ("TFT", torch.tensor([[[0.99, 0.99],
                                                   [0.01, 0.01]],
                                                  [[0.01, 0.01],
                                                   [0.99, 0.99]]], device=self.B_params.device).flatten()),
                            ("Grim", torch.tensor([[[0.99, 0.01],
                                                    [0.01, 0.99]],
                                                   [[0.01, 0.01],
                                                    [0.99, 0.99]]], device=self.B_params.device).flatten()),
                            ("Pavlov", torch.tensor([[[0.99, 0.01],
                                                      [0.01, 0.99]],
                                                     [[0.01, 0.99],
                                                      [0.99, 0.01]]], device=self.B_params.device).flatten()),
                            ("Reduce", a_red),
                        ]
                        if name in self.B_candidates
                    ]
                ]
                delta_F_vector = torch.stack([
                    delta_free_energy(a_post_full, a_prior, a_red_i) for a_red_i in a_red_candidates
                ])
                assert delta_F_vector[0].abs() < 1e-2, f"Delta F for full model ({delta_F_vector[0]}) should be zero."
                weight_vector = torch.softmax(self.gamma_r * delta_F_vector, dim=0).to(a_post_full.dtype)
                a_post_candidates = torch.stack([
                    (a_post_full + a_red_i - a_prior).clamp_min(EPSILON) for a_red_i in a_red_candidates
                ], dim=0)
                self.delta_F[factor_idx, :] = delta_F_vector.detach()
                self.B_model_weights[factor_idx, :] = weight_vector.detach()
                self.B_params[factor_idx] = (weight_vector @ a_post_candidates).view_as(self.B_params[factor_idx])
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
    """Construct a reduced Dirichlet prior for a single 1D concentration vector.

    Args:
        a_prior: 1D concentration parameters, shape (K,)
        method: 'epsilon' (hard prune below quantile) or 'softmax' (smooth sharpening)
        alpha: quantile cutoff for epsilon; sharpening strength for softmax
        preserve_total: if True, rescale retained entries to preserve original sum (non-Friston)
    Returns:
        a_red: reduced concentrations, clamped >= EPSILON
    """
    if a_prior.ndim != 1:
        raise ValueError("a_prior must be a 1D tensor.")
    assert alpha is not None, "alpha must be specified."

    a = a_prior.clone().detach().float().clamp_min(EPSILON)
    total = a.sum()
    p_prior = a / total.clamp_min(EPSILON)

    if method == "epsilon":
        cutoff = torch.quantile(p_prior, alpha)
        mask = p_prior < cutoff
        a_red = a.clone()
        a_red[mask] = EPSILON
        if preserve_total:
            retained_mask = ~mask
            retained_sum = a_red[retained_mask].sum()
            if retained_sum > 0:
                a_red[retained_mask] *= (total - EPSILON * mask.sum()) / retained_sum

    elif method == "softmax":
        p_red = F.softmax(alpha * torch.log(p_prior + EPSILON), dim=-1)
        a_red = (p_red * a).clone()
        if preserve_total:
            sum_a_red = a_red.sum()
            if sum_a_red > 0:
                a_red *= total / sum_a_red

    elif method == "flat":
        a_red = torch.ones_like(a_prior)

    elif method == "identity":
        a_red = torch.eye(a_prior.shape[0] // 2).view_as(a_prior)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'epsilon' or 'softmax'.")

    return a_red.clamp_min(EPSILON)


def log_MBF(alpha):
    """Log multivariate Beta function: log B(alpha) = sum(lgamma(alpha)) - lgamma(sum(alpha))."""
    a = alpha.clamp_min(EPSILON)
    return torch.lgamma(a).sum() - torch.lgamma(a.sum())


def delta_free_energy(a_posterior, a_prior, a_reduced):
    """Compute ΔF = log p(y|M_red) - log p(y|M_full) via Dirichlet-Categorical BMR identity.

    ΔF = [log B(a_comb) - log B(a_red)] - [log B(a_post) - log B(a_prior)]
    where a_comb = a_post + a_red - a_prior.
    Positive ΔF → prefer reduced model. (Friston et al. 2016, Eq. 12)
    """
    a_post = a_posterior.to(torch.float64).clamp_min(EPSILON)
    a_prior = a_prior.to(torch.float64).clamp_min(EPSILON)
    a_reduced = a_reduced.to(torch.float64).clamp_min(EPSILON)
    a_comb = (a_post + a_reduced - a_prior).clamp_min(EPSILON)
    return (log_MBF(a_comb) - log_MBF(a_reduced)) - (log_MBF(a_post) - log_MBF(a_prior))


# ==============================================================================
# Policy tree
# ==============================================================================

class TreeNode:
    def __init__(self, action=None, depth=0):
        self.u = action
        self.EFE_u = torch.tensor(0)
        self.children = []
        self.depth = depth

    def add_child(self, action):
        child = TreeNode(action, self.depth + 1)
        self.children.append(child)
        return child

    def __repr__(self):
        return f"TreeNode(action={self.u}, depth={self.depth}, children={len(self.children)})"


def build_policy_tree(action_space, max_depth, node=None):
    '''Recursively build a full policy tree of depth max_depth over action_space.'''
    if node is None:
        node = TreeNode()
    if node.depth == max_depth:
        return node
    for action in action_space:
        child = node.add_child(torch.tensor([action]))
        build_policy_tree(action_space, max_depth, node=child)
    return node
