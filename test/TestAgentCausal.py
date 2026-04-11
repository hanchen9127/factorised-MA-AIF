'''
Unit tests for causal alignment and planning/inference bookkeeping.

Author: Hanchen
Date: 2026-04
'''

import os
import sys
import unittest

import torch
import torch.nn.functional as F

# Allow imports from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent import Agent, build_policy_tree


class TestAgentCausal(unittest.TestCase):

    def test_policy_prior_normalized(self):
        game = torch.zeros((2, 2))
        agent = Agent(
            id=0,
            game_matrix=game,
            policy_length=2,
            deterministic_actions=True,
            compute_novelty=False,
            A_learning=False,
            B_learning=False,
        )
        self.assertEqual(agent.E.numel(), 4)
        self.assertTrue(torch.allclose(agent.E, torch.ones(4) / 4))

    def test_u_prev_updates_on_select_action(self):
        torch.manual_seed(0)
        game = torch.zeros((2, 2))
        agent = Agent(
            id=0,
            game_matrix=game,
            policy_length=1,
            deterministic_actions=True,
            compute_novelty=False,
            A_learning=False,
            B_learning=False,
        )
        old_u = agent.u
        agent.select_action()
        self.assertEqual(agent.u_prev, old_u)

    def test_collect_policies_uses_previous_action_for_transition(self):
        torch.manual_seed(0)
        game = torch.zeros((2, 2))
        agent = Agent(
            id=0,
            game_matrix=game,
            policy_length=1,
            deterministic_actions=True,
            compute_novelty=False,
            A_learning=False,
            B_learning=False,
        )

        # Custom deterministic transitions:
        # action 0 -> identity, action 1 -> flip
        B = torch.zeros((2, 2, 2, 2), dtype=torch.float32)
        B[:, 0] = torch.eye(2)
        B[:, 1] = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        agent.B = B

        agent.q_s = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)
        agent.u = 1  # previous action

        root = build_policy_tree(torch.arange(agent.num_actions), agent.policy_length)
        agent.collect_policies(root, q_s=agent.q_s)

        expected = torch.einsum('fnk,fk->fn', B[:, 1], agent.q_s)
        for child in root.children:
            self.assertTrue(torch.allclose(child.q_s_u, expected))

    def test_compute_efe_depth_gates_epistemic_terms(self):
        torch.manual_seed(0)
        game = torch.zeros((2, 2))
        agent = Agent(
            id=0,
            game_matrix=game,
            policy_length=2,
            deterministic_actions=True,
            compute_novelty=False,
            A_learning=False,
            B_learning=False,
        )

        A = torch.stack([torch.eye(2), torch.eye(2)])
        q_s_u = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        u = torch.tensor(0, dtype=torch.int64)

        efe1, terms1, _ = agent.compute_efe(u, q_s_u, A, agent.log_C, depth=1)
        efe2, terms2, _ = agent.compute_efe(u, q_s_u, A, agent.log_C, depth=2)

        ambiguity1, risk1, salience1, pragmatic1, novelty1 = terms1
        self.assertTrue(torch.allclose(salience1, torch.tensor(0.0)))
        self.assertTrue(torch.allclose(risk1, -pragmatic1))
        self.assertTrue(torch.allclose(novelty1, torch.tensor(0.0)))

        salience2 = terms2[2].item()
        self.assertGreater(salience2, 0.1)

        self.assertEqual(efe1.shape, torch.Size([1]))
        self.assertEqual(efe2.shape, torch.Size([1]))

    def test_q_s_history_is_detached_clone(self):
        torch.manual_seed(0)
        game = torch.zeros((2, 2))
        agent = Agent(
            id=0,
            game_matrix=game,
            inference_num_iterations=1,
            inference_num_samples=2,
            inference_learning_rate=1e-2,
            deterministic_actions=True,
            compute_novelty=False,
            A_learning=False,
            B_learning=False,
        )

        o = F.one_hot(torch.tensor([0, 1]), num_classes=2).to(torch.float32)
        agent.infer_state(o)

        history_snapshot = agent.q_s_history[0].clone()
        agent.q_s[0, 0] = 0.0
        agent.q_s[0, 1] = 1.0

        self.assertTrue(torch.allclose(agent.q_s_history[0], history_snapshot))


if __name__ == '__main__':
    unittest.main()
