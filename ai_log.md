# Conversation Log (Tutor Summary)

Date: 2026-03-17 (Australia/Sydney)
Repo: d:\Projects\factorised-MA-AIF

## User request
- Understand the codebase and verify whether the current `agent.py` implementation matches the tutor's causal diagram in `Formula.png`.
- Propose a minimal change to make planning two-step consistent with the diagram.
- Verify whether `agent.py` `collect_policies` needs code changes.

## Evidence reviewed
- `README.md` for high-level structure.
- `agent.py` for generative model, inference, EFE planning, and learning.
- `utils/simulation.py` for the time-step loop and how observations/actions are passed.
- `scripts/INFG-simulate.py` for experiment config.
- `dummy.py` for opponent behavior.
- `generator2.py` for game loading and transitions.
- `scripts/INFG-plot.py` for plotting and use of EFE outputs.
- `Formula.png` (image) with tutor's causal diagram and equations.

## Key message from `Formula.png` (summary)
- The diagram notes a temporal mismatch: the observation at time t+1 (denoted \bar{o}_2 in the screenshot) is causally dependent on the previous action (a_1), not on the next action (a_2).
- Therefore, evaluating action a_2 should account for its effect on the next state and observation (s_3, \bar{o}_3), which implies two-step planning when computing EFE.

## Findings in `agent.py`
- `select_action()` builds a policy tree of length `policy_length` and calls `collect_policies()`.
- `collect_policies()` recursively propagates beliefs via `B` and computes EFE for each node's action using `compute_efe()`.
- When `policy_length = 2`, the EFE for the second action is computed after one transition step, which aligns with the tutor's two-step logic.

Relevant references:
- `agent.py:609` (`collect_policies`): uses `B` to compute `q_s_u` before EFE.
- `agent.py:428` (`compute_efe`): evaluates EFE from predicted state -> predicted observation.
- `agent.py:660` (`select_action`): builds policy tree using `policy_length`.

## Minimal change proposed
- Set `policy_length=2` for the AIF agent in the experiment configuration.
- This enables two-step planning without changing `agent.py`.

Applied patch:
- `scripts/INFG-simulate.py`: added `policy_length=2` to the AIF agent kwargs.

Rationale:
- Two-step planning makes the evaluation of a_2 depend on the subsequent predicted observation (\bar{o}_3), consistent with the tutor diagram.

## Verification request: Does `collect_policies` need changes?
- Conclusion: No code change required in `collect_policies` if `policy_length` is set to 2.
- The recursion already accounts for multi-step planning when `policy_length > 1`.

## Notes / optional future work
- If the model must explicitly encode that observations at time t depend on actions at time t-1 (a hard one-step delay), this would require changes to the generative model timing or inference flow rather than `collect_policies`.
- This was not implemented in the current change.

## Files referenced
- `README.md`
- `agent.py`
- `utils/simulation.py`
- `scripts/INFG-simulate.py`
- `dummy.py`
- `generator2.py`
- `scripts/INFG-plot.py`
- `Formula.png`
