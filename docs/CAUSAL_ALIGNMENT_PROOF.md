# Causal Alignment Proof: `agent.py` vs `causal-delay-formula.png`

This document is a precise, code-referenced proof that `agent.py` correctly
implements the causal diagram in `causal-delay-formula.png` and the generative model in
the AAMAS 2025 paper (arXiv:2411.07362v2). It records every bug found in
`agent_old.py`, the fix applied, and the justification. It is intended for
technical readers who want to verify claims against the source.

---

## 0. The causal diagram: what it requires

`causal-delay-formula.png` specifies a two-step planning horizon with the following causal
structure at planning time t:

```
s₀ → s₁ → s̄₂ → s̄₃
          ↑         ↑
     a₁ (past)   â₂ (proposed)

o₀   o₁   ō₂   ō₃
          ↑         ↑
     a₁ (past)   â₂ (proposed)

μ₀ → μ₁ → μ₂ → μ₃
```

The key causal constraints are:

1. **ō₂ depends on a₁, not â₂.** The opponent's observable action at the
   next step is caused by ego's *past committed action*, because both agents
   chose simultaneously and the opponent responds to history.

2. **G₂(â₂) — "respond to the past"** evaluates the pragmatic value of â₂
   given that ō₂ is already causally fixed by a₁. Epistemic terms (ambiguity,
   salience, novelty) are constant w.r.t. the policy here and must be dropped.

3. **G₃(â₂, â₃) — "shape the future"** evaluates full EFE including epistemic
   terms, because â₂ genuinely causes s̄₃ and ō₃.

4. **G(â₂, â₃) = G₂(â₂) + G₃(â₂, â₃)** — the policy tree sums across depths.

5. **Novelty (Eq. 19, paper §3.5.1):**
   `B̄[û,n] = B[û,n] + α_l (s̄_n ⊗ s_n)` where `s̄_n` is the predicted
   *next* state and `s_n` is the state *from which* the transition originates.

---

## 1. Bug 1 — Inference prior indexed by the wrong action

### What `agent_old.py` did (wrong)

`agent_old.py`, `infer_state()`, line 381:

```python
log_prior = torch.log(self.B[factor_idx, self.u] @ s_prev + EPSILON)
```

At the time `infer_state()` runs, `self.u` has already been overwritten by
`select_action()` with the *newly committed* action â_t:

```python
self.u = policies[policy_idx][0].item()   # â_t written here
self.u_history.append(self.u)
# select_action() returns; sim loop calls infer_state() — self.u is â_t
```

So `B[factor_idx, self.u]` was computing:

```
q(s̄₂ | â₂)   ← planning counterfactual, WRONG for perception
```

when the diagram requires:

```
q(s̄₂ | a₁)   ← causal prior from past committed action, CORRECT
```

Whenever the agent switched action (a₁ ≠ â₂), the perception prior was
causally incorrect.

### What `agent.py` does (correct)

Three coordinated changes:

**`__init__`, lines 229–230:**
```python
self.u = torch.randint(0, num_actions, (1,)).item()
self.u_prev = self.u  # a₁ initialised to a valid action index
```

**`select_action()`, lines 705–707:**
```python
policy_idx = torch.multinomial(q_u, 1).item() if not self.deterministic_actions else torch.argmax(q_u).item()
self.u_prev = self.u   # capture a₁ BEFORE overwriting
self.u = policies[policy_idx][0].item()
```

`u_prev` is captured immediately before `self.u` is overwritten. This is the
only safe location — after this point a₁ is lost.

**`infer_state()`, line 395:**
```python
log_prior = torch.log(self.B[factor_idx, self.u_prev] @ s_prev + EPSILON)
```

The prior is now `B[a₁] @ s_prev`, computing `q(s̄₂ | a₁)` as required.
The likelihood is unchanged:

```python
log_likelihood = torch.log(self.A[factor_idx].T @ o[factor_idx] + EPSILON)
```

`o[factor_idx]` for the alter factor contains the opponent's current action,
which in a synchronous game is causally downstream of a₁ (not â₂), so it is
consistent with the corrected prior. The VFE being minimised is now:

```
VFE = KL[q(s̄₂) || B[a₁] @ q(s₁)] - E_{q(s̄₂)}[log A.T @ o₂]
       ↑ correct causal prior           ↑ consistent observation
```

---

## 2. Bug 2 — EFE computed without depth awareness

### What `agent_old.py` did (wrong)

`agent_old.py`, `compute_efe()`, line 428:

```python
def compute_efe(self, u, q_s_u, A, log_C):
```

No `depth` parameter. Every policy tree node received the same full EFE
computation including ambiguity, salience, and novelty at depth 1. At depth 1
the diagram is explicit: s̄₂ is already determined by a₁; â₂ cannot reduce
uncertainty about ō₂. Including epistemic terms at depth 1 causes the agent
to seek information it cannot causally obtain.

### What `agent.py` does (correct)

**`compute_efe()` signature, line 442:**
```python
def compute_efe(self, u, q_s_u, A, log_C, depth, q_s_parent=None):
```

`depth` is passed from `collect_policies()` via `node.depth`.

**At `depth == 1` — G₂: respond to the past:**
```python
if depth == 1:
    risk = -pragmatic_value
    salience = torch.tensor(0.0)
    # ambiguity, novelty remain 0.0
```

Only pragmatic value survives, matching G₂(â₂). The entropy term
H[q(ō₂|a₁)] is constant w.r.t. â₂ and is correctly dropped.

**At `depth > 1` — G₃: shape the future:**
```python
else:
    # ambiguity, salience, risk all computed in full
    salience = ppo_entropy - ambiguity
```

Full epistemic terms are computed, matching G₃(â₂, â₃).

**Novelty is also depth-gated:**
```python
if self.compute_novelty and depth > 1:
```

Novelty is suppressed at depth 1 for the same causal reason.

---

## 3. Bug 3 — Policy tree state transitioned inside the wrong node

### What `agent_old.py` did (wrong)

`agent_old.py`, `collect_policies()`, lines 627–634:

```python
else:
    node.q_s_u = torch.einsum('funk,fk->fun', self.B, q_s)[:, node.u].squeeze()
    node.EFE_u, ... = self.compute_efe(node.u, node.q_s_u, self.A, self.log_C)
```

The B-transition was performed *inside* each non-root node using `node.u`
(the candidate action) before calling `compute_efe`. Depth-1 nodes therefore
evaluated EFE on a state that had already been transitioned by â₂, when the
diagram requires the state to have been transitioned by a₁. Depth-2 nodes
received this corrupted state from their parent.

### What `agent.py` does (correct)

**Non-root nodes accept the state without re-transitioning:**
```python
else:
    node.q_s_u = q_s   # already transitioned by parent
    node.EFE_u, node.EFE_terms, node.q_o_u = self.compute_efe(
        node.u, node.q_s_u, self.A, self.log_C, node.depth,
        q_s_parent=parent_q_s_u
    )
```

**The transition is computed at the parent before iterating children:**
```python
causal_action = self.u if node.u is None else node.u.item()

next_q_s = torch.einsum('funk,fk->fun', self.B, node.q_s_u)[:, causal_action]

for child in node.children:
    self.collect_policies(child, next_q_s, ..., parent_q_s_u=node.q_s_u)
```

- At the **root** (`node.u is None`): `causal_action = self.u = â_t`.
  Depth-1 children receive `B[â_t] @ q_s` = `q(s̄₂ | â_t)` — correct because
  the planning rollout starts from the committed action.
- At **depth-1 nodes**: `causal_action = node.u = â₂`. Depth-2 children
  receive `B[â₂] @ q_s_u` = `q(s̄₃ | â₂)` — correct for G₃.

---

## 4. Bug 4 — Novelty outer product used wrong "from" state

### What `agent_old.py` / early `agent.py` did (wrong)

`compute_efe()` called `compute_B_novelty` as:

```python
novelty += self.compute_B_novelty(self.q_s, q_s_u, action_idx)
```

Inside `compute_B_novelty`, the outer product is:

```python
outer_product = torch.einsum('fn,fk->fnk', q_s_u, q_s)
#                                           ↑ s̄_n   ↑ s_n
```

The paper (Eq. 19) requires `s_n` to be the state *from which* the transition
originates at the current planning step. `self.q_s` is the pre-planning
posterior — the correct "from" state only at depth 1 (which is skipped by the
depth gate). At depth 2, the correct "from" state is the depth-1 node's
`q_s_u` (i.e. `B[â_t] @ self.q_s`), not `self.q_s` itself.

Using `self.q_s` at depth 2 miscalibrates the novelty magnitude: it is
evaluated against the pre-planning belief rather than the intermediate
predicted state, causing the epistemic exploration pressure at depth 2 to be
incorrect once B has been learned and `B[â_t] @ self.q_s ≠ self.q_s`.

### What `agent.py` does (correct)

The "from" state is threaded through the call chain explicitly via a new
`parent_q_s_u` argument in `collect_policies` and a new `q_s_parent` argument
in `compute_efe`.

**`collect_policies()` passes `parent_q_s_u` to children:**
```python
subtree_EFEs, ... = self.collect_policies(
    child, next_q_s, ..., parent_q_s_u=node.q_s_u
)
```

`node.q_s_u` at the root is `self.q_s`; at depth-1 nodes it is
`B[â_t] @ self.q_s`. This is exactly the "from" state for depth-2 transitions.

**`compute_efe()` receives `q_s_parent` and passes it to novelty:**
```python
prior_state = q_s_parent if q_s_parent is not None else self.q_s
novelty += self.compute_B_novelty(prior_state, q_s_u, action_idx)
```

The outer product in `compute_B_novelty` is now:

```python
outer_product = torch.einsum('fn,fk->fnk', q_s_u, prior_state)
#                                           ↑ s̄_n   ↑ s_n (correct per Eq. 19)
```

At depth 2, `prior_state = parent_q_s_u = B[â_t] @ self.q_s` — the state the
agent predicts it will be in after executing the depth-1 action, which is the
correct causal "from" state for the depth-2 novelty computation.

The `None` fallback is never reached in practice since novelty is only
computed at `depth > 1` where `q_s_parent` is always explicitly set.

---

## 5. Bug 5 — Initial action sampled from policy-index distribution

### What `agent_old.py` did (wrong)

`agent_old.py`, `__init__`, line 224:

```python
self.u = torch.multinomial(self.E, 1).item()
```

`self.E` has `num_actions ** policy_length` entries. With `policy_length=2`
and `num_actions=2`, `torch.multinomial` returns values in `{0, 1, 2, 3}`.
But `self.u` is immediately used to index into `self.B` of shape
`(num_agents, num_actions, num_actions, num_actions)` where the action
dimension has size `num_actions=2`. Any initial sample of 2 or 3 raises an
`IndexError` at the first `infer_state()` call — affecting ~50% of runs.

### What `agent.py` does (correct)

`agent.py`, `__init__`, line 229:

```python
self.u = torch.randint(0, num_actions, (1,)).item()
self.u_prev = self.u
```

`torch.randint(0, num_actions, (1,))` always returns a valid action index in
`{0, ..., num_actions-1}` regardless of `policy_length`. After the first
`select_action()` call, `self.u` is always set from
`policies[policy_idx][0].item()` which correctly extracts the first action
from the chosen policy sequence, so the bug only existed at initialisation.

---

## 6. What is not fixed, and why it does not need to be

### Observation timing

The simulator loop (`utils/simulation.py`, lines 265–292) is same-step:

```python
u_all = [agent.select_action() for agent in self.agents]
# o_t built from u_all (current step)
for agent in self.agents:
    agent.infer_state(o_t)
```

`o_t` is constructed from the current joint action. A strict causal delay
would buffer observations so that `infer_state` at time t receives the joint
action from time t-1. This is **not fixed**, but it does not introduce a
causal error for this game. In a synchronous iterated game, both agents choose
simultaneously. The opponent's action at time t is chosen without observing
ego's action at time t — it is a response to history up to t-1. Therefore
`o_t` is causally downstream of a_{t-1}, not of â_t. The likelihood
`A[1].T @ o[1]` and the corrected prior `B[a₁] @ q_s` are consistent: both
are consequences of the same causal parent a_{t-1}. A hard observation delay
would be required only for sequential (non-simultaneous) games.

### Ego factor inference

When `interoception=False` (current config), the ego factor is inferred
through the same VFE loop as alter factors, using `B[u_prev] @ q_s[0]` as its
prior. With `A_prior=99` (near-identity), the likelihood dominates the prior
and the ego posterior converges to `one_hot(own_action)` regardless. This does
not affect alter factor inference correctness.

### `self.q_u` is a policy-sequence distribution, not an action marginal

`self.q_u` has `num_actions ** policy_length` entries — one per full policy
sequence. The paper's Fig. 2 policy plots and Eq. 16 treat `q(û)` as a
2-entry action distribution. The marginalisation
`q_action = q_u.reshape(num_actions, num_actions).sum(dim=1)` is **not
computed in `agent.py`** by design: `self.q_u` is used correctly at full
length for policy selection (`torch.multinomial(q_u, 1)`) and precision
updates (`torch.dot(q_u, EFE_policies)`). The marginalisation is handled in
the plotting module, which reads `q_u` from the database and performs the sum
before display. No change to `agent.py` is needed or appropriate here.

---

## 7. Summary table

| Issue | `agent_old.py` | `agent.py` | Requirement |
|---|---|---|---|
| Inference prior action | `B[self.u]` — uses â_t | `B[self.u_prev]` — uses a_{t-1} | `q(s̄₂ \| a₁)` per Eq. 5 |
| EFE at depth 1 | Full epistemic terms | Epistemic terms zeroed | G₂ = pragmatic only |
| EFE at depth > 1 | Same as depth 1 | Full ambiguity + salience + novelty | G₃ = full EFE |
| Novelty at depth 1 | Computed if enabled | Suppressed | Constant w.r.t. policy, drop |
| State transition in tree | Inside node using `node.u` | At parent using `causal_action` | s̄₂ caused by a₁, not â₂ |
| Novelty "from" state | `self.q_s` at all depths | `parent_q_s_u` at depth > 1 | s_n = parent node's q_s_u per Eq. 19 |
| Initial action index | `multinomial(E)` → policy idx 0–3 | `randint(0, num_actions)` → valid action idx | Valid B index at initialisation |
| Observation timing | Same-step | Same-step (unchanged) | Correct for synchronous game |
| `q_u` marginalisation | Not in agent.py | Not in agent.py | Handled in plotting module |

All five bugs are fixed in `agent.py`. Observation timing and `q_u`
marginalisation are unchanged and correctly handled outside `agent.py`.
