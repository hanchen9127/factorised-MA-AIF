'''
Plotting utilities for MA-AIF ensemble experiments.

Authors: Pat Sweeney, Jaime Ruiz Serra
Date:    2024/08

Extended and refactored by: Hanchen Wang
Date: 2026/06

Design contract
---------------
Every public plot_*_ensemble function in this file follows the same contract:

    plot_foo_ensemble(data, ..., ifLegend=True, ax=None) -> None

- `data` is always a list-of-seeds structure, exactly as stored in ExperimentData.
- `ax` is a matplotlib Axes object provided by the caller (INFG-plot-gui.py).
- The function draws on `ax` and returns nothing.
- No plt.show(), no figure creation, no global state mutation.

Adding a new panel
------------------
1. Write a function here following the contract above.
2. Add one entry to `panel_spec()` in INFG-plot-gui.py.
3. Add one checkbox entry to DEFAULT_PANELS in INFG-plot-gui.py.
That is all — the GUI wires everything else automatically.

Legacy functions
----------------
The original single-seed, per-agent plotting system (make_default_config, plot,
plot_vfe, plot_efe, plot_cooperation_prob, etc.) has been moved to:

    utils/plotting_legacy.py

Those functions are preserved unchanged for any notebooks or scripts that still
call them directly. This file imports nothing from the legacy module.
'''

import logging

import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Section 1 — Global style constants
# ==============================================================================

DPI = 72                  # Used by the GUI renderer (_DPI_RENDER=500 overrides at save time)
SHOW_LEGEND = True        # Kept for back-compat; ensemble functions use ifLegend kwarg
ONLY_LEFT_Y_LABEL = False
TIGHT_LAYOUT = False
MARGIN = 0.05             # Fractional y-axis padding on data-driven ylim

# Colour palette — all ensemble functions draw from these names so a single
# change here propagates everywhere.
ELBO_COLOR          = '#000080'  # VFE mean line
EFE_COLOR           = '#990000'  # EFE mean line
ENTROPY_COLOR       = '#468499'  # Agent i entropy
PRECISION_COLOR     = '#333333'  # Precision γ

# Two-action colour convention used consistently across all panels:
#   ACTION_COLORS[0] = cooperate (cyan)
#   ACTION_COLORS[1] = defect    (pink)
ACTION_COLORS = [
    '#03dffc',  # 0 — cooperate
    '#f70ce4',  # 1 — defect
    '#fcad03',  # 2 — (third action, if used)
    '#03fcad',
    '#ad03fc',
    '#adfc03',
    '#fc03ad',
    '#03adfc',
]

SIZE_MULTIPLIER  = 1.4
label_font_size  = 8 * SIZE_MULTIPLIER
LINEWIDTH        = 0.7 * SIZE_MULTIPLIER

# Apply rcParams once at import time so all figures share the same style.
plt.rc('font',   size=label_font_size, family='serif')
plt.rc('axes',   titlesize=14, labelsize=label_font_size)
plt.rc('xtick',  labelsize=label_font_size)
plt.rc('ytick',  labelsize=label_font_size)
plt.rc('legend', fontsize=label_font_size)
plt.rc('text',   usetex=False)


# ==============================================================================
# Section 2 — Data coercion helpers
# ==============================================================================
# These helpers normalise the varied shapes that q_u history can arrive in
# (torch tensors, object arrays, mixed-length per-agent vectors) into a clean
# float64 numpy array before any plotting logic runs.

def _coerce_q_u_history(q_u_history):
    '''Convert nested q_u history into a float64 array.

    Supports both:
      - multi-seed  : list[seed][t][agent] -> vector  →  (S, T, agents, K)
      - single-seed : list[t][agent] -> vector         →  (T, agents, K)
    '''
    arr = np.array(q_u_history, dtype=object)
    if arr.dtype != object:
        return arr.astype(float)

    try:
        # Multi-seed path
        return np.array(
            [[[np.array(a, dtype=float) for a in step] for step in seed]
             for seed in q_u_history],
            dtype=float
        )
    except Exception:
        # Single-seed fallback
        return np.array(
            [[np.array(a, dtype=float) for a in step] for step in q_u_history],
            dtype=float
        )


def _coerce_q_u_action_history(q_u_history, game_transitions):
    '''Normalise q_u history to action-level probabilities.

    When policy_length > 1 the agent stores a distribution over action
    *sequences* (length num_actions ** policy_length).  This function
    marginalises back to num_actions by summing over sequence suffixes.

    Returns ndarray of shape (S, T, agents, num_actions).
    '''
    num_actions = game_transitions[0][1].shape[0] if game_transitions else None

    def to_action_vec(q):
        q_arr = np.array(q, dtype=float)
        if num_actions is None or q_arr.shape[0] == num_actions:
            return q_arr
        if q_arr.shape[0] % num_actions != 0:
            raise ValueError(
                f'q_u length {q_arr.shape[0]} is not divisible by '
                f'num_actions {num_actions}; cannot marginalise.'
            )
        return q_arr.reshape(num_actions, -1).sum(axis=1)

    def convert_seed(seed_hist):
        return np.array(
            [[to_action_vec(q) for q in step] for step in seed_hist],
            dtype=float
        )

    # Already a clean numeric array
    if isinstance(q_u_history, np.ndarray) and q_u_history.dtype != object:
        arr = q_u_history
        if num_actions is not None and arr.shape[-1] != num_actions:
            arr = arr.reshape(*arr.shape[:-1], num_actions, -1).sum(axis=-1)
        return arr

    # Multi-seed stored as list of arrays (one per seed)
    if (isinstance(q_u_history, list)
            and q_u_history
            and isinstance(q_u_history[0], np.ndarray)):
        arr = np.array(q_u_history, dtype=float)
        if num_actions is not None and arr.shape[-1] != num_actions:
            arr = arr.reshape(*arr.shape[:-1], num_actions, -1).sum(axis=-1)
        return arr

    # Multi-seed: list[seed][t][agent]
    if (isinstance(q_u_history, list)
            and q_u_history
            and isinstance(q_u_history[0], list)
            and q_u_history[0]
            and isinstance(q_u_history[0][0], list)):
        return np.array([convert_seed(seed) for seed in q_u_history], dtype=float)

    # Single-seed: list[t][agent]
    return convert_seed(q_u_history)


# ==============================================================================
# Section 3 — Shared axis decorators
# ==============================================================================

def highlight_transitions(game_transitions, ax, t_min=0, t_max=None):
    '''Shade alternating game phases and annotate with game-name labels.

    Called by the GUI renderer after every panel's plot function returns,
    so plot functions themselves never need to call this.

    Returns a dash-joined string of game labels (used as part of the saved
    figure filename).
    '''
    durations = [g[-1] for g in game_transitions]
    t_max = sum(durations) if t_max is None else t_max
    t = np.arange(t_min, t_max)
    ylo, yhi = ax.get_ylim()
    labels = []

    for game_idx, (raw_label, _, duration) in enumerate(game_transitions):
        t_start = sum(durations[:game_idx])
        t_end   = t_start + duration

        # Shade every other phase for easy visual separation
        if game_idx % 2 == 0:
            ax.fill_between(
                t,
                ylo, yhi,
                where=(t_start <= t) & (t < t_end),
                color='gray', edgecolor='none', alpha=0.1
            )

        # Extract the human-readable part of IDs like "1_2_2_Chicken"
        label = raw_label.split('_')[-1] if raw_label.split('_')[-1] != 'NA' else raw_label
        labels.append(label)

        ax.text(
            t_start + 0.5 * duration,
            ylo + 0.5 * (yhi - ylo),
            r'$\text{' + label + '}$',
            color='gray', alpha=0.4, ha='center'
        )

    return '-'.join(labels)


def get_action_labels(num_actions):
    '''Return human-readable action labels for axis tick marks.'''
    if num_actions == 2:
        return ['$q(u = c)$', '$q(u = d)$']
    if num_actions == 3:
        return ['$q(u = 0)$', '$q(u = 1)$', '$q(u = 2)$']
    # Fallback: binary representation
    bits = int(np.log2(num_actions))
    return [f'{i:0{bits}b}' for i in range(num_actions)]


# ==============================================================================
# Section 4 — Ensemble plot functions  (active surface used by the GUI)
# ==============================================================================
# Naming convention  : plot_<variable>_ensemble
# Signature contract : (data_arg, ..., ifLegend=True, ax=None) -> None
#
# All functions draw individual-seed traces at low alpha plus a bold mean
# line, following the ensemble visualisation style of the project.


def plot_vfe_ensemble(vfe_history, game_transitions, ifLegend=True, ax=None):
    '''Plot ensemble of total VFE (summed over agents and factors) across seeds.

    Args:
        vfe_history    : list of seeds; each seed is (T, agents, factors) array.
        game_transitions: standard game_transitions list (used only to infer
                          axis limits automatically).
        ifLegend       : show legend.
        ax             : matplotlib Axes to draw on.
    '''
    vfe = np.array(vfe_history)                        # (S, T, agents, factors)
    per_seed = vfe.sum(axis=(2, 3))                    # (S, T)
    mean_vfe = per_seed.mean(axis=0)                   # (T,)

    # Data-driven y-limits (replaces the old hardcoded ylim=[0, 60])
    vmin = per_seed.min()
    vmax = per_seed.max()
    span = max(vmax - vmin, 1e-6)

    for s in range(vfe.shape[0]):
        ax.plot(per_seed[s],
                color='blue', alpha=0.15, linewidth=LINEWIDTH,
                label='Individual seeds' if s == 0 else None)

    ax.plot(mean_vfe,
            color=ELBO_COLOR, linewidth=LINEWIDTH * 2, label='Mean')

    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('VFE', fontsize=label_font_size)
    ax.set_ylim(vmin - MARGIN * span, vmax + MARGIN * span)
    if ifLegend:
        ax.legend(loc='upper right', fontsize=label_font_size)


def plot_expected_efe_ensemble(q_u_history, efe_history, ifLegend=True, ax=None):
    '''Plot ensemble of expected EFE = sum_u q(u) * G(u), summed over agents.

    Handles the mixed-shape case (AIF agent with policy_length > 1 paired
    with a DummyAgent) via a try/except fallback to element-wise conversion.

    Args:
        q_u_history : list of seeds; each element is (T, agents, K) array.
        efe_history : list of seeds; each element is (T, agents, K) array.
        ifLegend    : show legend.
        ax          : matplotlib Axes.
    '''
    try:
        q_u = _coerce_q_u_history(q_u_history)        # (S, T, agents, K)
        efe = _coerce_q_u_history(efe_history)         # (S, T, agents, K)
        # Vectorised dot product over the K dimension
        expected = (q_u * efe).sum(axis=-1)            # (S, T, agents)

    except ValueError:
        # Fallback: element-wise conversion for mixed agent types
        S = len(q_u_history)
        T = len(q_u_history[0])
        A = len(q_u_history[0][0])
        expected = np.empty((S, T, A))
        for s in range(S):
            for t in range(T):
                for a in range(A):
                    q = np.array(q_u_history[s][t][a], dtype=float)
                    e = np.array(efe_history[s][t][a], dtype=float)
                    expected[s, t, a] = np.dot(q, e)

    per_seed = expected.sum(axis=-1)                   # (S, T) — sum over agents
    mean_line = per_seed.mean(axis=0)                  # (T,)

    for s in range(per_seed.shape[0]):
        ax.plot(per_seed[s],
                color=EFE_COLOR, alpha=0.2, linewidth=LINEWIDTH,
                label='Individual seeds' if s == 0 else None)

    ax.plot(mean_line,
            color=EFE_COLOR, alpha=1.0, linewidth=LINEWIDTH * 2, label='Mean')

    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('Expected EFE', fontsize=label_font_size)
    if ifLegend:
        ax.legend(loc='upper right', fontsize=label_font_size)


def plot_policies_ensemble(q_u_history, game_transitions, nash_strategy,
                           ifLegend=True, single=False, ax=None):
    '''Plot P(u=c) trajectories for all agents across seeds.

    Agents are colour-coded by their initial behaviour in the first game:
      - cyan  (#03dffc) : initial cooperators (mean q(u=c) ≥ 0.5)
      - pink  (#f70ce4) : initial defectors

    Nash equilibrium cooperation probabilities are overlaid as dashed grey
    horizontal lines, scoped to each game's time window.

    Args:
        q_u_history     : list of seeds; each (T, agents, num_actions).
        game_transitions: standard list of (label, payoff_matrix, duration).
        nash_strategy   : dict mapping game label → NE/NBS probability dicts.
        ifLegend        : show legend (currently suppressed for compactness).
        single          : if True, plot only seed 0 and show both agents.
        ax              : matplotlib Axes.
    '''
    q_u = _coerce_q_u_action_history(q_u_history, game_transitions)
    num_seeds, T, num_agents, _ = q_u.shape
    first_duration = game_transitions[0][-1]

    if single:
        # Single-seed view: distinguish agents by line style
        for a in range(num_agents):
            ax.plot(q_u[0, :, a, 0],
                    alpha=0.85 if a == 0 else 0.6,
                    color='red' if a == 0 else 'black',
                    linewidth=LINEWIDTH,
                    linestyle='-' if a == 0 else '--',
                    label='Agent i' if a == 0 else 'Agent j')
    else:
        # Ensemble view: colour by initial role
        coop_labeled = defect_labeled = False
        for s in range(num_seeds):
            for a in range(num_agents):
                avg = q_u[s, :first_duration, a, 0].mean()
                if avg >= 0.5:
                    color = ACTION_COLORS[0]  # cooperate — cyan
                    label = 'Initial cooperators' if not coop_labeled else None
                    coop_labeled = True
                else:
                    color = ACTION_COLORS[1]  # defect — pink
                    label = 'Initial defectors' if not defect_labeled else None
                    defect_labeled = True
                ax.plot(q_u[s, :, a, 0],
                        alpha=0.5, color=color, linewidth=LINEWIDTH, label=label)

    # Nash equilibrium reference lines — scoped per game phase
    t_start = 0
    for game_name, _, duration in game_transitions:
        if game_name in nash_strategy:
            ne_info = nash_strategy[game_name]
            coop_probs = set()
            for arr in ne_info.get('Row_NE', []):
                coop_probs.add(float(arr[0]))
            for arr in ne_info.get('Col_NE', []):
                coop_probs.add(float(arr[0]))
            for p in coop_probs:
                ax.hlines(p, xmin=t_start, xmax=t_start + duration,
                          colors='gray', linestyles='--',
                          linewidth=LINEWIDTH * 2, alpha=0.3)
        t_start += duration

    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('P(u=c)', fontsize=label_font_size)
    ax.set_ylim(-MARGIN, 1 + MARGIN)


def plot_B_state_ensemble(B_history, q_s_history,
                          ifLegend=True, single=False, collapse=False, ax=None):
    '''Plot ensemble of P(s\'=1 | u) — the B-matrix predicted next state.

    P(s\'=1 | u) is computed as the second element of  B[u] @ q_s  for
    factor=0 (the self-state factor), vectorised over all seeds, timesteps,
    agents, and actions in a single einsum call.

    Args:
        B_history   : list of seeds; each (T, agents, factors, actions, s, s).
        q_s_history : list of seeds; each (T, agents, factors, s).
        ifLegend    : show legend.
        single      : if True, plot only seed 0 at full opacity.
        collapse    : if True, average over actions before plotting.
        ax          : matplotlib Axes.
    '''
    B   = np.array(B_history)    # (S, T, agents, factors, actions, s, s)
    q_s = np.array(q_s_history)  # (S, T, agents, factors, s)

    # Vectorised: P(s'|u) = B[..., u, :, :] @ q_s[..., :]
    # B shape for factor=0: (S, T, agents, actions, s_next, s_curr)
    B_f0   = B[:, :, :, 0, :, :, :]                  # (S, T, agents, actions, 2, 2)
    q_s_f0 = q_s[:, :, :, 0, :]                       # (S, T, agents, 2)

    # einsum: for each (s, t, a, u): sum over s_curr → p_s_next
    p_next = np.einsum('...uij,...j->...ui', B_f0, q_s_f0)  # (S, T, agents, actions, 2)
    p_s1   = p_next[..., 1]                           # (S, T, agents, actions)

    colors = {0: ACTION_COLORS[0], 1: ACTION_COLORS[1]}
    collapse_color = '#7a42f5'
    num_seeds, T, num_agents, num_actions = p_s1.shape

    if collapse:
        p_collapsed = p_s1.mean(axis=-1)              # (S, T, agents)
        for s in range(num_seeds):
            if single and s != 0:
                continue
            for a in range(num_agents):
                ax.plot(p_collapsed[s, :, a],
                        color=collapse_color,
                        alpha=0.9 if single else 0.3,
                        linewidth=LINEWIDTH,
                        label='Individuals' if (s == 0 and a == 0) else None)
    else:
        for s in range(num_seeds):
            if single and s != 0:
                continue
            for a in range(num_agents):
                for u in range(num_actions):
                    label = None
                    if s == 0 and a == 0:
                        label = 'u=cooperate' if u == 0 else 'u=defect'
                    ax.plot(p_s1[s, :, a, u],
                            color=colors[u],
                            alpha=0.9 if single else 0.3,
                            linewidth=LINEWIDTH,
                            label=label)

    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel("Mean P(s'=1|u)" if collapse else "P(s'=1|u)",
                  fontsize=label_font_size)
    ax.set_ylim(-MARGIN, 1 + MARGIN)
    if ifLegend:
        ax.legend(loc='upper right', fontsize=label_font_size)


def plot_B_separation_degree_ensemble(B_history, q_u_history, game_transitions,
                                      factor=0, selectedRole='Cooperators',
                                      ifLegend=True, single=False, average=True,
                                      ax=None):
    '''Plot belief separation degree across seeds, filtered by initial role.

    Separation degree for state s at time t is defined as:

        sep(t, s) = ( |B[u=0, s, s] - B[u=0, 1-s, s]|
                    + |B[u=1, s, s] - B[u=1, 1-s, s]| ) / 2

    A value near 1.0 means the agent has a confident, differentiated model
    of the opponent's response to cooperation vs defection.  A value near 0
    means the agent's B-matrix is still near-uniform — no learned model.

    Args:
        B_history    : list of seeds; each (T, agents, factors, actions, s, s).
        q_u_history  : list of seeds; each (T, agents, K) — used for role tagging.
        game_transitions: standard list.
        factor       : which B factor to evaluate (0 = opponent model).
        selectedRole : 'Cooperators' or 'Defectors' — filters which seeds/agents
                       are included based on mean q(u=c) in the first game.
        ifLegend     : show legend.
        single       : if True and average=False, plot only one seed.
        average      : if True, plot mean over all matching (seed, agent) pairs
                       rather than individual trajectories.
        ax           : matplotlib Axes.
    '''
    B   = np.array(B_history)    # (S, T, agents, factors, actions, s, s)
    q_u = _coerce_q_u_action_history(q_u_history, game_transitions)
    num_seeds, T, num_agents, _, num_actions, num_states, _ = B.shape
    first_duration = game_transitions[0][-1]

    # --- Role assignment: tag each (seed, agent) pair by initial behaviour ---
    # avg q(u=c) over the first game window determines the role label.
    avg_coop = q_u[:, :first_duration, :, 0].mean(axis=1)  # (S, agents)
    roles = np.where(avg_coop >= 0.5, 'Cooperators', 'Defectors')

    # --- Vectorised separation degree computation ---
    # B_f: (S, T, agents, actions, s_next, s_curr)
    B_f = B[:, :, :, factor, :, :, :]

    # For each state s: diag = B[u, s, s], off_diag = B[u, 1-s, s]
    # sep_s = (|diag_c - off_c| + |diag_d - off_d|) / 2
    sep = np.empty((num_seeds, T, num_agents, num_states))
    for s in range(num_states):
        diag_c   = B_f[:, :, :, 0, s,     s]   # (S, T, agents)
        off_c    = B_f[:, :, :, 0, 1 - s, s]
        diag_d   = B_f[:, :, :, 1, s,     s]
        off_d    = B_f[:, :, :, 1, 1 - s, s]
        sep[:, :, :, s] = (np.abs(diag_c - off_c) + np.abs(diag_d - off_d)) / 2

    # --- Filter to the requested role ---
    # role_mask[seed, agent] is True when that pair matches selectedRole
    role_mask = (roles == selectedRole)            # (S, agents)
    matched = [(s, a)
               for s in range(num_seeds)
               for a in range(num_agents)
               if role_mask[s, a]]

    if not matched:
        logging.warning(f'plot_B_separation_degree_ensemble: no {selectedRole} found.')
        return

    color = ACTION_COLORS[0] if selectedRole == 'Cooperators' else ACTION_COLORS[1]

    if average:
        # Mean trajectory over all matched (seed, agent) pairs
        stacked  = np.stack([sep[s, :, a, :] for s, a in matched])  # (N, T, states)
        mean_sep = stacked.mean(axis=0)                              # (T, states)
        for s_idx in range(num_states):
            ax.plot(mean_sep[:, s_idx],
                    color=color,
                    linewidth=LINEWIDTH * 2.2,
                    linestyle='-' if s_idx == 0 else ':',
                    label=f's={s_idx}, factor={factor}' if ifLegend else None)
    else:
        # Individual trajectories (optionally just one seed)
        labeled = [False] * num_states
        for s, a in matched:
            for s_idx in range(num_states):
                label = (f's={s_idx}, factor={factor}'
                         if not labeled[s_idx] and ifLegend else None)
                ax.plot(sep[s, :, a, s_idx],
                        alpha=0.9 if single else 0.3,
                        color=color,
                        linewidth=LINEWIDTH * 1.5,
                        linestyle='-' if s_idx == 0 else ':',
                        label=label)
                if label:
                    labeled[s_idx] = True
            if single:
                break

    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('Separation degree  |diag − off-diag|', fontsize=label_font_size)
    ax.set_ylim(-MARGIN, 1 + MARGIN)
    if ifLegend:
        ax.legend(loc='lower right', fontsize=label_font_size)


def plot_delta_F_ensemble(delta_F_history, candidates,
                          agent_idx=0, factor_idx=0,
                          ifLegend=True, ax=None):
    '''Stacked-area plot of ΔF (BMR free-energy change) per candidate model.

    Shows the mean ΔF across all seeds, rather than only seed 0.

    Each candidate model (e.g. TFT, Grim, Full) contributes a band.
    The relative size of each band over time shows which model the agent is
    converging toward.

    Args:
        delta_F_history : list of seeds; each is list[t][agent][factor][model].
        candidates      : list of seeds; each is list[t][agent] of space-split
                          model-name strings (e.g. "Full TFT Grim").
        agent_idx       : which agent's ΔF to display.
        factor_idx      : which B factor to display.
        ifLegend        : show legend.
        ax              : matplotlib Axes.
    '''
    num_seeds = len(delta_F_history)
    T         = len(delta_F_history[0])
    model_names = candidates[0][0][agent_idx].split()
    num_models  = len(model_names)

    # Collect ΔF over all seeds → shape (S, T, num_models)
    all_data = np.zeros((num_seeds, T, num_models))
    for s in range(num_seeds):
        for t in range(T):
            for m in range(num_models):
                all_data[s, t, m] = float(
                    delta_F_history[s][t][agent_idx][factor_idx][m]
                )

    mean_data = all_data.mean(axis=0).T   # (num_models, T)

    if ifLegend:
        ax.stackplot(range(T), mean_data, labels=model_names, alpha=0.7)
    else:
        ax.stackplot(range(T), mean_data, alpha=0.7)

    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('ΔF (mean across seeds)', fontsize=label_font_size)
    ax.set_ylim(-MARGIN, 1 + MARGIN)
    ax.grid(True, alpha=0.3)
    if ifLegend:
        ax.legend(loc='upper right', fontsize=label_font_size * 0.8)


def plot_B_model_weights_ensemble(weights_history, candidates, agent_idx,
                                  ifLegend=True, ax=None):
    '''Stacked-area plot of B-model softmax weights per candidate strategy.

    Shows the mean weight across all seeds, rather than only seed 0.
    The weight for each candidate (TFT, Grim, Full, …) reflects how well
    that transition template explains the agent's current B-matrix under BMR.

    Args:
        weights_history : list of seeds; each is list[t][agent][factor][model].
        candidates      : list of seeds; each is list[t][agent] of space-split
                          model-name strings.
        agent_idx       : which agent's weights to display (0=agent i, 1=agent j).
        ifLegend        : show legend.
        ax              : matplotlib Axes.
    '''
    num_seeds   = len(weights_history)
    T           = len(weights_history[0])
    factor_idx  = 0
    model_names = candidates[0][0][agent_idx].split()
    num_models  = len(model_names)

    # Collect weights over all seeds → shape (S, T, num_models)
    all_data = np.zeros((num_seeds, T, num_models))
    for s in range(num_seeds):
        for t in range(T):
            for m in range(num_models):
                all_data[s, t, m] = float(
                    weights_history[s][t][agent_idx][factor_idx][m]
                )

    mean_data = all_data.mean(axis=0).T   # (num_models, T)

    if ifLegend:
        ax.stackplot(range(T), mean_data, labels=model_names, alpha=0.5)
    else:
        ax.stackplot(range(T), mean_data, alpha=0.5)

    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('Model weight (mean across seeds)', fontsize=label_font_size)
    ax.set_ylim(-MARGIN, 1 + MARGIN)
    ax.grid(True, alpha=0.3)
    if ifLegend:
        ax.legend(loc='lower right', fontsize=label_font_size)


def plot_entropy_ensemble(entropy_history, ifLegend=True, ax=None):
    '''Plot ensemble of H(q_s) — state-belief entropy — for each agent.

    Only the self-state factor (factor=0) is shown, because the opponent-state
    factor (factor=1) is redundant for the ego-centric analysis.

    Args:
        entropy_history : list of seeds; each (T, agents, factors) array.
        ifLegend        : show legend.
        ax              : matplotlib Axes.
    '''
    entropy = np.array(entropy_history)     # (S, T, agents, factors)
    num_seeds, T, num_agents, _ = entropy.shape
    linestyles = ['-', '--', ':', '-.']
    agent_colors = [ENTROPY_COLOR, 'red', 'green', 'orange']

    individual_alpha = 0.3 if num_seeds > 1 else 1.0

    for s in range(num_seeds):
        for a in range(num_agents):
            label = f'Agent {a} (ego)' if s == 0 else None
            ax.plot(entropy[s, :, a, 0],       # factor=0 only
                    color=agent_colors[a % len(agent_colors)],
                    alpha=individual_alpha,
                    linewidth=LINEWIDTH,
                    linestyle=linestyles[a % len(linestyles)],
                    label=label)

    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('H(q_s)  (nats)', fontsize=label_font_size)
    ax.grid(True, alpha=0.3)
    if ifLegend:
        ax.legend(loc='upper right', fontsize=label_font_size)


# ==============================================================================
# Section 5 — Ensemble plots: gamma and realised actions
# ==============================================================================
# Moved here from INFG-plot-gui.py (Sections 6b / 6c).
# Both follow the standard ensemble contract and have no GUI dependency.


def plot_gamma_ensemble(all_gamma: list, game_transitions: list,
                        ifLegend: bool = True, ax=None):
    '''Plot ensemble of precision γ across seeds, one line per agent.

    Faint per-seed traces are overlaid with a bold per-agent mean line,
    colour-coded so agent i (AIF) and agent j (dummy) are visually distinct.

    Args:
        all_gamma       : list of seeds; each entry is (T, num_agents) array
                          holding γ at every timestep.
        game_transitions: standard list (not used for data, kept for API
                          consistency with other ensemble functions).
        ifLegend        : show legend.
        ax              : matplotlib Axes.
    '''
    if ax is None:
        _, ax = plt.subplots()

    try:
        gamma_arr = np.array(all_gamma)          # (S, T, num_agents)
    except ValueError:
        gamma_arr = np.stack(
            [np.array(g) for g in all_gamma], axis=0)

    num_seeds, T, num_agents = gamma_arr.shape
    agent_colors = [PRECISION_COLOR, '#888888', '#4477aa', '#cc6677'][:num_agents]

    for a in range(num_agents):
        color      = agent_colors[a]
        label_stem = f'Agent {chr(105 + a)}'
        for s in range(num_seeds):
            ax.plot(gamma_arr[s, :, a],
                    color=color, alpha=0.15, linewidth=LINEWIDTH)
        mean_gamma = gamma_arr[:, :, a].mean(axis=0)
        ax.plot(mean_gamma,
                color=color, alpha=1.0,
                linewidth=LINEWIDTH * 2,
                label=f'{label_stem} mean')

    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('Precision γ',   fontsize=label_font_size)
    if ifLegend:
        ax.legend(loc='upper right', fontsize=label_font_size)


def _coerce_u_history(all_u: list) -> np.ndarray:
    '''Convert per-seed realised-action histories to a float64 array.

    Returns ndarray of shape (num_seeds, T, num_agents) with values in {0, 1}.
    Values may arrive as plain Python ints or torch scalar tensors; both are
    handled by the defensive float() fallback path.
    '''
    try:
        return np.array(all_u, dtype=float)
    except (ValueError, TypeError):
        pass
    converted = [
        [[float(a) for a in step] for step in seed_hist]
        for seed_hist in all_u
    ]
    return np.array(converted, dtype=float)


def plot_actions_single(all_u: list, all_q_u: list, game_transitions: list,
                        ifLegend: bool = True, seed_idx: int = 0, ax=None):
    '''Overlay realised discrete actions on the continuous policy P(u=c).

    Displays one seed (default: seed 0) showing both the belief trajectory
    q(u=c) as a continuous line and the committed actions as isolated tick
    marks in dedicated rows outside the [0, 1] band.

    Args:
        all_u           : list of seeds; each (T, num_agents) of {0, 1} ints.
                          Confirmed to be the actual committed action from
                          agent.py: self.u = policies[idx][0].item().
        all_q_u         : list of seeds; same format as plot_policies_ensemble.
        game_transitions: standard list; passed to _coerce_q_u_action_history.
        ifLegend        : show legend.
        seed_idx        : which seed to render (default 0).
        ax              : matplotlib Axes.

    Rendering note
    --------------
    Actions are drawn as isolated tick marks (marker="|") rather than a
    connected step-line through {0, 1}.  When q(u=c) ≈ 0.5 the agent
    toggles almost every step; a connected binary line would fill the
    entire belief band solidly, making it indistinguishable from the
    probability curve itself.  Tick marks outside [0, 1] stay readable
    at any toggling frequency.
    '''
    if ax is None:
        _, ax = plt.subplots()

    u_arr = _coerce_u_history(all_u)              # (S, T, agents)
    num_seeds, T, num_agents = u_arr.shape
    s = min(seed_idx, num_seeds - 1)

    q_u_arr = _coerce_q_u_action_history(all_q_u, game_transitions)

    agent_colors = ['red', 'black', '#4477aa', '#cc6677'][:num_agents]
    t_axis   = np.arange(T)
    row_step = 0.10   # vertical gap between agents' tick rows

    for a in range(num_agents):
        color      = agent_colors[a]
        label_stem = f'Agent {chr(105 + a)}'

        # Continuous belief
        ax.plot(q_u_arr[s, :, a, 0],
                color=color, alpha=0.85,
                linewidth=LINEWIDTH,
                linestyle='-' if a == 0 else '--',
                label=f'{label_stem}  P(u=c)')

        actions     = u_arr[s, :, a]
        coop_mask   = actions == 0   # u=0 → cooperate
        defect_mask = actions == 1   # u=1 → defect
        coop_row    = 1.0 + row_step * (a + 1)
        defect_row  = 0.0 - row_step * (a + 1)

        ax.scatter(t_axis[coop_mask],
                   np.full(coop_mask.sum(), coop_row),
                   marker='|', s=25, linewidths=0.8,
                   color=color, alpha=0.7,
                   label=f'{label_stem}  action' if ifLegend else None)
        ax.scatter(t_axis[defect_mask],
                   np.full(defect_mask.sum(), defect_row),
                   marker='|', s=25, linewidths=0.8,
                   color=color, alpha=0.7)

    top_row    = 1.0 + row_step * (num_agents + 0.6)
    bottom_row = 0.0 - row_step * (num_agents + 0.6)
    ax.set_ylim(bottom_row, top_row)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.text(0.0, top_row    - row_step * 0.3, 'cooperate',
            fontsize=max(label_font_size - 2, 6), color='gray', va='top')
    ax.text(0.0, bottom_row + row_step * 0.3, 'defect',
            fontsize=max(label_font_size - 2, 6), color='gray', va='bottom')

    ax.set_xlabel('Time step', fontsize=label_font_size)
    ax.set_ylabel('P(u=c)',    fontsize=label_font_size)
    if ifLegend:
        ax.legend(loc='right', fontsize=max(label_font_size - 1, 6))


# ==============================================================================
# Section 6 — Utility plot: state-space trajectory
# ==============================================================================
# This function is not an ensemble panel but is used by external notebooks
# for phase-space visualisation.

def plot_state_space_trajectory(
        trajectory,
        ax=None,
        legend=False,
        gradient=False,
        lines=True,
        vertex_markers=True,
        additional_trajectory_kwargs=None):
    '''Plot a 2D or 3D state-space trajectory of agent cooperation probabilities.

    Args:
        trajectory : ndarray of shape (T, num_agents) — columns are q(u=c) per agent.
        ax         : matplotlib Axes (2D) or Axes3D (3D).  Created if None.
        legend     : show vertex labels.
        gradient   : colour the trajectory by time using the viridis colormap.
        lines      : draw connecting line segments between steps.
        vertex_markers : mark the corners of the cooperation simplex.
        additional_trajectory_kwargs : merged into the trajectory line kwargs.

    Example::

        q_u = np.array(variables_history['q_u'])
        plot_state_space_trajectory(q_u[:, :, 0])
    '''
    if additional_trajectory_kwargs is None:
        additional_trajectory_kwargs = {}

    num_agents = trajectory.shape[1]
    T          = trajectory.shape[0]

    corner_kwargs = dict(alpha=1., marker='^', s=50, zorder=3)
    trajectory_kwargs = dict(
        label='Trajectory', linewidth=0.5, marker='o', markersize=2
    ) | additional_trajectory_kwargs

    colors = None
    if gradient:
        from matplotlib.cm import viridis
        from matplotlib.colors import Normalize
        norm   = Normalize(vmin=0, vmax=T - 1)
        colors = viridis(norm(np.arange(T)))

    if num_agents == 3:
        from mpl_toolkits.mplot3d import Axes3D
        if ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')

        if gradient:
            for t in range(T - 1):
                ax.plot(trajectory[t, 0], trajectory[t, 1], trajectory[t, 2],
                        color=colors[t], **trajectory_kwargs)
                if lines:
                    ax.plot(trajectory[t:t+2, 0], trajectory[t:t+2, 1],
                            trajectory[t:t+2, 2], color=colors[t], alpha=0.4)
        else:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                    **trajectory_kwargs)

        if vertex_markers:
            ax.scatter(0, 0, 0, label='0C', color='#fc03ad', **corner_kwargs)
            ax.scatter([1, 0, 0], [0, 1, 0], [0, 0, 1],
                       label='1C', color='#c90411', **corner_kwargs)
            ax.scatter([0, 1, 1], [1, 0, 1], [1, 1, 0],
                       label='2C', color='#fc6f03', **corner_kwargs)
            ax.scatter(1, 1, 1, label='3C', color='#39c904', **corner_kwargs)

        ax.set_xlabel('$q(u_i = c)$')
        ax.set_ylabel('$q(u_j = c)$')
        ax.set_zlabel('$q(u_k = c)$')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])

    elif num_agents == 2:
        if ax is None:
            _, ax = plt.subplots()

        if gradient:
            for t in range(T - 1):
                ax.plot(trajectory[t, 0], trajectory[t, 1],
                        color=colors[t], **trajectory_kwargs)
                if lines:
                    ax.plot(trajectory[t:t+2, 0], trajectory[t:t+2, 1],
                            color=colors[t], alpha=0.4)
        else:
            ax.plot(trajectory[:, 0], trajectory[:, 1], **trajectory_kwargs)

        if vertex_markers:
            ax.scatter(0, 0, label='0C', color='#fc03ad', **corner_kwargs)
            ax.scatter([1, 0], [0, 1], label='1C', color='#fc6f03', **corner_kwargs)
            ax.scatter(1, 1,   label='2C', color='#39c904', **corner_kwargs)

        ax.set_xlabel('$q(u_i = c)$')
        ax.set_ylabel('$q(u_j = c)$')
        ax.set_aspect('equal', adjustable='datalim')

    if legend:
        ax.legend()
    ax.set_title('State-space trajectory')