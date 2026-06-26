'''
Functions for plotting results

Authors: Pat Sweeney, Jaime Ruiz Serra
Date:    2024/08

Extended by: Hanchen Wang
Date: 2025/11
'''
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import dirichlet
import torch

EPSILON = torch.finfo().eps

DPI = 72
SHOW_LEGEND = True
ONLY_LEFT_Y_LABEL = False
TIGHT_LAYOUT = False
MARGIN = 0.05  # Percentage margin for ylim
LEGEND_IDX = 1

# Define consistent color settings
VFE_COLOR = '#ff00ff'
ENERGY_COLOR = '#ccccff'
COMPLEXITY_COLOR = '#8a2be2'
ELBO_COLOR = '#000080'
ENTROPY_COLOR = '#468499'
ACCURACY_COLOR = '#00ced1'
EFE_COLOR = '#990000'
RISK_COLOR = '#ff7f50'
AMBIGUITY_COLOR = '#f6546a'
EXP_ELBO_COLOR = '#065535'
PRAGMATIC_COLOR = '#66cdaa'
SALIENCE_COLOR = '#00ff00'
NOVELTY_COLOR = '#fc03d0'
PRECISION_COLOR = '#000000'
POLICY_ENTROPY_COLOR = '#1F948B'
ACTION_COLORS = ['#03dffc', '#f70ce4', '#fcad03', '#fc03d0', '#03fcad', '#ad03fc', '#adfc03', '#fcad03', '#fc03ad',
                 '#03adfc']

SIZE_MULTIPLIER = 1.4
label_font_size = 8 * SIZE_MULTIPLIER  # Define a consistent font size for labels
LINEWIDTH = 0.7 * SIZE_MULTIPLIER

# Set consistent font size and style for all labels and titles
plt.rc('font', size=label_font_size)  # Default text size
plt.rc('axes', titlesize=14, labelsize=label_font_size)  # Axes titles and labels
plt.rc('xtick', labelsize=label_font_size)  # X-tick labels
plt.rc('ytick', labelsize=label_font_size)  # Y-tick labels
plt.rc('legend', fontsize=label_font_size)  # Legend font size
plt.rc('font', family='serif')  # Use serif fonts
plt.rc('text', usetex=False)  # Disable LaTeX rendering for simplicity (enable if needed)


# ==============================================================================
# Helper functions
# ==============================================================================

def unstack(a, axis=0):
    '''Source: https://stackoverflow.com/a/64097936/21740893'''
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis=axis)]


def get_action_labels(num_actions):
    '''Define action labels based on the number of actions'''
    if num_actions == 2:
        return ['$q(u = c)$', '$q(u = d)$']
    elif num_actions == 3:
        return ['$q(u = 0)$', '$q(u = 1)$', '$q(u = 2)$']
    else:
        # return [f'Action {i}' for i in range(num_actions)]
        # Return binary representation of actions
        return [f'{i:0{int(np.log2(num_actions))}b}' for i in range(num_actions)]

def _coerce_q_u_history(q_u_history):
    """
    Convert nested q_u history (lists of torch tensors / arrays) into a numeric numpy array.
    Supports both single-seed (T, agents, K) and multi-seed (S, T, agents, K) layouts.
    """
    arr = np.array(q_u_history, dtype=object)
    if arr.dtype != object:
        return arr

    # Try multi-seed: list[seed][t][agent] -> vector
    try:
        return np.array(
            [[[np.array(a, dtype=float) for a in step] for step in seed] for seed in q_u_history],
            dtype=float
        )
    except Exception:
        # Fallback to single-seed: list[t][agent] -> vector
        return np.array(
            [[np.array(a, dtype=float) for a in step] for step in q_u_history],
            dtype=float
        )

def _marginalize_policy(q_u_history, game_transitions):
    """
    Convert policy-distribution q_u (over action sequences) into marginal action probabilities.
    If already action-level, returns unchanged.
    """
    if not game_transitions:
        return q_u_history
    num_actions = game_transitions[0][1].shape[0]
    last_dim = q_u_history.shape[-1]
    if last_dim == num_actions:
        return q_u_history
    if last_dim % num_actions != 0:
        return q_u_history
    return q_u_history.reshape(*q_u_history.shape[:-1], num_actions, -1).sum(axis=-1)

def _coerce_q_u_action_history(q_u_history, game_transitions):
    """
    Convert q_u history into action-level probabilities with uniform length per agent.
    Handles mixed agent types (e.g., active agent with policy-length>1 and dummy agent).
    """
    num_actions = game_transitions[0][1].shape[0] if game_transitions else None

    def to_action_vec(q):
        q_arr = np.array(q, dtype=float)
        if num_actions is None:
            return q_arr
        if q_arr.shape[0] == num_actions:
            return q_arr
        if q_arr.shape[0] % num_actions != 0:
            raise ValueError("q_u length is not divisible by num_actions; cannot marginalize.")
        return q_arr.reshape(num_actions, -1).sum(axis=1)

    def convert_seed(seed_hist):
        out = []
        for step in seed_hist:
            out.append([to_action_vec(q) for q in step])
        return np.array(out, dtype=float)

    if isinstance(q_u_history, np.ndarray) and q_u_history.dtype != object:
        arr = q_u_history
        if num_actions is not None and arr.shape[-1] != num_actions:
            arr = arr.reshape(*arr.shape[:-1], num_actions, -1).sum(axis=-1)
        return arr

    # Multi-seed stored as list of numpy arrays (one per seed)
    if isinstance(q_u_history, list) and q_u_history and isinstance(q_u_history[0], np.ndarray):
        arr = np.array(q_u_history, dtype=float)
        if num_actions is not None and arr.shape[-1] != num_actions:
            arr = arr.reshape(*arr.shape[:-1], num_actions, -1).sum(axis=-1)
        return arr

    # Multi-seed: list[seed][t][agent]
    if isinstance(q_u_history, list) and q_u_history and isinstance(q_u_history[0], list) and q_u_history[0] and isinstance(q_u_history[0][0], list):
        return np.array([convert_seed(seed) for seed in q_u_history], dtype=float)

    # Single-seed: list[t][agent]
    return convert_seed(q_u_history)

# def get_figure_size(num_players, num_actions, base_width=6, base_height=4):
#     '''Determine figure size based on the number of players and actions'''
#     width = base_width + num_players * 2
#     height = base_height + num_actions * 2
#     return (width, height)

def highlight_transitions(game_transitions, ax, t_min=0, t_max=None):
    durations = [g[-1] for g in game_transitions]
    t_max = sum(durations) if t_max is None else t_max
    t = np.arange(t_min, t_max)

    labels = []

    for game_idx, (raw_label, payoffs, duration) in enumerate(game_transitions):
        # Highlight game transitions (every other game)
        if game_idx % 2 == 0:
            ax.fill_between(
                t,
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                where=(sum(durations[:game_idx]) <= t) & (t < sum(durations[:game_idx + 1])),
                color='gray',
                edgecolor='none',
                alpha=0.1)

        label = raw_label.split("_")[-1] if raw_label.split("_")[-1] != "NA" else raw_label
        labels.append(label)

        # Game labels
        y_mid = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2
        ax.text(
            sum(durations[:game_idx]) + (1 / 2) * durations[game_idx],  # x position
            y_mid,  # y position: halfway up the y-axis
            r'$\text{' + label + '}$',
            color='gray',
            alpha=0.4,
            ha='center')

    # Pass extract game labels for naming to INFG-plot.py
    return "-".join(labels)


def compute_cumulative_sum(arr):
    cumulative_sum = np.int64(0)
    result = []
    for num in arr:
        cumulative_sum += num
        result.append(cumulative_sum)
    return result


def add_learning_marker(i, ax, t_min, t_max,
                        content, learn_record_refined,
                        point_offset=0.04, text_offset=0.08):
    # Ensure np array
    content = np.array(content)

    # Add markers for learning steps from learn_record_refined
    if learn_record_refined is not None and i < len(learn_record_refined):
        current_learn_steps = learn_record_refined[i]
        # Filter steps within the current plotting range
        valid_steps = [step for step in current_learn_steps if t_min <= step < t_max]
        if valid_steps:
            # Get corresponding y-values from content
            y_points = content[[step - t_min for step in valid_steps]]
            # Use a distinct marker and label
            ax.scatter(
                valid_steps,
                y_points - point_offset,
                color='yellow',
                marker='^',
                zorder=5,  # Ensure markers are on top
                label='Learning Step' if i == 0 else None,
                s=40,
                edgecolors='black',  # Add edge for contrast
                alpha=0.5
            )

            # Add text labels below markers
            for x, y in zip(valid_steps, y_points):
                ax.text(
                    x,
                    y - text_offset,  # Position below marker
                    str(x),  # Present x-value as string
                    ha='center',
                    va='top',
                    fontsize=label_font_size,
                    fontname="Calibri",
                    weight="bold",
                    color='red'
                )


# ==============================================================================
# Main plotting function
# ==============================================================================

def make_default_config(variables_history, nash_strategy, game_transitions):
    # Data preprocessing
    required_variables_names = [
        'VFE',
        'energy',
        'entropy',
        'accuracy',
        'complexity',
        'EFE',
        'EFE_terms',
        'gamma',
        'q_s',
        'q_u',
        'u',
        'A',
        'B',
        # 'payoff',
        'learn_record',
        'o_pred_record',
        # 'A_model_change',
        'B_model_change',
        'delta_F',
        'F_full',
        'F_red',
    ]
    # Extract learn intervals data
    learn_record = variables_history['learn_record']
    # Identify steps where learning happens
    learn_steps_i = []
    learn_steps_j = []
    for idx in range(0, len(learn_record)):
        value_i = learn_record[idx][0]
        value_j = learn_record[idx][1]
        if value_i != 0:
            learn_steps_i.append(value_i)
        if value_j != 0:
            learn_steps_j.append(value_j)
    learn_steps_i = compute_cumulative_sum(learn_steps_i)
    learn_steps_j = compute_cumulative_sum(learn_steps_j)
    # Reform learn_record_refined passing into the plot functions
    learn_record_refined = [learn_steps_i, learn_steps_j]

    # Extract o_pred data
    o_pred_record = variables_history['o_pred_record']

    for required_variable_name in required_variables_names:
        if not isinstance(variables_history[required_variable_name], np.ndarray):
            variables_history[required_variable_name] = np.array(
                variables_history[required_variable_name]
            ).squeeze()
        else:
            variables_history[required_variable_name] = (
                variables_history[required_variable_name].copy().squeeze()
            )

    (
        ambiguity,  # each of shape (T, num_actions**policy_length, policy_length)
        risk,
        salience,
        pragmatic_value,
        novelty
    ) = unstack(variables_history['EFE_terms'], axis=-1)

    # variables_history['EFE_terms'][0][0].shape == torch.Size([2, 1, 5])

    # Define what plots we want to compute and show
    plot_configs = [
        {'plot_fn': plot_precision,
         'args': (variables_history['gamma'],)},

        {'plot_fn': plot_efe,
         'args': (
             variables_history['EFE'],
             risk,
             ambiguity,
             pragmatic_value,
             salience,
             novelty,
             learn_record_refined,)},

        {'plot_fn': plot_efe_diff_decomposition,
         'args': (
             variables_history['EFE'],
             risk,
             ambiguity,
             pragmatic_value,
             salience,
             novelty,
             learn_record_refined,)},

        {'plot_fn': plot_vfe,
         'args': (
             variables_history['VFE'],
             variables_history['energy'],
             variables_history['complexity'],
             variables_history['entropy'],
             variables_history['accuracy'])},

        {'plot_fn': plot_cooperation_prob,
         'args': (
             variables_history['q_u'],
             nash_strategy,
             game_transitions,
             learn_record_refined,)},

        {'plot_fn': plot_policy_heatmap,
         'args': (variables_history['q_u'],)},

        {'plot_fn': plot_inferred_policy_heatmap,
         'args': (variables_history['q_s'],)},

        {'plot_fn': plot_B_overtime,
         'args': (variables_history['B'],
                  [0, 0])},
        {'plot_fn': plot_B_overtime,
         'args': (variables_history['B'],
                  [0, 1])},
        {'plot_fn': plot_B_overtime,
         'args': (variables_history['B'],
                  [1, 0])},
        {'plot_fn': plot_B_overtime,
         'args': (variables_history['B'],
                  [1, 1])},

        {'plot_fn': plot_learn_model_change,
         'args': (variables_history['B_model_change'],
                  learn_record_refined,)},

        {'plot_fn': plot_full_red_model_F,
         'args': (variables_history['delta_F'],
                  variables_history['F_red'],
                  variables_history['F_full'],)},

        {'plot_fn': plot_observation_prediction,
         'args': (
             o_pred_record,
             learn_record_refined,)},

        {'plot_fn': plot_expected_efe,
         'args': (
             variables_history['q_u'],
             variables_history['EFE'],
             risk,
             ambiguity,
             pragmatic_value,
             salience,
             novelty,)},

        {'plot_fn': plot_policy_entropy,
         'args': (variables_history['q_u'],)},

        # {'plot_fn': plot_A,
        # 'args': (variables_history['A'], )},

        # {'plot_fn': plot_B_snapshot,
        #  'args': (variables_history['B'],)},
    ]

    return plot_configs


def plot(plot_configs=None, num_players=0,
         suptitle=None, game_transitions=None, figsize=(14, 22), height_ratios=None,
         t_min=0, t_max=None):
    # Create a figure with subplots for each agent, arranged in a (num_plots)x(num_agents) grid
    n_rows = len(plot_configs)
    fig, axes = plt.subplots(n_rows, num_players, figsize=figsize, dpi=DPI, height_ratios=height_ratios)
    fig.subplots_adjust(hspace=1, wspace=0.1)  # Adjust spacing

    for row_idx in range(n_rows):
        for col_idx in range(num_players):
            ax = axes[row_idx, col_idx]
            plot_configs[row_idx]['plot_fn'](
                *plot_configs[row_idx]['args'],
                ax=ax, i=col_idx,
                t_min=t_min, t_max=t_max,
            )
            if (
                    game_transitions
                    and not t_max
                    and plot_configs[row_idx]['plot_fn'].__name__ not in [
                'plot_policy_heatmap',
                'plot_inferred_policy_heatmap',
                'plot_A'
            ]
            ):  # TODO: show transitions even if t_max provided
                highlight_transitions(game_transitions, ax)
            if TIGHT_LAYOUT and row_idx < n_rows - 1:
                ax.set_xlabel(None)

    # Adjust layout for better spacing and overall title
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make space for the suptitle
    if TIGHT_LAYOUT:
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
    # plt.show()

    return fig


# ==============================================================================
# Plotting functions
# ==============================================================================

def plot_vfe(
        vfe_history,
        energy_history, complexity_history,
        entropy_history, accuracy_history,
        ax, i,
        t_min=0, t_max=None
):
    t_max = len(vfe_history) if t_max is None else t_max
    x_range = range(t_min, t_max)

    summed_vfe = vfe_history[t_min:t_max, i].sum(axis=1)
    summed_energy = energy_history[t_min:t_max, i].sum(axis=1)
    summed_entropy = entropy_history[t_min:t_max, i].sum(axis=1)
    summed_complexity = complexity_history[t_min:t_max, i].sum(axis=1)
    summed_accuracy = accuracy_history[t_min:t_max, i].sum(axis=1)

    vfe_plot = ax.plot(
        x_range,
        summed_vfe,
        label='$F[q, o]$', color=ELBO_COLOR, linestyle='-', linewidth=LINEWIDTH, alpha=0)

    ax2 = ax  # .twinx()

    energy_plot = ax2.plot(
        x_range,
        summed_energy,
        label='Energy',
        color=ENERGY_COLOR,
        linestyle=':',
        linewidth=LINEWIDTH,
        alpha=0.7)
    entropy_plot = ax2.plot(
        x_range,
        summed_entropy,
        label='Entropy',
        color=ENTROPY_COLOR,
        linestyle=':',
        linewidth=LINEWIDTH,
        alpha=0.7)
    complexity_plot = ax2.plot(
        x_range,
        summed_complexity,
        label='Complexity',
        color=COMPLEXITY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    accuracy_plot = ax2.plot(
        x_range,
        -summed_accuracy,
        label='Accuracy', color=ACCURACY_COLOR, linestyle=':', linewidth=LINEWIDTH)

    ax.set_title(f'VFE Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    # ylim range based on all agents (not just the current agent i)
    ymin = vfe_history[t_min:t_max].sum(axis=2).min()
    ymax = vfe_history[t_min:t_max].sum(axis=2).max()
    margin = MARGIN * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('$F[q, o]$', color='black', fontsize=label_font_size)
        ax2.set_ylabel('Nats', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
        ax2.set_yticklabels([])

    # Combine both legends in one box
    lines = [
        vfe_plot[0],
        complexity_plot[0], accuracy_plot[0],
        energy_plot[0], entropy_plot[0]
    ]
    labels = [line.get_label() for line in lines]
    if i == 0:
        ax.legend(lines, labels)
    # ax.legend(loc='upper left')
    # ax2.legend(loc='upper right')


def plot_vfe_per_factor(
        vfe_history,
        energy_history, complexity_history,
        entropy_history, accuracy_history,
        ax, i,
        t_min=0, t_max=None
):
    t_max = len(vfe_history) if t_max is None else t_max
    x_range = range(t_min, t_max)

    for j in range(vfe_history.shape[-1]):
        vfe_plot = ax.plot(
            x_range,
            vfe_history[t_min:t_max, i, j],
            label='VFE', color=ELBO_COLOR, linestyle='-', linewidth=LINEWIDTH)

    ax.set_title(f'VFE Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    # ylim range based on all agents (not just the current agent i)
    ymin = vfe_history[t_min:t_max].min()
    ymax = vfe_history[t_min:t_max].max()
    margin = MARGIN * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('VFE', color='black', fontsize=label_font_size)
        # ax2.set_ylabel('Nats', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        # Combine both legends in one box
        lines = [vfe_plot[0]]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc='lower right')


def plot_vfe_complexity_energy(vfe_history, energy_history, complexity_history, ax, i):
    summed_vfe = vfe_history[:, i].sum(axis=1)
    summed_energy = energy_history[:, i].sum(axis=1)
    summed_complexity = complexity_history[:, i].sum(axis=1)

    ax.plot(
        range(len(vfe_history)),
        summed_vfe,
        label='VFE', color=VFE_COLOR, linewidth=LINEWIDTH)

    ax2 = ax  # .twinx()
    ax2.plot(
        range(len(summed_complexity)),
        summed_complexity,
        label='Complexity',
        color=COMPLEXITY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    ax2.plot(
        range(len(summed_energy)),
        summed_energy,
        label='Energy', color=ENERGY_COLOR, linestyle=':', linewidth=LINEWIDTH)

    ax.set_title(f'VFE Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('VFE', color='black', fontsize=label_font_size)
        ax2.set_ylabel('Nats', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')


def plot_vfe_accuracy_entropy(vfe_history, entropy_history, accuracy_history, ax, i):
    summed_vfe = vfe_history[:, i].sum(axis=1)
    summed_entropy = entropy_history[:, i].sum(axis=1)
    summed_accuracy = accuracy_history[:, i].sum(axis=1)

    ax.plot(
        range(len(summed_vfe)),
        summed_vfe,
        label='VFE', color=ELBO_COLOR, linestyle='-', linewidth=LINEWIDTH)

    ax2 = ax  # .twinx()
    ax2.plot(
        range(len(summed_accuracy)),
        -summed_accuracy,
        label='Accuracy', color=ACCURACY_COLOR, linestyle=':', linewidth=LINEWIDTH)
    ax2.plot(
        range(len(summed_entropy)),
        summed_entropy,
        label='Entropy', color=ENTROPY_COLOR, linestyle=':', linewidth=LINEWIDTH)

    ax.set_title(f'VFE Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('VFE', color='black', fontsize=label_font_size)
        ax2.set_ylabel('Nats', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')


def plot_efe(
        efe_history,
        risk,
        ambiguity,
        pragmatic_value,
        salience,
        novelty,
        learn_record_refined,
        ax, i,
        t_min=0, t_max=None
):
    t_max = len(efe_history) if t_max is None else t_max
    x_range = range(t_min, t_max)

    # Only plot EFE terms if they are non-zero (i.e. if actual data was provided)
    # (as a way to stub out the plotting of EFE terms)
    # FIXME: Use kwargs instead?
    plot_EFE_terms = (not np.all(risk == 0))

    if plot_EFE_terms:
        ax2 = ax  # .twinx()

    num_actions = efe_history.shape[-1]
    # Differentiate 2x2 and 2x3 games
    if num_actions == 2:
        action_labels = ["c", "d"]  # Cooperate/Defect
    else:
        action_labels = [f"s{j}" for j in range(num_actions)]  # Strategy 0/1/2
    for a in range(num_actions):
        efe_plot = ax.plot(
            x_range,
            efe_history[t_min:t_max, i, a],
            # label='$G[\mathtt{' + ['c', 'd'][a] + '}]$',
            label=f"$G[\\mathtt{{{action_labels[a]}}}]$",
            color=ACTION_COLORS[a],
            linewidth=LINEWIDTH
        )

        if plot_EFE_terms:
            # risk_plot = ax2.plot(
            #     x_range,
            #     risk[t_min:t_max, i, a],
            #     label='$\mathcal{R}[\mathtt{'+['c', 'd'][a]+'}]$',
            #     color=ACTION_COLORS[a],
            #     linestyle=':', linewidth=LINEWIDTH, alpha=0.5)
            # ambiguity_plot = ax2.plot(
            #     x_range,
            #     ambiguity[t_min:t_max, i, a],
            #     label='$\mathcal{A}[\mathtt{'+['c', 'd'][a]+'}]$',
            #     color=ACTION_COLORS[a],
            #     linestyle='--', linewidth=LINEWIDTH, alpha=0.5)
            pv_plot = ax2.plot(
                x_range,
                -pragmatic_value[t_min:t_max, i, a],
                # label=r'$-\rho[\mathtt{' + ['c', 'd'][a] + '}]$',
                label=rf'$-\rho[\mathtt{{{action_labels[a]}}}]$',
                color=ACTION_COLORS[a], linestyle=(0, (1, 5)), linewidth=LINEWIDTH, alpha=0.7)
            salience_plot = ax2.plot(
                x_range,
                salience[t_min:t_max, i, a],
                # label=r'$\varsigma[\mathtt{' + ['c', 'd'][a] + '}]$',
                label=rf'$\varsigma[\mathtt{{{action_labels[a]}}}]$',
                color=ACTION_COLORS[a], linestyle=(0, (5, 7)), linewidth=LINEWIDTH, alpha=0.7)
            novelty_plot = ax.plot(
                x_range,
                novelty[t_min:t_max, i, a],
                # label=r'$\eta[\mathtt{' + ['c', 'd'][a] + '}]$',
                label=rf'$\eta[\mathtt{{{action_labels[a]}}}]$',
                color=ACTION_COLORS[a],
                linestyle=':',
                linewidth=LINEWIDTH
            )

    ax.set_title(f'EFE Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    # ylim range based on all agents (not just the current agent i)
    ymin = min(
        efe_history[t_min:t_max].min(),
        risk[t_min:t_max].min(),
        ambiguity[t_min:t_max].min(),
        -pragmatic_value[t_min:t_max].max(),  # negative
        salience[t_min:t_max].min(),
        novelty[t_min:t_max].min()
    )
    ymax = max(
        efe_history[t_min:t_max].max(),
        risk[t_min:t_max].max(),
        ambiguity[t_min:t_max].max(),
        -pragmatic_value[t_min:t_max].min(),  # negative
        salience[t_min:t_max].max(),
        novelty[t_min:t_max].max()
    )
    margin = MARGIN * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    # ax.set_xlim(400, 800)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('$G[u]$', color='black', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        # Combine both legends in one box
        lines = [
            efe_plot[0],
            # risk_plot[0], ambiguity_plot[0],
            pv_plot[0],
            salience_plot[0],
            novelty_plot[0]
        ]
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc='upper right')
    if i == 0:
        ax.legend()
        ax2.legend()


def plot_efe_diff_decomposition(
        efe_history,
        risk,
        ambiguity,
        pragmatic_value,
        salience,
        novelty,
        learn_record_refined,
        ax, i,
        t_min=0, t_max=None
):
    # Specify the range to investigate
    t_max = len(efe_history) if t_max is None else t_max
    x_range = range(t_min, t_max)

    # Calculate difference between Cooperate (Strategy 0) and Defect (Strategy 1) for all components
    diff_efe = efe_history[t_min:t_max, i, 0] - efe_history[t_min:t_max, i, 1]
    diff_pragmatic = pragmatic_value[t_min:t_max, i, 0] - pragmatic_value[t_min:t_max, i, 1]
    diff_salience = salience[t_min:t_max, i, 0] - salience[t_min:t_max, i, 1]
    diff_novelty = novelty[t_min:t_max, i, 0] - novelty[t_min:t_max, i, 1]

    # # Check formula calculation: efe_history = - pragmatic_value - salience - novelty
    # print(efe_history[t_min:t_max, i, 0][0],
    #       pragmatic_value[t_min:t_max, i, 0][0],
    #       salience[t_min:t_max, i, 0][0],
    #       novelty[t_min:t_max, i, 0][0])
    #
    # print(efe_history[t_min:t_max, i, 1][0],
    #       pragmatic_value[t_min:t_max, i, 1][0],
    #       salience[t_min:t_max, i, 1][0],
    #       novelty[t_min:t_max, i, 1][0])
    #
    # # diff_efe = - diff_pragmatic - diff_salience - diff_novelty
    # print(diff_efe[0],
    #       diff_pragmatic[0],
    #       diff_salience[0],
    #       diff_novelty[0])

    # Stacked area plot of EFE component differences
    ax.stackplot(
        x_range,
        [-diff_pragmatic, -diff_salience, -diff_novelty],
        labels=['Pragmatic', 'Salience', 'Novelty'],
        colors=[PRAGMATIC_COLOR, SALIENCE_COLOR, NOVELTY_COLOR],
        alpha=0.7
    )

    # Plot total EFE difference line
    ax.plot(x_range, diff_efe,
            color=EFE_COLOR, linewidth=LINEWIDTH, label='Total EFE Difference')

    add_learning_marker(i, ax, t_min, t_max, diff_efe, learn_record_refined, point_offset=0.04, text_offset=0.1)

    ax.set_title(f'Agent {chr(105 + i)}: EFE Difference (Cooperate - Defect)', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('ΔEFE Components', fontsize=label_font_size)
    # ax.set_xlim(400, 800)

    # Add legend only for the first agent to avoid clutter
    if i == 0:
        ax.legend()


def plot_expected_efe(
        q_u_history, efe_history,
        risk_history, ambiguity_history,
        pragmatic_value_history, salience_history,
        novelty_history,
        ax, i,
        t_min=0, t_max=None
):
    t_max = len(efe_history) if t_max is None else t_max
    x_range = range(t_min, t_max)

    # Only plot EFE terms if they are non-zero (i.e. if actual data was provided)
    # (as a way to stub out the plotting of EFE terms)
    plot_EFE_terms = (not np.all(risk_history == 0))
    plot_EFE_terms = False

    # Get EFE, Pragmatic Value, Salience, Risk, and Ambiguity over all actions
    # Calculate the dot product with q_u_history for each agent (i.e. expected value undre action dist)
    expected_efe = np.empty((q_u_history.shape[0], q_u_history.shape[1]))
    weighted_pragmatic_value = np.empty_like(expected_efe)
    weighted_salience = np.empty_like(expected_efe)
    weighted_risk = np.empty_like(expected_efe)
    weighted_ambiguity = np.empty_like(expected_efe)
    weighted_novelty = np.empty_like(expected_efe)

    for t in range(q_u_history.shape[0]):
        for a in range(q_u_history.shape[1]):
            expected_efe[t, a] = np.dot(q_u_history[t, a], efe_history[t, a])
            weighted_pragmatic_value[t, a] = np.dot(q_u_history[t, a], pragmatic_value_history[t, a])
            weighted_salience[t, a] = np.dot(q_u_history[t, a], salience_history[t, a])
            weighted_risk[t, a] = np.dot(q_u_history[t, a], risk_history[t, a])
            weighted_ambiguity[t, a] = np.dot(q_u_history[t, a], ambiguity_history[t, a])
            weighted_novelty[t, a] = np.dot(q_u_history[t, a], novelty_history[t, a])

    summed_efe = expected_efe[t_min:t_max, i]
    summed_risk = weighted_risk[t_min:t_max, i]
    summed_ambiguity = weighted_ambiguity[t_min:t_max, i]
    summed_salience = weighted_salience[t_min:t_max, i]
    summed_pragmatic_value = -weighted_pragmatic_value[t_min:t_max, i]
    summed_novelty = weighted_novelty[t_min:t_max, i]

    # Plot EFE on the primary y-axis
    efe_plot = ax.plot(
        x_range,
        summed_efe,
        label=r'$\langle \boldsymbol{\mathsf{G}} \rangle$', color=EFE_COLOR, linewidth=LINEWIDTH)

    if plot_EFE_terms:
        # Create a secondary y-axis for Risk, Ambiguity, Salience, Pragmatic Value, and Novelty)
        ax2 = ax  # .twinx()
        # risk_plot = ax2.plot(
        #     x_range,
        #     summed_risk,
        #     label='r', color=RISK_COLOR, linestyle=':', linewidth=LINEWIDTH)
        # ambiguity_plot = ax2.plot(
        #     x_range,
        #     summed_ambiguity,
        #     label='a', color=AMBIGUITY_COLOR, linestyle='--', linewidth=LINEWIDTH)
        pragmatic_value_plot = ax2.plot(
            x_range,
            summed_pragmatic_value,
            label='$-$Pragmatic value', color=PRAGMATIC_COLOR, linestyle='--', linewidth=LINEWIDTH)
        salience_plot = ax2.plot(
            x_range,
            summed_salience,
            label='Salience', color=SALIENCE_COLOR, linestyle=':', linewidth=LINEWIDTH)
        # novelty_plot = ax2.plot(
        #     x_range,
        #     summed_novelty,
        #     label='Novelty', color=NOVELTY_COLOR, linestyle=':', linewidth=LINEWIDTH)

    # Set labels and title
    ax.set_title(f'Expected EFE Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    # ylim range based on all agents (not just the current agent i)
    ymin = min(
        expected_efe[t_min:t_max].min(),
        summed_risk[t_min:t_max].min(),
        summed_ambiguity[t_min:t_max].min(),
        summed_salience[t_min:t_max].min(),
        summed_pragmatic_value[t_min:t_max].min(),
        summed_novelty[t_min:t_max].min()
    )
    ymax = max(
        expected_efe[t_min:t_max].max(),
        summed_risk[t_min:t_max].max(),
        summed_ambiguity[t_min:t_max].max(),
        summed_salience[t_min:t_max].max(),
        summed_pragmatic_value[t_min:t_max].max(),
        summed_novelty[t_min:t_max].max()
    )
    margin = MARGIN * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel(r'$\langle \boldsymbol{\mathsf{G}} \rangle$', color='black', fontsize=label_font_size)
        # if plot_EFE_terms:
        #     ax2.set_ylabel('Nats', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        if plot_EFE_terms:
            lines = [
                efe_plot[0],
                # risk_plot[0], ambiguity_plot[0],
                pragmatic_value_plot[0], salience_plot[0],
                # novelty_plot[0]
            ]
            labels = [line.get_label() for line in lines]
            ax2.legend(lines, labels, loc='lower right')
            # ax2.legend(loc='upper right')
            # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        else:
            pass
            # ax.legend(loc='upper left')


def plot_efe_risk_ambiguity(expected_efe, weighted_risk, weighted_ambiguity, ax, i):
    summed_efe = expected_efe[:, i]
    summed_risk = weighted_risk[:, i]
    summed_ambiguity = weighted_ambiguity[:, i]

    ax.plot(
        range(len(summed_efe)),
        summed_efe,
        label='EFE', color=EFE_COLOR, linewidth=LINEWIDTH)

    ax2 = ax  # .twinx()
    ax2.plot(
        range(len(summed_risk)),
        summed_risk,
        label='Risk', color=RISK_COLOR, linestyle=':', linewidth=LINEWIDTH)
    ax2.plot(
        range(len(summed_ambiguity)),
        summed_ambiguity,
        label='Ambiguity', color=AMBIGUITY_COLOR, linestyle=':', linewidth=LINEWIDTH)

    ax.set_title(f'EFE Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('EFE', color='black', fontsize=label_font_size)
        # ax2.set_ylabel('Nats', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')


def plot_efe_salience_pragmatic_value(expected_efe, weighted_salience, weighted_pragmatic_value, weighted_novelty=None,
                                      ax=None, i=0):
    summed_efe = expected_efe[:, i]
    summed_salience = -weighted_salience[:, i]
    summed_pragmatic_value = -weighted_pragmatic_value[:, i]
    summed_novelty = -weighted_novelty[:, i] if weighted_novelty is not None else None

    # Plot EFE on the primary y-axis
    ax.plot(
        range(len(summed_efe)),
        summed_efe,
        label='EFE', color=EXP_ELBO_COLOR, linestyle='-', linewidth=LINEWIDTH)

    # Create a secondary y-axis for Salience, Pragmatic Value, and Novelty
    ax2 = ax  # .twinx()
    ax2.plot(
        range(len(summed_salience)),
        summed_salience,
        label='Salience', color=SALIENCE_COLOR, linestyle=':', linewidth=LINEWIDTH)
    ax2.plot(
        range(len(summed_pragmatic_value)),
        summed_pragmatic_value,
        label='Pragmatic value', color=PRAGMATIC_COLOR, linestyle=':', linewidth=LINEWIDTH)

    # Plot Novelty if provided
    if summed_novelty is not None:
        ax2.plot(
            range(len(summed_novelty)),
            summed_novelty,
            label='Novelty', color=NOVELTY_COLOR, linestyle=':', linewidth=LINEWIDTH)

    # Set labels and title
    ax.set_title(f'EFE Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('EFE', color='black', fontsize=label_font_size)
        # ax2.set_ylabel('Nats', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])
    if SHOW_LEGEND and i == LEGEND_IDX:
        # Set legends for both axes
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')


def plot_policy_heatmap(
        q_pi_history,
        ax, i,
        t_min=0, t_max=None
):
    t_max = len(q_pi_history) if t_max is None else t_max

    num_actions = q_pi_history.shape[-1]
    action_labels = get_action_labels(num_actions)
    cax = ax.imshow(
        q_pi_history[t_min:t_max, i, :].T,
        vmin=0, vmax=1,
        origin='upper',
        aspect='auto',
        interpolation='nearest',
        cmap='inferno',
    )

    ax.set_title(f'Policy Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_xticks(range(0, t_max - t_min, int((t_max - t_min) / 5)))
    ax.set_xticklabels(range(t_min, t_max, int((t_max - t_min) / 5)))
    ax.set_yticks(range(len(action_labels)))
    ax.set_yticklabels(action_labels)
    # if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
    #     ax.set_ylabel('$q(\hat u)$', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])


# Line coloring
def plot_cooperation_prob(q_pi_history,
                          nash_strategy,
                          game_transitions,
                          learn_record_refined,
                          ax, i, t_min=0, t_max=None):
    # print(nash_strategy)
    t_max = len(q_pi_history) if t_max is None else t_max

    x_range = np.arange(t_min, t_max)

    # Extract the cooperation probabilities from q_pi_history
    prob_cooperate = np.round(q_pi_history[t_min:t_max, i, 0], 2)
    # print(prob_cooperate)

    # Mark time points where the prob_cooperate aligns with NE
    nash_mask = np.zeros_like(x_range, dtype=bool)

    if nash_strategy is not None and game_transitions is not None:
        current_time = t_min
        game_start_times = []
        game_names = []

        # Calculate game phase boundaries
        for label, _, duration in game_transitions:
            game_start_times.append(current_time)
            game_names.append(label)
            current_time += duration

        # Plot Nash lines for each game phase
        for game_name, start_time, end_time in zip(game_names, game_start_times, game_start_times[1:] + [t_max]):
            # Find matching time points
            phase_slice = slice(max(t_min, start_time) - t_min, min(t_max, end_time) - t_min)
            phase_probs = np.round(prob_cooperate[phase_slice], 2)
            phase_times = x_range[phase_slice]

            # Identify key-value pair in nash_strategy
            agent_key = 'Row' if i == 0 else 'Col'
            stg_NE = nash_strategy[game_name].get(agent_key + "_NE", [])
            stg_NBS = nash_strategy[game_name].get(agent_key + "_NBS", [])

            # Get unique Nash cooperation probabilities
            probs_NE = np.unique(np.round([s[0] for s in stg_NE], 2))
            probs_NBS = np.unique(np.round([s[0] for s in stg_NBS], 2))

            for nash_value in probs_NE:
                matches = np.where(phase_probs == nash_value)[0]
                nash_mask[phase_times[matches]] = True

            # Plot horizontal lines for each NE and NBS
            for y in probs_NE:
                ax.hlines(y, xmin=start_time, xmax=end_time,
                          colors='orange', linestyles=':', linewidth=LINEWIDTH * 2, zorder=2)
            for y in probs_NBS:
                ax.hlines(y, xmin=start_time, xmax=end_time,
                          colors='green', linestyles='--', linewidth=LINEWIDTH * 5, zorder=2, alpha=0.3)

    # Plot main cooperation probability line
    ax.plot(x_range, prob_cooperate,
            color='blue', linewidth=LINEWIDTH, label=f'Agent {chr(105 + i)}', zorder=1)

    add_learning_marker(i, ax, t_min, t_max, prob_cooperate, learn_record_refined, point_offset=0.02, text_offset=0.06)

    # Find the indices where the mask changes
    change_points = np.where(np.diff(nash_mask))[0] + 1

    # Split the line into segments at change points
    segments = []
    start_idx = 0
    for end_idx in change_points:
        segments.append((start_idx, end_idx, nash_mask[start_idx]))
        start_idx = end_idx
    segments.append((start_idx, len(x_range), nash_mask[start_idx]))

    # Overplot Nash segments in red
    for start_idx, end_idx, is_nash in segments:
        if is_nash:
            segment_times = x_range[start_idx:end_idx - 1]
            segment_probs = prob_cooperate[start_idx:end_idx - 1]
            ax.plot(segment_times,
                    segment_probs,
                    color='red',
                    linewidth=LINEWIDTH * 1.5,
                    solid_capstyle='round',
                    solid_joinstyle='round',
                    zorder=3,
                    label='Nash-Strategy' if start_idx == 0 and is_nash else None)

    ax.set_title(f'Policy Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('Probability of Cooperation (Strategy 0)', fontsize=label_font_size)
    ax.set_ylim(0 - MARGIN, 1 + MARGIN)
    # ax.set_xticks(range(0, t_max - t_min, int((t_max - t_min) / 5)))
    # ax.set_xlim(400, 800)

    from matplotlib.lines import Line2D
    if i == 0:
        ax.legend(handles=[
            Line2D([0], [0], color='red', linewidth=LINEWIDTH * 1.5, label='Nash-Strategy'),
            Line2D([0], [0], color='blue', linewidth=LINEWIDTH, label='Non-Nash-Strategy'),
            Line2D([0], [0], color='orange', linestyle=':', linewidth=LINEWIDTH * 2, label='NE'),
            Line2D([0], [0], color='green', linestyle='--', linewidth=LINEWIDTH * 5, label='NBS', alpha=0.3, )
        ])

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')


def plot_observation_prediction(o_pred_record,
                                learn_record_refined,
                                ax, i, t_min=0, t_max=None):
    t_max = len(o_pred_record) if t_max is None else t_max
    x_range = np.arange(t_min, t_max)

    # Print out the top part of tensor
    # print(o_pred_record[0:5])
    # The result is mixed with [0, 1] and [1, 0]
    action_label_i = [step[0][0] for step in o_pred_record]
    # print(action_label_i)
    action_label_j = [step[1][0] for step in o_pred_record]
    # print(action_label_j)

    # Extract the bottom part of tensor for the current agent
    value_s0 = [step[i][1][0] for step in o_pred_record]  # The Strategy 0
    value_s1 = [step[i][1][1] for step in o_pred_record]  # The Strategy 1
    if len(o_pred_record[0][i][1]) == 3:
        value_s2 = [step[i][1][2] for step in o_pred_record]  # The Strategy 2

    ax.plot(x_range, value_s0, color='green', linewidth=LINEWIDTH, label="Strategy 0")

    value_s0 = [float(x) for x in value_s0]

    add_learning_marker(i, ax, t_min, t_max, value_s0, learn_record_refined, point_offset=0.02, text_offset=0.06)

    ax.set_title(f'Observation Prediction Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('o_pred of Strategy 0', fontsize=label_font_size)
    ax.set_ylim(0 - MARGIN, 1 + MARGIN)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')

    if i == 0:
        ax.legend()


def plot_learn_model_change(B_model_change,
                            learn_record_refined,
                            ax, i, t_min=0, t_max=None):
    t_max = len(B_model_change) if t_max is None else t_max
    x_range = np.arange(t_min, t_max)

    # print(B_model_change[200])

    if len(B_model_change[0].shape) == 2:  # Per factor
        for factor_idx in range(B_model_change[0].shape[0]):  # 0=ego, 1=opponent
            series = [step[factor_idx]
                      for step in B_model_change[t_min:t_max]]
            ax.plot(x_range, series,
                    color=f'{"green" if factor_idx == 0 else "red"}',  # 0  ego, 1 opponent
                    linewidth=LINEWIDTH * 2,
                    label=f'{"Ego" if factor_idx == 0 else "Opponent"}',
                    alpha=0.7)

    elif len(B_model_change[0].shape) == 3:  # Per factor * Per action
        # Collect per-action time series for agent i
        num_actions = B_model_change[0].shape[1]
        action_series = [[] for _ in range(num_actions)]

        for step in B_model_change[t_min:t_max]:
            agent_matrix = step[i]  # shape: (num_factors, num_actions)
            # print(agent_matrix)
            for idx in range(num_actions):
                val = agent_matrix[0][idx]
                # print(val)
                action_series[idx].append(val)

        # Plot each action’s trajectory
        for idx, series in enumerate(action_series):
            ax.plot(x_range, series,
                    color=ACTION_COLORS[idx],
                    linewidth=LINEWIDTH * 2,
                    label=f"Strategy {idx}",
                    alpha=0.7)
    else:
        raise NotImplementedError("Invalid granularity: B_model_change")

    ax.set_title(f'The reduction level of B matrix Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('Weight of the reduction model', fontsize=label_font_size)
    ax.set_ylim(0 - MARGIN * 2, 1 + MARGIN)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    if i == 0:
        ax.legend()


def plot_full_red_model_F(delta_F_history, F_red_history, F_full_history,
                          ax, i, t_min=0, t_max=None):
    # Time slicing
    T = len(delta_F_history)
    t_min = max(0, t_min)
    t_max = T if t_max is None else min(t_max, T)
    x_range = np.arange(t_min, t_max)

    # print(F_red_history[200])
    # print(F_red_history[200].shape)

    if len(delta_F_history[0].shape) == 2:  # Per factor: 2 agents * 2 factors
        for factor_idx in range(delta_F_history[0].shape[0]):
            delta_F_series = [step[i][factor_idx] for step in delta_F_history[t_min:t_max]]
            red_series = [step[i][factor_idx] for step in F_red_history[t_min:t_max]]
            full_series = [step[i][factor_idx] for step in F_full_history[t_min:t_max]]

            color = "green" if factor_idx == 0 else "red"
            label_prefix = "Ego" if factor_idx == 0 else "Opponent"

            ax.plot(x_range, red_series,
                    color=color, linewidth=LINEWIDTH * 2, linestyle=':',
                    label=f"F_red {label_prefix}", alpha=0.5)
            ax.plot(x_range, full_series,
                    color=color, linewidth=LINEWIDTH * 2, linestyle='-',
                    label=f"F_full {label_prefix}", alpha=0.7)

    elif len(delta_F_history[0].shape) == 3:  # Per factor Per action: 2 agents * 2 factors * 2 actions
        # Determine number of actions
        num_actions = delta_F_history[0].shape[1]

        # Initialize per-action series
        delta_series = [[] for _ in range(num_actions)]
        red_series = [[] for _ in range(num_actions)]
        full_series = [[] for _ in range(num_actions)]

        # Collect data per action over time
        for step_idx in range(t_min, t_max):
            # Extract matrix of shape: (num_factors, num_actions)
            delta_matrix = delta_F_history[step_idx][i]
            red_matrix = F_red_history[step_idx][i]
            full_matrix = F_full_history[step_idx][i]

            for action_idx in range(num_actions):
                delta_series[action_idx].append(delta_matrix[0][action_idx])
                red_series[action_idx].append(red_matrix[0][action_idx])
                full_series[action_idx].append(full_matrix[0][action_idx])

        # print(delta_F_history[100])
        # print(delta_series[0][100])
        # print(delta_series[1][100])

        # Plot each action's trajectory
        for action_idx in range(num_actions):
            ax.plot(x_range, red_series[action_idx],
                    color=ACTION_COLORS[action_idx],
                    linewidth=LINEWIDTH * 2,
                    linestyle=':',
                    label=f"F_red Strategy {action_idx}",
                    alpha=0.5)
            ax.plot(x_range, full_series[action_idx],
                    color=ACTION_COLORS[action_idx],
                    linewidth=LINEWIDTH * 2,
                    linestyle='-',
                    label=f"F_full Strategy {action_idx}",
                    alpha=0.7)
            # ax.plot(x_range, delta_series[action_idx],
            #         color=ACTION_COLORS[action_idx],
            #         linewidth=LINEWIDTH * 3,
            #         linestyle=':',
            #         label=f"ΔF Strategy {action_idx}",
            #         alpha=0.9)

    else:
        raise NotImplementedError("Invalid granularity: delta_F_history")

    ax.set_title(f'Log model evidence of reduced and full model Agent {chr(105 + i)}')
    ax.set_xlabel('Time step')
    ax.set_ylabel(r'$F$')

    if i == 0:
        ax.legend()


def plot_policy_entropy(
        q_pi_history,
        ax, i,
        t_min=0, t_max=None
):
    t_max = len(q_pi_history) if t_max is None else t_max
    x_range = range(t_min, t_max)

    # num_actions = q_pi_history.shape[-1]
    # action_labels = get_action_labels(num_actions)
    entropy = -np.sum(q_pi_history[t_min:t_max, i, :] * np.log(q_pi_history[t_min:t_max, i, :] + 1e-9), axis=1)
    ax.plot(
        x_range,
        entropy,
        label='Policy entropy', color=POLICY_ENTROPY_COLOR, linewidth=LINEWIDTH
    )
    max_ent = np.log(q_pi_history.shape[-1])
    ax.hlines(max_ent, t_min, t_max, color='#ccc', linestyle='--', linewidth=LINEWIDTH, label='Max entropy')

    ax.set_title(f'Policy entropy Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylim(-0.1, max_ent + 0.1)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('$H[q(u)]$', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])


def plot_inferred_policy_heatmap(
        s_history,
        ax, i,
        t_min=0, t_max=None,
        factor=0
):
    t_max = len(s_history) if t_max is None else t_max

    # print(s_history[50])
    # print(s_history[50, i, factor, :])
    # print("------")
    # print(s_history[550])
    # print(s_history[550, i, factor, :])
    # print("!!!!!!")

    # focus only on the chosen factor
    s_focus = s_history[t_min:t_max, i, factor, :]
    num_actions = s_history.shape[-1]
    action_labels = get_action_labels(num_actions)

    cax = ax.imshow(
        s_focus.T,
        vmin=0, vmax=1,
        origin='upper',
        aspect='auto',
        interpolation='nearest',
        cmap='inferno',
    )

    ax.set_title(f'Inferred hidden state Agent {chr(105 + i)} Factor={factor} (self)', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_xticks(range(0, t_max - t_min, int((t_max - t_min) / 10)))
    ax.set_xticklabels(range(t_min, t_max, int((t_max - t_min) / 10)))

    ax.set_yticks(range(num_actions))
    ax.set_yticklabels(action_labels)

    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('$q(s)$', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])


# def plot_log_C_modality(log_C_modality_history, ax, i, t_max=None, max_legend_items=9):

#     num_players = log_C_modality_history.shape[1]
#     num_actions = log_C_modality_history.shape[-1]
#     action_labels = get_action_labels(num_actions)
#     cax = ax.imshow(
#         log_C_modality_history[:, i, :, :].reshape(-1, num_players*num_actions).T,
#         # vmin=0, vmax=1,
#         origin='upper',
#         aspect='auto',
#         interpolation='nearest'
#     )
#     print(f'log_C_modality (agent {i}) (min, max):\t{log_C_modality_history[:, i, :, :].min():0.4f}, {log_C_modality_history[:, i, :, :].max():0.4f}')

#     ax.set_title(f'Agent {chr(105+i)}\'s preferred observation (per factor)', fontsize=label_font_size)
#     ax.set_xlabel('Time step (t)', fontsize=label_font_size)
#     ax.set_yticks(range(num_actions*num_players))
#     ax.set_yticklabels(action_labels * num_players)
#     if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
#         ax.set_ylabel('log_C_modality', fontsize=label_font_size)

# def plot_log_C_opp(log_C_opp_history, ax, i, t_max=None, max_legend_items=9):
#     num_players = log_C_opp_history.shape[1]
#     num_actions = log_C_opp_history.shape[2]
#     all_joint_actions_enum = np.array(np.meshgrid(*[np.arange(num_actions)] * num_players)).T.reshape(-1, num_players)

#     t_max = min(t_max, log_C_opp_history.shape[0]) if t_max else log_C_opp_history.shape[0]  # Ensure t_max is within bounds
#     log_C_values = log_C_opp_history[:t_max, i].reshape(-1, num_actions ** num_players)

#     # Calculate the average log C values for each joint action
#     avg_log_C = log_C_values.mean(axis=0)

#     # Get the indices of the top max_legend_items joint actions
#     top_indices = np.argsort(avg_log_C)[-max_legend_items:]

#     # Plot only the top max_legend_items joint actions
#     for j in top_indices:
#         current_action = all_joint_actions_enum[j]
#         ax.plot(
#             range(t_max),
#             log_C_values[:, j],
#             label=str(current_action.tolist()),
#             linewidth=LINEWIDTH
#         )

#     ax.set_title(f'Learnt Preferences for Agent {chr(105+i)}', fontsize=label_font_size)
#     ax.set_xlabel('Time step (t)', fontsize=label_font_size)
#     if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
#         ax.set_ylabel('log C (opponent)', fontsize=label_font_size)
#     if SHOW_LEGEND and i == LEGEND_IDX:
#         ax.legend(title='Joint actions', fontsize=8, loc='upper right')


def plot_utility(log_C_opp_history, ax, i, t_max=None, num_samples=1000, epsilon=np.finfo(float).eps,
                 max_legend_items=9):
    num_players = log_C_opp_history.shape[1]
    num_actions = log_C_opp_history.shape[2]
    all_joint_actions_enum = np.array(np.meshgrid(*[np.arange(num_actions)] * num_players)).T.reshape(-1, num_players)

    t_max = min(t_max, log_C_opp_history.shape[0]) if t_max else log_C_opp_history.shape[
        0]  # Ensure t_max is within bounds
    utility_mean = np.zeros((t_max, num_actions ** num_players))
    utility_std = np.zeros((t_max, num_actions ** num_players))

    for t in range(t_max):
        alpha = log_C_opp_history[t, i].reshape(-1)
        samples = np.array([dirichlet.rvs(alpha, size=1) for _ in range(num_samples)]).reshape(num_samples, -1)
        log_samples = np.log(samples + epsilon)
        utility_mean[t, :] = log_samples.mean(axis=0)
        utility_std[t, :] = log_samples.std(axis=0)

    # Calculate the average utility values for each joint action
    avg_utility = utility_mean.mean(axis=0)

    # Get the indices of the top max_legend_items joint actions
    top_indices = np.argsort(avg_utility)[-max_legend_items:]

    # Plot only the top max_legend_items joint actions
    for j in top_indices:
        current_action = all_joint_actions_enum[j]
        ax.plot(
            range(t_max),
            utility_mean[:, j],
            label=str(current_action.tolist()),
            color=f'C{j % 10}',
            linewidth=LINEWIDTH
        )
        ax.fill_between(
            range(t_max),
            utility_mean[:, j] - utility_std[:, j],
            utility_mean[:, j] + utility_std[:, j],
            color=f'C{j % 10}',
            alpha=0.3
        )

    ax.set_title(f'Learnt Reward for Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('Reward (log prob)', fontsize=label_font_size)
    if SHOW_LEGEND and i == LEGEND_IDX:
        ax.legend(title='Joint actions', fontsize=8, loc='upper right')


def plot_precision(
        gamma_history, ax, i,
        t_min=0, t_max=None
):
    t_max = len(gamma_history) if t_max is None else t_max
    x_range = range(t_min, t_max)
    ax.plot(
        x_range,
        gamma_history[t_min:t_max, i],
        label='Precision', color=PRECISION_COLOR, linewidth=LINEWIDTH
    )

    ax.set_title(f'Precision for Agent {chr(105 + i)}', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    # ylim range based on all agents (not just the current agent i)
    ymin = gamma_history[t_min:t_max].min()
    ymax = gamma_history[t_min:t_max].max()
    margin = MARGIN * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('Precision', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])


def plot_A(A_history, ax, i,
           t_min=0, t_max=None
           ):
    t_max = len(A_history) if t_max is None else t_max

    num_players = A_history.shape[1]
    num_actions = A_history.shape[-1]
    cax = ax.imshow(
        A_history[t_min:t_max, i].reshape(-1, num_players * num_actions * num_actions).T,
        vmin=0, vmax=1,
        origin='upper',
        aspect='auto',
        interpolation='nearest',
        cmap='inferno',
    )
    labels = [f'[{chr(105 + j)}, {o}, {s}]'
              for j in range(num_players)
              for o in range(num_actions)
              for s in range(num_actions)]

    ax.set_title(f'Likelihood Agent {chr(105 + i)} (factor, o, s)', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_xticks(range(0, t_max - t_min, int((t_max - t_min) / 10)))
    ax.set_xticklabels(range(t_min, t_max, int((t_max - t_min) / 10)))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    if not ONLY_LEFT_Y_LABEL or (ONLY_LEFT_Y_LABEL and i == 0):
        ax.set_ylabel('A', fontsize=label_font_size)
    if ONLY_LEFT_Y_LABEL and i > 0:
        ax.set_yticklabels([])


def plot_B_overtime(B_history, state, ax, i, t_min=0, t_max=None):
    t_max = len(B_history) if t_max is None else t_max
    x_range = range(t_min, t_max)

    B_history = np.array(B_history)

    agent_history = np.array(B_history[x_range, i])

    # T, num_actions, K, _ = agent_history.shape
    # print(B_history[234])
    # print(f"--------------------Agent{i}---------------------")
    # print(agent_history[234])
    T, num_factors, num_actions, _, _ = agent_history.shape

    for f in range(num_factors):  # loop over factors
        if f != 0:
            continue
        for u in range(num_actions):  # loop over actions
            series = agent_history[x_range, f, u, state[0], state[1]]
            # print(series[234])

            color = ACTION_COLORS[0] if u == 0 else ACTION_COLORS[1]
            linestyle = '-' if f == 0 else ':'
            linewidth = LINEWIDTH * 2 if f == 0 else LINEWIDTH * 3
            ax.plot(x_range, series, label=f"f={f}, u={u}", linestyle=linestyle,
                    linewidth=linewidth, color=color, alpha=0.7)

    ax.set_title(f'Time series of B[f=0,u,{state[0]},{state[1]}]', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel(f"P(next={state[0]} | curr={state[1]})", fontsize=label_font_size)
    ax.set_ylim([0 - MARGIN, 1 + MARGIN])
    ax.legend()


# def plot_B_snapshot(B_history, ax, i, t_min=0, t_max=None, t1=50, t2=550):
#     t_max = len(B_history) if t_max is None else t_max
#     x_range = range(t_min, t_max)
#
#     B_history = np.array(B_history[x_range, i])
#
#     T, num_factors, num_actions, _, _ = B_history.shape
#
#     fig, axes = plt.subplots(num_factors, num_actions, figsize=(6 * num_actions, 4 * num_factors))
#
#     if num_factors == 1:
#         axes = np.expand_dims(axes, 0)
#     if num_actions == 1:
#         axes = np.expand_dims(axes, 1)
#
#     for f in range(num_factors):
#         for u in range(num_actions):
#             ax = axes[f, u]
#
#             mat1 = B_history[t1, f, u, :, :]
#             mat2 = B_history[t2, f, u, :, :]
#
#             # Concatenate side-by-side (next states on rows, timesteps on columns)
#             mat_combined = np.hstack([mat1, mat2])
#
#             im = ax.imshow(mat_combined, vmin=0, vmax=1, cmap="Blues")
#             ax.set_title(f"f={f}, u={u}")
#             ax.set_xlabel(f"curr s (left=t{t1}, right=t{t2})")
#             ax.set_ylabel("next s'")
#
#             # grid lines to separate t1 and t2 blocks
#             ax.axvline(x=mat1.shape[1] - 0.5, color='black', linewidth=1)
#
#     fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Probability")
#     plt.suptitle(f"Transition matrices comparison at t={t1} vs t={t2}")
#     plt.tight_layout()
# plt.show()


# ==============================================================================
# ENSEMBLE PLOTS
# ==============================================================================


# def plot_ensemble_policies(q_u_hist, game_transitions, ax=None):
#     import utils.timeseries as timeseries
#
#     # Compute metrics
#     num_agents = q_u_hist.shape[1]
#     if num_agents == 2:  # Two agents
#         data = np.hstack((
#             q_u_hist[:, :, 0],
#             np.zeros_like(q_u_hist[:, 0, 0])[:, None],
#         ))
#     elif num_agents == 3:  # Three agents
#         data = q_u_hist[:, :, 0]
#
#     t_steady = timeseries.get_t_steady_state(data)
#     t_bifurcation = timeseries.get_t_bifurcation(data)
#
#     if ax is None:
#         plt.figure(figsize=(8, 2), dpi=DPI)
#         ax = plt.gca()
#
#     # Plot the ensemble policies
#     ax.plot(q_u_hist[:, :, 0])
#
#     # Highlight the steady state and bifurcation points
#     ax.vlines(
#         [t_bifurcation, t_steady],
#         0, 1,
#         linestyle='--', linewidth=LINEWIDTH, color='red', alpha=0.4, )
#
#     ax.set_xlabel('Time step')
#     ax.set_ylabel('q(u=c)')
#
def plot_vfe_ensemble(vfe_history, game_transitions,
                      ifLegend=True, ax=None):
    vfe_history = np.array(vfe_history)
    num_seeds = vfe_history.shape[0]
    # print(vfe_history.shape) # (4 Seed, 1000 Time, 2 agents, 2 actions)
    for s in range(num_seeds):
        summed_vfe = vfe_history[s].sum(axis=(1, 2))
        ax.plot(summed_vfe, color="blue", alpha=0.1, linewidth=LINEWIDTH, label="Individuals" if s == 0 else None)

    # Highlight mean trajectory across seeds
    mean_vfe = vfe_history.sum(axis=(2, 3)).mean(axis=0)
    ax.plot(mean_vfe, color=ELBO_COLOR, linewidth=LINEWIDTH * 2, label="Mean")

    # ax.set_title(f'VFE Ensemble', fontsize=label_font_size)
    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('VFE', color='black', fontsize=label_font_size)
    ax.set_ylim([0 - MARGIN, 10 + MARGIN])
    if ifLegend:
        ax.legend(loc='right', fontsize=label_font_size)


def plot_policies_ensemble(q_u_history, game_transitions, nash_strategy,
                           ifLegend=True, single=False, ax=None):
    q_u_history = _coerce_q_u_action_history(q_u_history, game_transitions)
    num_seeds, T, num_agents, num_actions = q_u_history.shape

    # Duration of the first game
    _, _, first_game_duration = game_transitions[0]

    if single:
        # --- Single-seed mode ---
        s = 0  # only use ONE seed
        for a in range(num_agents):
            ax.plot(
                q_u_history[s, :, a, 0],
                alpha=0.7 if a == 0 else 0.5,
                color="red" if a == 0 else "black",
                linewidth=LINEWIDTH,
                linestyle="-" if a == 0 else "--",
                label="Agent i" if a == 0 else "Agent j"
            )
    else:
        # --- Ensemble (multi-seed) mode ---
        coop_labeled = False
        defect_labeled = False

        for s in range(num_seeds):
            for a in range(num_agents):
                # average q(u=c) for this agent in first game
                avg = q_u_history[s, :first_game_duration, a, 0].mean().item()

                if avg >= 0.5:
                    color = "#03dffc"
                    label = "Initial Cooperators" if not coop_labeled else None
                    coop_labeled = True
                else:
                    color = "#f70ce4"
                    label = "Initial Defectors" if not defect_labeled else None
                    defect_labeled = True

                ax.plot(q_u_history[s, :, a, 0],
                        alpha=0.5,
                        color=color,
                        linewidth=LINEWIDTH,
                        label=label)

    # Add Nash equilibrium lines
    start_t = 0
    for game_name, _, duration in game_transitions:
        if game_name in nash_strategy:
            ne_info = nash_strategy[game_name]
            coop_probs = set()

            for arr in ne_info.get("Row_NE", []):
                coop_probs.add(float(arr[0]))
            for arr in ne_info.get("Col_NE", []):
                coop_probs.add(float(arr[0]))

            for p in coop_probs:
                ax.hlines(
                    y=p,
                    xmin=start_t,
                    xmax=start_t + duration,
                    colors="gray",
                    linestyles="--",
                    linewidth=LINEWIDTH * 2,
                    alpha=0.3,
                    # label="NE Strategy" if start_t == 0 and p == list(coop_probs)[0] else None,
                )

        start_t += duration

    # ax.set_title("Policy Ensemble", fontsize=label_font_size)
    ax.set_xlabel("Time step")
    ax.set_ylabel("P(u=c)")
    # if ifLegend:
    #     ax.legend(loc="right", fontsize=label_font_size)


def plot_expected_efe_ensemble(q_u_history, efe_history,
                               ifLegend=True, ax=None):
    try:
        q_u_history = _coerce_q_u_history(q_u_history)
        efe_history = _coerce_q_u_history(efe_history)
        num_seeds, T, num_agents, _ = q_u_history.shape
        expected_efe = np.empty((num_seeds, T, num_agents))
        for s in range(num_seeds):
            for t in range(T):
                for a in range(num_agents):
                    expected_efe[s, t, a] = np.dot(q_u_history[s, t, a], efe_history[s, t, a])

        # Plot all seeds × agents (was missing entirely — caused blank plot for 2 AIF agents)
        for s in range(num_seeds):
            ensemble_efe = expected_efe[s].sum(axis=-1)  # sum over agents → scalar per timestep
            ax.plot(ensemble_efe,
                    color=EFE_COLOR,
                    alpha=0.3,
                    linewidth=LINEWIDTH,
                    label="Individuals" if s == 0 else None)

        # Mean across seeds
        mean_efe = expected_efe.sum(axis=-1).mean(axis=0)  # (T,)
        ax.plot(mean_efe,
                color=EFE_COLOR,
                alpha=1.0,
                linewidth=LINEWIDTH * 2,
                label="Mean")

    except ValueError:
        # Fallback for mixed agent q_u sizes (e.g., policy-length agents + dummy agents)
        num_seeds = len(q_u_history)
        T = len(q_u_history[0])
        num_agents = len(q_u_history[0][0])
        expected_efe = np.empty((num_seeds, T, num_agents))
        for s in range(num_seeds):
            for t in range(T):
                for a in range(num_agents):
                    q = np.array(q_u_history[s][t][a], dtype=float)
                    e = np.array(efe_history[s][t][a], dtype=float)
                    expected_efe[s, t, a] = np.dot(q, e)

        # Plot all seeds (was using stale loop variable `s` = last seed only)
        for s in range(num_seeds):
            ensemble_efe = expected_efe[s].sum(axis=-1)
            ax.plot(ensemble_efe,
                    color=EFE_COLOR,
                    alpha=0.3,
                    linewidth=LINEWIDTH,
                    label="Individuals" if s == 0 else None)

        mean_efe = expected_efe.sum(axis=-1).mean(axis=0)
        ax.plot(mean_efe,
                color=EFE_COLOR,
                alpha=1.0,
                linewidth=LINEWIDTH * 2,
                label="Mean")

    # ax.set_title(f"Expected EFE Ensemble", fontsize=label_font_size)
    ax.set_xlabel("Time step (t)", fontsize=label_font_size)
    ax.set_ylabel("Expected EFE", color="black", fontsize=label_font_size)
    if ifLegend:
        ax.legend(loc="right", fontsize=label_font_size)


def plot_B_state_ensemble(B_history, q_s_history,
                          ifLegend=True, single=False, collapse=False, ax=None):
    """
    Plot ensemble of P(s'=1|a) for each agent and action across seeds.
    When collapse=True, plot the mean of P(s'=1|u=0) and P(s'=1|u=1).

    Parameters:
    - B_history: (num_seeds, T, num_agents, num_factors, num_actions, 2, 2) B-matrix history
    - q_s_history: (num_seeds, T, num_agents, num_factors, 2) state belief history
    - ax: Matplotlib axis to plot on
    """
    #print(B_history[0][500])
    B_history = np.array(B_history)
    q_s_history = np.array(q_s_history)
    num_seeds, T, num_agents, num_factors, num_actions, _, _ = B_history.shape

    # Colors for actions
    colors = {0: "#03dffc", 1: "#f70ce4"}  # blue for cooperate, pink for defect
    collapse_color = "#7a42f5"  # purple for collapsed mean

    # Compute P(s'=1|a) for each seed, time, agent, and action
    p_s1_given_a = np.empty((num_seeds, T, num_agents, num_actions))
    for s in range(num_seeds):
        for t in range(T):
            for a in range(num_agents):
                for u in range(num_actions):
                    # P(s'|u) = B @ q_s for factor=0 (self-state)
                    B = B_history[s, t, a, 0, u, :, :]  # Shape (2, 2)
                    q_s = q_s_history[s, t, a, 0, :]  # Shape (2,)
                    p_s1 = B @ q_s  # Shape (2,)
                    p_s1_given_a[s, t, a, u] = p_s1[1]  # P(s'=1|a)

    # --- COLLAPSE MODE ---
    if collapse:
        # Average across actions for each agent, time, and seed
        p_s1_collapsed = p_s1_given_a.mean(axis=-1)  # shape (num_seeds, T, num_agents)

        for s in range(num_seeds):
            if single and s != 9:
                continue
            for a in range(num_agents):
                label = "Individuals" if (s == 0 and a == 0) else None
                ax.plot(
                    p_s1_collapsed[s, :, a],
                    color=collapse_color,
                    alpha=0.9 if single else 0.3,
                    linewidth=LINEWIDTH,
                    label=label
                )

    # --- NORMAL MODE ---
    else:
        for s in range(num_seeds):
            if single and s != 9:
                continue
            for a in range(num_agents):
                for u in range(num_actions):
                    if s == 0 and a == 0:
                        label = "u=cooperate" if u == 0 else "u=defect"
                    else:
                        label = None
                    ax.plot(
                        p_s1_given_a[s, :, a, u],
                        color=colors[u],
                        alpha=0.9 if single else 0.3,
                        linewidth=LINEWIDTH,
                        label=label
                    )

    ax.set_xlabel("Time step (t)", fontsize=label_font_size)
    if collapse:
        ax.set_ylabel("Mean P(s'=1|u)", fontsize=label_font_size)
    else:
        ax.set_ylabel("P(s'=1|u)", fontsize=label_font_size)
    ax.set_ylim([0 - MARGIN, 1 + MARGIN])

    if ifLegend:
        ax.legend(loc="right", fontsize=label_font_size)


def plot_B_separation_degree_ensemble(B_history, q_u_history, game_transitions, factor=0, selectedRole="Cooperators",
                                      ifLegend=True, single=False, average=True, ax=None):
    """
    Plot ensemble of Separation Degree(t) = |P(s'=s | u, s) − P(s'≠s | u, s)|
    Supports:
        - factor       : 0 (ego) or 1 (alter-ego)
        - selectedRole : "Cooperators" or "Defectors"
        - single=True  : plot only one seed's trajectories
        - average=True : plot only the mean trajectory of all agents for selectedRole
    """
    B_history = np.array(B_history)
    q_u_history = _coerce_q_u_action_history(q_u_history, game_transitions)
    num_seeds, T, num_agents, _, num_actions, num_states, _ = B_history.shape
    factor_idx = factor

    # Duration of first game for initial role detection
    _, _, first_game_duration = game_transitions[0]

    # Identify each agent’s initial role based on cooperation probability
    agent_roles = []
    for seed in range(num_seeds):
        for agent in range(num_agents):
            avg_q_u_c = q_u_history[seed, :first_game_duration, agent, 0].mean()
            role = "Defectors" if avg_q_u_c < 0.5 else "Cooperators"
            agent_roles.append((seed, agent, role))

    # Compute Separation Degree
    sep_degree = np.empty((num_seeds, T, num_agents, num_states))
    for seed in range(num_seeds):
        for t in range(T):
            for agent in range(num_agents):
                for s in range(num_states):
                    diag_c = B_history[seed, t, agent, factor_idx, 0, s, s]  # P(s'=s|u=0,s)
                    off_diag_c = B_history[seed, t, agent, factor_idx, 0, 1 - s, s]  # P(s'≠s|u=0,s)
                    diag_d = B_history[seed, t, agent, factor_idx, 1, s, s]  # P(s'=s|u=1,s)
                    off_diag_d = B_history[seed, t, agent, factor_idx, 1, 1 - s, s]  # P(s'≠s|u=1,s)
                    sep_degree[seed, t, agent, s] = (abs(diag_c - off_diag_c) + abs(diag_d - off_diag_d)) / 2

    # Role-based filtering
    success = 0
    labeled = {f"{s}": False for s in range(num_states)}
    color = ACTION_COLORS[0] if selectedRole == "Cooperators" else ACTION_COLORS[1]

    if average:
        # --- Average trajectory mode ---
        role_indices = [(seed, agent) for seed, agent, role in agent_roles if role == selectedRole]
        if not role_indices:
            logging.warning(f"No agents found for role {selectedRole}")
            return

        # Stack all matching trajectories
        role_sep = np.stack([sep_degree[seed, :, agent, :] for seed, agent in role_indices])
        mean_sep = role_sep.mean(axis=0)  # (T, num_states)

        for s in range(num_states):
            label = f"s={s}, f={factor}" if ifLegend else None
            ax.plot(
                mean_sep[:, s],
                color=color,
                linewidth=LINEWIDTH * 2.2,
                linestyle="-" if s == 0 else ":",
                label=label
            )

    else:
        # --- Single-seed mode ---
        for seed in range(num_seeds):
            if single and success != 0:
                break
            for agent in range(num_agents):
                role = next(r for s, a, r in agent_roles if s == seed and a == agent)
                if role != selectedRole:
                    continue

                for s in range(num_states):
                    label_key = f"{s}"
                    label = f"s={s}, f={factor}" if not labeled[label_key] and ifLegend else None
                    ax.plot(
                        sep_degree[seed, :, agent, s],
                        alpha=0.9 if single else 0.3,
                        color=color,
                        linewidth=LINEWIDTH * 1.5,
                        linestyle="-" if s == 0 else ":",
                        label=label
                    )
                    if label:
                        labeled[label_key] = True
                    success += 1

    # Axis formatting
    ax.set_xlabel("Time step (t)")
    # ax.set_ylabel("| P(s'=s|s)−P(s'≠s|s) |")
    ax.set_ylabel(selectedRole)
    ax.set_ylim([0 - MARGIN, 1 + MARGIN])
    if ifLegend:
        ax.legend(loc="lower right", fontsize=label_font_size)


def plot_delta_F_ensemble(delta_F_history, candidates, agent_idx=0, factor_idx=0,
                          ifLegend=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    seed = 0
    T = len(delta_F_history[0])
    model_names = candidates[seed][0][agent_idx].split()
    #print(model_names)
    num_models = len(delta_F_history[0][0][agent_idx][factor_idx])
    #print(num_models)
    stack_data = [[] for _ in range(num_models)]
    for t in range(T):
        for m in range(num_models):
            stack_data[m].append(delta_F_history[seed][t][agent_idx][factor_idx][m].item())

    stack_data = np.array(stack_data, dtype=float)
    if ifLegend:
        ax.stackplot(range(T), stack_data, labels=model_names, alpha=0.7)
    else:
        ax.stackplot(range(T), stack_data, alpha=0.7)

    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('ΔF (Ego, factor=0)', fontsize=label_font_size)
    ax.set_ylim([0 - MARGIN, 1 + MARGIN])
    ax.grid(True, alpha=0.3)
    if ifLegend:
        ax.legend(loc='upper right', fontsize=label_font_size / 2)


def plot_entropy_ensemble(entropy_history,
                          ifLegend=True, ax=None):
    entropy_history = np.array(entropy_history)
    num_seeds, T, num_agents, num_factors = entropy_history.shape  # (4 seeds, 1000 steps, 2 agents, 2 factors)
    linestyles = ['-', '--']

    for s in range(num_seeds):  # 对每个seed
        alpha = 0.3 if num_seeds > 1 else 1.0  # 多seed时用低透明度表示个体轨迹
        for a in range(num_agents):  # 每个代理
            for f in range(num_factors):  # 每个因子（通常是自己和对手）
                if f == 1:
                    continue
                label = f'Agent {a} Ego' if s == 0 else None  # 只在第一个seed显示图例，避免重复
                ax.plot(range(T),
                        entropy_history[s, :, a, f],
                        color=ENTROPY_COLOR if a == 0 else "red",
                        alpha=alpha,
                        linewidth=LINEWIDTH,
                        linestyle=linestyles[a],
                        label=label)

    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('Entropy (Nats)', fontsize=label_font_size)
    ax.tick_params(axis='both', which='major', labelsize=label_font_size)
    ax.grid(True, alpha=0.3)

    if ifLegend:
        ax.legend(loc='upper right', fontsize=label_font_size)


def plot_B_model_weights_ensemble(weights_history, candidates, agent_idx,
                                  ifLegend=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    seed, agent, factor = 0, agent_idx, 0

    model_names = candidates[seed][0][agent].split()
    T = len(weights_history[0])

    #print(weights_history[seed][0])
    num_models = len(weights_history[seed][0][agent][factor])
    stack_data = [[] for _ in range(num_models)]

    for t in range(T):
        for m in range(num_models):
            stack_data[m].append(weights_history[seed][t][agent][factor][m].item())

    stack_data = np.array(stack_data, dtype=float)

    if ifLegend:
        ax.stackplot(range(T), stack_data, labels=model_names, alpha=0.5)
    else:
        ax.stackplot(range(T), stack_data, alpha=0.5)

    ax.set_xlabel('Time step (t)', fontsize=label_font_size)
    ax.set_ylabel('Model weights', fontsize=label_font_size)
    ax.set_ylim([0 - MARGIN, 1 + MARGIN])
    ax.grid(True, alpha=0.3)
    if ifLegend:
        ax.legend(loc='lower right', fontsize=label_font_size)


def plot_state_space_trajectory(
        trajectory,
        ax=None,
        legend=False,
        gradient=False,
        lines=True,
        vertex_markers=True,
        additional_trajectory_kwargs: dict = {}):
    '''
    Args:
        trajectory: np.ndarray of shape (T, num_agents)

    >>> q_u_hist = np.array(variables_history['q_u'])
    >>> plotting.plot_state_space_trajectory(q_u_hist[:, :, 0])
    '''

    num_agents = trajectory.shape[1]
    T = trajectory.shape[0]

    corner_kwargs = dict(
        alpha=1.,
        marker='^',
        s=50,
        zorder=3,
    )

    trajectory_kwargs = dict(
        label='Trajectory', linewidth=0.5, marker='o', markersize=2
    ) | additional_trajectory_kwargs  # Merge the defaults and user-provided dictionaries

    if gradient:
        from matplotlib.cm import viridis
        from matplotlib.colors import Normalize

        # Create a colormap (viridis) and normalize the time steps
        norm = Normalize(vmin=0, vmax=T - 1)
        colors = viridis(norm(np.arange(T)))

    if num_agents == 3:
        from mpl_toolkits.mplot3d import Axes3D

        # Plot the 3D trajectory -----------------------------------------------
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        if gradient:
            for t in range(T - 1):
                ax.plot(
                    trajectory[t, 0], trajectory[t, 1], trajectory[t, 2],
                    color=colors[t], **trajectory_kwargs)
                if lines:
                    ax.plot(trajectory[t:t + 2, 0], trajectory[t:t + 2, 1], trajectory[t:t + 2, 2],
                            color=colors[t], alpha=0.4)
        else:
            ax.plot(
                trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                **trajectory_kwargs)

        # Plot the corners of the cube -----------------------------------------
        if vertex_markers:
            ax.scatter(
                0, 0, 0,
                label='0C', color='#fc03ad', **corner_kwargs)
            ax.scatter(
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                label='1C', color='#c90411', **corner_kwargs)
            ax.scatter(
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
                label='2C', color='#fc6f03', **corner_kwargs)
            ax.scatter(
                1, 1, 1,
                label='3C', color='#39c904', **corner_kwargs)

        # Set labels and limits to define the cube -----------------------------
        ax.set_xlabel('$q(u_i = c)$')
        ax.set_ylabel('$q(u_j = c)$')
        ax.set_zlabel('$q(u_k = c)$')

        # Setting the limits of the cube to be between 0 and 1
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])

    elif num_agents == 2:

        if ax is None:
            fig, ax = plt.subplots()

        if gradient:
            for t in range(T - 1):
                ax.plot(
                    trajectory[t, 0], trajectory[t, 1],
                    color=colors[t], **trajectory_kwargs)
                if lines:
                    ax.plot(trajectory[t:t + 2, 0], trajectory[t:t + 2, 1],
                            color=colors[t], alpha=0.4)
        else:
            ax.plot(
                trajectory[:, 0], trajectory[:, 1],
                **trajectory_kwargs)

        # Plot the corners of the square ---------------------------------------
        if vertex_markers:
            ax.scatter(
                0, 0,
                label='0C', color='#fc03ad', **corner_kwargs)
            ax.scatter(
                [1, 0],
                [0, 1],
                label='1C', color='#fc6f03', **corner_kwargs)
            ax.scatter(
                1, 1,
                label='2C', color='#39c904', **corner_kwargs)

        # Set labels, limits, etc ----------------------------------------------
        ax.set_xlabel('$q(u_i = c)$')
        ax.set_ylabel('$q(u_j = c)$')

        # set axis ratio to equal (square)
        ax.set_aspect('equal', adjustable='datalim')

    if legend:
        ax.legend()
    ax.set_title('State-space trajectory')
    # plt.show()


def plot_rolling_payoffs(data, game_transitions, ax=None):
    '''
    Args:
        data (ndarray): Payoff history of shape (T, num_players)

    >>> plt.figure(figsize=(8, 2), dpi=plotting.DPI)
    >>> ax = plt.gca()
    >>> plotting.plot_rolling_payoffs(np.array(variables_history['payoff']), game_transitions, ax)
    '''

    num_players = data.shape[1]
    T = data.shape[0]
    if ax is None:
        plt.figure(figsize=(8, 2), dpi=DPI)
        ax = plt.gca()

    # Individual payoffs
    for i in range(num_players):
        # plt.scatter(np.arange(len(data[:, i])), data[:, i])
        window = 20
        rolling_mean = np.convolve(data[:, i], np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window, T + 1), rolling_mean)

    # Mean payoffs
    plt.plot(np.arange(len(data)), data.mean(axis=1),
             marker='o', color='gray', linewidth=0.5, markersize=3, alpha=0.2)

    # Plot rolling mean
    window = 20
    rolling_mean = np.convolve(data.mean(axis=1), np.ones(window) / window, mode='valid')
    plt.plot(np.arange(window, T + 1), rolling_mean, color='red')

    highlight_transitions(game_transitions, ax)

    plt.title(f'Payoffs: mean and rolling mean ({window}-step window)')
    plt.xlabel('Time step')
    plt.ylabel('Payoff')
    # plt.show()


def plot_action_history(action_history, num_players, num_actions, ax=None):
    '''Plot the action history of the agents with custom colors.
    
    Args:
        action_history (np.array): Action history of the agents shape (num_players, T)
        num_players (int): Number of agents.
        num_actions (int): Number of actions.
    '''

    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    # Custom colors
    colors = ACTION_COLORS[:num_actions]
    cmap = mcolors.ListedColormap(colors)

    if ax is None:
        plt.figure(figsize=(8, 2), dpi=DPI)
        ax = plt.gca()

    # Display the action history with custom colors
    im = ax.imshow(
        action_history.T,
        origin='upper',
        aspect='auto',
        interpolation='nearest',
        cmap=cmap  # Apply the custom colormap
    )

    # Set y-ticks and labels
    ax.set_yticks(np.arange(num_players))
    ax.set_yticklabels([f'Agent {i + 1}' for i in range(num_players)])
    ax.set_xlabel('Time step')

    # Create custom legend
    defect_patch = mpatches.Patch(color=colors[0], label='Cooperate')
    cooperate_patch = mpatches.Patch(color=colors[1], label='Defect')
    ax.legend(handles=[defect_patch, cooperate_patch], loc='upper right')

    ax.set_title('Action history')


# ==============================================================================
# State-space trajectory plots (3D)
# ==============================================================================
def plot_reference_hull(
        ax,
        face_colors=['#cfaf59', '#ffdc7d', '#ffffff', '#ffffff'],
        face_alphas=[1, 1, 0, 0],
        legend=True,
):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    # Define the 8 vertices of a cube
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [1, 1, 0],
                         [1, 0, 1],
                         [0, 1, 1],
                         [1, 1, 1]])

    # Define the edges of the cube by connecting vertices
    edges = [
        (0, 1),
        (1, 4),
        (0, 4),
        (4, 7),
        (1, 7),
        (0, 7),
    ]

    # Extract the edge coordinates from the vertices
    edge_lines = [(vertices[start], vertices[end]) for start, end in edges]

    # Define triangular faces to shade (based on vertex indices)
    faces = [
        [vertices[0], vertices[1], vertices[4]],  # Bottom
        [vertices[1], vertices[4], vertices[7]],  # Side
        [vertices[0], vertices[4], vertices[7]],  # Ceiling back
        [vertices[0], vertices[1], vertices[7]],  # Ceiling front
    ]

    # Create a Poly3DCollection for the faces and add shading
    for i, face in enumerate(faces):
        face_collection = Poly3DCollection(
            [face],
            color=face_colors[i], alpha=face_alphas[i],
            linewidths=1, edgecolors='k', zsort='min')
        face_collection.set_edgecolor('k')
        face_collection.set_zorder(1)
        ax.add_collection3d(face_collection)

    # Plot each edge as a line segment
    for edge in edge_lines:
        ax.plot(*zip(*edge), color="k", linewidth=LINEWIDTH, zorder=2)

    # Plot the corners of the cube
    CORNER_MARKER = '^'
    CORNER_MARKER_SIZE = 80
    ax.scatter(0, 0, 0, label='0C', color='#fc03ad', alpha=1.,
               marker=CORNER_MARKER, s=CORNER_MARKER_SIZE, zorder=3)
    ax.scatter(1, 0, 0, label='1C', color='#c90411', alpha=1.,
               marker=CORNER_MARKER, s=CORNER_MARKER_SIZE, zorder=3)
    ax.scatter(1, 1, 0, label='2C', color='#fc6f03', alpha=1.,
               marker=CORNER_MARKER, s=CORNER_MARKER_SIZE, zorder=3)
    ax.scatter(1, 1, 1, label='3C', color='#39c904', alpha=1.,
               marker=CORNER_MARKER, s=CORNER_MARKER_SIZE, zorder=3)

    # Set the labels and aspect ratio
    ax.set_xlabel('$q(u_i = c)$')
    ax.set_ylabel('$q(u_j = c)$')
    ax.set_zlabel('$q(u_k = c)$')

    # Setting the limits of the cube to be between 0 and 1
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_zlim([-0.05, 1.05])

    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    if legend:
        ax.legend()


def plot_trajectory_flat(trajectory):
    boundary_kwargs = dict(
        color='gray', linestyle='--', zorder=-1, alpha=0.5,
    )
    trajectory_kwargs = dict(
        alpha=0.6,
        marker='o',
        linewidth=LINEWIDTH,
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(10, 5),
        gridspec_kw={'width_ratios': [1, 1], 'wspace': 0})

    # First plot (left side)
    ax1.plot(
        [p[0] for p in trajectory],
        [p[1] for p in trajectory],
        c='#323bf0',
        **trajectory_kwargs
    )
    ax1.plot([0, 1], [0, 1], **boundary_kwargs)
    ax1.plot([0, 1], [0, 0], **boundary_kwargs)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Second plot (right side)
    ax2.plot(
        [p[2] for p in trajectory],
        [p[1] for p in trajectory],
        c='#1b2080',
        **trajectory_kwargs
    )
    ax2.plot([0, 1], [0, 1], **boundary_kwargs)
    ax2.plot([0, 1], [1, 1], **boundary_kwargs)
    ax2.plot([0, 0], [0, 1], **boundary_kwargs)
    # ax2.set_ylabel('Y')
    ax2.set_xlabel('Z')

    # Hide the right spine of the first plot and the left spine of the second plot
    ax1.set_xlim(None, 1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.set_xlim(0, None)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    # Share the y-axis between the plots by aligning their limits
    ax2.set_yticks([])  # Optionally hide the y-axis ticks on the second plot

    # Display the plot
    # plt.show()