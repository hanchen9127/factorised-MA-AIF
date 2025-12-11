'''
Retrieve simulation data from database, plot, and store plots to disk.

Original Author: Jaime Ruiz Serra
Date:   2024-09-23

Extended by: Hanchen Wang
Date:
'''

import argparse
import os
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
import utils.plotting
import utils.database
import utils.timeseries
import generator2 as gen
import matplotlib.colors as mcolors

utils.plotting.DPI = 300
utils.plotting.SHOW_LEGEND = False
utils.plotting.ONLY_LEFT_Y_LABEL = True
utils.plotting.TIGHT_LAYOUT = True
SIZE_FACTOR = 1.5


def main(args):
    # Retrieve data from database --------------------------------------------------
    logging.info(f'Retrieving data matching {args.timestamp} from database {args.db_path}')
    metadata = utils.database.retrieve_timeseries_matching(
        db_path=args.db_path,
        sql_query=(
            'SELECT * FROM metadata '
            f'WHERE timestamp LIKE "%{args.timestamp}%" '
            # 'AND description LIKE "%{DESCRIPTION}%" '
        )
    )
    if len(metadata) == 0:
        logging.info(f'❌ No data found matching timestamp {args.timestamp}. Aborting.')
        exit()
    game_transitions = pickle.loads(metadata.iloc[0]['game_transitions'])
    agent_kwargs = pickle.loads(metadata.iloc[0]['agent_kwargs'])
    commit_sha = metadata.iloc[0]['commit_sha']
    description = metadata.iloc[0]['description']
    timestamp = metadata.iloc[0]['timestamp']
    num_players = metadata.iloc[0]['num_agents']
    num_actions = metadata.iloc[0]['num_actions']
    # KEY measurement of plot_cooperation_prob()
    nash_strategy = pickle.loads(metadata.iloc[0]['nash_strategy'])

    experiments = utils.database.retrieve_timeseries_matching(
        db_path=args.db_path,
        sql_query=(
            'SELECT * FROM timeseries '
            f'WHERE timestamp LIKE "%{timestamp}%" '
            f'AND commit_sha LIKE "%{commit_sha}%" '
        )
    )
    logging.info(f'Found {len(experiments)} matching experiments')

    # Create output directory ------------------------------------------------------
    if len(experiments) > 0:
        output_dir = os.path.join(
            args.figures_dir,
            f'{timestamp}-{num_players}x{num_actions}-{description.replace(" ", "_").replace("/", "_")[:100]}'
        )
        try:
            os.makedirs(output_dir, exist_ok=False)
        except:
            raise FileExistsError(f'❌ Output directory {output_dir} already exists. Aborting.')
        logging.info(f'Created output directory {output_dir}')
    else:
        raise ValueError('No matching experiments found.')

    # Store metadata in readable format --------------------------------------------
    with open(os.path.join(output_dir, 'metadata.md'), 'w') as f:
        f.write(f'# Metadata\n\n')
        f.write(f'**Description**: {description}\n\n')
        f.write(f'**Timestamp**: `{timestamp}`\n\n')
        f.write(f'**Commit SHA**: `{commit_sha}`\n\n')
        f.write(f'**Game transitions**:\n\n')
        for i, (name, _, duration) in enumerate(game_transitions):
            f.write(f'{i + 1}. {name} ({duration} steps)\n')
        f.write('\n')
        f.write(f'**Agent kwargs**:\n\n')
        for i, kwargs in enumerate(agent_kwargs):
            f.write(f'- Agent _{chr(105 + i)}_\n')
            for k, v in kwargs.items():
                f.write(f'    - {k}: {v}\n')
        f.write('\n')

    # --------------------------------------------------------------------------
    # Ensemble analysis (All seeds in one figure)
    # --------------------------------------------------------------------------
    ensemble_ouput = f'ts-ensemble-all-{len(experiments)}-seeds'
    suptitle = 'Factorised MA-AIF Ensemble'
    all_vfe, all_q_u, all_efe, all_B, all_q_s = [], [], [], [], []

    for i in range(len(experiments)):
        loaded_vars = utils.database.load_single_timeseries(experiments, i)

        all_vfe.append(np.array(loaded_vars['VFE']))
        all_q_u.append(np.array(loaded_vars['q_u']))
        all_efe.append(np.array(loaded_vars['EFE']))
        all_B.append(np.array(loaded_vars['B']))
        all_q_s.append(np.array(loaded_vars['q_s']))

    plot_configs = [
        {'plot_fn': utils.plotting.plot_vfe_ensemble,
         'args': (np.stack(all_vfe),
                  game_transitions)
         },
        {'plot_fn': utils.plotting.plot_expected_efe_ensemble,
         'args': (np.stack(all_q_u),
                  np.stack(all_efe))
         },
        {'plot_fn': utils.plotting.plot_policies_ensemble,
         'args': (np.stack(all_q_u),
                  game_transitions,
                  nash_strategy),
         },
        {'plot_fn': utils.plotting.plot_B_state_ensemble,
         'args': (np.stack(all_B),
                  np.stack(all_q_s))
         },
    ]

    # Create a figure with subplots for each agent, arranged in a (num_plots)x(num_agents) grid
    t_min = 0
    t_max = None
    n_rows = len(plot_configs)
    figsize = (10, int(n_rows * 2.2) + 2)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, dpi=utils.plotting.DPI)
    fig.subplots_adjust(hspace=0.4, wspace=0.1)  # Adjust spacing

    for row_idx in range(n_rows):
        ax = axes[row_idx] if n_rows > 1 else axes
        plot_configs[row_idx]['plot_fn'](
            *plot_configs[row_idx]['args'],
            ax=ax,
        )
        if game_transitions and not t_max:
            game_labels = utils.plotting.highlight_transitions(game_transitions, ax)
            if row_idx == 0:
                suptitle += f" ({game_labels})"
                ensemble_ouput += f" ({game_labels})"
        if utils.plotting.TIGHT_LAYOUT and row_idx < n_rows - 1:
            ax.set_xlabel(None)

    # Adjust layout for better spacing and overall title
    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make space for the suptitle
    if utils.plotting.TIGHT_LAYOUT:
        plt.subplots_adjust(wspace=0.15, hspace=0.25)

    # Save the figure to disk
    ensemble_ouput += ".png"
    fig.savefig(os.path.join(output_dir, ensemble_ouput))
    plt.close(fig)
    logging.info(f'✅ Saved {output_dir}/{ensemble_ouput}')

    # Run statistical test
    logging.info("Running statistical test")
    consistency = gen.strategic_consistency_test(experiments,
                                                 game_transitions,
                                                 defection_threshold=0.5,
                                                 save_path=output_dir)

    # --------------------------------------------------------------------------
    # Transition Matrix B analysis (Individual)
    # --------------------------------------------------------------------------
    logging.info("Plotting Transition Matrix B analysis")

    t1, t2 = 200, 550
    for i in range(3):
        loaded_vars = utils.database.load_single_timeseries(experiments, i)
        # q_pi_history = np.round(loaded_vars['q_u'], 2)
        seed = loaded_vars['seed']
        B_all = np.array(loaded_vars['B'])

        # print("Seed", seed)
        # print(B_all.shape)
        # print(B_all[t1])
        # print("------------------")
        # print(B_all[t2])

        T, num_agents, num_factors, num_actions, _, _ = B_all.shape
        cmap_action = {
            0: mcolors.LinearSegmentedColormap.from_list('blue_map', ['#ffffff', '#03dffc']),
            1: mcolors.LinearSegmentedColormap.from_list('pink_map', ['#ffffff', '#f70ce4'])
        }

        fig, axes = plt.subplots(
            4, 4,
            figsize=(4 * 4, 3 * 4),
            squeeze=False, dpi=300
        )

        for row_idx, (agent, time) in enumerate([(0, t1), (0, t2), (1, t1), (1, t2)]):
            for factor in range(num_factors):
                for action in range(num_actions):
                    col_idx = factor * num_actions + action
                    ax = axes[row_idx, col_idx]

                    B_plot = B_all[time, agent, factor, action, :, :]

                    im = ax.imshow(B_plot, vmin=0, vmax=1, cmap=cmap_action[action])
                    for i in range(2):
                        for j in range(2):
                            ax.text(j, i, f'{B_plot[i, j]:.2f}', ha='center', va='center', color='black')

                    ax.set_xticks([0, 1])
                    ax.set_xticklabels([0, 1])
                    ax.set_yticks([0, 1])
                    ax.set_yticklabels([0, 1])
                    ax.set_title(f'factor={factor}, action={action}')

            test_result = "Consistent" if consistency[seed][agent] else "Inconsistent"
            text_color = "green" if consistency[seed][agent] else "red"
            axes[row_idx, 0].set_ylabel(f'Agent {agent}, t={time} \n({test_result})',
                                        rotation=0,
                                        labelpad=50,
                                        va='center',
                                        color=text_color,
                                        fontsize=12)

        # Add a colorbar on the right
        cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04)
        cbar.set_label('Transition Probability')
        plt.suptitle(f'B-matrix Heatmaps Seed:{seed}', fontsize=16)
        out_file = os.path.join(output_dir, f'B-seed{seed}.png')
        fig.savefig(out_file, dpi=utils.plotting.DPI)
        plt.close(fig)
        logging.info(f'✅ Saved {out_file}')

    # Generate the video for B
    gen.animate_transition_matrix_B(experiments, consistency,
                                    save_path=output_dir,
                                    q_s_computation=True,
                                    selected_seed=[1, 2])

    # Plot per-agent timeseries ----------------------------------------------------
    plot_configs = utils.plotting.make_default_config(loaded_vars, nash_strategy, game_transitions)
    seed = loaded_vars['seed']

    fig = utils.plotting.plot(
        plot_configs=plot_configs,
        game_transitions=game_transitions,
        num_players=num_players,
        suptitle=(
            f'Factorised MA-AIF (seed={seed})'
        ),
        figsize=(14 * SIZE_FACTOR, 22 * SIZE_FACTOR),
        # t_min=725, t_max=745,
    )

    # # Save the figure to disk
    # fig.savefig(os.path.join(output_dir, f'ts-individual-{seed}.png'))
    # plt.close(fig)
    # logging.info(f'✅ Saved {output_dir}/ts-individual-{seed}.png')

    # Plot individual figures for ALL seeds ----------------------------------------
    for i in range(3):
        loaded_vars = utils.database.load_single_timeseries(experiments, i)
        seed = loaded_vars['seed']
        num_actions = metadata.iloc[0]['num_actions']
        num_players = metadata.iloc[0]['num_agents']

        plot_configs = utils.plotting.make_default_config(loaded_vars, nash_strategy, game_transitions)

        fig = utils.plotting.plot(
            plot_configs=plot_configs,
            game_transitions=game_transitions,
            num_players=num_players,
            suptitle=f'Factorised MA-AIF (seed={seed})',
            figsize=(14 * SIZE_FACTOR, 28 * SIZE_FACTOR),
            # t_min=args.t_min,
            # t_max=args.t_max  # optional time window
        )

        fig.savefig(os.path.join(output_dir, f'ts-individual-{seed}.png'), dpi=300)
        plt.close(fig)
        logging.info(f'✅ Saved {output_dir}/ts-individual-{seed}.png')

    # # Plot state-space trajectory --------------------------------------------------
    # from functools import reduce
    #
    # logging.info(f'Processing {len(experiments)} experiment trajectories')
    #
    # fig = plt.figure()
    # if num_players == 2:
    #     # raise NotImplementedError('This code is not ready for 2-player games')
    #     ax = fig.add_subplot(211)
    #     ax2 = fig.add_subplot(212)
    # elif num_players == 3:
    #     ax = fig.add_subplot(211, projection='3d')
    #     ax2 = fig.add_subplot(212, projection='3d')
    #
    # agent_kwargs_same = reduce(
    #     lambda x, y: x == y and x,
    #     agent_kwargs
    # ) or (agent_kwargs == [{}] * num_players)
    # logging.info((
    #     'Homogeneous agents: will permute space where applicable.'
    #     if agent_kwargs_same else
    #     'Heterogeneous agents: will not permute space.'
    # ))

    # # Reference trajectory
    # i = 0
    # loaded_vars = utils.database.load_single_timeseries(experiments, i)
    # seed = loaded_vars['seed']
    # q_u_hist = loaded_vars['q_u']
    #
    # dataset = np.empty((len(experiments), len(q_u_hist[args.t_min:args.t_max]), num_players))
    # distances = []
    # seeds = []
    # for i in range(len(experiments)):
    #
    #     logging.info(f'{i + 1}/{len(experiments)}')
    #
    #     # Load data
    #     loaded_vars = utils.database.load_single_timeseries(experiments, i)
    #     seed = loaded_vars['seed']
    #     q_u_hist = loaded_vars['q_u']
    #     trajectory = q_u_hist[args.t_min:args.t_max, :, 0].copy()
    #
    #     # Plot ORIGINAL trajectory
    #     utils.plotting.plot_state_space_trajectory(
    #         np.array(trajectory),
    #         ax,
    #         gradient=True,
    #         # lines=False,
    #         additional_trajectory_kwargs={
    #             'alpha': 0.3,
    #             # 'color': 'red',
    #         })
    #
    #     # Use reduce operation to check if all dicts in agent_kargs are the same
    #     if agent_kwargs_same:
    #         # Permute agents to a "canonical" order
    #         trajectory = utils.timeseries.permute_agents(trajectory.copy())
    #
    #     # Save trajectory and seed
    #     dataset[i] = trajectory
    #     seeds.append(seed)
    #
    #     # Plot trajectory
    #     utils.plotting.plot_state_space_trajectory(
    #         np.array(trajectory),
    #         ax2,
    #         gradient=True,
    #         # lines=False,
    #         additional_trajectory_kwargs={
    #             'alpha': 0.3,
    #             # 'color': 'red',
    #         })
    #
    #     # Highlight specific points
    #     # ax.scatter(*trajectory[300], color='red', s=40, zorder=10)
    #     # ax.scatter(*trajectory[950], color='orange', s=40, zorder=10)
    #
    # args.t_min = 0 if args.t_min is None else args.t_min
    # args.t_max = len(q_u_hist) if args.t_max is None else args.t_max
    # ax.set_title(f"Original trajectories ({args.t_min} < t < {args.t_max})")
    # ax2.set_title(f"Normalised trajectories ({args.t_min} < t < {args.t_max})")
    # if num_players == 3:
    #     # Change azimuth and elevation
    #     ax.view_init(azim=40, elev=30)
    #     ax2.view_init(azim=40, elev=30)
    #
    # # Save the figure to disk
    # fig.savefig(os.path.join(output_dir, f'ss-transform.png'))
    # plt.close(fig)
    # logging.info(f'✅ Saved {output_dir}/ss-transform.png')
    #
    # # Clustering -------------------------------------------------------------------
    # from tslearn.clustering import TimeSeriesKMeans
    # from tslearn.metrics import dtw
    # from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    #
    # n_clusters = min(args.n_clusters, len(dataset))
    #
    # # Normalize the data
    # scaler = TimeSeriesScalerMeanVariance()
    # time_series_data_scaled = scaler.fit_transform(dataset)
    #
    # # KMeans clustering with DTW
    # model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10)
    # labels = model.fit_predict(time_series_data_scaled)
    #
    # unique_labels, counts = np.unique(labels, return_counts=True)
    # logging.info(f'Cluster member counts {[(u, c) for u, c in zip(unique_labels, counts)]}')
    #
    # # Plot clusters
    # if num_players == 3:
    #     from sklearn.decomposition import PCA
    #
    #     # Fit PCA to all trajectories at once
    #     all_data_3d = dataset.reshape(-1, num_players)
    #     pca = PCA(n_components=2)
    #     pca.fit(all_data_3d)

    # Plotting --------------------------------------------------------------------
    # fig, axs = plt.subplots(1, n_clusters, figsize=(3 * n_clusters, 3))
    # fig.suptitle('Clusters')
    # axs = axs.flatten()
    #
    # for i in range(len(experiments)):
    #
    #     if num_players == 3:
    #         # Retrieve 3D trajectory data and project to 2D
    #         data_3d = dataset[i]
    #         data_2d = pca.transform(data_3d)
    #     elif num_players == 2:
    #         data_2d = dataset[i]
    #
    #     cluster = labels[i]
    #
    #     # Plot the 2D projection of the trajectory in its cluster plot
    #     axs[cluster].plot(data_2d[:, 0], data_2d[:, 1], alpha=0.3, marker='o', markersize=1)
    #     axs[cluster].set_title(f'{counts[cluster]} match' + ('es' if counts[cluster] > 1 else ''))
    #     axs[cluster].axis('off')
    #
    #     # Plot (the 2D projection of) the vertices of the unit cube
    #     corner_kwargs = dict(
    #         marker='^', s=40, zorder=10, alpha=1.,
    #     )
    #
    #     if num_players == 3:
    #         # 3C -------------------------------------------------
    #         reference_points = pca.transform(np.array([
    #             [1., 1., 1.],
    #         ]))
    #         axs[cluster].scatter(
    #             reference_points[:, 0], reference_points[:, 1],
    #             label='3C', color='#39c904', **corner_kwargs)
    #
    #         # 2C -------------------------------------------------
    #         reference_points = pca.transform(np.array([
    #             [1., 1., 0.],
    #             [1., 0., 1.],
    #             [0., 1., 1.],
    #         ]))
    #         axs[cluster].scatter(
    #             reference_points[:, 0], reference_points[:, 1],
    #             label='2C', color='#fc6f03', **corner_kwargs)
    #
    #         # 1C -------------------------------------------------
    #         reference_points = pca.transform(np.array([
    #             [1., 0., 0.],
    #             [0., 1., 0.],
    #             [0., 0., 1.],
    #         ]))
    #         axs[cluster].scatter(
    #             reference_points[:, 0], reference_points[:, 1],
    #             label='1C', color='#c90411', **corner_kwargs)
    #
    #         # 0C -------------------------------------------------
    #         reference_points = pca.transform(np.array([
    #             [0., 0., 0.],
    #         ]))
    #         axs[cluster].scatter(
    #             reference_points[:, 0], reference_points[:, 1],
    #             label='0C', color='#fc03ad', **corner_kwargs)
    #
    #     elif num_players == 2:
    #         # 2C -------------------------------------------------
    #         reference_points = np.array([
    #             [1., 1.],
    #         ])
    #         axs[cluster].scatter(
    #             reference_points[:, 0], reference_points[:, 1],
    #             label='2C', color='#39c904', **corner_kwargs)
    #
    #         # 1C -------------------------------------------------
    #         reference_points = np.array([
    #             [1., 0.],
    #             [0., 1.],
    #         ])
    #         axs[cluster].scatter(
    #             reference_points[:, 0], reference_points[:, 1],
    #             label='1C', color='#fc6f03', **corner_kwargs)
    #
    #         # 0C -------------------------------------------------
    #         reference_points = np.array([
    #             [0., 0.],
    #         ])
    #         axs[cluster].scatter(
    #             reference_points[:, 0], reference_points[:, 1],
    #             label='0C', color='#fc03ad', **corner_kwargs)
    #
    # # Save the figure to disk
    # fig.savefig(os.path.join(output_dir, f'ss-clusters.png'))
    # plt.close(fig)
    # logging.info(f'✅ Saved {output_dir}/ss-clusters.png')

    logging.info('\n😎 INFG-plot.py Done!')
