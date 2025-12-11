'''
Retrieve simulation data from database, plot, and store plots to disk.

Original Author: Jaime Ruiz Serra
Date: 2024-09-23

Extended by: Hanchen Wang
Date: 2025-11
'''

import argparse
import os
import logging
import pickle

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sys
import importlib
from tqdm import tqdm

plotsie = importlib.import_module("INFG-plot")

sys.path.append('../')

import utils.database
import utils.plotting

utils.plotting.DPI = 120
utils.plotting.SHOW_LEGEND = False
utils.plotting.ONLY_LEFT_Y_LABEL = True
utils.plotting.TIGHT_LAYOUT = True


def load_timestamp_data(args, timestamp):
    """Load data for a single timestamp, optimized for parallel execution."""
    args_copy = argparse.Namespace(**vars(args))  # Copy args to avoid shared state
    args_copy.timestamp = timestamp
    logging.info(f'Loading data for timestamp {timestamp}')
    metadata = utils.database.retrieve_timeseries_matching(
        db_path=args_copy.db_path,
        sql_query=f'SELECT * FROM metadata WHERE timestamp LIKE "%{timestamp}%"'
    )
    if len(metadata) == 0:
        return None
    commit_sha = metadata.iloc[0]['commit_sha']
    experiments = utils.database.retrieve_timeseries_matching(
        db_path=args_copy.db_path,
        sql_query=f'SELECT * FROM timeseries WHERE timestamp LIKE "%{timestamp}%" AND commit_sha LIKE "%{commit_sha}%"'
    )
    all_vfe, all_q_u, all_efe, all_B, all_q_s, all_delta_F = [], [], [], [], [], []
    for i in range(len(experiments)):
        loaded_vars = utils.database.load_single_timeseries(experiments, i)
        all_vfe.append(np.array(loaded_vars['VFE']))
        all_q_u.append(np.array(loaded_vars['q_u']))
        all_efe.append(np.array(loaded_vars['EFE']))
        all_B.append(np.array(loaded_vars['B']))
        all_q_s.append(np.array(loaded_vars['q_s']))
        all_delta_F.append(np.array(loaded_vars['delta_F']))
    return {
        'timestamp': timestamp,
        'game_transitions': pickle.loads(metadata.iloc[0]['game_transitions']),
        'nash_strategy': pickle.loads(metadata.iloc[0]['nash_strategy']),
        'all_vfe': np.stack(all_vfe),
        'all_q_u': np.stack(all_q_u),
        'all_efe': np.stack(all_efe),
        'all_B': np.stack(all_B),
        'all_q_s': np.stack(all_q_s),
        'all_delta_F': np.stack(all_delta_F),
    }


def plot_all_games_ensemble_for_one_file(data_list, output_dir, filename):
    """Generate a single ensemble figure with one column per timestamp (matching per game_transition)."""
    num_timestamps = len(data_list)
    n_rows = 7  # Adjust to the length of plot_configs
    figsize = (5 * num_timestamps, int(n_rows * 2))
    fig, axes = plt.subplots(n_rows, num_timestamps, figsize=figsize, dpi=utils.plotting.DPI, squeeze=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.15)

    for col_idx, data in enumerate(tqdm(data_list)):
        timestamp = data['timestamp']
        game_transitions = data['game_transitions']
        nash_strategy = data['nash_strategy']
        ifLegend = False if col_idx != len(data_list) - 1 else True

        plot_configs = [
            {'plot_fn': utils.plotting.plot_vfe_ensemble,
             'args': (data['all_vfe'],
                      game_transitions,
                      ifLegend)},
            {'plot_fn': utils.plotting.plot_expected_efe_ensemble,
             'args': (data['all_q_u'],
                      data['all_efe'],
                      ifLegend)},
            {'plot_fn': utils.plotting.plot_policies_ensemble,
             'args': (data['all_q_u'],
                      game_transitions,
                      nash_strategy,
                      ifLegend)},
            {'plot_fn': utils.plotting.plot_B_state_ensemble,
             'args': (data['all_B'],
                      data['all_q_s'],
                      ifLegend,
                      False,
                      True)},
            {'plot_fn': utils.plotting.plot_B_separation_degree_ensemble,
             'args': (data['all_B'],
                      data['all_q_u'],
                      game_transitions,
                      0,  # Ego
                      "Cooperators",
                      ifLegend,
                      False,  # Not single -> All seeds
                      True  # Use average for all seeds
                      )},
            {'plot_fn': utils.plotting.plot_B_separation_degree_ensemble,
             'args': (data['all_B'],
                      data['all_q_u'],
                      game_transitions,
                      0,  # Ego
                      "Defectors",
                      ifLegend,
                      False,
                      True)},
            {'plot_fn': utils.plotting.plot_delta_F_ensemble,
             'args': (data['all_delta_F'],
                      game_transitions,
                      False)},
            # {'plot_fn': utils.plotting.plot_B_separation_degree_ensemble,  # If uncomment, remember update n_rows to 8
            #  'args': (data['all_B'],
            #           data['all_q_u'],
            #           game_transitions,
            #           1,  # Alter-Ego
            #           "Cooperators",
            #           ifLegend,
            #           False,  # Not single -> All seeds
            #           True  # Use average for all seeds
            #           )},
            # {'plot_fn': utils.plotting.plot_B_separation_degree_ensemble,
            #  'args': (data['all_B'],
            #           data['all_q_u'],
            #           game_transitions,
            #           1,  # Alter-Ego
            #           "Defectors",
            #           ifLegend,
            #           False,
            #           True)}
        ]
        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]
            plot_configs[row_idx]['plot_fn'](*plot_configs[row_idx]['args'], ax=ax)
            # Remove all spines to make the plot borderless
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            game_labels = utils.plotting.highlight_transitions(game_transitions, ax)
            if row_idx == 0:
                ax.set_title(f'{game_labels}', fontsize=12)
            if row_idx < n_rows - 1:
                ax.set_xlabel(None)
            if col_idx > 0:
                ax.set_ylabel(None)

    fig.suptitle(f'Factorised MA-AIF Belief Separation ({filename})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    try:
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
        output_path = os.path.join(output_dir, f'Ensemble-All-Games-{filename}.png')
        fig.savefig(output_path)
        plt.close(fig)
        logging.info(f'✅ Saved {output_path}')
    except Exception as e:
        logging.error(f'Error saving Ensemble-AllGames-{filename}.png: {e}')
        plt.close(fig)
        exit()


def plot_all_games_ensemble_for_all_files(filenames, base_dir="BMR-Study", output_dir="MAAIF-Ensembles"):
    """
    Generate 3 ensemble figures (VFE, EFE, Policy) across all BMR files.
    Each row corresponds to one BMR method, and columns correspond to timestamps (games).
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f'Output directory ensured: {output_dir}')

    # Define which plots we want
    plot_configs = [
        {
            'name': 'Ensemble — VFE',
            'plot_fn': utils.plotting.plot_vfe_ensemble,
            'get_args': lambda data, ifLegend: (
                data['all_vfe'],
                data['game_transitions'],
                ifLegend
            )
        },
        {
            'name': 'Ensemble — EFE',
            'plot_fn': utils.plotting.plot_expected_efe_ensemble,
            'get_args': lambda data, ifLegend: (
                data['all_q_u'],
                data['all_efe'],
                ifLegend
            )
        },
        {
            'name': "Ensemble — State P(s'=1)",
            'plot_fn': utils.plotting.plot_B_state_ensemble,
            'get_args': lambda data, ifLegend: (
                data['all_B'],
                data['all_q_s'],
                ifLegend,
                False,  # Enable all seeds
                True
            )
        },
        {
            'name': 'Ensemble — Policy P(u=c)',
            'plot_fn': utils.plotting.plot_policies_ensemble,
            'get_args': lambda data, ifLegend: (
                data['all_q_u'],
                data['game_transitions'],
                data['nash_strategy'],
                ifLegend,
                False  # Enable all seeds
            )
        },
        {
            'name': 'Single — Policy P(u=c)',
            'plot_fn': utils.plotting.plot_policies_ensemble,
            'get_args': lambda data, ifLegend: (
                data['all_q_u'],
                data['game_transitions'],
                data['nash_strategy'],
                ifLegend,
                True  # Single seed
            )
        },
        {
            'name': 'Mean — Belief Separation (Cooperators)',
            'plot_fn': utils.plotting.plot_B_separation_degree_ensemble,
            'get_args': lambda data, ifLegend: (
                data['all_B'],
                data['all_q_u'],
                data['game_transitions'],
                0,
                "Cooperators",
                ifLegend,
                False,
                True
            )
        },
        {
            'name': 'Mean — Belief Separation (Defectors)',
            'plot_fn': utils.plotting.plot_B_separation_degree_ensemble,
            'get_args': lambda data, ifLegend: (
                data['all_B'],
                data['all_q_u'],
                data['game_transitions'],
                0,
                "Defectors",
                ifLegend,
                False,
                True
            )
        },
    ]

    # Load data for all BMR files
    all_BMR_data = {}
    for filename in filenames:
        db_path = os.path.join(base_dir, f"{filename}.db")
        metadata = utils.database.retrieve_timeseries_matching(
            db_path=db_path,
            sql_query='SELECT * FROM metadata WHERE timestamp LIKE "%2025%"'
        )
        timestamps = metadata['timestamp'].values
        logging.info(f'🔍 [{filename}] Found {len(timestamps)} timestamps: {timestamps}')

        # Parallel load
        with Pool(processes=min(4, len(timestamps))) as pool:
            data_list = pool.starmap(load_timestamp_data,
                                     [(argparse.Namespace(db_path=db_path), t) for t in timestamps])
        data_list = [d for d in data_list if d is not None]
        if not data_list:
            logging.warning(f'⚠️ No data loaded for {filename}')
            continue
        all_BMR_data[filename] = data_list

    # --- Create one figure per metric ---
    for config in plot_configs:
        metric = config['name']
        logging.info(f'🧩 Creating ensemble plot for {metric}')

        # Determine subplot layout
        n_rows = len(all_BMR_data)
        n_cols = max(len(v) for v in all_BMR_data.values())
        figsize = (4 * n_cols, 2.5 * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=utils.plotting.DPI, squeeze=False)
        fig.subplots_adjust(hspace=0.4, wspace=0.15)

        for row_idx, (filename, data_list) in enumerate(all_BMR_data.items()):
            for col_idx, data in enumerate(data_list):
                ax = axes[row_idx, col_idx]
                # ifLegend = (col_idx == len(data_list) - 1)                    # Add legend for all rows
                ifLegend = (row_idx == 0 and col_idx == len(data_list) - 1)  # Only for the first row (.db file)
                args = config['get_args'](data, ifLegend)
                config['plot_fn'](*args, ax=ax)

                # Remove spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                game_labels = utils.plotting.highlight_transitions(data['game_transitions'], ax)
                if row_idx == 0:
                    ax.set_title(game_labels, fontsize=12)
                if row_idx < n_rows - 1:
                    ax.set_xlabel(None)
                if col_idx > 0:
                    ax.set_ylabel(None)

            # Label each row by filename
            axes[row_idx, 0].set_ylabel(filename, fontsize=12, rotation=90, labelpad=10)

        fig.suptitle(f'Factorised MA-AIF {metric}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        # Save output
        output_path = os.path.join(output_dir, f'{metric}.png')
        fig.savefig(output_path)
        plt.close(fig)
        logging.info(f'✅ Saved {output_path}')


def generate_plots(timestamp, args):
    """Generate plots for a single timestamp using INFG-plot."""
    args.timestamp = timestamp
    logging.info(f'📊 Plotting data for timestamp {timestamp}')
    plotsie.main(args)


if __name__ == '__main__':
    # Manage all BMR files here
    bmr_methods = ["NoBMR", "Baseline_0.5", "Epsilon_0.25"]

    # Select the focused one
    selected_bmr = bmr_methods[-1]
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--db-path', type=str, default=f'BMR-Study/{selected_bmr}.db')
    argparser.add_argument('--timestamp', type=str, default='2025')
    argparser.add_argument('--figures-dir', type=str, default=f'MAAIF-Ensembles/{selected_bmr}')
    argparser.add_argument('--t-min', type=int, default=None)
    argparser.add_argument('--t-max', type=int, default=None)
    argparser.add_argument('--n-clusters', type=int, default=6)
    args = argparser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Create output directory
    try:
        os.makedirs(args.figures_dir, exist_ok=True)
        logging.info(f'Created output directory {args.figures_dir}')
    except Exception as e:
        logging.error(f'Error creating output directory {args.figures_dir}: {e}')
        exit()

    # Retrieve metadata
    metadata = utils.database.retrieve_timeseries_matching(
        db_path=args.db_path,
        sql_query=f'SELECT * FROM metadata WHERE timestamp LIKE "%{args.timestamp}%"'
    )
    timestamps = metadata['timestamp'].values
    logging.info(f'Found {len(timestamps)} timestamps: {timestamps}')

    # Parallel data loading
    with Pool(processes=min(4, len(timestamps))) as pool:
        data_list = pool.starmap(load_timestamp_data, [(args, timestamp) for timestamp in timestamps])
    data_list = [d for d in data_list if d is not None]
    if not data_list:
        logging.error('❌ No data loaded. Aborting.')
        exit()

    # Generate combined ensemble figure for .db files
    plot_all_games_ensemble_for_all_files(bmr_methods)
    plot_all_games_ensemble_for_one_file(data_list, args.figures_dir, selected_bmr)

    # Parallel generation for individual experiments
    # with Pool(processes=4) as pool:
    #     pool.starmap(generate_plots, [(timestamp, args) for timestamp in timestamps])

    logging.info('\n😎 INFG-plot-batch.py Done!')

    exit()
