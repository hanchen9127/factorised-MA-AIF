"""
INFG-plot-gui.py
================
Self-contained GUI for the MA-AIF plotting pipeline.

All data-loading, figure-building, and rendering logic that was previously
in INFG-plot-refined.py is inlined here.  No external pipeline module is
imported at runtime; only the project's own utility packages (utils.database,
utils.plotting, generator2) are required.

Layout of this file
-------------------
  Section 1  — GUI palette & constants
  Section 2  — Panel registry
  Section 3  — Data structures  (ExperimentData, PlotConfig)
  Section 4  — Data loading     (load_experiment, load_all_timestamps_parallel)
  Section 5  — Figure helpers   (build_figure, save_figure, _style_ax, …)
  Section 6  — B-matrix heatmap helper
  Section 7  — Panel-spec dispatch
  Section 8  — DB file discovery
  Section 9  — Styled widget helpers (_label, _button, _card, …)
  Section 10 — ScrollableCheckList widget
  Section 11 — DbFileTree widget
  Section 12 — ProgressPanel widget
  Section 13 — PlottingGUI  (main window, pipeline thread, renderer)
  Section 14 — Entry point

Usage
-----
1. Run commandline below to start.
    ''' 
    cd scripts
    python INFG-plot-gui.py <database_directory>
    
    ''' 
2. Hold CTRL to select multiple db files.
3. Click on checkbox to select various variables.
4. Press button at the bottom-right corner to generate figures.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import datetime
import logging
import os
import pickle
import sys
import threading
import tkinter as tk
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Third-party / project imports
# ---------------------------------------------------------------------------
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
import generator2 as gen
import utils.database
import utils.plotting

utils.plotting.DPI              = 500
utils.plotting.SHOW_LEGEND      = False
utils.plotting.ONLY_LEFT_Y_LABEL = True
utils.plotting.TIGHT_LAYOUT    = True

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


# ===========================================================================
# Section 1 — GUI palette & typography
# ===========================================================================

BG          = "#0f1117"
SURFACE     = "#1a1d27"
SURFACE2    = "#22263a"
BORDER      = "#2e3350"
ACCENT      = "#5c7cfa"
ACCENT2     = "#38d9a9"
TEXT        = "#e8eaf0"
TEXT_DIM    = "#7a8099"
DANGER      = "#ff6b6b"
SUCCESS     = "#38d9a9"
PROGRESS_BG = "#22263a"
PRECISION_COLOR = "#000000"

FONT_TITLE = ("Georgia",     20, "bold")
FONT_LABEL = ("Courier New", 10)
FONT_SMALL = ("Courier New",  9)
FONT_MONO  = ("Courier New", 10)
FONT_BTN   = ("Georgia",     11, "bold")

_DPI_RENDER = 500   # DPI for saved figures
_COL_W      = 5.0   # inches per game-sequence column
_ROW_H      = 3.2   # inches per db-file row


# ===========================================================================
# Section 2 — Panel registry
# ===========================================================================

DEFAULT_PANELS = [
    {"key": "vfe",           "label": "VFE Ensemble",                    "default": True},
    {"key": "efe",           "label": "Expected EFE Ensemble",            "default": True},
    {"key": "policy",        "label": "Policy  P(u=c)  Ensemble",         "default": True},
    {"key": "policy_single", "label": "Policy  P(u=c)  Single Seed",      "default": True},
    {"key": "state",         "label": "State   P(s′=1)  Ensemble",        "default": False},
    {"key": "sep_coop",      "label": "Belief Separation — Cooperators",  "default": False},
    {"key": "sep_def",       "label": "Belief Separation — Defectors",    "default": False},
    {"key": "delta_f",       "label": "ΔF  Ensemble",                     "default": False},
    {"key": "weights_i",     "label": "Model Weights  Agent i",           "default": True},
    {"key": "weights_j",     "label": "Model Weights  Agent j",           "default": False},
    {"key": "entropy",       "label": "Entropy  Ensemble",                "default": False},
    {"key": "gamma",         "label": "Precision γ  Ensemble",            "default": False},
]


# ===========================================================================
# Section 3 — Data structures
# ===========================================================================

@dataclass
class ExperimentData:
    """All timeseries fields for one (db_path, timestamp) combination."""
    timestamp:         str
    game_transitions:  list
    nash_strategy:     dict
    agent_kwargs:      list
    commit_sha:        str
    description:       str
    num_players:       int
    num_actions:       int
    all_vfe:           list = field(default_factory=list)
    all_q_u:           list = field(default_factory=list)
    all_efe:           list = field(default_factory=list)
    all_B:             list = field(default_factory=list)
    all_q_s:           list = field(default_factory=list)
    all_delta_F:       list = field(default_factory=list)
    all_entropy:       list = field(default_factory=list)
    all_gamma:         list = field(default_factory=list)
    all_candidates:    list = field(default_factory=list)
    all_model_weights: list = field(default_factory=list)
    experiments:       Any  = None   # raw DataFrame, for heatmaps/anim/ts


@dataclass
class PlotConfig:
    """Declarative description of one subplot panel."""
    plot_fn:          Callable
    args:             tuple = field(default_factory=tuple)
    kwargs:           dict  = field(default_factory=dict)
    label:            str   = ""
    skip_transitions: bool  = False


# ===========================================================================
# Section 4 — Data loading
# ===========================================================================

def _load_metadata(db_path: str, timestamp_filter: str) -> Any:
    metadata = utils.database.retrieve_timeseries_matching(
        db_path=db_path,
        sql_query=(
            "SELECT * FROM metadata "
            f'WHERE timestamp LIKE "%{timestamp_filter}%"'
        ),
    )
    if len(metadata) == 0:
        raise ValueError(
            f"No metadata for timestamp '{timestamp_filter}' in {db_path}")
    return metadata


def _resolve_metadata_row(db_path: str, metadata: Any,
                           timestamp_filter: str) -> Any:
    """Exact match first, then latest match, then first row."""
    ts    = str(timestamp_filter)
    exact = metadata[metadata["timestamp"].astype(str) == ts]
    if len(exact) >= 1:
        if len(exact) > 1:
            logging.warning(
                f"Multiple exact matches for '{ts}' in {db_path}; "
                "using the first.")
        return exact.iloc[0]

    if len(metadata) > 1:
        idx    = metadata["timestamp"].astype(str).sort_values().index[-1]
        chosen = metadata.loc[idx]
        logging.warning(
            f"Filter '{timestamp_filter}' matched {len(metadata)} rows in "
            f"{db_path}; using latest '{chosen['timestamp']}'.")
        return chosen

    return metadata.iloc[0]


def _load_timeseries_rows(db_path: str, timestamp: str,
                           commit_sha: str) -> Any:
    return utils.database.retrieve_timeseries_matching(
        db_path=db_path,
        sql_query=(
            "SELECT * FROM timeseries "
            f'WHERE timestamp = "{timestamp}" '
            f'AND commit_sha = "{commit_sha}"'
        ),
    )


def _unpack_seed(loaded_vars: dict) -> dict:
    return {
        "VFE":             loaded_vars["VFE"],
        "q_u":             loaded_vars["q_u"],
        "EFE":             loaded_vars["EFE"],
        "B":               loaded_vars["B"],
        "q_s":             loaded_vars["q_s"],
        "delta_F":         loaded_vars["delta_F"],
        "entropy":         loaded_vars["entropy"],
        "gamma":           loaded_vars["gamma"],
        "B_candidates":    loaded_vars["B_candidates"],
        "B_model_weights": loaded_vars["B_model_weights"],
    }


def load_experiment(db_path: str, timestamp: str) -> ExperimentData | None:
    """
    Load one game-sequence from *db_path* matching *timestamp*.
    Returns None on failure so callers can skip gracefully.
    """
    try:
        metadata = _load_metadata(db_path, timestamp)
    except ValueError as exc:
        logging.warning(str(exc))
        return None

    row              = _resolve_metadata_row(db_path, metadata, timestamp)
    chosen_timestamp = str(row["timestamp"])
    commit_sha       = str(row["commit_sha"])

    experiments = _load_timeseries_rows(db_path, chosen_timestamp, commit_sha)
    if len(experiments) == 0:
        logging.warning(
            f"No timeseries rows for '{chosen_timestamp}' in {db_path}")
        return None

    data = ExperimentData(
        timestamp        = chosen_timestamp,
        game_transitions = pickle.loads(row["game_transitions"]),
        nash_strategy    = pickle.loads(row["nash_strategy"]),
        agent_kwargs     = pickle.loads(row["agent_kwargs"]),
        commit_sha       = commit_sha,
        description      = str(row["description"]),
        num_players      = int(row["num_agents"]),
        num_actions      = int(row["num_actions"]),
        experiments      = experiments,
    )
    for i in range(len(experiments)):
        sv = utils.database.load_single_timeseries(experiments, i)
        u  = _unpack_seed(sv)
        data.all_vfe.append(u["VFE"])
        data.all_q_u.append(u["q_u"])
        data.all_efe.append(u["EFE"])
        data.all_B.append(u["B"])
        data.all_q_s.append(u["q_s"])
        data.all_delta_F.append(u["delta_F"])
        data.all_entropy.append(u["entropy"])
        data.all_gamma.append(u["gamma"])
        data.all_candidates.append(u["B_candidates"])
        data.all_model_weights.append(u["B_model_weights"])

    logging.info(
        f"Loaded {len(experiments)} seeds  ·  {db_path} [{chosen_timestamp}]")
    return data


def _load_experiment_star(args: tuple) -> ExperimentData | None:
    """Picklable wrapper for multiprocessing.Pool."""
    db_path, timestamp = args
    return load_experiment(db_path, timestamp)


def load_all_timestamps_parallel(
        db_path: str,
        timestamp_filter: str,
        n_workers: int = 4) -> list[ExperimentData]:
    """
    Load every distinct timestamp in *db_path* that matches *timestamp_filter*.
    Returns one ExperimentData per game-transition sequence found.
    """
    metadata = utils.database.retrieve_timeseries_matching(
        db_path=db_path,
        sql_query=(
            "SELECT * FROM metadata "
            f'WHERE timestamp LIKE "%{timestamp_filter}%"'
        ),
    )
    timestamps = list(metadata["timestamp"].values)
    logging.info(f"Found {len(timestamps)} timestamp(s) in {db_path}")
    if not timestamps:
        return []

    tasks = [(db_path, ts) for ts in timestamps]
    with Pool(processes=min(n_workers, len(tasks))) as pool:
        results = pool.map(_load_experiment_star, tasks)
    return [r for r in results if r is not None]


# ===========================================================================
# Section 5 — Figure helpers
# ===========================================================================

def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"✅  Saved  {path}")


# ===========================================================================
# Section 6 — B-matrix heatmap helper
# ===========================================================================

def plot_B_heatmaps(data: ExperimentData, consistency: dict,
                    output_dir: str, n_seeds: int = 3) -> None:
    """4-row × 4-col heatmap grid per seed (agent × time, factor × action)."""
    t1, t2 = 200, 550
    cmap = {
        0: mcolors.LinearSegmentedColormap.from_list(
            "blue", ["#ffffff", "#03dffc"]),
        1: mcolors.LinearSegmentedColormap.from_list(
            "pink", ["#ffffff", "#f70ce4"]),
    }
    for i in range(min(n_seeds, len(data.experiments))):
        sv    = utils.database.load_single_timeseries(data.experiments, i)
        seed  = sv["seed"]
        B_all = np.array(sv["B"])
        _, num_agents, num_factors, num_actions, _, _ = B_all.shape[1:]

        fig, axes = plt.subplots(4, 4, figsize=(16, 12),
                                 squeeze=False, dpi=_DPI_RENDER)
        im = None
        for row_idx, (agent, time) in enumerate(
                [(0, t1), (0, t2), (1, t1), (1, t2)]):
            for factor in range(num_factors):
                for action in range(num_actions):
                    col_idx = factor * num_actions + action
                    ax      = axes[row_idx, col_idx]
                    B_plot  = B_all[time, agent, factor, action, :, :]
                    im = ax.imshow(B_plot, vmin=0, vmax=1, cmap=cmap[action])
                    for r in range(2):
                        for c in range(2):
                            ax.text(c, r, f"{B_plot[r, c]:.2f}",
                                    ha="center", va="center", color="black")
                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])
                    ax.set_title(f"factor={factor}, action={action}")

            is_ok = consistency.get(seed, {}).get(agent, False)
            axes[row_idx, 0].set_ylabel(
                f"Agent {agent}, t={time}\n"
                f"({'Consistent' if is_ok else 'Inconsistent'})",
                rotation=0, labelpad=50, va="center",
                color="green" if is_ok else "red", fontsize=12,
            )

        if im is not None:
            fig.colorbar(im, ax=axes, fraction=0.03,
                         pad=0.04).set_label("Transition Probability")
        plt.suptitle(f"B-matrix Heatmaps  seed={seed}", fontsize=16)
        save_figure(fig, os.path.join(output_dir, f"B-seed{seed}.png"))


# ===========================================================================
# Section 6b — Gamma (precision) ensemble plot
# ===========================================================================

def plot_gamma_ensemble(all_gamma: list, game_transitions: list,
                        ifLegend: bool = True, ax=None):
    """
    Ensemble precision plot.

    all_gamma : list of per-seed gamma histories.
                Each entry has shape (T, num_agents) — a 2-D array where
                column i is agent i's γ at each timestep.
    Plots one faint line per seed per agent, plus a bold per-agent mean.
    """
    if ax is None:
        _, ax = plt.subplots()

    try:
        gamma_arr = np.array(all_gamma)   # (num_seeds, T, num_agents)
    except ValueError:
        # Ragged shapes — stack manually
        gamma_arr = np.stack(
            [np.array(g) for g in all_gamma], axis=0)

    num_seeds, T, num_agents = gamma_arr.shape
    agent_colors = [PRECISION_COLOR, "#888888",
                    "#4477aa", "#cc6677"][:num_agents]

    for agent_idx in range(num_agents):
        color = agent_colors[agent_idx]
        label_stem = f"Agent {chr(105 + agent_idx)}"
        for s in range(num_seeds):
            ax.plot(gamma_arr[s, :, agent_idx],
                    color=color, alpha=0.15,
                    linewidth=utils.plotting.LINEWIDTH)
        mean_gamma = gamma_arr[:, :, agent_idx].mean(axis=0)
        ax.plot(mean_gamma,
                color=color, alpha=1.0,
                linewidth=utils.plotting.LINEWIDTH * 2,
                label=f"{label_stem} mean")

    ax.set_xlabel("Time step (t)", fontsize=utils.plotting.label_font_size)
    ax.set_ylabel("Precision γ",   fontsize=utils.plotting.label_font_size)
    if ifLegend:
        ax.legend(loc="upper right",
                  fontsize=utils.plotting.label_font_size)


# ===========================================================================
# Section 7 — Panel-spec dispatch
# ===========================================================================

def panel_spec(panel_key: str,
               data: ExperimentData) -> tuple[Callable, tuple] | None:
    """
    Return (plot_fn, positional_args) for one grid cell.
    Returns None for panels that do not use the standard grid figure.
    """
    gt = data.game_transitions
    ns = data.nash_strategy

    specs: dict[str, tuple] = {
        "vfe": (utils.plotting.plot_vfe_ensemble,
                (data.all_vfe, gt, True)),
        "efe": (utils.plotting.plot_expected_efe_ensemble,
                (data.all_q_u, data.all_efe, True)),
        "policy": (utils.plotting.plot_policies_ensemble,
                   (data.all_q_u, gt, ns, True, False)),
        "policy_single": (utils.plotting.plot_policies_ensemble,
                          (data.all_q_u, gt, ns, True, True)),
        "state": (utils.plotting.plot_B_state_ensemble,
                  (data.all_B, data.all_q_s, True, False, True)),
        "sep_coop": (utils.plotting.plot_B_separation_degree_ensemble,
                     (data.all_B, data.all_q_u, gt,
                      0, "Cooperators", True, False, True)),
        "sep_def": (utils.plotting.plot_B_separation_degree_ensemble,
                    (data.all_B, data.all_q_u, gt,
                     0, "Defectors", True, False, True)),
        "delta_f": (utils.plotting.plot_delta_F_ensemble,
                    (data.all_delta_F, data.all_candidates)),
        "weights_i": (utils.plotting.plot_B_model_weights_ensemble,
                      (data.all_model_weights, data.all_candidates, 0, True)),
        "weights_j": (utils.plotting.plot_B_model_weights_ensemble,
                      (data.all_model_weights, data.all_candidates, 1, True)),
        "entropy": (utils.plotting.plot_entropy_ensemble,
                    (data.all_entropy, True)),
        "gamma":   (plot_gamma_ensemble,
                    (data.all_gamma, gt, True)),
    }
    return specs.get(panel_key)


# ===========================================================================
# Section 8 — DB file discovery
# ===========================================================================

def discover_db_files(root: str) -> dict[str, list[str]]:
    """
    Walk *root* recursively.
    Return { group_name -> [abs_path, ...] }.
    Files in root itself use group "."; files in a sub-folder use that name.
    """
    result: dict[str, list[str]] = {}
    root_path = Path(root)
    if not root_path.exists():
        return result
    for item in sorted(root_path.rglob("*.db")):
        rel   = item.relative_to(root_path)
        group = str(rel.parts[0]) if len(rel.parts) > 1 else "."
        result.setdefault(group, []).append(str(item.resolve()))
    return result


# ===========================================================================
# Section 9 — Styled widget helpers
# ===========================================================================

def _entry(parent, textvariable, width: int = 40) -> tk.Entry:
    return tk.Entry(
        parent, textvariable=textvariable, width=width,
        bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
        relief="flat", font=FONT_MONO,
        highlightthickness=1, highlightcolor=ACCENT,
        highlightbackground=BORDER,
    )


def _label(parent, text="", font=FONT_LABEL, fg=TEXT, **kw) -> tk.Label:
    kw.setdefault("bg", BG)
    return tk.Label(parent, text=text, fg=fg, font=font, **kw)


def _button(parent, text, command, accent: bool = True, **kw) -> tk.Button:
    bg = ACCENT if accent else SURFACE2
    return tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=TEXT, activebackground=ACCENT2, activeforeground=BG,
        relief="flat", font=FONT_BTN, cursor="hand2",
        padx=18, pady=7, **kw,
    )


def _separator(parent) -> tk.Frame:
    return tk.Frame(parent, bg=BORDER, height=1)


def _card(parent, **kw) -> tk.Frame:
    return tk.Frame(parent, bg=SURFACE, relief="flat",
                    highlightthickness=1, highlightbackground=BORDER, **kw)


# ===========================================================================
# Section 10 — ScrollableCheckList
# ===========================================================================

class ScrollableCheckList(tk.Frame):
    """Vertically scrollable column of labelled checkboxes."""

    def __init__(self, parent, items: list[dict], **kw):
        super().__init__(parent, bg=SURFACE, **kw)
        self._vars: dict[str, tk.BooleanVar] = {}

        canvas    = tk.Canvas(self, bg=SURFACE, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical",
                                   command=canvas.yview)
        self._inner = tk.Frame(canvas, bg=SURFACE)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        win = canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._inner.bind(
            "<Configure>",
            lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfig(win, width=e.width))
        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        for item in items:
            self._add_row(item)

    def _add_row(self, item: dict):
        var = tk.BooleanVar(value=item.get("default", True))
        self._vars[item["key"]] = var

        row = tk.Frame(self._inner, bg=SURFACE)
        row.pack(fill="x", padx=12, pady=2)
        tk.Checkbutton(
            row, variable=var, text=item["label"],
            bg=SURFACE, fg=TEXT, selectcolor=SURFACE2,
            activebackground=SURFACE, activeforeground=ACCENT2,
            font=FONT_LABEL, anchor="w", cursor="hand2",
        ).pack(side="left", fill="x", expand=True)
        row.bind("<Enter>", lambda _, r=row: r.config(bg=SURFACE2))
        row.bind("<Leave>", lambda _, r=row: r.config(bg=SURFACE))

    def get_selected(self) -> list[str]:
        return [k for k, v in self._vars.items() if v.get()]

    def select_all(self):
        for v in self._vars.values():
            v.set(True)

    def deselect_all(self):
        for v in self._vars.values():
            v.set(False)


# ===========================================================================
# Section 11 — DbFileTree
# ===========================================================================

class DbFileTree(tk.Frame):
    """
    Treeview of .db files grouped by sub-folder; supports multi-select.

    Selection order
    ---------------
    Each click appends to an ordered list so the user can control which
    file becomes row 1, row 2, … in the output figure.  A circled number
    badge (① ② ③ …) is shown next to the file name inside the tree.
    The badge is UI-only and never written to output figures.

    select_all() uses default tree order (no badges shown).
    deselect_all() / refresh() reset the order completely.
    """

    # Circled digit glyphs for positions 1-20; fall back to plain "(N)" beyond
    _BADGES = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","⑩",
               "⑪","⑫","⑬","⑭","⑮","⑯","⑰","⑱","⑲","⑳"]

    def __init__(self, parent, groups: dict[str, list[str]], **kw):
        super().__init__(parent, bg=SURFACE, **kw)
        self._path_map:  dict[str, str]  = {}   # iid → absolute path
        self._stem_map:  dict[str, str]  = {}   # iid → bare stem text
        self._order:     list[str]       = []   # iids in click order

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "DB.Treeview",
            background=SURFACE, foreground=TEXT,
            fieldbackground=SURFACE, rowheight=26,
            font=FONT_SMALL, borderwidth=0)
        style.configure(
            "DB.Treeview.Heading",
            background=SURFACE2, foreground=ACCENT,
            font=("Georgia", 10, "bold"), relief="flat")
        style.map("DB.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", BG)])

        self._tree = ttk.Treeview(self, style="DB.Treeview",
                                   selectmode="extended",
                                   show="tree headings")
        self._tree.heading("#0", text="  Available databases", anchor="w")
        self._tree.column("#0", stretch=True)

        sb = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Track every click to maintain selection order
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        self._populate(groups)

    # ------------------------------------------------------------------

    def _populate(self, groups: dict[str, list[str]]):
        for group, paths in sorted(groups.items()):
            label  = f"📁  {group}" if group != "." else "📁  root"
            parent = self._tree.insert("", "end", text=label,
                                        open=True, tags=("group",))
            self._tree.tag_configure("group", foreground=TEXT_DIM)
            for path in sorted(paths):
                stem = Path(path).stem
                iid  = self._tree.insert(parent, "end",
                                          text=f"   🗄  {stem}",
                                          tags=("file",))
                self._tree.tag_configure("file", foreground=TEXT)
                self._path_map[iid] = path
                self._stem_map[iid] = stem

    def _on_select(self, _event):
        """
        Called on every <<TreeviewSelect>> event.
        Reconcile self._order with the current treeview selection:
          - remove iids that were deselected
          - append newly selected file iids (preserving first-click order)
        Then redraw the badges.
        """
        current = set(
            iid for iid in self._tree.selection()
            if iid in self._path_map)

        # Remove deselected items, preserving order of remaining ones
        self._order = [iid for iid in self._order if iid in current]

        # Append newly selected items in tree (top-to-bottom) order
        existing = set(self._order)
        for iid in self._path_map:          # iterates in insertion order
            if iid in current and iid not in existing:
                self._order.append(iid)
                existing.add(iid)

        self._redraw_badges()

    def _redraw_badges(self):
        """Update the label text of every file node with its order badge."""
        # Build a position lookup for selected iids
        pos = {iid: i for i, iid in enumerate(self._order)}
        for iid, stem in self._stem_map.items():
            if iid in pos:
                n     = pos[iid] + 1          # 1-based
                badge = (self._BADGES[n - 1]
                         if n <= len(self._BADGES)
                         else f"({n})")
                self._tree.item(iid, text=f"   🗄  {stem}  {badge}")
            else:
                self._tree.item(iid, text=f"   🗄  {stem}")

    # ------------------------------------------------------------------

    def get_selected_paths_ordered(self) -> list[str]:
        """Return paths in the user's click order."""
        return [self._path_map[iid] for iid in self._order]

    def get_all_paths_default(self) -> list[str]:
        """Return all file paths in default tree (alphabetical) order."""
        return [self._path_map[iid] for iid in self._path_map]

    def select_all(self):
        """Select all files in default order; no numbered badges."""
        self._order.clear()
        for iid in self._path_map:
            self._tree.selection_add(iid)
        # _on_select fires automatically; clear order again so no badges show
        self._order.clear()
        self._redraw_badges()

    def deselect_all(self):
        self._order.clear()
        self._tree.selection_set([])
        self._redraw_badges()

    def refresh(self, groups: dict[str, list[str]]):
        self._order.clear()
        self._path_map.clear()
        self._stem_map.clear()
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._populate(groups)


# ===========================================================================
# Section 12 — ProgressPanel
# ===========================================================================

class ProgressPanel(tk.Frame):
    """Progress bar + scrollable log pane."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._total = 1

        bar_row = tk.Frame(self, bg=BG)
        bar_row.pack(fill="x", padx=20, pady=(14, 4))
        self._progress_label = _label(
            bar_row, "Idle", font=FONT_SMALL, fg=TEXT_DIM)
        self._progress_label.pack(side="left")
        self._pct_label = _label(
            bar_row, "0%", font=FONT_SMALL, fg=ACCENT)
        self._pct_label.pack(side="right")

        style = ttk.Style()
        style.configure(
            "Accent.Horizontal.TProgressbar",
            troughcolor=PROGRESS_BG, background=ACCENT,
            bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT)
        self._bar = ttk.Progressbar(
            self, orient="horizontal", mode="determinate",
            style="Accent.Horizontal.TProgressbar")
        self._bar.pack(fill="x", padx=20, pady=(0, 10))

        log_card = _card(self)
        log_card.pack(fill="both", expand=True, padx=20, pady=(0, 14))
        self._log = tk.Text(
            log_card, bg=SURFACE, fg=TEXT_DIM, font=FONT_MONO,
            relief="flat", state="disabled", wrap="word",
            padx=10, pady=8)
        sb = ttk.Scrollbar(log_card, command=self._log.yview)
        self._log.configure(yscrollcommand=sb.set)
        self._log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self._log.tag_configure("ok",   foreground=SUCCESS)
        self._log.tag_configure("err",  foreground=DANGER)
        self._log.tag_configure("head", foreground=ACCENT,
                                font=("Georgia", 10, "bold"))
        self._log.tag_configure("dim",  foreground=TEXT_DIM)

    def reset(self, total: int):
        self._total = max(total, 1)
        self._bar["maximum"] = self._total
        self._bar["value"]   = 0
        self._pct_label.config(text="0%")
        self._progress_label.config(text="Starting…", fg=TEXT_DIM)
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    def step(self, message: str = "", tag: str = "dim"):
        self._bar["value"] += 1
        pct = int(100 * self._bar["value"] / self._total)
        self._pct_label.config(text=f"{pct}%")
        self._progress_label.config(
            text=message[:60] or
                 f"Step {int(self._bar['value'])}/{self._total}",
            fg=SUCCESS if pct == 100 else TEXT_DIM,
        )
        if message:
            self._append(message + "\n", tag)

    def log(self, message: str, tag: str = "dim"):
        self._append(message + "\n", tag)

    def _append(self, text: str, tag: str):
        self._log.config(state="normal")
        self._log.insert("end", text, tag)
        self._log.see("end")
        self._log.config(state="disabled")

    def finish(self, success: bool = True):
        if success:
            self._progress_label.config(text="✓  Done", fg=SUCCESS)
            self._pct_label.config(text="100%", fg=SUCCESS)
            self._bar["value"] = self._total
            self.log("\n✓  All plots saved successfully.", "ok")
        else:
            self._progress_label.config(text="✗  Error", fg=DANGER)
            self.log("\n✗  Pipeline encountered an error.", "err")


# ===========================================================================
# Section 13 — PlottingGUI
# ===========================================================================

class PlottingGUI(tk.Tk):

    def __init__(self, bma_root: str = "BMA-Study"):
        super().__init__()
        self._bma_root = bma_root
        self._running  = False

        self.title("MA-AIF  ·  Plotting Interface")
        self.geometry("1080x780")
        self.minsize(860, 620)
        self.configure(bg=BG)
        self.resizable(True, True)

        self._build_ui()
        self._refresh_files()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── header ────────────────────────────────────────────────────
        header = tk.Frame(self, bg=SURFACE, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)

        _label(header, "MA-AIF", font=FONT_TITLE, fg=ACCENT,
               bg=SURFACE).pack(side="left", padx=22, pady=12)
        _label(header,
               "Factorised Active Inference  ·  Visualisation Pipeline",
               font=FONT_SMALL, fg=TEXT_DIM,
               bg=SURFACE).pack(side="left", pady=18)

        right_hdr = tk.Frame(header, bg=SURFACE)
        right_hdr.pack(side="right", padx=16, pady=10)
        _button(right_hdr, "⟳  Refresh", self._refresh_files,
                accent=False).pack(side="right", padx=4)

        # ── bottom bar — packed before body so expand=True never hides it ──
        bottom = tk.Frame(self, bg=SURFACE, height=58)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        self._status_var = tk.StringVar(
            value="Ready  ·  Select databases and panels to begin")
        _label(bottom, "", textvariable=self._status_var,
               font=FONT_SMALL, fg=TEXT_DIM, bg=SURFACE).pack(
               side="left", padx=18, pady=18)

        self._run_btn = _button(
            bottom, "  Generate Plots  →", self._on_run, accent=True)
        self._run_btn.pack(side="right", padx=18, pady=10)

        # ── three-column body ─────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3, minsize=260)
        body.columnconfigure(1, weight=2, minsize=240)
        body.columnconfigure(2, weight=4, minsize=320)
        body.rowconfigure(0, weight=1)

        # ── column 0: file browser ────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(16, 6), pady=16)

        _label(left, "DATABASES",
               font=("Georgia", 9, "bold"), fg=ACCENT).pack(
               anchor="w", padx=2, pady=(0, 6))

        path_row = tk.Frame(left, bg=BG)
        path_row.pack(fill="x", pady=(0, 8))
        self._root_var = tk.StringVar(value=self._bma_root)
        _entry(path_row, self._root_var, width=22).pack(
            side="left", fill="x", expand=True)
        _button(path_row, "…", self._browse_root,
                accent=False).pack(side="left", padx=(4, 0))

        tree_card = _card(left)
        tree_card.pack(fill="both", expand=True)
        self._file_tree = DbFileTree(tree_card, groups={})
        self._file_tree.pack(fill="both", expand=True, padx=1, pady=1)

        fb = tk.Frame(left, bg=BG)
        fb.pack(fill="x", pady=(6, 0))
        _button(fb, "All",  self._file_tree.select_all,
                accent=False).pack(side="left", padx=(0, 4))
        _button(fb, "None", self._file_tree.deselect_all,
                accent=False).pack(side="left")

        # ── column 1: panel checkboxes ────────────────────────────────
        mid = tk.Frame(body, bg=BG)
        mid.grid(row=0, column=1, sticky="nsew", padx=6, pady=16)

        _label(mid, "PLOT PANELS",
               font=("Georgia", 9, "bold"), fg=ACCENT).pack(
               anchor="w", padx=2, pady=(0, 6))

        panel_card = _card(mid)
        panel_card.pack(fill="both", expand=True)
        self._checklist = ScrollableCheckList(panel_card, DEFAULT_PANELS)
        self._checklist.pack(fill="both", expand=True, padx=1, pady=1)

        pb = tk.Frame(mid, bg=BG)
        pb.pack(fill="x", pady=(6, 0))
        _button(pb, "All",  self._checklist.select_all,
                accent=False).pack(side="left", padx=(0, 4))
        _button(pb, "None", self._checklist.deselect_all,
                accent=False).pack(side="left")

        # ── column 2: settings + progress ────────────────────────────
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 16), pady=16)

        _label(right, "SETTINGS",
               font=("Georgia", 9, "bold"), fg=ACCENT).pack(
               anchor="w", padx=2, pady=(0, 6))
        settings_card = _card(right)
        settings_card.pack(fill="x")
        self._build_settings(settings_card)

        _separator(right).pack(fill="x", pady=12)

        _label(right, "PROGRESS",
               font=("Georgia", 9, "bold"), fg=ACCENT).pack(
               anchor="w", padx=2, pady=(0, 6))
        prog_card = _card(right)
        prog_card.pack(fill="both", expand=True)
        self._progress = ProgressPanel(prog_card)
        self._progress.pack(fill="both", expand=True)

    def _build_settings(self, parent: tk.Frame):
        rows = [
            ("Timestamp filter", "timestamp",   "2026"),
            ("Output directory", "figures_dir", "figures/"),
        ]
        self._settings: dict[str, tk.StringVar] = {}
        for label, key, default in rows:
            row = tk.Frame(parent, bg=SURFACE)
            row.pack(fill="x", padx=14, pady=6)
            _label(row, label, font=FONT_SMALL, fg=TEXT_DIM, bg=SURFACE,
                   width=18, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            self._settings[key] = var
            _entry(row, var, width=22).pack(side="left", fill="x", expand=True)
            if key == "figures_dir":
                _button(
                    row, "…",
                    lambda: self._settings["figures_dir"].set(
                        filedialog.askdirectory()
                        or self._settings["figures_dir"].get()),
                    accent=False,
                ).pack(side="left", padx=(4, 0))

        # ── x-range: start / end time ─────────────────────────────────
        _separator(parent).pack(fill="x", padx=14, pady=(8, 0))

        xrange_label = tk.Frame(parent, bg=SURFACE)
        xrange_label.pack(fill="x", padx=14, pady=(6, 2))
        _label(xrange_label, "X-axis range  (blank = full)",
               font=FONT_SMALL, fg=TEXT_DIM, bg=SURFACE).pack(side="left")

        xrange_row = tk.Frame(parent, bg=SURFACE)
        xrange_row.pack(fill="x", padx=14, pady=(0, 8))

        _label(xrange_row, "Start t", font=FONT_SMALL,
               fg=TEXT_DIM, bg=SURFACE, width=8, anchor="w").pack(side="left")
        self._settings["t_start"] = tk.StringVar(value="")
        _entry(xrange_row, self._settings["t_start"],
               width=7).pack(side="left", padx=(0, 12))

        _label(xrange_row, "End t", font=FONT_SMALL,
               fg=TEXT_DIM, bg=SURFACE, width=6, anchor="w").pack(side="left")
        self._settings["t_end"] = tk.StringVar(value="")
        _entry(xrange_row, self._settings["t_end"],
               width=7).pack(side="left")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _browse_root(self):
        chosen = filedialog.askdirectory(initialdir=self._bma_root)
        if chosen:
            self._bma_root = chosen
            self._root_var.set(chosen)
            self._refresh_files()

    def _refresh_files(self):
        root   = self._root_var.get() or self._bma_root
        groups = discover_db_files(root)
        self._file_tree.refresh(groups)
        n = sum(len(v) for v in groups.values())
        self._status_var.set(f"Found {n} database file(s) in  {root}")

    def _on_run(self):
        if self._running:
            return

        # get_selected_paths_ordered() returns [] when select_all() was used
        # (it clears _order so no badges are shown).
        # In that case fall back to all files in default alphabetical order.
        ordered = self._file_tree.get_selected_paths_ordered()
        selected_files = ordered if ordered \
            else self._file_tree.get_all_paths_default()

        selected_panels = self._checklist.get_selected()

        if not selected_files:
            messagebox.showwarning(
                "No files selected",
                "Please select at least one .db file from the tree.")
            return
        if not selected_panels:
            messagebox.showwarning(
                "No panels selected",
                "Please select at least one plot panel.")
            return

        # Parse optional x-range — validate that values are non-negative
        # integers and that start < end when both are given.
        def _parse_t(key: str) -> int | None:
            raw = self._settings[key].get().strip()
            if not raw:
                return None
            try:
                v = int(raw)
                if v < 0:
                    raise ValueError
                return v
            except ValueError:
                messagebox.showerror(
                    "Invalid time range",
                    f"'{raw}' is not a valid non-negative integer for {key}.")
                return -1   # sentinel: abort

        t_start = _parse_t("t_start")
        t_end   = _parse_t("t_end")
        if t_start == -1 or t_end == -1:
            return
        if t_start is not None and t_end is not None and t_start >= t_end:
            messagebox.showerror(
                "Invalid time range",
                f"Start t ({t_start}) must be less than End t ({t_end}).")
            return

        self._running = True
        self._run_btn.config(state="disabled", bg=SURFACE2)
        self._status_var.set("Running pipeline…")

        threading.Thread(
            target=self._run_pipeline,
            args=(selected_files, selected_panels,
                  self._settings["timestamp"].get(),
                  self._settings["figures_dir"].get(),
                  t_start, t_end),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Pipeline  (background thread)
    # ------------------------------------------------------------------

    def _run_pipeline(self, db_paths: list[str], panels: list[str],
                      timestamp: str, figures_dir: str,
                      t_start: int | None = None,
                      t_end:   int | None = None):
        """
        Output layout
        -------------
        figures/<run_timestamp>/
            <panel_key>.png   one figure per selected panel
                rows  = selected .db files
                cols  = game-transition sequences in that file

        t_start / t_end : optional x-axis display window applied after
                          plotting. None means use the full range.

        Progress: n_files load steps  +  n_panels render steps.
        """
        run_ts  = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(figures_dir, run_ts)
        os.makedirs(out_dir, exist_ok=True)

        total = len(db_paths) + len(panels)
        self.after(0, self._progress.reset, total)
        self.after(0, self._progress.log, f"Output:  {out_dir}", "head")

        try:
            import matplotlib
            matplotlib.use("Agg")   # non-interactive; safe in threads

            # ── Phase 1: load ─────────────────────────────────────────
            # loaded: list[ (stem, [ExperimentData, ...]) ]
            #   inner list = one ExperimentData per game sequence / timestamp
            loaded: list[tuple[str, list[ExperimentData]]] = []

            for db_path in db_paths:
                stem = Path(db_path).stem
                self.after(0, self._progress.step,
                            f"Loading  {stem}…", "dim")

                data_list = load_all_timestamps_parallel(
                    db_path, timestamp, n_workers=4)

                if not data_list:
                    self.after(0, self._progress.log,
                               f"⚠  No data  {stem} @ '{timestamp}'", "err")
                    continue

                n_seq   = len(data_list)
                n_seeds = len(data_list[0].experiments) if data_list else 0
                self.after(0, self._progress.log,
                           f"✓  {stem}  ·  {n_seq} sequence(s)"
                           f"  ·  {n_seeds} seed(s)", "ok")
                loaded.append((stem, data_list))

            if not loaded:
                raise RuntimeError(
                    "No data could be loaded from any selected file.")

            # ── Phase 2: render ───────────────────────────────────────
            for panel_key in panels:
                panel_label = next(
                    (p["label"] for p in DEFAULT_PANELS
                     if p["key"] == panel_key),
                    panel_key,
                )
                self.after(0, self._progress.step,
                            f"Rendering  {panel_label}…", "dim")
                try:
                    self._render_stacked(
                        loaded, panel_key, panel_label, out_dir,
                        t_start, t_end)
                    self.after(0, self._progress.log,
                               f"  ✓  {panel_label}", "ok")
                except Exception as exc:
                    self.after(0, self._progress.log,
                               f"  ✗  {panel_label}: {exc}", "err")

            self.after(0, self._progress.finish, True)
            self.after(0, self._status_var.set,
                       f"Done  ·  {len(panels)} plot(s)  →  {out_dir}")

        except Exception as exc:
            self.after(0, self._progress.finish, False)
            self.after(0, self._progress.log, str(exc), "err")
            self.after(0, self._status_var.set, f"Error: {exc}")

        finally:
            self._running = False
            self.after(0, lambda: self._run_btn.config(
                state="normal", bg=ACCENT))

    # ------------------------------------------------------------------
    # Stacked figure renderer
    # ------------------------------------------------------------------

    def _render_stacked(self,
                        loaded: list[tuple[str, list[ExperimentData]]],
                        panel_key: str,
                        panel_label: str,
                        out_dir: str,
                        t_start: int | None = None,
                        t_end:   int | None = None):
        """
        Build and save one output figure.

        Grid: rows = .db files x cols = game-transition sequences.

        t_start / t_end : optional absolute x-axis window [t_start, t_end]
                          applied after plotting. Full data is still plotted,
                          then view is clipped to avoid ragged slicing issues.
        """
        import matplotlib.pyplot as plt

        def _apply_time_window(ax: plt.Axes) -> None:
            if t_start is None and t_end is None:
                return

            x_lo, x_hi = ax.get_xlim()
            left = x_lo if t_start is None else max(float(t_start), x_lo)
            right = x_hi if t_end is None else min(float(t_end), x_hi)

            if right <= left:
                right = left + 1.0

            ax.set_xlim(left, right)

        show_game_transitions = (t_start is None and t_end is None)

        n_rows = len(loaded)
        n_cols = max(len(dl) for _, dl in loaded)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(_COL_W * n_cols, _ROW_H * n_rows),
            dpi=_DPI_RENDER,
            squeeze=False,
        )
        fig.subplots_adjust(hspace=0.55, wspace=0.25)

        for row_idx, (stem, data_list) in enumerate(loaded):
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]

                if col_idx >= len(data_list):
                    ax.set_visible(False)
                    continue

                data = data_list[col_idx]
                spec = panel_spec(panel_key, data)
                if spec is None:
                    ax.set_visible(False)
                    continue

                plot_fn, args = spec
                plot_fn(*args, ax=ax)

                _style_ax(ax)

                if show_game_transitions:
                    try:
                        utils.plotting.highlight_transitions(
                            data.game_transitions, ax)
                    except Exception:
                        pass

                _apply_time_window(ax)

                if row_idx == 0 and show_game_transitions:
                    ax.set_title(
                        "-".join(g[0].split("_")[-1]
                                 for g in data.game_transitions),
                        fontsize=10, pad=5)

                if col_idx == 0:
                    ax.set_ylabel(
                        f"{stem}\n{ax.get_ylabel()}",
                        fontsize=8, rotation=90, labelpad=8)
                else:
                    ax.set_ylabel(None)

                if row_idx < n_rows - 1:
                    ax.set_xlabel(None)

        range_str = ""
        if t_start is not None or t_end is not None:
            lo = t_start if t_start is not None else 0
            hi = t_end if t_end is not None else "end"
            range_str = f"  [t={lo}-{hi}]"

        fig.suptitle(f"{panel_label}{range_str}", fontsize=13, y=1.01)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        suffix = (f"_t{t_start}-{t_end}"
                  if (t_start is not None or t_end is not None) else "")
        save_figure(fig, os.path.join(out_dir, f"{panel_key}{suffix}.png"))


# ===========================================================================
# Section 14 — Entry point
# ===========================================================================

def main():
    bma_root = sys.argv[1] if len(sys.argv) > 1 else "BMA-Study"
    PlottingGUI(bma_root=bma_root).mainloop()


if __name__ == "__main__":
    main()
