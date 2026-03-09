"""
DMML Skill Tree — wide poster format, game-like dark aesthetic.
Run with:  ./env/bin/python skill_tree.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe
import numpy as np
import textwrap

# ── Palette ───────────────────────────────────────────────────────────────────
BG          = "#0d1117"
TEXT_BRIGHT = "#ffffff"
TEXT_DIM    = "#8b949e"

COL = {
    "root":         "#58a6ff",   # blue
    "tab":          "#3fb950",   # green
    "ts":           "#f78166",   # orange-red
    "unsup":        "#2dd4bf",   # teal  (was too close to tab green)
    "img":          "#bc8cff",   # purple
    "seq":          "#ffa657",   # amber
    "shap_special": "#e3b341",   # gold  – cross-cutting tool
}

BADGE_BG = {
    "sklearn":      "#1f6feb",
    "statsmodels":  "#2d6a9f",
    "xgboost":      "#b08000",
    "lightgbm":     "#2e7d32",
    "catboost":     "#b85c00",
    "pytorch":      "#b93221",
    "huggingface":  "#b07300",
    "shap":         "#276e27",
}

# ── Canvas & node dimensions (data units) ─────────────────────────────────────
CW, CH = 76, 52            # canvas width / height

HW, HH = 3.6, 1.25         # header node half-w / half-h
MW, MH = 3.0, 2.35         # method node half-w / half-h

BAR_H = 0.72               # top accent bar inside method card

HEADERS = {"root", "tab", "ts", "unsup", "img", "seq"}

# ── Node definitions ──────────────────────────────────────────────────────────
#  (id, display_label, use_when, [bullets], [libs], branch_key)
NODES = [
    ("root",  "What is your\ndata?", None, [], [], "root"),

    ("tab",   "Tabular /\nStructured",  None, [], [], "tab"),
    ("ts",    "Time Series",            None, [], [], "ts"),
    ("unsup", "Unlabelled Data",        None, [], [], "unsup"),
    ("img",   "Images",                 None, [], [], "img"),
    ("seq",   "Text / Sequences",       None, [], [], "seq"),

    # ── Tabular ───────────────────────────────────────────────────────────────
    ("linreg", "Linear Regression",
     "Predict a continuous value",
     ["Interpretable coefficients", "Assumes linearity", "Fast & scalable"],
     ["sklearn"], "tab"),

    ("logreg", "Logistic Regression",
     "Binary or multiclass classification",
     ["Probabilistic output", "Interpretable weights", "Strong baseline"],
     ["sklearn"], "tab"),

    ("knn", "k-NN",
     "Non-parametric, instance-based tasks",
     ["No training phase needed", "Sensitive to scale & dims", "Slow at inference"],
     ["sklearn"], "tab"),

    ("svm", "SVM",
     "High-dim or small-dataset classification",
     ["Max-margin boundary", "Kernel trick (RBF, poly)", "Poor scaling to big N"],
     ["sklearn"], "tab"),

    ("tree", "Decision Tree",
     "Interpretable rule-based decisions",
     ["Handles mixed types natively", "Prone to overfitting", "Fully visualisable"],
     ["sklearn"], "tab"),

    ("rf", "Random Forest",
     "Robust classification or regression",
     ["Bagging reduces variance", "Feature importance scores", "Parallelisable"],
     ["sklearn"], "tab"),

    ("boost", "Gradient Boosting",
     "Best accuracy on structured data",
     ["Boosting reduces bias", "SHAP-interpretable", "Tune n_estimators & lr"],
     ["xgboost", "lightgbm", "catboost"], "tab"),

    ("shap_node", "SHAP",
     "Explain any black-box model",
     ["Model-agnostic", "Local & global views", "Pairs best with boosting"],
     ["shap"], "shap_special"),

    ("mlp", "MLP / Neural Net",
     "General learner for tabular tasks",
     ["Flexible architecture", "Needs careful tuning", "Bridge to deep learning"],
     ["pytorch"], "tab"),

    # ── Time Series ───────────────────────────────────────────────────────────
    ("arima", "ARIMA",
     "Forecast univariate stationary series",
     ["AR + MA + differencing", "Needs stationarity check", "ACF/PACF for order"],
     ["statsmodels"], "ts"),

    ("lstm_ts", "LSTM  (for TS)",
     "Complex or multivariate series",
     ["Captures long-range deps", "Needs more data", "See Sequences branch"],
     ["pytorch"], "ts"),

    # ── Unsupervised ──────────────────────────────────────────────────────────
    ("kmeans", "K-Means Clustering",
     "Partition data into k groups",
     ["Fast and scalable", "Assumes spherical clusters", "Elbow / silhouette for k"],
     ["sklearn"], "unsup"),

    ("dbscan", "DBSCAN",
     "Arbitrary-shape clusters, noisy data",
     ["No k to set", "Detects outliers natively", "Tune eps and min_samples"],
     ["sklearn"], "unsup"),

    ("pca", "PCA",
     "Reduce dims, visualise high-dim data",
     ["Maximises explained variance", "Linear transform only", "Check scree plot"],
     ["sklearn"], "unsup"),

    # ── Images ────────────────────────────────────────────────────────────────
    ("cnn", "CNN",
     "Image classification or detection",
     ["Learns spatial hierarchies", "Data-hungry, GPU needed", "Strong baseline"],
     ["pytorch"], "img"),

    ("tl", "Transfer Learning",
     "Image task with limited data",
     ["Fine-tune pretrained backbone", "Fast convergence", "ResNet / EfficientNet"],
     ["pytorch"], "img"),

    # ── Sequences ─────────────────────────────────────────────────────────────
    ("rnn", "RNN / LSTM",
     "Ordered sequences with temporal deps",
     ["Hidden state carries memory", "LSTM fixes vanishing grad", "Good for short seqs"],
     ["pytorch"], "seq"),

    ("transformer", "Transformer",
     "NLP or long-range sequence tasks",
     ["Self-attention, parallel training", "State-of-the-art NLP", "Fine-tune via HuggingFace"],
     ["pytorch", "huggingface"], "seq"),
]

NODE_MAP = {n[0]: n for n in NODES}

# ── Positions (data units, origin bottom-left) ────────────────────────────────
POS = {
    "root":         (38,  45.5),

    "tab":          (11,  40),
    "ts":           (28,  40),
    "unsup":        (40,  40),
    "img":          (54,  40),
    "seq":          (67,  40),

    # Tabular row 1  (y=33)
    "linreg":       ( 4,  33),
    "logreg":       (11,  33),
    "knn":          (18,  33),

    # Tabular row 2  (y=26)
    "svm":          ( 7,  26),
    "tree":         (15,  26),

    # Tabular row 3  (y=19)
    "mlp":          ( 4,  19),
    "rf":           (11,  19),
    "boost":        (18,  19),

    # SHAP — bottom centre, below the whole tree
    "shap_node":    (38,   6),

    # Time series
    "arima":        (28,  33),
    "lstm_ts":      (28,  26),

    # Unsupervised
    "kmeans":       (36,  33),
    "dbscan":       (44,  33),
    "pca":          (40,  26),

    # Images
    "cnn":          (54,  33),
    "tl":           (54,  26),

    # Sequences
    "rnn":          (64,  33),
    "transformer":  (70,  26),
}

EDGES = [
    # Root → branches
    ("root", "tab"),
    ("root", "ts"),
    ("root", "unsup"),
    ("root", "img"),
    ("root", "seq"),

    # Tabular
    ("tab", "linreg"),
    ("tab", "logreg"),
    ("tab", "knn"),
    ("tab", "svm"),
    ("tab", "tree"),
    ("tab", "mlp"),
    ("tree", "rf"),
    ("tree", "boost"),

    # SHAP edges — converge from across branches to signal it's universal
    ("boost",       "shap_node"),
    ("rf",          "shap_node"),
    ("mlp",         "shap_node"),
    ("tl",          "shap_node"),
    ("transformer", "shap_node"),

    # Time Series
    ("ts", "arima"),
    ("arima", "lstm_ts"),

    # Unsupervised
    ("unsup", "kmeans"),
    ("unsup", "dbscan"),
    ("unsup", "pca"),

    # Images
    ("img", "cnn"),
    ("cnn", "tl"),

    # Sequences
    ("seq", "rnn"),
    ("rnn", "transformer"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def node_hh(nid):
    return HH if nid in HEADERS else MH

def branch_col(key):
    return COL.get(key, COL["tab"])

def wrap(text, width=28):
    return textwrap.wrap(text, width=width) if text else []

# ── Drawing primitives ────────────────────────────────────────────────────────

def bezier_edge(ax, src, dst):
    is_shap_edge = (dst == "shap_node")
    nd_dst = NODE_MAP[dst]
    color  = COL["shap_special"] if is_shap_edge else branch_col(nd_dst[5])

    xs, ys = POS[src]
    xd, yd = POS[dst]

    # attach at bottom of source, top of destination
    y0 = ys - node_hh(src)
    y1 = yd + node_hh(dst)

    dy      = abs(y0 - y1)
    tension = max(dy * 0.48, 1.8)

    verts = [(xs, y0), (xs, y0 - tension), (xd, y1 + tension), (xd, y1)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

    if is_shap_edge:
        # dashed gold lines sweeping in from multiple branches
        patch = mpatches.PathPatch(
            Path(verts, codes),
            fc="none", ec=color, lw=1.3, alpha=0.55, zorder=1,
            linestyle=(0, (5, 4)), capstyle="round",
        )
    else:
        patch = mpatches.PathPatch(
            Path(verts, codes),
            fc="none", ec=color, lw=1.4, alpha=0.50, zorder=1,
            capstyle="round",
        )
    ax.add_patch(patch)

    # small dot at destination entry (skip for SHAP — too many dots)
    if not is_shap_edge:
        ax.plot(xd, y1, "o", ms=3, color=color, alpha=0.7, zorder=2)


def header_node(ax, nid):
    x, y = POS[nid]
    nd   = NODE_MAP[nid]
    col  = branch_col(nd[5])

    # glow
    ax.add_patch(FancyBboxPatch(
        (x - HW, y - HH), 2*HW, 2*HH,
        boxstyle="round,pad=0.12", lw=7,
        ec=col, fc="none", alpha=0.14, zorder=2,
    ))
    # card
    ax.add_patch(FancyBboxPatch(
        (x - HW, y - HH), 2*HW, 2*HH,
        boxstyle="round,pad=0.12", lw=2,
        ec=col, fc="#161b22", zorder=3,
    ))
    ax.text(
        x, y, nd[1],
        ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=col, zorder=4,
        multialignment="center",
        path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")],
    )


def method_node(ax, nid):
    x, y = POS[nid]
    nd   = NODE_MAP[nid]
    _, label, use_when, bullets, libs, branch = nd
    col      = branch_col(branch)
    is_shap  = (nid == "shap_node")

    # SHAP: outer dashed ring to signal "cross-cutting"
    if is_shap:
        ax.add_patch(FancyBboxPatch(
            (x - MW - 0.30, y - MH - 0.30), 2*(MW+0.30), 2*(MH+0.30),
            boxstyle="round,pad=0.12", lw=1.5,
            ec=col, fc="none", alpha=0.60, zorder=2,
            linestyle=(0, (5, 3)),
        ))
        # brighter outer glow
        ax.add_patch(FancyBboxPatch(
            (x - MW - 0.15, y - MH - 0.15), 2*(MW+0.15), 2*(MH+0.15),
            boxstyle="round,pad=0.12", lw=10,
            ec=col, fc="none", alpha=0.16, zorder=2,
        ))

    # standard glow
    ax.add_patch(FancyBboxPatch(
        (x - MW, y - MH), 2*MW, 2*MH,
        boxstyle="round,pad=0.1", lw=6,
        ec=col, fc="none", alpha=0.10, zorder=2,
    ))
    # card background
    ax.add_patch(FancyBboxPatch(
        (x - MW, y - MH), 2*MW, 2*MH,
        boxstyle="round,pad=0.1", lw=1.5,
        ec=col, fc="#1a1608" if is_shap else "#161b22", zorder=3,
    ))

    # ── top accent bar ────────────────────────────────────────────────────
    bar_top    = y + MH - 0.08
    bar_bottom = bar_top - BAR_H
    ax.add_patch(FancyBboxPatch(
        (x - MW + 0.08, bar_bottom), 2*MW - 0.16, BAR_H,
        boxstyle="round,pad=0.06", lw=0,
        fc=col, alpha=0.88, zorder=4,
    ))

    # method name (centred in bar)
    bar_cy = (bar_top + bar_bottom) / 2
    ax.text(
        x, bar_cy, label,
        ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=TEXT_BRIGHT, zorder=5,
        multialignment="center",
        path_effects=[pe.withStroke(linewidth=1.5, foreground=col)],
    )

    # ── body text ─────────────────────────────────────────────────────────
    cur_y = bar_bottom - 0.13   # start just below the bar

    # "use when" line(s)
    if use_when:
        uw_lines = wrap(f"► {use_when}", width=30)[:2]
        for line in uw_lines:
            ax.text(
                x, cur_y, line,
                ha="center", va="top",
                fontsize=6.0, color=col, fontstyle="italic", zorder=5,
                path_effects=[pe.withStroke(linewidth=1, foreground="#0d1117")],
            )
            cur_y -= 0.40

    cur_y -= 0.10  # small gap before bullets

    # bullet points
    for b in bullets[:3]:
        blines = wrap(f"· {b}", width=32)[:2]
        for bl in blines:
            ax.text(
                x - MW + 0.22, cur_y, bl,
                ha="left", va="top",
                fontsize=5.5, color=TEXT_DIM, zorder=5,
                path_effects=[pe.withStroke(linewidth=1, foreground="#0d1117")],
            )
            cur_y -= 0.35
        cur_y -= 0.06  # inter-bullet gap

    # ── library badges ────────────────────────────────────────────────────
    if libs:
        badge_cy = y - MH + 0.38
        bw       = 1.55
        gap      = 0.22
        total    = len(libs) * bw + (len(libs) - 1) * gap
        bx       = x - total / 2 + bw / 2

        for lib in libs:
            bc = BADGE_BG.get(lib, "#30363d")
            ax.add_patch(FancyBboxPatch(
                (bx - bw/2, badge_cy - 0.20), bw, 0.38,
                boxstyle="round,pad=0.06", lw=0,
                fc=bc, alpha=0.92, zorder=5,
            ))
            ax.text(
                bx, badge_cy, lib,
                ha="center", va="center",
                fontsize=5.0, fontweight="bold", color=TEXT_BRIGHT, zorder=6,
            )
            bx += bw + gap


# ── Main ──────────────────────────────────────────────────────────────────────

def build():
    # Wide landscape — comfortable screen/print size
    fig, ax = plt.subplots(figsize=(28, 18))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, CW)
    ax.set_ylim(0, CH)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── subtle dot-grid ───────────────────────────────────────────────────
    xs = np.arange(1.5, CW, 1.5)
    ys = np.arange(1.5, CH, 1.5)
    gx, gy = np.meshgrid(xs, ys)
    ax.scatter(gx.ravel(), gy.ravel(), s=0.6, color="#1c2128", zorder=0)

    # ── title block ───────────────────────────────────────────────────────
    ax.text(
        CW / 2, 51.4,
        "DATA MINING & MACHINE LEARNING",
        ha="center", va="top",
        fontsize=19, fontweight="bold", color=TEXT_BRIGHT,
        fontfamily="monospace", zorder=5,
        path_effects=[pe.withStroke(linewidth=3, foreground=BG)],
    )
    ax.text(
        CW / 2, 50.2,
        "Model Selection Skill Tree  ·  DMML Course Reference",
        ha="center", va="top",
        fontsize=9, color=TEXT_DIM, zorder=5,
    )

    # ── edges (drawn first, behind nodes) ────────────────────────────────
    for src, dst in EDGES:
        bezier_edge(ax, src, dst)

    # ── nodes ─────────────────────────────────────────────────────────────
    for nid in POS:
        if nid in HEADERS:
            header_node(ax, nid)
        else:
            method_node(ax, nid)

    # ── SHAP legend note (bottom-left) ────────────────────────────────────
    shap_col = COL["shap_special"]
    ax.add_patch(FancyBboxPatch(
        (1.2, 1.5), 0.7, 0.7,
        boxstyle="round,pad=0.06", lw=1.2,
        ec=shap_col, fc="#1a1608", alpha=0.9, zorder=5,
        linestyle=(0, (5, 3)),
    ))
    ax.text(2.2, 1.85,
            "Dashed gold border  =  cross-cutting tool (applies to any model)",
            fontsize=6.0, color=TEXT_DIM, va="center", zorder=5)

    # ── separator line below title ────────────────────────────────────────
    ax.plot([1, CW - 1], [49.5, 49.5],
            color="#21262d", lw=1.0, zorder=1)

    plt.tight_layout(pad=0.3)

    out_pdf = "/home/czx/Code/dmml/skill_tree.pdf"
    out_png = "/home/czx/Code/dmml/skill_tree.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor=BG)
    fig.savefig(out_png, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    print(f"✓  {out_pdf}")
    print(f"✓  {out_png}")


if __name__ == "__main__":
    build()
