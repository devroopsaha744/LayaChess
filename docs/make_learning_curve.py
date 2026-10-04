"""Learning curve of LayaChess v2 from the EVAL lines of the two Kaggle runs (300 held-out test positions)."""
import matplotlib.pyplot as plt

# (training examples, top-move accuracy, win-chance mean abs error)
EVALS = [
    (0,         0.0600, 0.2865),   # base Laya, before training
    (128_000,   0.1633, 0.1285),
    (256_000,   0.1967, 0.1205),
    (512_000,   0.2167, 0.1043),
    (768_000,   0.1967, 0.0964),
    (896_000,   0.2367, 0.0943),
    (1_033_280, 0.2367, 0.0910),   # end of run 1
    (1_152_000, 0.2133, 0.0945),   # run 2: learning rate restarted
    (1_280_000, 0.2067, 0.0942),
    (2_048_000, 0.2667, 0.0820),   # end of run 2
]
SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6"

x = [e / 1e6 for e, _, _ in EVALS]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
panels = [
    (axes[0], [a * 100 for _, a, _ in EVALS], "Picks Stockfish's best move (%)", "{:.0f}%", (0, 32)),
    (axes[1], [m * 100 for _, _, m in EVALS], "Win-chance error (percentage points, lower is better)", "{:.1f}", (0, 31)),
]
for ax, y, title, fmt, ylim in panels:
    ax.set_facecolor(SURFACE)
    ax.axvline(1.033, color=GRID, lw=1.5, ls="--", zorder=0)
    ax.text(1.06, ylim[1] * 0.97, "run 2 starts", color=INK2, fontsize=9, va="top")
    ax.plot(x, y, color=SERIES, lw=2, zorder=2)
    ax.plot(x, y, "o", ms=8, color=SERIES, mec=SURFACE, mew=2, zorder=3)
    for i in (0, len(x) - 1):
        ax.annotate(fmt.format(y[i]), (x[i], y[i]), textcoords="offset points", xytext=(0, 10),
                    ha="center", color=INK, fontsize=11, fontweight="bold")
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_ylim(*ylim); ax.set_xlim(-0.08, 2.18)
    ax.set_xlabel("training examples (millions)", color=INK2, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
fig.suptitle("LayaChess v2: a general decision model (Laya, 421M) learning chess on 2 free Kaggle GPU sessions",
             color=INK, fontsize=12.5, x=0.01, ha="left", y=0.99)
fig.text(0.01, 0.005, "Evaluated on 300 held-out ChessBench test positions (best-move accuracy wobbles about ±2.5 points); "
         "random-legal-move baseline ≈ 3%.", color=INK2, fontsize=8.5)
plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.savefig("docs/learning_curve.png", dpi=180, facecolor=SURFACE)
print("saved docs/learning_curve.png")
