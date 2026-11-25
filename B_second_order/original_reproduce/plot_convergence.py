"""Create the two-curve slide figure and its source table."""

import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

from experiment_io import ROOT, read_csv, read_json, write_csv


def main():
    dataset = "ijcnn1_seed19"
    inputs = {
        "CSMO": (ROOT / "reference" / dataset, "#547F9E", "-"),
        "Second-order SMO": (ROOT / "results" / "python" / "smo2" / dataset, "#826AA2", "--"),
    }
    epsilon = read_json(ROOT / "reference" / dataset / "config.json")["epsilon"]
    rows = []
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "Carlito", "Liberation Sans", "DejaVu Sans"],
        "font.size": 14,
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#58636E",
        "ytick.color": "#58636E",
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
    }):
        fig, ax = plt.subplots(figsize=(8.6, 4.2), layout="constrained")
        for name, (folder, color, line) in inputs.items():
            history = read_csv(folder / "history.csv")
            iterations = [int(row["iteration"]) for row in history]
            violations = [float(row["kkt_violation"]) for row in history]
            ax.semilogy(iterations, violations, color=color, linestyle=line, linewidth=2.25,
                        label=f"{name} ({iterations[-1]})")
            ax.scatter(iterations[-1], violations[-1], s=31, color=color, zorder=4)
            rows.extend({"algorithm": name, **row} for row in history)
        ax.axhline(epsilon, color="#58636E", linewidth=1.1, linestyle=":",
                   label=f"Tolerance = {epsilon:g}")
        max_updates = max(int(row["iteration"]) for row in rows)
        ax.set(xlabel="Updates", ylabel="KKT violation (log scale)",
               xlim=(-0.01 * max_updates, 1.08 * max_updates), ylim=(0.00065, 3.6))
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.grid(axis="y", color="#E2E7EA", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#AEB7BC")
        ax.tick_params(length=3)
        ax.legend(loc="upper right", frameon=False, fontsize=12, labelspacing=0.6)
        fig.savefig(figure_dir / "ijcnn1_kkt_convergence.svg", facecolor="white")
        fig.savefig(figure_dir / "ijcnn1_kkt_convergence.pdf", facecolor="white")
        plt.close(fig)
    write_csv(figure_dir / "ijcnn1_kkt_convergence_data.csv", rows, tuple(rows[0]))
    print(f"Plotted {len(rows)} measured points from B and C.")


if __name__ == "__main__":
    main()
