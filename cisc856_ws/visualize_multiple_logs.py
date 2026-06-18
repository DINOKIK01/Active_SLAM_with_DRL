#!/usr/bin/env python3

import re
import csv
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


LOG_PATTERN = re.compile(
    r"Total Epi:\s*(\d+)\s+"
    r"Steps:\s*(\d+)\s+"
    r"Episode Steps:\s*(\d+)\s+"
    r"Return:\s*([-+]?\d+\.\d+)\s+"
    r"Coverage Node:\s*([-+]?\d+\.\d+)\s+"
    r"Coverage Edge\s*([-+]?\d+\.\d+)\s+"
    r"Eps:\s*([-+]?\d+\.\d+)"
)


def parse_log(logfile):
    data = []

    with open(logfile, "r", encoding="utf-8") as f:
        for line in f:
            match = LOG_PATTERN.search(line)

            if not match:
                continue

            episode_steps = int(match.group(3))
            return_value = float(match.group(4))

            # Nachträgliche Korrektur
            if episode_steps == 1250:
                return_value -= 50

            coverage_node = float(match.group(5))
            coverage_edge = float(match.group(6))

            data.append({
                "episode": int(match.group(1)),
                "steps": int(match.group(2)),
                "episode_steps": episode_steps,
                "return": return_value,
                "coverage_node": coverage_node,
                "coverage_edge": coverage_edge,
                "coverage_overall": (
                    coverage_node + coverage_edge
                ) / 2,
                "coverage_per_step": (
                    (coverage_node + coverage_edge) / 2
                ) * 144 / episode_steps,
                "epsilon": float(match.group(7))
            })

    return data


def save_csv(data, filename):

    with open(filename, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "steps",
                "episode_steps",
                "return",
                "coverage_node",
                "coverage_edge",
                "coverage_overall",
                "coverage_per_step",
                "epsilon"
            ]
        )

        writer.writeheader()
        writer.writerows(data)


def moving_average(values, window=20):

    result = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        subset = values[start:i + 1]
        result.append(sum(subset) / len(subset))

    return result


def print_summary(name, data):

    if not data:
        return

    returns = [d["return"] for d in data]

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Episoden:           {len(data)}")
    print(f"Max Return:         {max(returns):.4f}")
    print(f"Min Return:         {min(returns):.4f}")
    print(f"Durchschnitt Return:{sum(returns)/len(returns):.4f}")

    print("\nLetzte Episode:")
    print(data[-1])


def create_comparison_plots(all_runs, ma_window=20):

    if not HAS_MATPLOTLIB:
        print("matplotlib nicht installiert.")
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    for run_name, data in all_runs.items():

        episodes = [d["episode"] for d in data]

        returns = [d["return"] for d in data]
        coverage_node = [d["coverage_node"] for d in data]
        coverage_edge = [d["coverage_edge"] for d in data]
        episode_steps = [d["episode_steps"] for d in data]

        returns_ma = moving_average(returns, ma_window)
        node_ma = moving_average(coverage_node, ma_window)
        edge_ma = moving_average(coverage_edge, ma_window)
        steps_ma = moving_average(episode_steps, ma_window)

        axes[0].plot(
            episodes,
            returns_ma,
            linewidth=2,
            label=run_name
        )

        axes[1].plot(
            episodes,
            node_ma,
            linewidth=2,
            label=run_name
        )

        axes[2].plot(
            episodes,
            edge_ma,
            linewidth=2,
            label=run_name
        )

        axes[3].plot(
            episodes,
            steps_ma,
            linewidth=2,
            label=run_name
        )

    axes[0].set_title(f"Return (MA{ma_window})")
    axes[1].set_title(f"Node Coverage (MA{ma_window})")
    axes[2].set_title(f"Edge Coverage (MA{ma_window})")
    axes[3].set_title(f"Episode Steps (MA{ma_window})")

    for ax in axes:
        ax.set_xlabel("Episode")
        ax.grid(True)
        ax.legend()

    axes[0].set_ylabel("Return")
    axes[1].set_ylabel("Coverage")
    axes[2].set_ylabel("Coverage")
    axes[3].set_ylabel("Steps")

    plt.tight_layout()

    plt.savefig(
        "training_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def main():

    if len(sys.argv) < 2:

        print("Verwendung:")
        print(
            f"python {sys.argv[0]} "
            "run1.log run2.log run3.log ..."
        )
        sys.exit(1)

    all_runs = {}

    for logfile in sys.argv[1:]:

        print(f"\nLese Logdatei: {logfile}")

        data = parse_log(logfile)

        run_name = Path(logfile).stem

        print(f"{len(data)} Episoden gefunden.")

        csv_name = f"{run_name}.csv"

        save_csv(data, csv_name)

        print(f"CSV gespeichert: {csv_name}")

        print_summary(run_name, data)

        all_runs[run_name] = data

    create_comparison_plots(all_runs)


if __name__ == "__main__":
    main()