"""Reusable GA helper functions."""

from __future__ import annotations

from statistics import mean


def compile_population_stats(population) -> dict[str, float]:
    """Compile max, avg, and min fitness values for a DEAP population."""

    values = [individual.fitness.values[0] for individual in population]
    return {
        "max": float(max(values)),
        "avg": float(mean(values)),
        "min": float(min(values)),
    }


def record_generation_stats(population, logbook: list[dict[str, float | int]], gen: int) -> None:
    """Record and print per-generation fitness stats with improvement deltas."""

    stats = compile_population_stats(population)
    previous = logbook[-1] if logbook else None
    previous_max = float(previous["max"]) if previous else stats["max"]
    previous_avg = float(previous["avg"]) if previous else stats["avg"]
    previous_best = float(previous["best_so_far"]) if previous else stats["max"]
    best_so_far = max(previous_best, stats["max"])

    entry = {
        "gen": gen,
        **stats,
        "delta_max": float(stats["max"] - previous_max),
        "delta_avg": float(stats["avg"] - previous_avg),
        "best_so_far": float(best_so_far),
        "delta_best": float(best_so_far - previous_best),
    }
    logbook.append(entry)
    print(
        f"gen={gen:03d} max={entry['max']:.6f} "
        f"delta_max={entry['delta_max']:+.6f} "
        f"best={entry['best_so_far']:.6f} "
        f"delta_best={entry['delta_best']:+.6f} "
        f"avg={entry['avg']:.6f} delta_avg={entry['delta_avg']:+.6f} "
        f"min={entry['min']:.6f}"
    )
