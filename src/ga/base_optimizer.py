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
