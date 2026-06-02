"""DEAP-based optimizer for ROC indicator parameters."""

from __future__ import annotations

import random
from functools import partial
from typing import Any, Sequence

import numpy as np
import pandas as pd
from deap import base, creator, tools

from src.ga.base_optimizer import record_generation_stats
from src.ga.fitness import calculate_price_error_buy_sell_fitness
from src.indicators.roc import DEFAULT_ROC_BOUNDS, generate_roc_signals, repair_roc_params


def _bounds(bounds: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = DEFAULT_ROC_BOUNDS.copy()
    if bounds:
        merged.update(bounds)
    return merged


def create_roc_individual(bounds: dict | None = None) -> list[float]:
    """Create one random ROC individual."""

    b = _bounds(bounds)
    individual = [
        random.randint(int(b["period_min"]), int(b["period_max"])),
        random.uniform(float(b["buy_threshold_min"]), float(b["buy_threshold_max"])),
        random.uniform(float(b["sell_threshold_min"]), float(b["sell_threshold_max"])),
    ]
    return repair_roc_params(individual, bounds=b)


def mate_roc_individual(ind1, ind2, bounds: dict | None = None):
    """Crossover two ROC individuals and repair them."""

    if random.random() < 0.5:
        ind1[0], ind2[0] = ind2[0], ind1[0]

    alpha = 0.5
    for idx in (1, 2):
        if random.random() < 0.5:
            x1 = ind1[idx]
            x2 = ind2[idx]
            gamma = random.uniform(-alpha, 1.0 + alpha)
            ind1[idx] = (1.0 - gamma) * x1 + gamma * x2
            ind2[idx] = gamma * x1 + (1.0 - gamma) * x2

    ind1[:] = repair_roc_params(ind1, bounds=bounds)
    ind2[:] = repair_roc_params(ind2, bounds=bounds)
    return ind1, ind2


def mutate_roc_individual(
    individual,
    mutation_prob: float = 0.2,
    bounds: dict | None = None,
):
    """Mutate one ROC individual and repair it."""

    b = _bounds(bounds)
    if random.random() < mutation_prob:
        individual[0] = int(round(individual[0])) + random.randint(-3, 3)
    if random.random() < mutation_prob:
        individual[1] = float(
            np.clip(
                individual[1] + random.gauss(0.0, 1.0),
                float(b["buy_threshold_min"]),
                float(b["buy_threshold_max"]),
            )
        )
    if random.random() < mutation_prob:
        individual[2] = float(
            np.clip(
                individual[2] + random.gauss(0.0, 1.0),
                float(b["sell_threshold_min"]),
                float(b["sell_threshold_max"]),
            )
        )

    individual[:] = repair_roc_params(individual, bounds=b)
    return (individual,)


def evaluate_roc_individual(
    individual: Sequence[float],
    df: pd.DataFrame,
    label_col: str = "turning_label",
    close_col: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
    window: int = 5,
    bounds: dict | None = None,
    fitness_config: dict[str, Any] | None = None,
) -> tuple[float]:
    """Evaluate a ROC individual and return a DEAP fitness tuple."""

    period, buy_threshold, sell_threshold = repair_roc_params(individual, bounds=bounds)
    signal_df = generate_roc_signals(
        df,
        period=period,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        close_col=close_col,
    )
    fitness_kwargs = dict(fitness_config or {})
    window = int(fitness_kwargs.pop("max_time_window", window))
    fitness, _ = calculate_price_error_buy_sell_fitness(
        signal_df,
        label_col=label_col,
        buy_signal_col="roc_buy_signal",
        sell_signal_col="roc_sell_signal",
        price_col=close_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        max_time_window=window,
        **fitness_kwargs,
    )
    return (fitness,)


def setup_roc_toolbox(
    df: pd.DataFrame,
    label_col: str = "turning_label",
    close_col: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
    window: int = 5,
    bounds: dict | None = None,
    fitness_config: dict[str, Any] | None = None,
) -> base.Toolbox:
    """Set up a DEAP toolbox for ROC parameter optimization."""

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "ROCIndividual"):
        creator.create("ROCIndividual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.ROCIndividual,
        partial(create_roc_individual, bounds=bounds),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate",
        partial(
            evaluate_roc_individual,
            df=df,
            label_col=label_col,
            close_col=close_col,
            high_col=high_col,
            low_col=low_col,
            window=window,
            bounds=bounds,
            fitness_config=fitness_config,
        ),
    )
    toolbox.register("mate", mate_roc_individual, bounds=bounds)
    toolbox.register("mutate", mutate_roc_individual, bounds=bounds)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def run_roc_ga(
    df: pd.DataFrame,
    label_col: str = "turning_label",
    close_col: str = "Close",
    high_col: str = "High",
    low_col: str = "Low",
    window: int = 5,
    population_size: int = 50,
    generations: int = 30,
    cx_prob: float = 0.7,
    mut_prob: float = 0.2,
    seed: int = 42,
    bounds: dict | None = None,
    fitness_config: dict[str, Any] | None = None,
) -> tuple[list[float], float, list[dict[str, float | int]]]:
    """Run a manual DEAP GA loop for ROC parameters."""

    random.seed(seed)
    np.random.seed(seed)

    toolbox = setup_roc_toolbox(
        df,
        label_col=label_col,
        close_col=close_col,
        high_col=high_col,
        low_col=low_col,
        window=window,
        bounds=bounds,
        fitness_config=fitness_config,
    )
    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    logbook: list[dict[str, float | int]] = []

    invalid = [individual for individual in population if not individual.fitness.valid]
    for individual, fitness in zip(invalid, map(toolbox.evaluate, invalid)):
        individual.fitness.values = fitness
    hof.update(population)

    def record(gen: int) -> None:
        record_generation_stats(population, logbook, gen)

    record(0)

    for gen in range(1, generations + 1):
        offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        for individual in offspring:
            individual[:] = repair_roc_params(individual, bounds=bounds)

        invalid = [individual for individual in offspring if not individual.fitness.valid]
        for individual, fitness in zip(invalid, map(toolbox.evaluate, invalid)):
            individual.fitness.values = fitness

        population[:] = offspring
        hof.update(population)
        record(gen)

    best_params = repair_roc_params(list(hof[0]), bounds=bounds)
    best_fitness = float(hof[0].fitness.values[0])
    return best_params, best_fitness, logbook
