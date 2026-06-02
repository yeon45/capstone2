"""DEAP-based optimizer for Stochastic Oscillator parameters."""

from __future__ import annotations

import random
from functools import partial
from typing import Any, Sequence

import numpy as np
import pandas as pd
from deap import base, creator, tools

from src.ga.base_optimizer import record_generation_stats
from src.ga.fitness import calculate_price_error_buy_sell_fitness
from src.indicators.stochastic import (
    DEFAULT_STOCHASTIC_BOUNDS,
    generate_stochastic_signals,
    repair_stochastic_params,
)


def _bounds(bounds: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = DEFAULT_STOCHASTIC_BOUNDS.copy()
    if bounds:
        merged.update(bounds)
    return merged


def create_stochastic_individual(bounds: dict | None = None) -> list[float]:
    """Create one random Stochastic individual."""

    b = _bounds(bounds)
    individual = [
        random.randint(int(b["k_period_min"]), int(b["k_period_max"])),
        random.randint(int(b["d_period_min"]), int(b["d_period_max"])),
        random.uniform(float(b["lower_min"]), float(b["lower_max"])),
        random.uniform(float(b["upper_min"]), float(b["upper_max"])),
    ]
    return repair_stochastic_params(individual, bounds=b)


def mate_stochastic_individual(ind1, ind2, bounds: dict | None = None):
    """Crossover two Stochastic individuals and repair them."""

    for idx in (0, 1):
        if random.random() < 0.5:
            ind1[idx], ind2[idx] = ind2[idx], ind1[idx]

    alpha = 0.5
    for idx in (2, 3):
        if random.random() < 0.5:
            x1 = ind1[idx]
            x2 = ind2[idx]
            gamma = random.uniform(-alpha, 1.0 + alpha)
            ind1[idx] = (1.0 - gamma) * x1 + gamma * x2
            ind2[idx] = gamma * x1 + (1.0 - gamma) * x2

    ind1[:] = repair_stochastic_params(ind1, bounds=bounds)
    ind2[:] = repair_stochastic_params(ind2, bounds=bounds)
    return ind1, ind2


def mutate_stochastic_individual(
    individual,
    mutation_prob: float = 0.2,
    bounds: dict | None = None,
):
    """Mutate one Stochastic individual and repair it."""

    b = _bounds(bounds)
    if random.random() < mutation_prob:
        individual[0] = int(round(individual[0])) + random.randint(-3, 3)
    if random.random() < mutation_prob:
        individual[1] = int(round(individual[1])) + random.randint(-2, 2)
    if random.random() < mutation_prob:
        individual[2] = float(
            np.clip(
                individual[2] + random.gauss(0.0, 3.0),
                float(b["lower_min"]),
                float(b["lower_max"]),
            )
        )
    if random.random() < mutation_prob:
        individual[3] = float(
            np.clip(
                individual[3] + random.gauss(0.0, 3.0),
                float(b["upper_min"]),
                float(b["upper_max"]),
            )
        )

    individual[:] = repair_stochastic_params(individual, bounds=b)
    return (individual,)


def evaluate_stochastic_individual(
    individual: Sequence[float],
    df: pd.DataFrame,
    label_col: str = "turning_label",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    window: int = 5,
    use_cross: bool = False,
    bounds: dict | None = None,
    fitness_config: dict[str, Any] | None = None,
) -> tuple[float]:
    """Evaluate a Stochastic individual and return a DEAP fitness tuple."""

    k_period, d_period, lower_threshold, upper_threshold = repair_stochastic_params(
        individual,
        bounds=bounds,
    )
    signal_df = generate_stochastic_signals(
        df,
        k_period=k_period,
        d_period=d_period,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        use_cross=use_cross,
    )
    fitness_kwargs = dict(fitness_config or {})
    window = int(fitness_kwargs.pop("max_time_window", window))
    fitness, _ = calculate_price_error_buy_sell_fitness(
        signal_df,
        label_col=label_col,
        buy_signal_col="stoch_buy_signal",
        sell_signal_col="stoch_sell_signal",
        price_col=close_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        max_time_window=window,
        **fitness_kwargs,
    )
    return (fitness,)


def setup_stochastic_toolbox(
    df: pd.DataFrame,
    label_col: str = "turning_label",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    window: int = 5,
    use_cross: bool = False,
    bounds: dict | None = None,
    fitness_config: dict[str, Any] | None = None,
) -> base.Toolbox:
    """Set up a DEAP toolbox for Stochastic parameter optimization."""

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "StochasticIndividual"):
        creator.create("StochasticIndividual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.StochasticIndividual,
        partial(create_stochastic_individual, bounds=bounds),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate",
        partial(
            evaluate_stochastic_individual,
            df=df,
            label_col=label_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            window=window,
            use_cross=use_cross,
            bounds=bounds,
            fitness_config=fitness_config,
        ),
    )
    toolbox.register("mate", mate_stochastic_individual, bounds=bounds)
    toolbox.register("mutate", mutate_stochastic_individual, bounds=bounds)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def run_stochastic_ga(
    df: pd.DataFrame,
    label_col: str = "turning_label",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    window: int = 5,
    population_size: int = 50,
    generations: int = 30,
    cx_prob: float = 0.7,
    mut_prob: float = 0.2,
    seed: int = 42,
    use_cross: bool = False,
    bounds: dict | None = None,
    fitness_config: dict[str, Any] | None = None,
) -> tuple[list[float], float, list[dict[str, float | int]]]:
    """Run a manual DEAP GA loop for Stochastic parameters."""

    random.seed(seed)
    np.random.seed(seed)

    toolbox = setup_stochastic_toolbox(
        df,
        label_col=label_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        window=window,
        use_cross=use_cross,
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
            individual[:] = repair_stochastic_params(individual, bounds=bounds)

        invalid = [individual for individual in offspring if not individual.fitness.valid]
        for individual, fitness in zip(invalid, map(toolbox.evaluate, invalid)):
            individual.fitness.values = fitness

        population[:] = offspring
        hof.update(population)
        record(gen)

    best_params = repair_stochastic_params(list(hof[0]), bounds=bounds)
    best_fitness = float(hof[0].fitness.values[0])
    return best_params, best_fitness, logbook
