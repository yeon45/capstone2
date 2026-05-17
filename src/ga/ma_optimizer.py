"""DEAP-based optimizer for Moving Average indicator parameters."""

from __future__ import annotations

import random
from functools import partial
from typing import Any, Sequence

import numpy as np
import pandas as pd
from deap import base, creator, tools

from src.ga.base_optimizer import compile_population_stats
from src.ga.fitness import calculate_signal_fitness
from src.indicators.ma import DEFAULT_MA_BOUNDS, generate_ma_signals, repair_ma_params


def _bounds(bounds: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = DEFAULT_MA_BOUNDS.copy()
    if bounds:
        merged.update(bounds)
    return merged


def create_ma_individual(bounds: dict | None = None) -> list[float]:
    """Create one random MA individual."""

    b = _bounds(bounds)
    n = random.randint(int(b["n_min"]), int(b["n_max"]))
    N = random.randint(max(n + 1, int(b["N_min"])), int(b["N_max"]))
    individual = [
        n,
        N,
        random.uniform(b["a_min"], b["a_max"]),
        random.uniform(b["b_min"], b["b_max"]),
        random.uniform(b["c_min"], b["c_max"]),
        random.uniform(b["a_min"], b["a_max"]),
        random.uniform(b["b_min"], b["b_max"]),
        random.uniform(b["c_min"], b["c_max"]),
    ]
    return repair_ma_params(individual, bounds=b)


def mate_ma_individual(ind1, ind2, bounds: dict | None = None):
    """Crossover two MA individuals and repair them."""

    for idx in (0, 1):
        if random.random() < 0.5:
            ind1[idx], ind2[idx] = ind2[idx], ind1[idx]

    alpha = 0.5
    for idx in range(2, 8):
        if random.random() < 0.5:
            x1 = ind1[idx]
            x2 = ind2[idx]
            gamma = random.uniform(-alpha, 1.0 + alpha)
            ind1[idx] = (1.0 - gamma) * x1 + gamma * x2
            ind2[idx] = gamma * x1 + (1.0 - gamma) * x2

    ind1[:] = repair_ma_params(ind1, bounds=bounds)
    ind2[:] = repair_ma_params(ind2, bounds=bounds)
    return ind1, ind2


def mutate_ma_individual(
    individual,
    mutation_prob: float = 0.2,
    bounds: dict | None = None,
):
    """Mutate one MA individual and repair it."""

    b = _bounds(bounds)
    if random.random() < mutation_prob:
        individual[0] = int(round(individual[0])) + random.randint(-5, 5)
    if random.random() < mutation_prob:
        individual[1] = int(round(individual[1])) + random.randint(-10, 10)

    float_specs = {
        2: ("a_min", "a_max", 0.4),
        3: ("b_min", "b_max", 0.25),
        4: ("c_min", "c_max", 0.01),
        5: ("a_min", "a_max", 0.4),
        6: ("b_min", "b_max", 0.25),
        7: ("c_min", "c_max", 0.01),
    }
    for idx, (lower_key, upper_key, sigma) in float_specs.items():
        if random.random() < mutation_prob:
            individual[idx] = float(
                np.clip(individual[idx] + random.gauss(0.0, sigma), b[lower_key], b[upper_key])
            )

    individual[:] = repair_ma_params(individual, bounds=b)
    return (individual,)


def evaluate_ma_individual(
    individual: Sequence[float],
    df: pd.DataFrame,
    label_col: str = "turning_label",
    close_col: str = "Close",
    window: int = 5,
    bounds: dict | None = None,
) -> tuple[float]:
    """Evaluate an MA individual and return a DEAP fitness tuple."""

    params = repair_ma_params(individual, bounds=bounds)
    signal_df = generate_ma_signals(df, params, close_col=close_col, normalize=True)
    fitness, _ = calculate_signal_fitness(
        signal_df,
        label_col=label_col,
        buy_signal_col="ma_buy_signal",
        sell_signal_col="ma_sell_signal",
        window=window,
    )
    return (fitness,)


def setup_ma_toolbox(
    df: pd.DataFrame,
    label_col: str = "turning_label",
    close_col: str = "Close",
    window: int = 5,
    bounds: dict | None = None,
) -> base.Toolbox:
    """Set up a DEAP toolbox for MA parameter optimization."""

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "MAIndividual"):
        creator.create("MAIndividual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.MAIndividual,
        partial(create_ma_individual, bounds=bounds),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate",
        partial(
            evaluate_ma_individual,
            df=df,
            label_col=label_col,
            close_col=close_col,
            window=window,
            bounds=bounds,
        ),
    )
    toolbox.register("mate", mate_ma_individual, bounds=bounds)
    toolbox.register("mutate", mutate_ma_individual, bounds=bounds)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def run_ma_ga(
    df: pd.DataFrame,
    label_col: str = "turning_label",
    close_col: str = "Close",
    window: int = 5,
    population_size: int = 50,
    generations: int = 30,
    cx_prob: float = 0.7,
    mut_prob: float = 0.3,
    seed: int = 42,
    bounds: dict | None = None,
) -> tuple[list[float], float, list[dict[str, float | int]]]:
    """Run a manual DEAP GA loop for MA parameters."""

    random.seed(seed)
    np.random.seed(seed)

    toolbox = setup_ma_toolbox(
        df,
        label_col=label_col,
        close_col=close_col,
        window=window,
        bounds=bounds,
    )
    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    logbook: list[dict[str, float | int]] = []

    invalid = [individual for individual in population if not individual.fitness.valid]
    for individual, fitness in zip(invalid, map(toolbox.evaluate, invalid)):
        individual.fitness.values = fitness
    hof.update(population)

    def record(gen: int) -> None:
        stats = compile_population_stats(population)
        entry = {"gen": gen, **stats}
        logbook.append(entry)
        print(
            f"gen={gen:03d} max={entry['max']:.6f} "
            f"avg={entry['avg']:.6f} min={entry['min']:.6f}"
        )

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
            individual[:] = repair_ma_params(individual, bounds=bounds)

        invalid = [individual for individual in offspring if not individual.fitness.valid]
        for individual, fitness in zip(invalid, map(toolbox.evaluate, invalid)):
            individual.fitness.values = fitness

        population[:] = offspring
        hof.update(population)
        record(gen)

    best_params = repair_ma_params(list(hof[0]), bounds=bounds)
    best_fitness = float(hof[0].fitness.values[0])
    return best_params, best_fitness, logbook
