"""DEAP-based optimizer for candlestick pattern parameters."""

from __future__ import annotations

import random
from functools import partial
from typing import Any, Sequence

import numpy as np
import pandas as pd
from deap import base, creator, tools

from src.ga.base_optimizer import record_generation_stats
from src.ga.fitness import calculate_price_error_signal_fitness
from src.indicators.candle import (
    CANDLE_SIGNAL_COLUMNS,
    DEFAULT_CANDLE_BOUNDS,
    generate_candle_signals,
    repair_candle_params,
)


def _bounds(bounds: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = DEFAULT_CANDLE_BOUNDS.copy()
    if bounds:
        merged.update(bounds)
    return merged


def _match_events_with_forward_window(
    true_indices: list[int],
    pred_indices: list[int],
    window: int,
) -> tuple[int, int, int]:
    true_list = sorted(int(i) for i in true_indices)
    pred_list = sorted(int(i) for i in pred_indices)
    used_pred_positions: set[int] = set()
    tp = 0
    fn = 0

    for true_idx in true_list:
        matched_position = None
        for pos, pred_idx in enumerate(pred_list):
            if pos in used_pred_positions:
                continue
            if pred_idx < true_idx:
                continue
            if pred_idx > true_idx + window:
                break
            matched_position = pos
            break
        if matched_position is None:
            fn += 1
        else:
            used_pred_positions.add(matched_position)
            tp += 1

    fp = len(pred_list) - len(used_pred_positions)
    return tp, fp, fn


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    return (2 * tp) / ((2 * tp) + fp + fn)


def calculate_candle_fitness(
    df: pd.DataFrame,
    label_col: str = "turning_label",
    signal_cols: Sequence[str] = CANDLE_SIGNAL_COLUMNS,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    window: int = 5,
    buy_label: int = -1,
    sell_label: int = 1,
    false_signal_penalty: float = 0.5,
    fitness_config: dict[str, Any] | None = None,
) -> tuple[float, dict]:
    """Calculate price-error fitness from independent signed candle pattern columns."""

    required = [label_col, *signal_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for candle fitness: {missing}")
    if window < 0:
        raise ValueError("window must be non-negative")

    signal_values = df[list(signal_cols)].to_numpy(dtype=float)
    signed_signal = np.sign(signal_values.sum(axis=1)).astype(int)
    temp_df = df.copy()
    temp_df["__candle_signed_signal__"] = signed_signal
    fitness_kwargs = dict(fitness_config or {})
    window = int(fitness_kwargs.pop("max_time_window", window))
    false_signal_penalty = float(fitness_kwargs.pop("false_signal_penalty", false_signal_penalty))
    fitness, details = calculate_price_error_signal_fitness(
        temp_df,
        signal_col="__candle_signed_signal__",
        label_col=label_col,
        price_col=close_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        max_time_window=window,
        false_signal_penalty=false_signal_penalty,
        **fitness_kwargs,
    )
    details["signal_columns"] = list(signal_cols)
    details["buy_label"] = buy_label
    details["sell_label"] = sell_label
    return fitness, details


def create_candle_individual(bounds: dict | None = None) -> list[float]:
    """Create one random candle individual."""

    b = _bounds(bounds)
    individual = [
        random.uniform(float(b["a_min"]), float(b["a_max"])),
        random.uniform(float(b["b_min"]), float(b["b_max"])),
        random.uniform(float(b["c_min"]), float(b["c_max"])),
        random.uniform(float(b["d_min"]), float(b["d_max"])),
        random.uniform(float(b["e_min"]), float(b["e_max"])),
        random.uniform(float(b["f_min"]), float(b["f_max"])),
        random.uniform(float(b["g_min"]), float(b["g_max"])),
    ]
    return repair_candle_params(individual, bounds=b)


def mate_candle_individual(ind1, ind2, bounds: dict | None = None):
    """Crossover two candle individuals and repair them."""

    alpha = 0.5
    for idx in range(7):
        if random.random() < 0.5:
            x1 = ind1[idx]
            x2 = ind2[idx]
            gamma = random.uniform(-alpha, 1.0 + alpha)
            ind1[idx] = (1.0 - gamma) * x1 + gamma * x2
            ind2[idx] = gamma * x1 + (1.0 - gamma) * x2

    ind1[:] = repair_candle_params(ind1, bounds=bounds)
    ind2[:] = repair_candle_params(ind2, bounds=bounds)
    return ind1, ind2


def mutate_candle_individual(
    individual,
    mutation_prob: float = 0.2,
    bounds: dict | None = None,
):
    """Mutate one candle individual and repair it."""

    b = _bounds(bounds)
    sigmas = [0.4, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]
    keys = [
        ("a_min", "a_max"),
        ("b_min", "b_max"),
        ("c_min", "c_max"),
        ("d_min", "d_max"),
        ("e_min", "e_max"),
        ("f_min", "f_max"),
        ("g_min", "g_max"),
    ]
    for idx, ((lower_key, upper_key), sigma) in enumerate(zip(keys, sigmas)):
        if random.random() < mutation_prob:
            individual[idx] = float(
                np.clip(
                    individual[idx] + random.gauss(0.0, sigma),
                    float(b[lower_key]),
                    float(b[upper_key]),
                )
            )

    individual[:] = repair_candle_params(individual, bounds=b)
    return (individual,)


def evaluate_candle_individual(
    individual: Sequence[float],
    df: pd.DataFrame,
    label_col: str = "turning_label",
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    window: int = 5,
    signal_cols: Sequence[str] = CANDLE_SIGNAL_COLUMNS,
    bounds: dict | None = None,
    fitness_config: dict[str, Any] | None = None,
) -> tuple[float]:
    """Evaluate a candle individual and return a DEAP fitness tuple."""

    params = repair_candle_params(individual, bounds=bounds)
    signal_df = generate_candle_signals(
        df,
        params=params,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
    )
    fitness, _ = calculate_candle_fitness(
        signal_df,
        label_col=label_col,
        signal_cols=signal_cols,
        window=window,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        fitness_config=fitness_config,
    )
    return (fitness,)


def setup_candle_toolbox(
    df: pd.DataFrame,
    label_col: str = "turning_label",
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    window: int = 5,
    signal_cols: Sequence[str] = CANDLE_SIGNAL_COLUMNS,
    bounds: dict | None = None,
    fitness_config: dict[str, Any] | None = None,
) -> base.Toolbox:
    """Set up a DEAP toolbox for candle parameter optimization."""

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "CandleIndividual"):
        creator.create("CandleIndividual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.CandleIndividual,
        partial(create_candle_individual, bounds=bounds),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate",
        partial(
            evaluate_candle_individual,
            df=df,
            label_col=label_col,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            window=window,
            signal_cols=signal_cols,
            bounds=bounds,
            fitness_config=fitness_config,
        ),
    )
    toolbox.register("mate", mate_candle_individual, bounds=bounds)
    toolbox.register("mutate", mutate_candle_individual, bounds=bounds)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def run_candle_ga(
    df: pd.DataFrame,
    label_col: str = "turning_label",
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    window: int = 5,
    signal_cols: Sequence[str] = CANDLE_SIGNAL_COLUMNS,
    population_size: int = 50,
    generations: int = 30,
    cx_prob: float = 0.7,
    mut_prob: float = 0.2,
    seed: int = 42,
    bounds: dict | None = None,
    fitness_config: dict[str, Any] | None = None,
) -> tuple[list[float], float, list[dict[str, float | int]]]:
    """Run a manual DEAP GA loop for candle parameters."""

    random.seed(seed)
    np.random.seed(seed)

    toolbox = setup_candle_toolbox(
        df,
        label_col=label_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        window=window,
        signal_cols=signal_cols,
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
            individual[:] = repair_candle_params(individual, bounds=bounds)

        invalid = [individual for individual in offspring if not individual.fitness.valid]
        for individual, fitness in zip(invalid, map(toolbox.evaluate, invalid)):
            individual.fitness.values = fitness

        population[:] = offspring
        hof.update(population)
        record(gen)

    best_params = repair_candle_params(list(hof[0]), bounds=bounds)
    best_fitness = float(hof[0].fitness.values[0])
    return best_params, best_fitness, logbook
