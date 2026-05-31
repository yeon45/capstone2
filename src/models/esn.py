"""Echo State Network classifier."""

from __future__ import annotations

import contextlib
import io

import numpy as np

try:
    with contextlib.redirect_stderr(io.StringIO()):
        from sklearn.linear_model import RidgeClassifier
except Exception:  # pragma: no cover - depends on local binary package state
    RidgeClassifier = None


class _NumpyRidgeClassifier:
    """Small one-vs-rest Ridge classifier fallback when sklearn is unavailable."""

    def __init__(self, alpha: float = 1.0, class_weight: str | dict | None = None) -> None:
        self.alpha = float(alpha)
        self.class_weight = class_weight
        self.classes_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyRidgeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        X_aug = np.column_stack([np.ones(len(X)), X])
        targets = np.column_stack(
            [np.where(y == label, 1.0, -1.0) for label in self.classes_]
        )
        sample_weight = self._sample_weight(y)
        weighted_X = X_aug * sample_weight[:, None]

        penalty = np.eye(X_aug.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        lhs = X_aug.T @ weighted_X + penalty
        rhs = X_aug.T @ (targets * sample_weight[:, None])
        self.coef_ = np.linalg.solve(lhs, rhs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None or self.coef_ is None:
            raise RuntimeError("Classifier must be fitted before predict().")
        X = np.asarray(X, dtype=float)
        X_aug = np.column_stack([np.ones(len(X)), X])
        scores = X_aug @ self.coef_
        return self.classes_[np.argmax(scores, axis=1)]

    def _sample_weight(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            return np.ones(len(y), dtype=float)
        if self.class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            weights = {
                label: len(y) / (len(classes) * count)
                for label, count in zip(classes, counts)
            }
            return np.array([weights[label] for label in y], dtype=float)
        if isinstance(self.class_weight, dict):
            return np.array([self.class_weight.get(label, 1.0) for label in y], dtype=float)
        raise ValueError(f"Unsupported class_weight: {self.class_weight}")


class ESNClassifier:
    """Echo State Network with a RidgeClassifier readout."""

    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 200,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        leaking_rate: float = 0.3,
        ridge_alpha: float = 1.0,
        random_state: int = 42,
        washout: int = 30,
        class_weight: str | dict | None = "balanced",
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if reservoir_size <= 0:
            raise ValueError("reservoir_size must be positive.")
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError("sparsity must be between 0 and 1.")
        if not 0.0 < leaking_rate <= 1.0:
            raise ValueError("leaking_rate must be in (0, 1].")
        if washout < 0:
            raise ValueError("washout must be non-negative.")

        self.input_dim = int(input_dim)
        self.reservoir_size = int(reservoir_size)
        self.spectral_radius = float(spectral_radius)
        self.sparsity = float(sparsity)
        self.leaking_rate = float(leaking_rate)
        self.ridge_alpha = float(ridge_alpha)
        self.random_state = int(random_state)
        self.washout = int(washout)
        self.class_weight = class_weight

        self.rng_ = np.random.default_rng(self.random_state)
        self.W_in = self.rng_.uniform(
            low=-1.0,
            high=1.0,
            size=(self.reservoir_size, self.input_dim + 1),
        )
        self.W = self._initialize_reservoir()
        self.state = np.zeros(self.reservoir_size, dtype=float)
        self.readout_ = None

    def _initialize_reservoir(self) -> np.ndarray:
        weights = self.rng_.uniform(
            low=-1.0,
            high=1.0,
            size=(self.reservoir_size, self.reservoir_size),
        )
        mask = self.rng_.random(size=weights.shape) < self.sparsity
        weights *= mask

        eigenvalues = np.linalg.eigvals(weights)
        current_radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
        if current_radius > 0.0:
            weights *= self.spectral_radius / current_radius
        return weights

    def _reset_state(self) -> None:
        """Reset the reservoir state to zeros."""

        self.state = np.zeros(self.reservoir_size, dtype=float)

    def _update_state(self, x: np.ndarray) -> np.ndarray:
        """Advance the reservoir by one input vector."""

        x = np.asarray(x, dtype=float)
        if x.shape != (self.input_dim,):
            raise ValueError(f"Expected input shape {(self.input_dim,)}, got {x.shape}.")

        x_with_bias = np.concatenate(([1.0], x))
        pre_activation = self.W_in @ x_with_bias + self.W @ self.state
        candidate_state = np.tanh(pre_activation)
        self.state = (
            (1.0 - self.leaking_rate) * self.state
            + self.leaking_rate * candidate_state
        )
        return self.state.copy()

    def predict_states(self, X: np.ndarray) -> np.ndarray:
        """Return reservoir states for X in time order."""

        X = self._validate_X(X)
        self._reset_state()
        states = np.empty((len(X), self.reservoir_size), dtype=float)
        for idx, x_t in enumerate(X):
            states[idx] = self._update_state(x_t)
        return states

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ESNClassifier":
        """Fit the RidgeClassifier readout on washout-trimmed reservoir states."""

        X = self._validate_X(X)
        y = np.asarray(y)
        if len(X) != len(y):
            raise ValueError(f"X and y lengths differ: {len(X)} != {len(y)}.")
        if len(X) <= self.washout:
            raise ValueError(
                f"Need more samples than washout ({self.washout}); got {len(X)}."
            )

        states = self.predict_states(X)
        train_states = states[self.washout :]
        train_y = y[self.washout :]

        readout_class = RidgeClassifier or _NumpyRidgeClassifier
        self.readout_ = readout_class(alpha=self.ridge_alpha, class_weight=self.class_weight)
        self.readout_.fit(train_states, train_y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X in time order."""

        if self.readout_ is None:
            raise RuntimeError("ESNClassifier must be fitted before predict().")

        states = self.predict_states(X)
        predictions = self.readout_.predict(states)
        return predictions.astype(int)

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}.")
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected X with {self.input_dim} features, got {X.shape[1]}."
            )
        return X
