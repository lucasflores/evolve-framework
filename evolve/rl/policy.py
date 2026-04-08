"""
Policy implementations for reinforcement learning.

Policies map observations to actions. They are phenotypes
that can be decoded from genomes and evaluated in environments.

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

import numpy as np

from evolve.representation.network import tanh

Observation = TypeVar("Observation", bound=np.ndarray)
Action = TypeVar("Action", bound=np.ndarray | int)


@runtime_checkable
class Policy(Protocol[Observation, Action]):  # type: ignore[misc]
    """
    Policy maps observations to actions.

    A Policy IS A Phenotype (evaluable decoded genome).

    Policies MAY be:
    - Deterministic (always same action for same obs)
    - Stochastic (sample from action distribution)
    - Stateful (e.g., RNN with hidden state)
    """

    def __call__(self, observation: Observation) -> Action:
        """
        Select action given observation.

        Args:
            observation: Current observation

        Returns:
            Action to take
        """
        ...


@runtime_checkable
class StatefulPolicy(Policy[Observation, Action], Protocol):  # type: ignore[misc]
    """Policy with internal state (RNN, LSTM)."""

    def reset_state(self) -> None:
        """Reset hidden state to initial values."""
        ...

    def get_state(self) -> Any:
        """Get current state for checkpointing."""
        ...

    def set_state(self, state: Any) -> None:
        """Restore state from checkpoint."""
        ...


@dataclass
class LinearPolicy:
    """
    Simple linear policy: action = W @ obs + b

    CPU reference implementation for basic control tasks.
    Can output discrete actions (argmax) or continuous actions.

    Attributes:
        weights: Weight matrix, shape (obs_dim, action_dim)
        bias: Bias vector, shape (action_dim,)
        discrete: If True, return argmax of output (for discrete actions)

    Example:
        >>> # CartPole: 4 observations -> 2 discrete actions
        >>> policy = LinearPolicy(
        ...     weights=np.random.randn(4, 2),
        ...     bias=np.zeros(2),
        ...     discrete=True
        ... )
        >>> action = policy(observation)  # Returns 0 or 1
    """

    weights: np.ndarray
    bias: np.ndarray
    discrete: bool = False

    def __call__(self, observation: np.ndarray) -> np.ndarray | int:
        """
        Compute action from observation.

        Args:
            observation: Observation array

        Returns:
            Action (int if discrete, array if continuous)
        """
        output = observation @ self.weights + self.bias
        if self.discrete:
            return int(np.argmax(output))
        return cast(np.ndarray, output)

    @property
    def n_parameters(self) -> int:
        """Total number of parameters."""
        return self.weights.size + self.bias.size

    def get_parameters(self) -> np.ndarray:
        """Flatten all parameters into a single vector."""
        return np.concatenate([self.weights.flatten(), self.bias.flatten()])

    def set_parameters(self, params: np.ndarray) -> None:
        """Set parameters from a flattened vector."""
        w_size = self.weights.size
        self.weights = params[:w_size].reshape(self.weights.shape)
        self.bias = params[w_size:]

    @classmethod
    def from_parameters(
        cls,
        params: np.ndarray,
        obs_dim: int,
        action_dim: int,
        discrete: bool = False,
    ) -> LinearPolicy:
        """
        Create policy from flattened parameter vector.

        Args:
            params: Flattened parameters
            obs_dim: Observation dimension
            action_dim: Action dimension
            discrete: Whether to use discrete actions

        Returns:
            LinearPolicy instance
        """
        w_size = obs_dim * action_dim
        weights = params[:w_size].reshape(obs_dim, action_dim)
        bias = params[w_size:]
        return cls(weights=weights, bias=bias, discrete=discrete)


@dataclass
class MLPPolicy:
    """
    Multi-layer perceptron policy.

    CPU reference implementation using NumPy.
    Supports configurable hidden layers and activations.

    Attributes:
        weights: List of weight matrices, one per layer
        biases: List of bias vectors, one per layer
        activation: Hidden layer activation function
        output_activation: Output layer activation (None = linear)
        discrete: If True, return argmax of output

    Example:
        >>> # CartPole: 4 -> 32 -> 2 with tanh hidden
        >>> policy = MLPPolicy(
        ...     weights=[np.random.randn(4, 32), np.random.randn(32, 2)],
        ...     biases=[np.zeros(32), np.zeros(2)],
        ...     activation=np.tanh,
        ...     discrete=True
        ... )
    """

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    activation: Callable[[np.ndarray], np.ndarray] = field(default=tanh)
    output_activation: Callable[[np.ndarray], np.ndarray] | None = None
    discrete: bool = False

    def __call__(self, observation: np.ndarray) -> np.ndarray | int:
        """
        Forward pass through network.

        Args:
            observation: Observation array

        Returns:
            Action (int if discrete, array if continuous)
        """
        x = observation

        # Hidden layers
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self.activation(x @ w + b)

        # Output layer
        x = x @ self.weights[-1] + self.biases[-1]
        if self.output_activation is not None:
            x = self.output_activation(x)

        if self.discrete:
            return int(np.argmax(x))
        return x

    @property
    def n_parameters(self) -> int:
        """Total number of parameters."""
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    @property
    def layer_sizes(self) -> list[int]:
        """List of layer sizes including input and output."""
        if not self.weights:
            return []
        sizes = [self.weights[0].shape[0]]
        for w in self.weights:
            sizes.append(w.shape[1])
        return sizes

    def get_parameters(self) -> np.ndarray:
        """Flatten all parameters into a single vector."""
        params = []
        for w, b in zip(self.weights, self.biases):
            params.append(w.flatten())
            params.append(b.flatten())
        return np.concatenate(params)

    def set_parameters(self, params: np.ndarray) -> None:
        """Set parameters from a flattened vector."""
        idx = 0
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            b_size = self.biases[i].size

            self.weights[i] = params[idx : idx + w_size].reshape(self.weights[i].shape)
            idx += w_size

            self.biases[i] = params[idx : idx + b_size]
            idx += b_size

    @classmethod
    def from_parameters(
        cls,
        params: np.ndarray,
        layer_sizes: list[int],
        activation: Callable[[np.ndarray], np.ndarray] = tanh,
        output_activation: Callable[[np.ndarray], np.ndarray] | None = None,
        discrete: bool = False,
    ) -> MLPPolicy:
        """
        Create policy from flattened parameter vector.

        Args:
            params: Flattened parameters
            layer_sizes: Sizes of each layer (including input/output)
            activation: Hidden layer activation
            output_activation: Output activation
            discrete: Whether to use discrete actions

        Returns:
            MLPPolicy instance
        """
        weights = []
        biases = []
        idx = 0

        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]

            w_size = in_size * out_size
            weights.append(params[idx : idx + w_size].reshape(in_size, out_size))
            idx += w_size

            biases.append(params[idx : idx + out_size])
            idx += out_size

        return cls(
            weights=weights,
            biases=biases,
            activation=activation,
            output_activation=output_activation,
            discrete=discrete,
        )


@dataclass
class RecurrentPolicy:
    """
    Simple recurrent policy with hidden state.

    Implements Elman-style recurrence:
        h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
        y_t = W_hy @ h_t + b_y

    Attributes:
        w_xh: Input to hidden weights, shape (input_dim, hidden_dim)
        w_hh: Hidden to hidden weights, shape (hidden_dim, hidden_dim)
        w_hy: Hidden to output weights, shape (hidden_dim, output_dim)
        b_h: Hidden bias, shape (hidden_dim,)
        b_y: Output bias, shape (output_dim,)
        discrete: If True, return argmax of output

    Example:
        >>> # Policy with memory: 4 inputs, 16 hidden, 2 outputs
        >>> policy = RecurrentPolicy(
        ...     w_xh=np.random.randn(4, 16) * 0.1,
        ...     w_hh=np.random.randn(16, 16) * 0.1,
        ...     w_hy=np.random.randn(16, 2) * 0.1,
        ...     b_h=np.zeros(16),
        ...     b_y=np.zeros(2),
        ...     discrete=True
        ... )
        >>> policy.reset_state()  # Reset hidden state
        >>> action = policy(observation)  # Uses and updates hidden state
    """

    w_xh: np.ndarray  # (input_dim, hidden_dim)
    w_hh: np.ndarray  # (hidden_dim, hidden_dim)
    w_hy: np.ndarray  # (hidden_dim, output_dim)
    b_h: np.ndarray  # (hidden_dim,)
    b_y: np.ndarray  # (output_dim,)
    discrete: bool = False
    _hidden: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize hidden state."""
        self.reset_state()

    @property
    def hidden_size(self) -> int:
        """Size of hidden state."""
        return int(self.w_hh.shape[0])

    @property
    def input_size(self) -> int:
        """Size of input."""
        return int(self.w_xh.shape[0])

    @property
    def output_size(self) -> int:
        """Size of output."""
        return int(self.w_hy.shape[1])

    def reset_state(self) -> None:
        """Reset hidden state to zeros."""
        self._hidden = np.zeros(self.hidden_size)

    def get_state(self) -> np.ndarray:
        """Get current hidden state."""
        if self._hidden is None:
            self.reset_state()
        return self._hidden.copy()  # type: ignore

    def set_state(self, state: np.ndarray) -> None:
        """Set hidden state."""
        self._hidden = state.copy()

    def __call__(self, observation: np.ndarray) -> np.ndarray | int:
        """
        Forward pass with state update.

        Args:
            observation: Observation array

        Returns:
            Action (int if discrete, array if continuous)
        """
        if self._hidden is None:
            self.reset_state()

        # Update hidden state
        self._hidden = np.tanh(observation @ self.w_xh + self._hidden @ self.w_hh + self.b_h)

        # Compute output
        output = self._hidden @ self.w_hy + self.b_y

        if self.discrete:
            return int(np.argmax(output))
        return cast(np.ndarray, output)

    @property
    def n_parameters(self) -> int:
        """Total number of parameters."""
        return self.w_xh.size + self.w_hh.size + self.w_hy.size + self.b_h.size + self.b_y.size

    def get_parameters(self) -> np.ndarray:
        """Flatten all parameters into a single vector."""
        return np.concatenate(
            [
                self.w_xh.flatten(),
                self.w_hh.flatten(),
                self.w_hy.flatten(),
                self.b_h.flatten(),
                self.b_y.flatten(),
            ]
        )

    def set_parameters(self, params: np.ndarray) -> None:
        """Set parameters from a flattened vector."""
        idx = 0

        size = self.w_xh.size
        self.w_xh = params[idx : idx + size].reshape(self.w_xh.shape)
        idx += size

        size = self.w_hh.size
        self.w_hh = params[idx : idx + size].reshape(self.w_hh.shape)
        idx += size

        size = self.w_hy.size
        self.w_hy = params[idx : idx + size].reshape(self.w_hy.shape)
        idx += size

        size = self.b_h.size
        self.b_h = params[idx : idx + size]
        idx += size

        self.b_y = params[idx:]

    @classmethod
    def from_parameters(
        cls,
        params: np.ndarray,
        input_size: int,
        hidden_size: int,
        output_size: int,
        discrete: bool = False,
    ) -> RecurrentPolicy:
        """
        Create policy from flattened parameter vector.

        Args:
            params: Flattened parameters
            input_size: Input dimension
            hidden_size: Hidden state dimension
            output_size: Output dimension
            discrete: Whether to use discrete actions

        Returns:
            RecurrentPolicy instance
        """
        idx = 0

        size = input_size * hidden_size
        w_xh = params[idx : idx + size].reshape(input_size, hidden_size)
        idx += size

        size = hidden_size * hidden_size
        w_hh = params[idx : idx + size].reshape(hidden_size, hidden_size)
        idx += size

        size = hidden_size * output_size
        w_hy = params[idx : idx + size].reshape(hidden_size, output_size)
        idx += size

        b_h = params[idx : idx + hidden_size]
        idx += hidden_size

        b_y = params[idx : idx + output_size]

        return cls(
            w_xh=w_xh,
            w_hh=w_hh,
            w_hy=w_hy,
            b_h=b_h,
            b_y=b_y,
            discrete=discrete,
        )
