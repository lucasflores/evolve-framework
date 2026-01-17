"""
Neural Network phenotypes using NumPy.

Provides CPU-only reference implementations of neural networks
for neuroevolution. These are decoded from graph genomes.

NO ML FRAMEWORK IMPORTS ALLOWED (except NumPy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np


# Activation functions
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation."""
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified linear unit: max(0, x)."""
    return np.maximum(0, x)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU: x if x > 0 else alpha * x."""
    return np.where(x > 0, x, alpha * x)


def identity(x: np.ndarray) -> np.ndarray:
    """Identity activation (no-op)."""
    return x


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation (for output layer)."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def step(x: np.ndarray) -> np.ndarray:
    """Step function: 1 if x > 0 else 0."""
    return np.where(x > 0, 1.0, 0.0)


def gaussian(x: np.ndarray) -> np.ndarray:
    """Gaussian activation: exp(-x^2)."""
    return np.exp(-x * x)


# Activation function registry
ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "identity": identity,
    "linear": identity,
    "softmax": softmax,
    "step": step,
    "gaussian": gaussian,
}


def get_activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get activation function by name.
    
    Args:
        name: Activation function name
        
    Returns:
        Activation function
        
    Raises:
        KeyError: If activation not found
    """
    return ACTIVATIONS[name.lower()]


@dataclass
class NumpyNetwork:
    """
    Simple feedforward network using NumPy.
    
    CPU reference implementation for neuroevolution.
    Supports variable-depth networks with different activations per layer.
    
    Attributes:
        weights: List of weight matrices, one per layer
        biases: List of bias vectors, one per layer
        activations: List of activation functions, one per layer
    
    Example:
        >>> # 2-layer network: 3 inputs -> 5 hidden -> 2 outputs
        >>> net = NumpyNetwork(
        ...     weights=[np.random.randn(3, 5), np.random.randn(5, 2)],
        ...     biases=[np.zeros(5), np.zeros(2)],
        ...     activations=[relu, sigmoid]
        ... )
        >>> output = net(np.array([1.0, 2.0, 3.0]))
    """
    
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    activations: list[Callable[[np.ndarray], np.ndarray]]
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input array, shape (n_inputs,) or (batch, n_inputs)
            
        Returns:
            Output array, shape (n_outputs,) or (batch, n_outputs)
        """
        for w, b, act in zip(self.weights, self.biases, self.activations):
            x = act(x @ w + b)
        return x
    
    @property
    def n_layers(self) -> int:
        """Number of layers (weight matrices)."""
        return len(self.weights)
    
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
            
            self.weights[i] = params[idx:idx + w_size].reshape(self.weights[i].shape)
            idx += w_size
            
            self.biases[i] = params[idx:idx + b_size]
            idx += b_size
    
    @property
    def n_parameters(self) -> int:
        """Total number of parameters."""
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))


@dataclass
class RecurrentNumpyNetwork:
    """
    Simple recurrent network using NumPy.
    
    Implements Elman-style recurrence with hidden state feedback.
    
    Attributes:
        input_weights: Weight matrix for inputs, shape (n_inputs, n_hidden)
        recurrent_weights: Weight matrix for hidden state, shape (n_hidden, n_hidden)
        output_weights: Weight matrix to outputs, shape (n_hidden, n_outputs)
        hidden_bias: Bias for hidden layer, shape (n_hidden,)
        output_bias: Bias for output layer, shape (n_outputs,)
        hidden_activation: Activation for hidden layer
        output_activation: Activation for output layer
        hidden_state: Current hidden state
    """
    
    input_weights: np.ndarray
    recurrent_weights: np.ndarray
    output_weights: np.ndarray
    hidden_bias: np.ndarray
    output_bias: np.ndarray
    hidden_activation: Callable[[np.ndarray], np.ndarray] = field(default=tanh)
    output_activation: Callable[[np.ndarray], np.ndarray] = field(default=identity)
    hidden_state: np.ndarray | None = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize hidden state if not provided."""
        if self.hidden_state is None:
            n_hidden = self.recurrent_weights.shape[0]
            self.hidden_state = np.zeros(n_hidden)
    
    def reset(self) -> None:
        """Reset hidden state to zeros."""
        n_hidden = self.recurrent_weights.shape[0]
        self.hidden_state = np.zeros(n_hidden)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with state update.
        
        Args:
            x: Input array, shape (n_inputs,)
            
        Returns:
            Output array, shape (n_outputs,)
        """
        # Compute new hidden state
        h_input = x @ self.input_weights
        h_recurrent = self.hidden_state @ self.recurrent_weights
        self.hidden_state = self.hidden_activation(
            h_input + h_recurrent + self.hidden_bias
        )
        
        # Compute output
        output = self.output_activation(
            self.hidden_state @ self.output_weights + self.output_bias
        )
        
        return output
    
    @property
    def n_hidden(self) -> int:
        """Number of hidden units."""
        return self.recurrent_weights.shape[0]


@dataclass
class NEATNetwork:
    """
    Network decoded from NEAT graph genome.
    
    Supports arbitrary topology with forward-only evaluation
    using topological ordering.
    
    Attributes:
        node_order: Topologically sorted list of node IDs
        node_biases: Dict of node_id -> bias
        node_activations: Dict of node_id -> activation function
        connections: Dict of (from_id, to_id) -> weight
        input_ids: Tuple of input node IDs
        output_ids: Tuple of output node IDs
    """
    
    node_order: list[int]
    node_biases: dict[int, float]
    node_activations: dict[int, Callable[[np.ndarray], np.ndarray]]
    connections: dict[tuple[int, int], float]
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input array, shape (n_inputs,)
            
        Returns:
            Output array, shape (n_outputs,)
        """
        # Initialize node values
        node_values: dict[int, float] = {}
        
        # Set input values
        for i, input_id in enumerate(self.input_ids):
            node_values[input_id] = float(x[i])
        
        # Process nodes in topological order
        for node_id in self.node_order:
            if node_id in self.input_ids:
                continue  # Already set
            
            # Sum weighted inputs
            total = self.node_biases.get(node_id, 0.0)
            for (from_id, to_id), weight in self.connections.items():
                if to_id == node_id and from_id in node_values:
                    total += node_values[from_id] * weight
            
            # Apply activation
            activation = self.node_activations.get(node_id, sigmoid)
            node_values[node_id] = float(activation(np.array([total]))[0])
        
        # Collect outputs
        outputs = np.array([
            node_values.get(output_id, 0.0) 
            for output_id in self.output_ids
        ])
        
        return outputs
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network (alias for __call__).
        
        Args:
            x: Input array, shape (n_inputs,)
            
        Returns:
            Output array, shape (n_outputs,)
        """
        return self(x)
