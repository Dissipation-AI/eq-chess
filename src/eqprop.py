"""
Holomorphic Equilibrium Propagation implementation in PyTorch.

Based on the holomorphic EqProp algorithm from:
https://github.com/Laborieux-Axel/holomorphic_eqprop

This implements a single-layer energy-based model that uses:
- Energy function: phi = 0.5 * state^T @ W @ state - beta * loss
- Dynamics: state evolves via gradient descent on phi
- Learning: Holomorphic EP with N-phase sampling on unit circle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class LinearHolomorphicEQProp(nn.Module):
    """
    Single-layer holomorphic equilibrium propagation module.

    The state vector is partitioned into three parts:
    - Input units: Clamped to input data, don't update during dynamics
    - Hidden units: Update via energy minimization
    - Output units: Update via energy minimization, compared with targets during learning

    Args:
        input_size: Dimension of input
        hidden_size: Dimension of hidden layer
        output_size: Dimension of output
        weight_std: Standard deviation for weight initialization (default: 0.1)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_std: float = 0.1):
        super().__init__()

        # Unit sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.total_size = input_size + hidden_size + output_size

        # Weight matrix - symmetric for energy-based model
        # Initialize with truncated normal
        self.W = nn.Parameter(torch.randn(self.total_size, self.total_size) * weight_std)

        # State vector [input, hidden, output]
        # This is not a parameter, it's the dynamic state
        self.register_buffer('state', torch.zeros(self.total_size))

        # Activation function
        self.activation = torch.tanh

    @property
    def input_state(self):
        """View of input portion of state."""
        return self.state[:self.input_size]

    @property
    def hidden_state(self):
        """View of hidden portion of state."""
        return self.state[self.input_size:self.input_size + self.hidden_size]

    @property
    def output_state(self):
        """View of output portion of state."""
        return self.state[self.input_size + self.hidden_size:]

    def set_inputs(self, x: torch.Tensor):
        """
        Clamp input units to provided input.

        Args:
            x: Input tensor of shape (input_size,)
        """
        assert x.shape[-1] == self.input_size, f"Input size mismatch: expected {self.input_size}, got {x.shape[-1]}"
        self.state[:self.input_size] = x

    def clear_input(self, noise_std: float = 0.0):
        """
        Reset input state to zeros or noise.

        Args:
            noise_std: Standard deviation of Gaussian noise (0 = zeros)
        """
        if noise_std > 0:
            self.state[:self.input_size] = torch.randn_like(self.state[:self.input_size]) * noise_std
        else:
            self.state[:self.input_size] = 0.0

    def phi(self, state: torch.Tensor, beta: torch.Tensor, target: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute energy function.

        Energy: phi = 0.5 * s^T @ W @ s - beta * loss(output, target)

        Supports both real and complex tensors for holomorphic gradients.

        Args:
            state: State vector (real or complex)
            beta: Nudging parameter (scalar or complex, 0 for free phase, >0 for nudged phase)
            target: Target output (required if beta != 0)
            mask: Binary mask indicating which output units to nudge (default: all)

        Returns:
            Scalar energy value (complex if inputs are complex)
        """
        # Quadratic energy term: 0.5 * s^T @ W @ s
        # For complex: use s^H @ W @ s (Hermitian) but W should be symmetric
        energy = 0.5 * torch.sum(state * (self.W @ state))

        # Loss term (only if beta != 0 and target provided)
        if beta != 0 and target is not None:
            # Extract output from state
            output = state[self.input_size + self.hidden_size:]

            # Cross-entropy loss (assuming target is one-hot)
            # For complex outputs, use real part of logits for classification
            logits = output.real if output.is_complex() else output
            log_softmax = F.log_softmax(logits, dim=-1)

            # Apply mask if provided (only nudge masked positions)
            if mask is not None:
                loss = -torch.sum(mask * target * log_softmax)
            else:
                loss = -torch.sum(target * log_softmax)

            energy = energy - beta * loss

        return energy

    def tick(self, steps: int = 1, beta: torch.Tensor = None, target: Optional[torch.Tensor] = None,
             mask: Optional[torch.Tensor] = None, lr: float = 0.5, noise_std: float = 0.0):
        """
        Iterate dynamics: evolve state via gradient descent on energy.

        Only hidden and output units are updated. Input units remain clamped.
        Supports both real and complex beta for holomorphic dynamics.

        Args:
            steps: Number of time steps to iterate
            beta: Nudging parameter (scalar float, int, or complex tensor)
            target: Target output (required if beta != 0)
            mask: Binary mask for partial supervision
            lr: Learning rate for dynamics (step size)
            noise_std: Standard deviation of noise added during dynamics
        """
        # Default beta to 0.0
        if beta is None:
            beta = 0.0

        # Convert beta to tensor if needed
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta, dtype=self.state.dtype)

        for _ in range(steps):
            # Compute gradient of energy w.r.t. state
            # We need to use autograd on a copy to avoid in-place issues
            state = self.state.clone().requires_grad_(True)

            energy = self.phi(state, beta, target, mask)

            # For complex dynamics, PyTorch requires real-valued loss
            # We use the real part of the energy for gradient computation
            if energy.is_complex():
                energy = energy.real

            grad_state = torch.autograd.grad(energy, state, create_graph=False)[0]

            # Update only hidden and output units (not input)
            # Gradient descent: s_new = activation(s - lr * grad_phi)
            with torch.no_grad():
                # Update hidden units
                h_start = self.input_size
                h_end = self.input_size + self.hidden_size
                new_hidden = self.state[h_start:h_end] - lr * grad_state[h_start:h_end]

                # Add noise if specified (only for real-valued dynamics)
                if noise_std > 0 and not self.state.is_complex():
                    new_hidden = new_hidden + torch.randn_like(new_hidden) * noise_std

                self.state[h_start:h_end] = self.activation(new_hidden)

                # Update output units
                o_start = self.input_size + self.hidden_size
                new_output = self.state[o_start:] - lr * grad_state[o_start:]

                if noise_std > 0 and not self.state.is_complex():
                    new_output = new_output + torch.randn_like(new_output) * noise_std

                self.state[o_start:] = self.activation(new_output)

    def learn(self, target: torch.Tensor, mask: Optional[torch.Tensor] = None,
              beta: float = 0.1, T1: int = 20, T2: int = 4, N: int = 1,
              lr_dynamics: float = 0.5, lr_learning: float = 0.01, noise_std: float = 0.0):
        """
        Perform equilibrium propagation learning with holomorphic N-phase sampling.

        EP algorithm:
        1. Free phase: Evolve to equilibrium with beta=0
        2. Nudged phase(s): Evolve with complex beta on unit circle (holomorphic)
        3. Compute gradient: avg(-dφ/dW / β_k) across N phases
        4. Update weights via gradient descent

        Args:
            target: Target output (one-hot)
            mask: Binary mask indicating which output units to supervise (default: all).
                  Shape should match target. Only masked positions contribute to nudging.
            beta: Nudging strength
            T1: Time steps for free phase
            T2: Time steps for nudged phase(s)
            N: Number of phases for holomorphic sampling (1=standard EP, >1=holomorphic)
            lr_dynamics: Learning rate for state dynamics
            lr_learning: Learning rate for weight updates
            noise_std: Noise level during dynamics
        """
        # Free phase: beta = 0
        self.tick(steps=T1, beta=0.0, target=None, mask=None, lr=lr_dynamics, noise_std=noise_std)
        state_free = self.state.clone()

        if N == 1:
            # Standard EP (single nudged phase)
            # Compute gradient at free equilibrium
            state_1 = state_free.clone().requires_grad_(True)
            phi_1 = self.phi(state_1, beta=torch.tensor(0.0), target=None, mask=None)
            grad_phi_1 = torch.autograd.grad(phi_1, self.W, create_graph=False)[0]

            # Nudged phase: beta != 0
            self.state = state_free.clone()
            self.tick(steps=T2, beta=beta, target=target, mask=mask, lr=lr_dynamics, noise_std=noise_std)

            # Compute gradient at nudged equilibrium
            state_2 = self.state.clone().requires_grad_(True)
            # Need to temporarily set state for phi computation
            old_state = self.state.clone()
            self.state = state_2
            phi_2 = self.phi(state_2, beta=torch.tensor(beta), target=target, mask=mask)
            grad_phi_2 = torch.autograd.grad(phi_2, self.W, create_graph=False)[0]
            self.state = old_state

            # EP gradient: (grad_phi_1 - grad_phi_2) / beta
            grad_W = (grad_phi_1 - grad_phi_2) / beta

        else:
            # Holomorphic EP: sample N phases on unit circle
            # β_k = β * exp(2πik/N) for k = 0, 1, ..., N-1
            # Gradient: dE/dW = -dφ/dW / β_k
            # We average these gradients and take the real part

            # Save original real-valued parameters
            W_real = self.W.data.clone()
            state_real = state_free.clone()

            # Convert to complex domain
            # PyTorch requires explicit dtype for complex conversion
            self.W.data = self.W.data.to(dtype=torch.complex64)
            self.state = state_free.to(dtype=torch.complex64)

            # Initialize gradient accumulator (complex)
            grad_W_accum = torch.zeros_like(self.W, dtype=torch.complex64)

            for k in range(N):
                # Complex beta on unit circle: β_k = β * exp(2πik/N)
                angle = 2.0 * math.pi * k / N
                beta_k = beta * torch.exp(torch.tensor(1j * angle, dtype=torch.complex64))

                # Reset to free state (complex) and evolve with complex beta
                self.state = state_real.to(dtype=torch.complex64).clone()

                # Run complex dynamics
                self.tick(steps=T2, beta=beta_k, target=target, mask=mask,
                         lr=lr_dynamics, noise_std=0.0)  # No noise in complex dynamics

                # Compute gradient: dE/dW = -dφ/dW / β_k
                # Need to compute dφ/dW at current complex state
                state_k = self.state.clone().requires_grad_(True)
                old_state = self.state.clone()
                self.state = state_k

                # Compute phi at this complex equilibrium
                phi_k = self.phi(state_k, beta=beta_k, target=target, mask=mask)

                # For complex gradients, PyTorch requires real scalar output
                # Take real part for gradient computation
                phi_k_real = phi_k.real if phi_k.is_complex() else phi_k

                # Compute dφ/dW using PyTorch's autograd
                # We compute d(Re(φ))/dW which gives us the holomorphic gradient
                grad_phi_k = torch.autograd.grad(phi_k_real, self.W, create_graph=False)[0]

                self.state = old_state

                # Gradient contribution: -dφ/dW / β_k
                grad_k = -grad_phi_k / beta_k

                # Accumulate
                grad_W_accum += grad_k

            # Average over N phases and take real part
            grad_W = (grad_W_accum / N).real

            # Restore real weights and state
            self.W.data = W_real
            self.state = state_real

            # Convert gradient to real dtype for update
            grad_W = grad_W.to(dtype=W_real.dtype)

        # Update weights via gradient descent
        with torch.no_grad():
            self.W -= lr_learning * grad_W

    def evaluate(self, x: torch.Tensor, noise_std: float = 0.0, T_noise: int = 100,
                 T_settle: int = 500, lr_dynamics: float = 0.5) -> torch.Tensor:
        """
        Evaluate the model on input x.

        1. Noise phase: Clear input, settle with noise
        2. Input phase: Clamp input, settle to equilibrium
        3. Return output

        Args:
            x: Input tensor of shape (input_size,)
            noise_std: Noise level during noise phase
            T_noise: Time steps for noise phase
            T_settle: Time steps to settle after input clamping
            lr_dynamics: Learning rate for dynamics

        Returns:
            Output tensor of shape (output_size,)
        """
        # Noise phase
        self.clear_input(noise_std=noise_std)
        self.tick(steps=T_noise, beta=0.0, target=None, lr=lr_dynamics, noise_std=noise_std)

        # Input phase
        self.set_inputs(x)
        self.tick(steps=T_settle, beta=0.0, target=None, lr=lr_dynamics)

        # Return copy of output state
        return self.output_state.clone()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass (alias for evaluate).

        Args:
            x: Input tensor
            **kwargs: Additional arguments passed to evaluate()

        Returns:
            Output tensor
        """
        return self.evaluate(x, **kwargs)


class BatchedLinearHolomorphicEQProp(nn.Module):
    """
    Batched version of LinearHolomorphicEQProp for efficient parallel processing.

    Maintains separate states for each sample in the batch.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 batch_size: int, weight_std: float = 0.1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.total_size = input_size + hidden_size + output_size

        # Shared weights across batch
        self.W = nn.Parameter(torch.randn(self.total_size, self.total_size) * weight_std)

        # Batched states: (batch_size, total_size)
        self.register_buffer('states', torch.zeros(batch_size, self.total_size))

        self.activation = torch.tanh

    def set_inputs(self, x: torch.Tensor):
        """
        Set inputs for entire batch.

        Args:
            x: Input tensor of shape (batch_size, input_size)
        """
        assert x.shape[0] == self.batch_size
        assert x.shape[1] == self.input_size
        self.states[:, :self.input_size] = x

    def clear_inputs(self, noise_std: float = 0.0):
        """Clear all input states."""
        if noise_std > 0:
            self.states[:, :self.input_size] = torch.randn(self.batch_size, self.input_size) * noise_std
        else:
            self.states[:, :self.input_size] = 0.0

    def phi_batch(self, states: torch.Tensor, beta: float,
                  targets: Optional[torch.Tensor] = None,
                  masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute energy for entire batch.

        Args:
            states: (batch_size, total_size)
            beta: Nudging parameter
            targets: (batch_size, output_size)
            masks: (batch_size, output_size) binary masks for partial supervision

        Returns:
            Energy tensor of shape (batch_size,)
        """
        # Quadratic term: 0.5 * s^T @ W @ s for each sample
        Ws = torch.matmul(states, self.W)  # (batch_size, total_size)
        energy = 0.5 * torch.sum(states * Ws, dim=1)  # (batch_size,)

        if beta != 0 and targets is not None:
            # Extract outputs
            outputs = states[:, self.input_size + self.hidden_size:]

            # Cross-entropy loss
            log_softmax = F.log_softmax(outputs, dim=1)

            # Apply mask if provided
            if masks is not None:
                loss = -torch.sum(masks * targets * log_softmax, dim=1)
            else:
                loss = -torch.sum(targets * log_softmax, dim=1)

            energy = energy - beta * loss

        return energy

    def tick_batch(self, steps: int = 1, beta: float = 0.0,
                   targets: Optional[torch.Tensor] = None,
                   masks: Optional[torch.Tensor] = None,
                   lr: float = 0.5, noise_std: float = 0.0):
        """
        Batched dynamics iteration.

        Args:
            steps: Number of time steps
            beta: Nudging parameter
            targets: (batch_size, output_size) or None
            masks: (batch_size, output_size) binary masks or None
            lr: Dynamics learning rate
            noise_std: Noise level
        """
        for _ in range(steps):
            states = self.states.clone().requires_grad_(True)
            energy = self.phi_batch(states, beta, targets, masks).sum()
            grad_states = torch.autograd.grad(energy, states, create_graph=False)[0]

            with torch.no_grad():
                # Update hidden
                h_start = self.input_size
                h_end = self.input_size + self.hidden_size
                new_hidden = self.states[:, h_start:h_end] - lr * grad_states[:, h_start:h_end]
                if noise_std > 0:
                    new_hidden += torch.randn_like(new_hidden) * noise_std
                self.states[:, h_start:h_end] = self.activation(new_hidden)

                # Update output
                o_start = self.input_size + self.hidden_size
                new_output = self.states[:, o_start:] - lr * grad_states[:, o_start:]
                if noise_std > 0:
                    new_output += torch.randn_like(new_output) * noise_std
                self.states[:, o_start:] = self.activation(new_output)

    def forward(self, x: torch.Tensor, T_settle: int = 500, lr_dynamics: float = 0.5) -> torch.Tensor:
        """
        Batched forward pass.

        Args:
            x: (batch_size, input_size)
            T_settle: Settling time
            lr_dynamics: Dynamics learning rate

        Returns:
            Outputs of shape (batch_size, output_size)
        """
        self.set_inputs(x)
        self.tick_batch(steps=T_settle, beta=0.0, targets=None, lr=lr_dynamics)
        return self.states[:, self.input_size + self.hidden_size:].clone()
