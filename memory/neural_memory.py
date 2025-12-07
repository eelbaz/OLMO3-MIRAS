# olmo3_miras/memory/neural_memory.py
"""
Neural Long-Term Memory Module implementing Titans-LMM from MIRAS framework.
Learns to memorize at test time through gradient-based surprise metrics.

Based on:
- "Titans: Learning to Memorize at Test Time" (arXiv:2501.00663)
- "MIRAS: A Framework for Designing Deep Learning Architectures" (arXiv:2504.13173)

Key equations (Titans-LMM variant from MIRAS Table 1):
    Memory update:  M_t = (1 - α_t) * M_{t-1} + S_t      [Titans Eq 13-14]
    Momentum:       S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M; x) [Titans Eq 10, MIRAS Titans-LMM]
    Loss:           ℓ(M; k, v) = ||M(k) - v||²_2         [L2-MSE from MIRAS]
    Retrieval:      y_t = M*(q_t)

    Parallel form:  M_t = β_t * M_0 + Σ_{i=1}^{t} (β_t / β_i) * S_i  [Titans Eq 16]
    where β_t = ∏_{j=1}^{t} (1 - α_j)

MIRAS Design Choices (for Titans-LMM):
    - Memory Structure: Deep MLP (L_M >= 2 layers)
    - Attentional Bias: L2-MSE loss
    - Retention Gate: Local (weight decay) + Global (norm)
    - Memory Algorithm: Gradient descent with momentum
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class MIRASMemoryConfig:
    """Configuration for MIRAS Neural Memory Module."""

    # Memory architecture
    hidden_size: int = 4096
    memory_hidden_size: int = 2048  # Internal memory dimension
    memory_depth: int = 2  # Number of MLP layers (L_M >= 2 recommended)
    num_memory_heads: int = 8

    # Learning dynamics
    use_momentum: bool = True
    momentum_decay: float = 0.9  # η_t base value
    learning_rate: float = 0.1   # θ_t base value
    forget_gate: bool = True     # Enable adaptive forgetting (α_t)

    # Chunk-based parallelization
    chunk_size: int = 512

    # Persistent memory
    num_persistent_tokens: int = 16

    # Data-dependent gates
    data_dependent_gates: bool = True

    # Numerical stability
    eps: float = 1e-6
    max_grad_norm: float = 1.0
    grad_scale: float = 0.1  # Scale gradients for stability


class AssociativeMemoryMLP(nn.Module):
    """
    Deep MLP serving as the memory module.
    Learns key-value associations through test-time optimization.

    The memory is parameterized as a deep network M(k) that maps
    keys to values. During test-time learning, the weights are
    updated based on surprise (gradient of prediction error).
    """

    def __init__(self, config: MIRASMemoryConfig):
        super().__init__()
        self.config = config

        # Build deep memory network
        layers = []
        in_dim = config.memory_hidden_size

        for i in range(config.memory_depth):
            out_dim = config.memory_hidden_size
            layers.extend([
                nn.Linear(in_dim, out_dim, bias=False),
                nn.SiLU() if i < config.memory_depth - 1 else nn.Identity()
            ])
            in_dim = out_dim

        self.memory_net = nn.Sequential(*layers)
        self.num_params = sum(p.numel() for p in self.memory_net.parameters())

        # Store layer shapes for manual weight application
        self.layer_shapes = []
        for layer in self.memory_net:
            if isinstance(layer, nn.Linear):
                self.layer_shapes.append(layer.weight.shape)

        # Initialize weights for stable gradient-based learning
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable test-time learning."""
        for module in self.memory_net.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight, gain=0.1)

    def forward(self, keys: Tensor) -> Tensor:
        """Forward pass without weight update (retrieval)."""
        return self.memory_net(keys)

    def compute_loss(self, keys: Tensor, values: Tensor) -> Tensor:
        """
        Compute associative memory loss.
        ℓ(M; x) = ||M(k) - v||²
        """
        predictions = self.forward(keys)
        return F.mse_loss(predictions, values, reduction='none').sum(dim=-1)

    def get_flat_params(self) -> Tensor:
        """Get flattened parameters for efficient updates."""
        return torch.cat([p.flatten() for p in self.parameters()])

    def set_flat_params(self, flat_params: Tensor):
        """Set parameters from flattened tensor."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view(p.shape))
            offset += numel


class NeuralLongTermMemory(nn.Module):
    """
    MIRAS Neural Long-Term Memory Module.

    Implements surprise-based learning with:
    - Momentary surprise: gradient of associative memory loss
    - Past surprise: momentum-accumulated surprise
    - Adaptive forgetting: weight decay for memory management

    The key insight from Titans is that the inner-loop gradient descent
    can be reformulated using only matmuls, enabling efficient GPU parallelization.
    """

    def __init__(self, config: MIRASMemoryConfig):
        super().__init__()
        self.config = config

        # Projection layers: x -> (k, v, q)
        self.key_proj = nn.Linear(config.hidden_size, config.memory_hidden_size, bias=False)
        self.value_proj = nn.Linear(config.hidden_size, config.memory_hidden_size, bias=False)
        self.query_proj = nn.Linear(config.hidden_size, config.memory_hidden_size, bias=False)
        self.output_proj = nn.Linear(config.memory_hidden_size, config.hidden_size, bias=False)

        # 1D causal convolutions (following modern linear RNN practices)
        self.key_conv = nn.Conv1d(
            config.memory_hidden_size, config.memory_hidden_size,
            kernel_size=4, padding=3, groups=config.memory_hidden_size
        )
        self.value_conv = nn.Conv1d(
            config.memory_hidden_size, config.memory_hidden_size,
            kernel_size=4, padding=3, groups=config.memory_hidden_size
        )
        self.query_conv = nn.Conv1d(
            config.memory_hidden_size, config.memory_hidden_size,
            kernel_size=4, padding=3, groups=config.memory_hidden_size
        )

        # Memory module (the deep MLP being optimized at test time)
        self.memory = AssociativeMemoryMLP(config)

        # Data-dependent gates for α, η, θ
        if config.data_dependent_gates:
            self.alpha_gate = nn.Sequential(
                nn.Linear(config.hidden_size, config.memory_hidden_size),
                nn.Sigmoid()  # α ∈ [0, 1] - forgetting rate
            )
            self.eta_gate = nn.Sequential(
                nn.Linear(config.hidden_size, config.memory_hidden_size),
                nn.Sigmoid()  # η ∈ [0, 1] - momentum decay
            )
            self.theta_gate = nn.Sequential(
                nn.Linear(config.hidden_size, config.memory_hidden_size),
                nn.Softplus()  # θ > 0 - learning rate
            )

        # Output normalization
        self.output_norm = nn.RMSNorm(config.hidden_size, eps=config.eps)

        # Learnable output gate (initialized to allow signal through)
        self.output_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        nn.init.zeros_(self.output_gate.weight)
        nn.init.ones_(self.output_gate.bias)  # Initialize bias to 1 so sigmoid ≈ 0.73

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        for module in [self.key_proj, self.value_proj, self.query_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)

    def _apply_conv(self, x: Tensor, conv: nn.Module) -> Tensor:
        """Apply causal convolution."""
        # x: (batch, seq, dim) -> (batch, dim, seq)
        x = x.transpose(1, 2)
        x = conv(x)[:, :, :-3]  # Remove right padding for causality
        return x.transpose(1, 2)

    def compute_surprise_metrics(
        self,
        hidden_states: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute data-dependent surprise control metrics.

        Returns:
            alpha: Forgetting gate ∈ [0, 1] - how much to forget
            eta: Momentum decay ∈ [0, 1] - surprise decay rate
            theta: Learning rate > 0 - how much to update from current surprise
        """
        if self.config.data_dependent_gates:
            alpha = self.alpha_gate(hidden_states) * 0.1  # Scale down forgetting
            eta = self.eta_gate(hidden_states) * 0.9 + 0.05  # Keep in [0.05, 0.95]
            theta = self.theta_gate(hidden_states) * self.config.learning_rate
        else:
            batch_size, seq_len, _ = hidden_states.shape
            device = hidden_states.device
            dtype = hidden_states.dtype

            alpha = torch.full(
                (batch_size, seq_len, self.config.memory_hidden_size),
                0.01, device=device, dtype=dtype
            )
            eta = torch.full(
                (batch_size, seq_len, self.config.memory_hidden_size),
                self.config.momentum_decay, device=device, dtype=dtype
            )
            theta = torch.full(
                (batch_size, seq_len, self.config.memory_hidden_size),
                self.config.learning_rate, device=device, dtype=dtype
            )

        return alpha, eta, theta

    def _apply_memory_weights(self, x: Tensor, flat_params: Tensor) -> Tensor:
        """
        Apply memory network weights manually for batched gradient computation.

        Args:
            x: Input tensor (batch, dim)
            flat_params: Flattened memory parameters (batch, num_params)

        Returns:
            Output tensor (batch, dim)
        """
        batch_size = x.shape[0]
        offset = 0

        for i, shape in enumerate(self.memory.layer_shapes):
            out_dim, in_dim = shape
            weight_numel = out_dim * in_dim

            # Extract weights for this layer: (batch, out_dim, in_dim)
            weights = flat_params[:, offset:offset + weight_numel]
            weights = weights.view(batch_size, out_dim, in_dim)

            # Batched matrix multiplication: (batch, in_dim) @ (batch, in_dim, out_dim) -> (batch, out_dim)
            x = torch.bmm(x.unsqueeze(1), weights.transpose(1, 2)).squeeze(1)

            offset += weight_numel

            # Apply activation (SiLU) except for last layer
            if i < len(self.memory.layer_shapes) - 1:
                x = F.silu(x)

        return x

    def _compute_memory_gradients(
        self,
        keys: Tensor,
        values: Tensor,
        memory_params: Tensor
    ) -> Tensor:
        """
        Compute gradients of associative memory loss w.r.t. memory parameters.

        ∇ℓ(M; x) = ∇||M(k) - v||²

        This is the key insight from Titans: test-time learning requires computing
        gradients even during inference. We use torch.enable_grad() to ensure
        gradients can be computed regardless of the outer context (e.g., torch.no_grad()).

        Args:
            keys: Key vectors (batch, seq, dim)
            values: Value vectors (batch, seq, dim)
            memory_params: Current memory parameters (batch, num_params)

        Returns:
            Gradients (batch, seq, num_params)
        """
        batch_size, seq_len, dim = keys.shape
        device = keys.device
        dtype = keys.dtype

        grads = []

        # Enable gradient computation for test-time learning
        # This is critical: even during inference, we need gradients for memory updates
        with torch.enable_grad():
            for t in range(seq_len):
                # Detach inputs to prevent gradients flowing back to main model during inference
                k_t = keys[:, t].detach().requires_grad_(False)  # (batch, dim)
                v_t = values[:, t].detach()  # (batch, dim)

                # Clone memory params with gradient tracking for this computation
                params = memory_params.detach().clone().requires_grad_(True)

                # Forward through memory with gradient-enabled params
                pred = self._apply_memory_weights(k_t, params)

                # Loss: ||pred - v||²
                loss = ((pred - v_t) ** 2).sum(dim=-1).sum()  # Scalar

                # Compute gradient w.r.t. params only
                grad = torch.autograd.grad(
                    loss, params,
                    retain_graph=False,
                    create_graph=False  # No second-order grads needed here
                )[0]

                grads.append(grad.detach())  # Detach to prevent graph accumulation

        grads = torch.stack(grads, dim=1)  # (batch, seq, num_params)

        # Scale gradients for stability
        grads = grads * self.config.grad_scale

        # Clip gradients
        grad_norm = grads.norm(dim=-1, keepdim=True).clamp(min=self.config.eps)
        grads = grads * torch.clamp(self.config.max_grad_norm / grad_norm, max=1.0)

        return grads

    def _parallel_momentum_scan(
        self,
        grads: Tensor,
        eta: Tensor,
        theta: Tensor,
        init_momentum: Tensor
    ) -> Tensor:
        """
        Parallel scan for momentum computation.

        S_t = η_t * S_{t-1} - θ_t * grad_t

        This is a linear recurrence that can be computed in O(log N) using
        associative scan (parallel prefix sum).

        Args:
            grads: Gradients (batch, seq, num_params)
            eta: Momentum decay (batch, seq, dim) or scalar
            theta: Learning rate (batch, seq, dim) or scalar
            init_momentum: Initial momentum state (batch, num_params)

        Returns:
            Momentum values at each timestep (batch, seq, num_params)
        """
        batch_size, seq_len, num_params = grads.shape
        device = grads.device
        dtype = grads.dtype

        # Reduce eta/theta to match param dimension
        if eta.dim() == 3 and eta.shape[-1] != num_params:
            eta = eta.mean(dim=-1, keepdim=True).expand(-1, -1, num_params)
            theta = theta.mean(dim=-1, keepdim=True).expand(-1, -1, num_params)

        # Input term: -θ_t * grad_t
        b = -theta * grads  # (batch, seq, num_params)
        a = eta  # (batch, seq, num_params)

        # For linear recurrence: y_t = a_t * y_{t-1} + b_t
        # Use sequential computation for correctness (can optimize with parallel scan later)
        outputs = []
        S = init_momentum  # (batch, num_params)

        for t in range(seq_len):
            S = a[:, t] * S + b[:, t]
            outputs.append(S)

        return torch.stack(outputs, dim=1)  # (batch, seq, num_params)

    def _chunk_forward(
        self,
        keys: Tensor,
        values: Tensor,
        queries: Tensor,
        alpha: Tensor,
        eta: Tensor,
        theta: Tensor,
        memory_state: Optional[Tensor] = None,
        momentum_state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Process a single chunk with parallelized gradient computation.

        Implements the parallel formulation from the Titans paper:
            M_t = β_t * M_0 + Σ_{i=1}^{t} (β_t / β_i) * S_i
            S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_0; x_t)

        Within a chunk, gradients are computed w.r.t. the chunk-start memory M_0.
        This enables parallelization while maintaining the learning dynamics.
        """
        batch_size, chunk_len, dim = keys.shape
        device = keys.device
        dtype = keys.dtype

        num_params = self.memory.num_params

        # Initialize states if needed
        if memory_state is None:
            # Each batch element gets its own copy of memory weights
            base_params = self.memory.get_flat_params()
            memory_state = base_params.unsqueeze(0).expand(batch_size, -1).clone()
        if momentum_state is None:
            momentum_state = torch.zeros(batch_size, num_params, device=device, dtype=dtype)

        # 1. Compute gradients w.r.t. chunk-start memory (parallelizable)
        grads = self._compute_memory_gradients(keys, values, memory_state)

        # 2. Compute momentum updates using parallel scan
        # S_t = η_t * S_{t-1} - θ_t * grad_t
        momentum_updates = self._parallel_momentum_scan(grads, eta, theta, momentum_state)

        # 3. Compute forgetting factors
        # β_t = ∏_{j=1}^{t} (1 - α_j)
        alpha_scalar = alpha.mean(dim=-1)  # (batch, chunk_len)
        alpha_scalar = alpha_scalar.clamp(0, 0.5)  # Prevent too much forgetting

        log_beta = torch.cumsum(
            torch.log((1 - alpha_scalar).clamp(min=self.config.eps)), dim=1
        )
        beta = torch.exp(log_beta.clamp(max=0))  # (batch, chunk_len), clamp to prevent explosion

        # 4. Compute cumulative memory updates
        # M_t = β_t * M_0 + Σ_{i=1}^{t} (β_t / β_i) * S_i

        # Compute β_t / β_i matrix (upper triangular for t >= i)
        beta_t = beta.unsqueeze(2)  # (batch, chunk_len, 1)
        beta_i = beta.unsqueeze(1)  # (batch, 1, chunk_len)
        beta_ratios = beta_t / (beta_i + self.config.eps)  # (batch, t, i)
        beta_ratios = torch.tril(beta_ratios)  # Zero out i > t

        # Weighted sum of momentum updates
        cumulative_S = torch.bmm(beta_ratios, momentum_updates)  # (batch, chunk_len, num_params)

        # 5. Compute memory states at each timestep
        # M_t = β_t * M_0 + cumulative_S[:, t]
        beta_expanded = beta.unsqueeze(-1)  # (batch, chunk_len, 1)
        M_0_expanded = memory_state.unsqueeze(1)  # (batch, 1, num_params)
        memory_at_t = beta_expanded * M_0_expanded + cumulative_S  # (batch, chunk_len, num_params)

        # 6. Retrieve from memory at each timestep
        # Use enable_grad for training gradient flow through retrieval
        outputs = []
        with torch.enable_grad():
            for t in range(chunk_len):
                q_t = queries[:, t]  # (batch, dim)
                mem_t = memory_at_t[:, t]  # (batch, num_params)
                # During training, allow gradients to flow through retrieval
                # During inference, this still works but gradients are ignored
                output = self._apply_memory_weights(q_t, mem_t)
                outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (batch, chunk_len, dim)

        # Final states for next chunk
        final_memory = memory_at_t[:, -1]  # (batch, num_params)
        final_momentum = momentum_updates[:, -1]  # (batch, num_params)

        return outputs, final_memory, final_momentum

    def forward(
        self,
        hidden_states: Tensor,
        memory_state: Optional[Tensor] = None,
        momentum_state: Optional[Tensor] = None,
        return_memory_state: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        Forward pass with test-time memory learning.

        Args:
            hidden_states: Input tensor (batch, seq, hidden_size)
            memory_state: Previous memory parameters (batch, num_params)
            momentum_state: Previous momentum state (batch, num_params)
            return_memory_state: Whether to return updated states

        Returns:
            Output tensor, optionally with (memory_state, momentum_state)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to memory dimension
        keys = self.key_proj(hidden_states)
        values = self.value_proj(hidden_states)
        queries = self.query_proj(hidden_states)

        # Apply causal convolutions
        keys = self._apply_conv(keys, self.key_conv)
        values = self._apply_conv(values, self.value_conv)
        queries = self._apply_conv(queries, self.query_conv)

        # Apply SiLU activation
        keys = F.silu(keys)
        values = F.silu(values)
        queries = F.silu(queries)

        # Normalize for stable dot products
        keys = F.normalize(keys, p=2, dim=-1)
        queries = F.normalize(queries, p=2, dim=-1)

        # Compute data-dependent surprise metrics
        alpha, eta, theta = self.compute_surprise_metrics(hidden_states)

        # Process in chunks for parallelization
        chunk_size = min(self.config.chunk_size, seq_len)
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        all_outputs = []
        current_memory = memory_state
        current_momentum = momentum_state

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, seq_len)

            chunk_keys = keys[:, start_idx:end_idx]
            chunk_values = values[:, start_idx:end_idx]
            chunk_queries = queries[:, start_idx:end_idx]
            chunk_alpha = alpha[:, start_idx:end_idx]
            chunk_eta = eta[:, start_idx:end_idx]
            chunk_theta = theta[:, start_idx:end_idx]

            chunk_output, current_memory, current_momentum = self._chunk_forward(
                chunk_keys, chunk_values, chunk_queries,
                chunk_alpha, chunk_eta, chunk_theta,
                current_memory, current_momentum
            )
            all_outputs.append(chunk_output)

        # Concatenate chunk outputs
        outputs = torch.cat(all_outputs, dim=1)

        # Project back to hidden dimension
        outputs = self.output_proj(outputs)
        outputs = self.output_norm(outputs)

        # Apply output gating
        gate = torch.sigmoid(self.output_gate(hidden_states))
        outputs = gate * outputs

        if return_memory_state:
            return outputs, current_memory, current_momentum
        return outputs


class PersistentMemory(nn.Module):
    """
    Learnable but data-independent persistent memory.

    From the paper: "[p_1, p_2, ..., p_{N_p}] || x"
    These are learnable parameters prepended to the input sequence.
    They store task-related knowledge that remains fixed at test time.
    """

    def __init__(self, config: MIRASMemoryConfig):
        super().__init__()
        self.config = config

        # Learnable persistent memory tokens
        self.memory_tokens = nn.Parameter(
            torch.randn(config.num_persistent_tokens, config.hidden_size) * 0.02
        )

    def forward(self, batch_size: int) -> Tensor:
        """
        Get persistent memory tokens for a batch.

        Returns:
            Tensor of shape (batch, num_persistent_tokens, hidden_size)
        """
        return self.memory_tokens.unsqueeze(0).expand(batch_size, -1, -1)
