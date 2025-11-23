# Holomorphic Equilibrium Propagation - Algorithm Notes

Based on: https://arxiv.org/abs/2209.00530
JAX Implementation: https://github.com/Laborieux-Axel/holomorphic_eqprop

## Key Algorithm (from JAX code)

### Energy Function
```
φ(x, h, y, β) = Σ W_i(h_i) · h_{i+1} - β · CrossEntropy(output, target)
```

### State Dynamics
States evolve via gradient descent on φ:
```
for t in T:
    h ← ∂φ/∂h  (with holomorphic=True for complex)
    h ← activation(h)
```

### Standard EP (N=1)
```python
# 1. Free phase (β=0)
h_free ← settle(x, h_init, β=0)

# 2. Nudged phase (β>0)
h_nudged ← settle(x, h_free, β=β)

# 3. Gradient
∇W = (∇_W φ(h_free, β=0) - ∇_W φ(h_nudged, β)) / β
```

### Holomorphic EP (N>1)
```python
# 1. Free phase
h_free ← settle(x, h_init, β=0)

# 2. Convert to complex domain
h ← to_complex(h_free)
W ← to_complex(W)
x ← to_complex(x)

# 3. Sample N phases on unit circle
grads ← 0
for k in range(N):
    # Complex beta
    β_k = β · exp(2πik/N)

    # Settle with complex beta
    h_k ← settle(x, h, β_k)  # Fully complex dynamics!

    # Compute gradient: dE/dW = -dφ/dW / β
    grad_k = -∇_W φ(x, h_k, y, β_k) / β_k

    # Accumulate
    grads += grad_k

# 4. Average and take real part
grads = real(grads / N)

# 5. Weight update
W ← W - lr · grads
```

## Key Implementation Details

### From `fast_ep_gradN` (lines 276-307)

**Optimization for even N:**
```python
if N == 1:
    # Standard EP
    grads = ep_grad1(...)
else:
    grads = 0

    # Sample β (real, k=0)
    h ← settle(x, h_free, β=β)
    grads += dE/dW(h, β)

    # Sample -β (real, k=N/2) if N is even
    if N % 2 == 0:
        h ← settle(x, h_free, β=-β)
        grads += dE/dW(h, -β)

    # Sample complex phases if N > 2
    if N > 2:
        # Convert to complex
        W ← complex(W)
        h ← complex(h)
        x ← complex(x)

        # Sample (N+1)//2 - 1 complex phases
        for k in range(1, (N+1)//2):
            β_k = β · exp(2πik/N)
            h ← settle(x, h_free, β_k)
            grad_k = dE/dW(h, β_k)

            # to_2real_dict: Exploit symmetry
            # β_k and β_{N-k} = β_k* (conjugate) give conjugate gradients
            # real(grad_k + grad_k*) = 2·real(grad_k)
            grads += 2 · real(grad_k)

    grads /= N
```

### From `_dEdw` (lines 236-242)

Energy gradient computation:
```python
def _dEdw(params, x, h, β, y):
    # Note: dE/dW = -dφ/dW / β
    dφ_dW = ∇_W φ(x, h, y, β)  # holomorphic=True if complex
    return -dφ_dW / β
```

## PyTorch Implementation Challenges

1. **Complex Autograd**: PyTorch's autograd for complex tensors is different from JAX
   - ✅ SOLVED: Use `.real` on energy before calling `torch.autograd.grad()`
   - PyTorch requires real-valued scalar outputs for gradient computation
   - Computing d(Re(φ))/dW gives correct holomorphic gradients

2. **Complex Activation**: `tanh(complex)` works in PyTorch natively

3. **State Management**: Need to carefully convert between real and complex
   - ✅ IMPLEMENTED: Convert W and state to complex64 before holomorphic phase
   - Restore to real after computing gradients

## Implementation Status

✅ **FIXED** (2025-11-23): Proper holomorphic EP with N>1 is now implemented!

The implementation now correctly:
- Converts weights and state to complex domain after free phase
- Uses full complex β_k = β * exp(2πik/N) for all N phases
- Runs complex dynamics with complex state and beta
- Computes holomorphic gradients: dE/dW = -dφ/dW / β_k
- Averages gradients and takes real part
- Restores real weights and state

Tested with N=1, 2, 4, 8 - all working correctly!

## Sources
- [Holomorphic EP Paper](https://arxiv.org/abs/2209.00530)
- [Zenke Lab Blog](https://zenkelab.org/2022/10/holomorphic-equilibrium-propagation/)
- [JAX Implementation](https://github.com/Laborieux-Axel/holomorphic_eqprop)
