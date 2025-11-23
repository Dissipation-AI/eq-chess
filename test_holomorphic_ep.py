"""
Test script for holomorphic EP implementation.

Tests that the implementation works correctly for both N=1 (standard EP)
and N>1 (holomorphic EP with complex dynamics).
"""

import torch
import sys
sys.path.insert(0, 'src')
from eqprop import LinearHolomorphicEQProp

def test_standard_ep():
    """Test standard EP with N=1."""
    print("Testing Standard EP (N=1)...")

    # Small model for testing
    model = LinearHolomorphicEQProp(
        input_size=10,
        hidden_size=20,
        output_size=5,
        weight_std=0.1
    )

    # Create dummy input and target
    x = torch.randn(10)
    target = torch.zeros(5)
    target[2] = 1.0  # One-hot target

    # Set input
    model.set_inputs(x)

    # Store initial weights
    W_before = model.W.data.clone()

    # Run one learning step with N=1
    model.learn(
        target=target,
        beta=0.1,
        T1=10,
        T2=4,
        N=1,
        lr_dynamics=0.5,
        lr_learning=0.01
    )

    # Check weights were updated
    W_after = model.W.data
    weight_change = torch.norm(W_after - W_before).item()

    print(f"  Weight change: {weight_change:.6f}")
    assert weight_change > 0, "Weights should have been updated"
    assert W_after.dtype == torch.float32, "Weights should remain real"
    print("  ✓ Standard EP works!\n")

    return True

def test_holomorphic_ep():
    """Test holomorphic EP with N=4."""
    print("Testing Holomorphic EP (N=4)...")

    # Small model for testing
    model = LinearHolomorphicEQProp(
        input_size=10,
        hidden_size=20,
        output_size=5,
        weight_std=0.1
    )

    # Create dummy input and target
    x = torch.randn(10)
    target = torch.zeros(5)
    target[3] = 1.0  # One-hot target

    # Set input
    model.set_inputs(x)

    # Store initial weights
    W_before = model.W.data.clone()

    # Run one learning step with N=4
    model.learn(
        target=target,
        beta=0.1,
        T1=10,
        T2=4,
        N=4,
        lr_dynamics=0.5,
        lr_learning=0.01
    )

    # Check weights were updated
    W_after = model.W.data
    weight_change = torch.norm(W_after - W_before).item()

    print(f"  Weight change: {weight_change:.6f}")
    assert weight_change > 0, "Weights should have been updated"
    assert W_after.dtype == torch.float32, "Weights should be real after holomorphic update"
    print("  ✓ Holomorphic EP works!\n")

    return True

def test_complex_dynamics():
    """Test that complex dynamics actually run with complex tensors."""
    print("Testing Complex Dynamics...")

    model = LinearHolomorphicEQProp(
        input_size=5,
        hidden_size=10,
        output_size=3,
        weight_std=0.1
    )

    # Set up state
    x = torch.randn(5)
    model.set_inputs(x)

    # Free phase
    model.tick(steps=5, beta=0.0)
    state_free = model.state.clone()

    # Convert to complex
    model.state = state_free.to(dtype=torch.complex64)
    model.W.data = model.W.data.to(dtype=torch.complex64)

    # Complex beta
    beta_complex = 0.1 * torch.exp(torch.tensor(1j * 3.14159/4, dtype=torch.complex64))

    # Target
    target = torch.zeros(3)
    target[1] = 1.0

    # Run tick with complex beta
    model.tick(steps=3, beta=beta_complex, target=target)

    # Check state is complex
    assert model.state.is_complex(), "State should be complex"
    assert model.W.is_complex(), "Weights should be complex during dynamics"

    print(f"  Complex state dtype: {model.state.dtype}")
    print(f"  Complex weight dtype: {model.W.dtype}")
    print("  ✓ Complex dynamics work!\n")

    return True

def test_multiple_N_values():
    """Test with different N values."""
    print("Testing Multiple N Values...")

    model = LinearHolomorphicEQProp(
        input_size=8,
        hidden_size=16,
        output_size=4,
        weight_std=0.1
    )

    x = torch.randn(8)
    target = torch.zeros(4)
    target[0] = 1.0

    for N in [1, 2, 4, 8]:
        print(f"  Testing N={N}...")
        model.set_inputs(x)
        W_before = model.W.data.clone()

        model.learn(
            target=target,
            beta=0.1,
            T1=5,
            T2=3,
            N=N,
            lr_dynamics=0.5,
            lr_learning=0.01
        )

        W_after = model.W.data
        weight_change = torch.norm(W_after - W_before).item()
        print(f"    Weight change: {weight_change:.6f}")

        assert weight_change > 0, f"Weights should update for N={N}"
        assert W_after.dtype == torch.float32, f"Weights should be real for N={N}"

    print("  ✓ All N values work!\n")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Holomorphic Equilibrium Propagation Tests")
    print("=" * 60)
    print()

    try:
        test_standard_ep()
        test_holomorphic_ep()
        test_complex_dynamics()
        test_multiple_N_values()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
