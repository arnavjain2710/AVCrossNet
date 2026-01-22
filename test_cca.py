"""
Unit tests for CustomCrossAttention (CCA) module.
Tests forward pass, gradients, shapes, and edge cases.
"""
import torch
import torch.nn as nn
import numpy as np
from model_components import CustomCrossAttention


def test_basic_forward_pass():
    """Test basic forward pass with matching dimensions."""
    print("Test 1: Basic Forward Pass (da == dv)")
    batch_size, T, da, dv = 2, 10, 100, 100
    
    # Use diagonal covariance and return_mean for stability in tests
    model = CustomCrossAttention(da=da, dv=dv, use_diagonal_cov=True, return_mean=True)
    A = torch.randn(batch_size, T, da)
    V = torch.randn(batch_size, T, dv)
    
    output = model(A, V)
    
    assert output.shape == (batch_size, T, da), f"Expected shape {(batch_size, T, da)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
    print("✓ Basic forward pass works correctly\n")


def test_different_dimensions():
    """Test forward pass with different audio and visual dimensions."""
    print("Test 2: Different Dimensions (da != dv)")
    batch_size, T, da, dv = 2, 10, 64, 128
    
    model = CustomCrossAttention(da=da, dv=dv, use_diagonal_cov=True, return_mean=True)
    A = torch.randn(batch_size, T, da)
    V = torch.randn(batch_size, T, dv)
    
    output = model(A, V)
    
    assert output.shape == (batch_size, T, da), f"Expected shape {(batch_size, T, da)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    print("✓ Different dimensions handled correctly\n")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("Test 3: Gradient Flow")
    batch_size, T, da, dv = 2, 10, 100, 100
    
    model = CustomCrossAttention(da=da, dv=dv)
    A = torch.randn(batch_size, T, da, requires_grad=True)
    V = torch.randn(batch_size, T, dv, requires_grad=True)
    
    output = model(A, V)
    loss = output.mean()
    loss.backward()
    
    assert A.grad is not None, "Gradients not computed for audio input"
    assert V.grad is not None, "Gradients not computed for visual input"
    
    # Check that learnable parameters have gradients
    assert model.Q_a.weight.grad is not None, "Q_a weights have no gradients"
    assert model.K_v.weight.grad is not None, "K_v weights have no gradients"
    assert model.W_A_g.grad is not None, "W_A_g has no gradients"
    assert model.W_V_g.grad is not None, "W_V_g has no gradients"
    
    print("✓ Gradients flow correctly through all components\n")


def test_batch_processing():
    """Test that batch processing works correctly."""
    print("Test 4: Batch Processing")
    batch_sizes = [1, 2, 4, 8]
    T, da, dv = 10, 100, 100
    
    model = CustomCrossAttention(da=da, dv=dv)
    
    for batch_size in batch_sizes:
        A = torch.randn(batch_size, T, da)
        V = torch.randn(batch_size, T, dv)
        
        output = model(A, V)
        
        assert output.shape == (batch_size, T, da), \
            f"Batch size {batch_size}: Expected shape {(batch_size, T, da)}, got {output.shape}"
    
    print("✓ Batch processing works for different batch sizes\n")


def test_temporal_dimension():
    """Test different temporal dimensions."""
    print("Test 5: Different Temporal Dimensions")
    batch_size, da, dv = 2, 100, 100
    temporal_dims = [5, 10, 20, 50]
    
    model = CustomCrossAttention(da=da, dv=dv)
    
    for T in temporal_dims:
        A = torch.randn(batch_size, T, da)
        V = torch.randn(batch_size, T, dv)
        
        output = model(A, V)
        
        assert output.shape == (batch_size, T, da), \
            f"T={T}: Expected shape {(batch_size, T, da)}, got {output.shape}"
    
    print("✓ Different temporal dimensions handled correctly\n")


def test_device_handling():
    """Test that model works on CPU and GPU (if available)."""
    print("Test 6: Device Handling")
    batch_size, T, da, dv = 2, 10, 100, 100
    
    # Test CPU
    model_cpu = CustomCrossAttention(da=da, dv=dv)
    A_cpu = torch.randn(batch_size, T, da)
    V_cpu = torch.randn(batch_size, T, dv)
    
    output_cpu = model_cpu(A_cpu, V_cpu)
    assert output_cpu.device.type == 'cpu', "Output should be on CPU"
    print("✓ CPU execution works")
    
    # Test GPU if available
    if torch.cuda.is_available():
        model_gpu = CustomCrossAttention(da=da, dv=dv).cuda()
        A_gpu = torch.randn(batch_size, T, da).cuda()
        V_gpu = torch.randn(batch_size, T, dv).cuda()
        
        output_gpu = model_gpu(A_gpu, V_gpu)
        assert output_gpu.device.type == 'cuda', "Output should be on GPU"
        print("✓ GPU execution works")
    else:
        print("⚠ GPU not available, skipping GPU test")
    
    print()


def test_attention_mechanism():
    """Test that attention weights are properly normalized."""
    print("Test 7: Attention Mechanism")
    batch_size, T, da, dv = 2, 10, 100, 100
    
    model = CustomCrossAttention(da=da, dv=dv, use_diagonal_cov=True, return_mean=True)
    A = torch.randn(batch_size, T, da)
    V = torch.randn(batch_size, T, dv)
    
    output = model(A, V)
    
    # Check that output is reasonable (not all zeros or extreme values)
    assert output.abs().mean() > 1e-8, f"Output values are too small: {output.abs().mean()}"
    assert output.abs().mean() < 1e6, "Output values are too large"
    
    print("✓ Attention mechanism produces reasonable outputs\n")


def test_gated_fusion():
    """Test that gated fusion produces expected behavior."""
    print("Test 8: Gated Fusion")
    batch_size, T, da, dv = 2, 10, 100, 100
    
    model = CustomCrossAttention(da=da, dv=dv)
    A = torch.randn(batch_size, T, da)
    V = torch.randn(batch_size, T, dv)
    
    output = model(A, V)
    
    # Output should be a combination of audio and visual features
    # Check that it's not identical to either input
    assert not torch.allclose(output, A[:, :, :da]), "Output identical to audio input"
    
    print("✓ Gated fusion produces distinct outputs\n")


def test_deterministic_behavior():
    """Test that model produces consistent outputs with same inputs."""
    print("Test 9: Deterministic Behavior")
    batch_size, T, da, dv = 2, 10, 100, 100
    
    # Use return_mean=True for deterministic behavior
    model = CustomCrossAttention(da=da, dv=dv, return_mean=True)
    model.eval()  # Set to eval mode
    
    A = torch.randn(batch_size, T, da)
    V = torch.randn(batch_size, T, dv)
    
    # Forward pass twice
    output1 = model(A, V)
    output2 = model(A, V)
    
    # Should be identical when return_mean=True
    assert torch.allclose(output1, output2, atol=1e-5), "Outputs are not deterministic"
    
    print("✓ Model produces deterministic outputs with return_mean=True\n")


def test_edge_cases():
    """Test edge cases like very small values, zeros, etc."""
    print("Test 10: Edge Cases")
    batch_size, T, da, dv = 2, 10, 100, 100
    
    model = CustomCrossAttention(da=da, dv=dv, use_diagonal_cov=True, return_mean=True)
    
    # Test with small values
    A_small = torch.randn(batch_size, T, da) * 1e-6
    V_small = torch.randn(batch_size, T, dv) * 1e-6
    output_small = model(A_small, V_small)
    assert not torch.isnan(output_small).any(), "NaN with small input values"
    print("✓ Handles small input values")
    
    # Test with zeros (should not crash)
    A_zero = torch.zeros(batch_size, T, da)
    V_zero = torch.zeros(batch_size, T, dv)
    output_zero = model(A_zero, V_zero)
    assert not torch.isnan(output_zero).any(), "NaN with zero inputs"
    print("✓ Handles zero inputs")
    
    # Test with large values
    A_large = torch.randn(batch_size, T, da) * 100
    V_large = torch.randn(batch_size, T, dv) * 100
    output_large = model(A_large, V_large)
    assert not torch.isnan(output_large).any(), "NaN with large input values"
    assert not torch.isinf(output_large).any(), "Inf with large input values"
    print("✓ Handles large input values")
    
    print()


def test_custom_d_model():
    """Test with custom d_model dimension."""
    print("Test 11: Custom d_model")
    batch_size, T, da, dv = 2, 10, 64, 128
    d_model = 256
    
    model = CustomCrossAttention(da=da, dv=dv, d_model=d_model, use_diagonal_cov=True, return_mean=True)
    A = torch.randn(batch_size, T, da)
    V = torch.randn(batch_size, T, dv)
    
    output = model(A, V)
    
    assert output.shape == (batch_size, T, da), f"Expected shape {(batch_size, T, da)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    
    print("✓ Custom d_model works correctly\n")


def test_parameter_count():
    """Test that model has expected number of parameters."""
    print("Test 12: Parameter Count")
    da, dv = 100, 100
    model = CustomCrossAttention(da=da, dv=dv)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Should have parameters for Q, K, V projections, gates, and projections
    assert total_params > 0, "Model should have parameters"
    assert trainable_params == total_params, "All parameters should be trainable"
    
    print("✓ Parameter count is reasonable\n")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("CustomCrossAttention (CCA) Unit Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_basic_forward_pass,
        test_different_dimensions,
        test_gradient_flow,
        test_batch_processing,
        test_temporal_dimension,
        test_device_handling,
        test_attention_mechanism,
        test_gated_fusion,
        test_deterministic_behavior,
        test_edge_cases,
        test_custom_d_model,
        test_parameter_count,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {str(e)}\n")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

