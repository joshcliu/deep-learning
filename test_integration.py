"""Comprehensive integration test to verify hierarchical probe works end-to-end."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

print("=" * 70)
print("HIERARCHICAL PROBE - INTEGRATION TEST")
print("=" * 70)

# Test 1: Imports
print("\n[1/6] Testing imports...")
try:
    from src.probes import BaseProbe, LinearProbe, HierarchicalProbe
    print("✓ Successfully imported all probe classes")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create instances
print("\n[2/6] Testing instantiation...")
try:
    linear = LinearProbe(input_dim=4096)
    hierarchical = HierarchicalProbe(input_dim=4096, hidden_dim=512, num_layers=3)
    print(f"✓ LinearProbe: {linear.get_model_info()['num_parameters']:,} parameters")
    print(f"✓ HierarchicalProbe: {hierarchical.get_model_info()['num_parameters']:,} parameters")
except Exception as e:
    print(f"✗ Instantiation failed: {e}")
    sys.exit(1)

# Test 3: Forward pass
print("\n[3/6] Testing forward pass...")
try:
    batch_size = 8
    input_dim = 4096
    X = torch.randn(batch_size, input_dim)

    with torch.no_grad():
        linear_out = linear(X)
        hierarchical_out = hierarchical(X)

    print(f"✓ LinearProbe: {X.shape} → {linear_out.shape}")
    print(f"✓ HierarchicalProbe: {X.shape} → {hierarchical_out.shape}")

    # Check output ranges
    assert linear_out.min() >= 0 and linear_out.max() <= 1, "Linear output not in [0,1]"
    assert hierarchical_out.min() >= 0 and hierarchical_out.max() <= 1, "Hierarchical output not in [0,1]"
    print("✓ All outputs in valid range [0, 1]")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Predict method
print("\n[4/6] Testing predict method...")
try:
    X_numpy = np.random.randn(20, 4096).astype(np.float32)

    linear_preds = linear.predict(X_numpy, batch_size=8)
    hierarchical_preds = hierarchical.predict(X_numpy, batch_size=8)

    print(f"✓ LinearProbe predictions: {linear_preds.shape}")
    print(f"✓ HierarchicalProbe predictions: {hierarchical_preds.shape}")
    print(f"  Linear range: [{linear_preds.min():.3f}, {linear_preds.max():.3f}]")
    print(f"  Hierarchical range: [{hierarchical_preds.min():.3f}, {hierarchical_preds.max():.3f}]")
except Exception as e:
    print(f"✗ Predict method failed: {e}")
    sys.exit(1)

# Test 5: Training (minimal)
print("\n[5/6] Testing training...")
try:
    # Create small training dataset
    n_train = 100
    n_val = 30
    X_train = np.random.randn(n_train, 4096).astype(np.float32)
    y_train = np.random.randint(0, 2, n_train).astype(np.float32)
    X_val = np.random.randn(n_val, 4096).astype(np.float32)
    y_val = np.random.randint(0, 2, n_val).astype(np.float32)

    print("  Training LinearProbe...")
    linear_history = linear.fit(
        X_train, y_train, X_val, y_val,
        num_epochs=5, batch_size=32, verbose=False
    )
    print(f"  ✓ Final train loss: {linear_history['train_loss'][-1]:.4f}")
    print(f"  ✓ Final val loss: {linear_history['val_loss'][-1]:.4f}")

    print("  Training HierarchicalProbe...")
    hierarchical_history = hierarchical.fit(
        X_train, y_train, X_val, y_val,
        num_epochs=5, batch_size=32, verbose=False
    )
    print(f"  ✓ Final train loss: {hierarchical_history['train_loss'][-1]:.4f}")
    print(f"  ✓ Final val loss: {hierarchical_history['val_loss'][-1]:.4f}")

except Exception as e:
    print(f"✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Evaluation after training
print("\n[6/6] Testing post-training predictions...")
try:
    X_test = np.random.randn(50, 4096).astype(np.float32)

    linear_test_preds = linear.predict(X_test)
    hierarchical_test_preds = hierarchical.predict(X_test)

    print(f"✓ LinearProbe test predictions: {linear_test_preds.shape}")
    print(f"✓ HierarchicalProbe test predictions: {hierarchical_test_preds.shape}")

    # Check that predictions are reasonable
    print(f"  Linear - Mean: {linear_test_preds.mean():.3f}, Std: {linear_test_preds.std():.3f}")
    print(f"  Hierarchical - Mean: {hierarchical_test_preds.mean():.3f}, Std: {hierarchical_test_preds.std():.3f}")

except Exception as e:
    print(f"✗ Post-training prediction failed: {e}")
    sys.exit(1)

# Success!
print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nYour hierarchical probe implementation is working correctly!")
print("It can:")
print("  - Be imported and instantiated")
print("  - Process batches of hidden states")
print("  - Train on labeled data")
print("  - Make predictions on new data")
print("\nNext steps:")
print("  1. Extract hidden states from a real LLM (e.g., Llama-3.1-8B)")
print("  2. Train probes on correctness labels")
print("  3. Evaluate calibration metrics (ECE, Brier score, AUROC)")
print("=" * 70)
