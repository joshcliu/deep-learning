"""Test script to verify hierarchical probe implementation.

Run this to check that all components are properly installed and importable.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.data import MMLUDataset
        print("✓ Data module imported successfully")
    except Exception as e:
        print(f"✗ Data module import failed: {e}")
        return False

    try:
        from src.probes import BaseProbe, LinearProbe, MLPProbe, HierarchicalProbe
        print("✓ Probe modules imported successfully")
    except Exception as e:
        print(f"✗ Probe modules import failed: {e}")
        return False

    try:
        from src.models import ModelLoader, HiddenStateExtractor
        print("✓ Model modules imported successfully")
    except Exception as e:
        print(f"✗ Model modules import failed: {e}")
        return False

    try:
        from src.evaluation import CalibrationMetrics
        print("✓ Evaluation modules imported successfully")
    except Exception as e:
        print(f"✗ Evaluation modules import failed: {e}")
        return False

    try:
        from src.utils import load_config, ExperimentLogger
        print("✓ Utility modules imported successfully")
    except Exception as e:
        print(f"✗ Utility modules import failed: {e}")
        return False

    return True


def test_config():
    """Test that configuration can be loaded."""
    print("\nTesting configuration...")

    try:
        from src.utils import load_config
        config_path = project_root / "configs" / "hierarchical_probe.yaml"

        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False

        config = load_config(config_path)
        print(f"✓ Configuration loaded successfully")
        print(f"  Experiment: {config.experiment.name}")
        print(f"  Model: {config.model.name}")
        print(f"  Dataset: {config.data.dataset}")
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False


def test_probe_instantiation():
    """Test that probes can be instantiated."""
    print("\nTesting probe instantiation...")

    try:
        from src.probes import LinearProbe, MLPProbe, HierarchicalProbe

        # Test linear probe
        linear = LinearProbe(input_dim=4096, dropout=0.0, device="cpu")
        print(f"✓ LinearProbe instantiated: {linear.get_num_parameters():,} parameters")

        # Test MLP probe
        mlp = MLPProbe(input_dim=4096, hidden_dim=512, num_layers=2, device="cpu")
        print(f"✓ MLPProbe instantiated: {mlp.get_num_parameters():,} parameters")

        # Test hierarchical probe
        hierarchical = HierarchicalProbe(
            input_dim=4096,
            hidden_dim=512,
            num_layers=4,
            device="cpu"
        )
        print(f"✓ HierarchicalProbe instantiated: {hierarchical.get_num_parameters():,} parameters")

        return True
    except Exception as e:
        print(f"✗ Probe instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test that dataset can be loaded."""
    print("\nTesting dataset loading...")

    try:
        from src.data import MMLUDataset

        # Load a small subset for testing
        print("  Loading MMLU dataset (this may take a moment on first run)...")
        dataset = MMLUDataset(split="test", max_samples=10)

        print(f"✓ MMLU dataset loaded: {len(dataset)} examples")

        # Test data access
        example = dataset[0]
        print(f"  Sample question: {example['question'][:60]}...")
        print(f"  Answer: {example['answer_letter']}")
        print(f"  Subject: {example['subject']}")

        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        print("  Note: First run downloads MMLU dataset (~200MB)")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """Test that required dependencies are installed."""
    print("\nTesting dependencies...")

    required = [
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "omegaconf",
        "loguru",
        "tqdm",
    ]

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg} (MISSING)")
            missing.append(pkg)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    return True


def main():
    """Run all tests."""
    print("="*60)
    print("Hierarchical Probe Setup Verification")
    print("="*60)

    results = []

    # Run tests
    results.append(("Dependencies", test_dependencies()))
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Probe Instantiation", test_probe_instantiation()))
    results.append(("Dataset Loading", test_dataset()))

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✓ All tests passed! You're ready to run the experiment.")
        print("\nNext step:")
        print("  python experiments/hierarchical_probe.py --debug")
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
