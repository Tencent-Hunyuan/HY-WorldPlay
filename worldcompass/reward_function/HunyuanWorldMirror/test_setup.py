#!/usr/bin/env python3
"""Test script to verify HunyuanWorldMirror inference-only module is properly set up.

This will check if all necessary files and imports are in place.
"""

import sys
import os


def test_imports():
    """Test if core modules can be imported."""
    print("Testing HunyuanWorldMirror inference module setup...")
    print("-" * 60)

    # Test 1: Check if main module can be imported
    try:
        from HunyuanWorldMirror import WorldMirror

        print("✓ WorldMirror class imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import WorldMirror: {e}")
        print("\nMissing dependencies detected. Please install:")
        print("  pip install torch einops gsplat huggingface_hub")
        return False


def test_structure():
    """Test if file structure is correct."""
    print("\nChecking file structure...")
    print("-" * 60)

    base_path = os.path.dirname(os.path.abspath(__file__))

    required_files = [
        "src/models/models/worldmirror.py",
        "src/models/models/visual_transformer.py",
        "src/models/models/rasterization.py",
        "src/models/heads/camera_head.py",
        "src/models/heads/dense_head.py",
        "src/utils/inference_utils.py",
        "infer.py",
    ]

    all_exist = True
    for filepath in required_files:
        full_path = os.path.join(base_path, filepath)
        if os.path.exists(full_path):
            print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath} - MISSING")
            all_exist = False

    return all_exist


def test_removed_files():
    """Test if training/demo files were removed."""
    print("\nChecking removed files...")
    print("-" * 60)

    base_path = os.path.dirname(os.path.abspath(__file__))

    should_not_exist = [
        "training",
        "demo",
        "examples",
        "app.py",
        "submodules",
    ]

    all_removed = True
    for item in should_not_exist:
        full_path = os.path.join(base_path, item)
        if not os.path.exists(full_path):
            print(f"✓ {item} removed")
        else:
            print(f"✗ {item} still exists")
            all_removed = False

    return all_removed


if __name__ == "__main__":
    print("=" * 60)
    print("HunyuanWorldMirror Inference Module Test")
    print("=" * 60)

    structure_ok = test_structure()
    removed_ok = test_removed_files()
    import_ok = test_imports()

    print("\n" + "=" * 60)
    if structure_ok and removed_ok:
        print("✓ File structure is correct")
        if import_ok:
            print("✓ All tests passed!")
            sys.exit(0)
        else:
            print("⚠ Structure OK, but missing Python dependencies")
            print(
                "  Install with: pip install torch einops gsplat huggingface_hub"
            )
            sys.exit(1)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
