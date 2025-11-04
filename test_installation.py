"""
Test script to verify OpenDPD installation and basic functionality

Run this script after installing OpenDPD to verify everything is working correctly:
    pip install -e .
    python test_installation.py
"""

import sys
import os

def test_import():
    """Test if opendpd can be imported"""
    print("Test 1: Importing opendpd...")
    try:
        import opendpd
        print(f"  ✓ Successfully imported opendpd v{opendpd.__version__}")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import opendpd: {e}")
        return False

def test_api_functions():
    """Test if API functions are accessible"""
    print("\nTest 2: Checking API functions...")
    try:
        import opendpd
        functions = ['train_pa', 'train_dpd', 'run_dpd', 'load_dataset', 'create_dataset', 'OpenDPDTrainer']
        missing = []
        for func in functions:
            if not hasattr(opendpd, func):
                missing.append(func)
        
        if missing:
            print(f"  ✗ Missing functions: {missing}")
            return False
        else:
            print(f"  ✓ All API functions are accessible: {functions}")
            return True
    except Exception as e:
        print(f"  ✗ Error checking API functions: {e}")
        return False

def test_dependencies():
    """Test if all dependencies are installed"""
    print("\nTest 3: Checking dependencies...")
    dependencies = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'rich': 'Rich'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name} installed")
        except ImportError:
            print(f"  ✗ {name} not installed")
            missing.append(name)
    
    if missing:
        print(f"\n  Missing dependencies: {', '.join(missing)}")
        print("  Install them with: pip install " + " ".join(missing).lower())
        return False
    return True

def test_dataset_loading():
    """Test if dataset loading works"""
    print("\nTest 4: Testing dataset loading...")
    try:
        from modules.data_collector import load_dataset
        
        # Check if any dataset exists
        datasets_dir = 'datasets'
        if not os.path.exists(datasets_dir):
            print(f"  ⚠ Datasets directory not found at {datasets_dir}")
            print("    Skipping dataset loading test")
            return True
        
        # Find first dataset
        available_datasets = [d for d in os.listdir(datasets_dir) 
                            if os.path.isdir(os.path.join(datasets_dir, d))]
        
        if not available_datasets:
            print("  ⚠ No datasets found in datasets/ directory")
            print("    Skipping dataset loading test")
            return True
        
        # Try to load the first dataset
        dataset_name = available_datasets[0]
        print(f"  → Trying to load dataset: {dataset_name}")
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name=dataset_name)
        print(f"  ✓ Successfully loaded {dataset_name}")
        print(f"    - Training samples: {len(X_train):,}")
        print(f"    - Validation samples: {len(X_val):,}")
        print(f"    - Test samples: {len(X_test):,}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spec_json():
    """Test if spec.json files have correct format"""
    print("\nTest 5: Checking spec.json format...")
    try:
        import json
        datasets_dir = 'datasets'
        
        if not os.path.exists(datasets_dir):
            print(f"  ⚠ Datasets directory not found")
            return True
        
        datasets = [d for d in os.listdir(datasets_dir) 
                   if os.path.isdir(os.path.join(datasets_dir, d))]
        
        if not datasets:
            print("  ⚠ No datasets found")
            return True
        
        all_valid = True
        for dataset_name in datasets:
            spec_path = os.path.join(datasets_dir, dataset_name, 'spec.json')
            if not os.path.exists(spec_path):
                print(f"  ⚠ {dataset_name}: spec.json not found")
                continue
            
            with open(spec_path, 'r') as f:
                spec = json.load(f)
            
            # Check for new fields (optional)
            has_format = 'dataset_format' in spec
            has_ratios = 'split_ratios' in spec
            
            if has_format and has_ratios:
                print(f"  ✓ {dataset_name}: spec.json has new format fields")
            else:
                print(f"  ℹ {dataset_name}: spec.json uses legacy format (still works)")
        
        return all_valid
        
    except Exception as e:
        print(f"  ✗ Error checking spec.json: {e}")
        return False

def test_project_structure():
    """Test if all necessary files exist"""
    print("\nTest 6: Checking project structure...")
    required_files = [
        'setup.py',
        'pyproject.toml',
        'MANIFEST.in',
        'opendpd/__init__.py',
        'opendpd/api.py',
        'opendpd/cli.py',
        'modules/data_collector.py',
        'arguments.py',
        'project.py',
        'main.py'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} not found")
            missing.append(file)
    
    if missing:
        print(f"\n  Missing files: {missing}")
        return False
    return True

def main():
    """Run all tests"""
    print("=" * 80)
    print("OpenDPD Installation Test")
    print("=" * 80)
    
    tests = [
        test_import,
        test_api_functions,
        test_dependencies,
        test_project_structure,
        test_dataset_loading,
        test_spec_json,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n  Unexpected error in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("\n✓ All tests passed! OpenDPD is ready to use.")
        print("\nTry running:")
        print("  python examples/api_usage_example.py")
        print("\nOr use the Python API:")
        print("  import opendpd")
        print("  opendpd.train_pa(dataset_name='DPA_200MHz', n_epochs=10)")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nIf you just installed, try:")
        print("  pip install -e .")
        return 1

if __name__ == '__main__':
    sys.exit(main())

