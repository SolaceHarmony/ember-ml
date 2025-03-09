#!/usr/bin/env python3
"""
Test script for the emberharmony library.
"""

import sys
import os

def test_imports():
    """Test importing the main modules."""
    try:
        import ember_ml
        print(f"Successfully imported emberharmony version {ember_ml.__version__}")
        
        # Test importing core modules
        from ember_ml import core
        print("Successfully imported emberharmony.core")
        
        from ember_ml import wave
        print("Successfully imported emberharmony.wave")
        
        from ember_ml import attention
        print("Successfully imported emberharmony.attention")
        
        from ember_ml import nn
        print("Successfully imported emberharmony.nn")
        
        from ember_ml import models
        print("Successfully imported emberharmony.models")
        
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def test_spherical_ltc():
    """Test the SphericalLTCChain class."""
    try:
        from ember_ml.core import SphericalLTCConfig, SphericalLTCChain
        
        # Create a config
        config = SphericalLTCConfig(
            tau=1.0,
            gleak=0.5,
            dt=0.01
        )
        
        # Create a chain
        chain = SphericalLTCChain(num_neurons=3, base_tau_or_config=config)
        
        print(f"Successfully created SphericalLTCChain with {chain.num_neurons} neurons")
        return True
    except Exception as e:
        print(f"Error testing SphericalLTCChain: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing emberharmony library...")
    
    import_success = test_imports()
    if not import_success:
        print("Import tests failed")
        return 1
    
    spherical_ltc_success = test_spherical_ltc()
    if not spherical_ltc_success:
        print("SphericalLTCChain tests failed")
        return 1
    
    print("All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
