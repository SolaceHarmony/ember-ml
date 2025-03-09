#!/usr/bin/env python3
"""
Run tests for the purified version of TerabyteFeatureExtractor.

This script runs the tests for the purified version of TerabyteFeatureExtractor
and reports the results.
"""

import os
import sys
import unittest
import logging
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('run_purification_tests')

def check_dependencies():
    """Check if required dependencies are installed."""
    dependencies = [
        ('numpy', 'NumPy'),
        ('pandas', 'pandas'),
        ('emberharmony', 'emberharmony')
    ]
    
    missing = []
    for module_name, display_name in dependencies:
        try:
            importlib.import_module(module_name)
            logger.info(f"{display_name} is available")
        except ImportError:
            missing.append(display_name)
            logger.warning(f"{display_name} is not available")
    
    # Check optional backends
    backends = [
        ('torch', 'PyTorch'),
        ('mlx', 'MLX')
    ]
    
    available_backends = []
    for module_name, display_name in backends:
        try:
            importlib.import_module(module_name)
            available_backends.append(display_name)
            logger.info(f"{display_name} backend is available")
        except ImportError:
            logger.warning(f"{display_name} backend is not available")
    
    if missing:
        logger.error(f"Missing required dependencies: {', '.join(missing)}")
        return False
    
    if not available_backends:
        logger.warning("No optional backends available. Tests will only run with NumPy backend.")
    
    return True

def run_tests():
    """Run the tests for the purified TerabyteFeatureExtractor."""
    # Add the current directory to the path
    sys.path.insert(0, os.path.abspath('.'))
    
    # Import the test module
    from tests.test_terabyte_feature_extractor_purified import TestTerabyteFeatureExtractorPurified
    
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTerabyteFeatureExtractorPurified)
    
    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()

def print_backend_info():
    """Print information about the available backends."""
    try:
        from ember_ml.utils import backend_utils
        backend_utils.print_backend_info()
    except ImportError:
        logger.error("Failed to import backend_utils. Make sure emberharmony is installed correctly.")

def main():
    """Main function."""
    logger.info("Running purification tests")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing required dependencies. Please install them and try again.")
        return 1
    
    # Print backend information
    print_backend_info()
    
    # Run tests
    success = run_tests()
    
    if success:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed.")
        return 1

if __name__ == '__main__':
    sys.exit(main())