#!/usr/bin/env python3
"""
Script to fix NumPy usage in Ember ML codebase.

This script scans Python files for direct NumPy imports and usage,
and replaces them with the appropriate ops functions.
"""

import os
import re
import argparse
from typing import List, Dict, Tuple, Set, Optional

# Mapping of NumPy functions to ops equivalents
NUMPY_TO_OPS = {
    # Array creation
    "np.array": "ops.convert_to_tensor",
    "np.zeros": "ops.zeros",
    "np.ones": "ops.ones",
    "np.eye": "ops.eye",
    "np.arange": "ops.arange",
    "np.linspace": "ops.linspace",
    "np.full": "ops.full",
    "np.zeros_like": "ops.zeros_like",
    "np.ones_like": "ops.ones_like",
    
    # Math operations
    "np.add": "ops.add",
    "np.subtract": "ops.subtract",
    "np.multiply": "ops.multiply",
    "np.divide": "ops.divide",
    "np.matmul": "ops.matmul",
    "np.dot": "ops.matmul",  # Note: behavior might differ for 1D arrays
    "np.sum": "ops.sum",
    "np.mean": "ops.mean",
    "np.max": "ops.max",
    "np.min": "ops.min",
    "np.abs": "ops.abs",
    "np.exp": "ops.exp",
    "np.log": "ops.log",
    "np.sqrt": "ops.sqrt",
    "np.sin": "ops.sin",
    "np.cos": "ops.cos",
    "np.tan": "ops.tan",
    "np.tanh": "ops.tanh",
    "np.sinh": "ops.sinh",
    "np.cosh": "ops.cosh",
    "np.clip": "ops.clip",
    "np.sign": "ops.sign",
    "np.square": "ops.square",
    "np.power": "ops.pow",
    
    # Comparison operations
    "np.equal": "ops.equal",
    "np.not_equal": "ops.not_equal",
    "np.greater": "ops.greater",
    "np.greater_equal": "ops.greater_equal",
    "np.less": "ops.less",
    "np.less_equal": "ops.less_equal",
    
    # Shape operations
    "np.reshape": "ops.reshape",
    "np.transpose": "ops.transpose",
    "np.concatenate": "ops.concatenate",
    "np.stack": "ops.stack",
    "np.split": "ops.split",
    "np.squeeze": "ops.squeeze",
    "np.expand_dims": "ops.expand_dims",
    
    # Type operations
    "np.cast": "ops.cast",
    "np.float32": "ops.float32",
    "np.float64": "ops.float64",
    "np.int32": "ops.int32",
    "np.int64": "ops.int64",
    "np.bool_": "ops.bool",
}

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def check_numpy_import(file_path: str) -> Tuple[bool, List[str]]:
    """Check if NumPy is imported in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for numpy imports
    numpy_imports = []
    
    # Regular expression patterns for different import styles
    patterns = [
        r'import\s+numpy\s+as\s+(\w+)',  # import numpy as np
        r'from\s+numpy\s+import\s+(.*)',  # from numpy import ...
        r'import\s+numpy\b',  # import numpy
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        if matches:
            if pattern == r'import\s+numpy\s+as\s+(\w+)':
                # For "import numpy as np" style, capture the alias
                numpy_imports.extend(matches)
            elif pattern == r'from\s+numpy\s+import\s+(.*)':
                # For "from numpy import ..." style, capture the imported names
                for match in matches:
                    imports = [name.strip() for name in match.split(',')]
                    numpy_imports.extend(imports)
            else:
                # For "import numpy" style, add "numpy" to the list
                numpy_imports.append("numpy")
    
    return bool(numpy_imports), numpy_imports

def fix_numpy_imports(file_path: str) -> bool:
    """Fix NumPy imports in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace NumPy imports with ops imports
    updated_content = re.sub(r'import\s+numpy\s+as\s+np', 'from ember_ml import ops', content)
    updated_content = re.sub(r'from\s+numpy\s+import\s+.*', 'from ember_ml import ops', updated_content)
    updated_content = re.sub(r'import\s+numpy\b', 'from ember_ml import ops', updated_content)
    
    if content != updated_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        return True
    return False

def fix_numpy_usage(file_path: str, numpy_aliases: List[str]) -> bool:
    """Fix NumPy usage in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    updated_content = content
    
    # Replace NumPy function calls with ops equivalents
    for alias in numpy_aliases:
        for np_func, ops_func in NUMPY_TO_OPS.items():
            if alias == "np":
                # Replace np.function with ops.function
                pattern = r'\b' + re.escape(np_func) + r'\b'
                updated_content = re.sub(pattern, ops_func, updated_content)
            elif alias == "numpy":
                # Replace numpy.function with ops.function
                numpy_func = np_func.replace("np.", "numpy.")
                pattern = r'\b' + re.escape(numpy_func) + r'\b'
                updated_content = re.sub(pattern, ops_func, updated_content)
            else:
                # Replace imported function directly
                if "." in np_func:
                    func_name = np_func.split(".")[-1]
                    if alias == func_name:
                        pattern = r'\b' + re.escape(alias) + r'\b'
                        updated_content = re.sub(pattern, ops_func, updated_content)
    
    # Replace tensor.numpy() with ops.to_numpy(tensor)
    pattern = r'(\w+)\.numpy\(\)'
    updated_content = re.sub(pattern, r'ops.to_numpy(\1)', updated_content)
    
    if content != updated_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        return True
    return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix NumPy usage in Ember ML codebase.")
    parser.add_argument("path", help="Directory or file to scan")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files, just report issues")
    args = parser.parse_args()
    
    # Check if the path is a file or directory
    if os.path.isfile(args.path) and args.path.endswith('.py'):
        # Fix a single file
        has_numpy, numpy_aliases = check_numpy_import(args.path)
        if has_numpy:
            print(f"Found NumPy imports in {args.path}: {numpy_aliases}")
            if not args.dry_run:
                fixed_imports = fix_numpy_imports(args.path)
                fixed_usage = fix_numpy_usage(args.path, numpy_aliases)
                if fixed_imports or fixed_usage:
                    print(f"Fixed NumPy usage in {args.path}")
        else:
            print(f"No NumPy imports found in {args.path}")
    else:
        # Fix a directory
        python_files = find_python_files(args.path)
        for file_path in python_files:
            has_numpy, numpy_aliases = check_numpy_import(file_path)
            if has_numpy:
                print(f"Found NumPy imports in {file_path}: {numpy_aliases}")
                if not args.dry_run:
                    fixed_imports = fix_numpy_imports(file_path)
                    fixed_usage = fix_numpy_usage(file_path, numpy_aliases)
                    if fixed_imports or fixed_usage:
                        print(f"Fixed NumPy usage in {file_path}")
    
    print("\nDone!")
    print("Note: This script performs basic replacements. Manual review is still necessary.")
    print("Run emberlint.py to verify the changes: python utils/emberlint.py path/to/file.py --verbose")

if __name__ == "__main__":
    main()