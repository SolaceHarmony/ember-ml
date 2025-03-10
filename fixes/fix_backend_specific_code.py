#!/usr/bin/env python3
"""
Script to fix backend-specific code in Ember ML codebase.

This script scans Python files for backend-specific imports and code
(torch, mlx, etc.) and replaces them with ops functions.
"""

import os
import re
import ast
import argparse
from typing import List, Dict, Tuple, Set, Optional, Any

class BackendSpecificVisitor(ast.NodeVisitor):
    """AST visitor to find backend-specific imports and code."""
    
    def __init__(self):
        self.backend_imports = set()
        self.backend_usage = []
        self.current_function = None
        self.current_line = 0
    
    def visit_Import(self, node):
        """Visit import statements."""
        for name in node.names:
            if name.name in ['torch', 'mlx']:
                self.backend_imports.add(f"import {name.name}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        if node.module in ['torch', 'mlx'] or node.module and node.module.startswith(('torch.', 'mlx.')):
            for name in node.names:
                self.backend_imports.add(f"from {node.module} import {name.name}")
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Track the current function being visited."""
        old_function = self.current_function
        self.current_function = node.name
        self.current_line = node.lineno
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Call(self, node):
        """Visit function calls to detect backend-specific usage."""
        if isinstance(node.func, ast.Attribute):
            # Check for backend-specific attribute access
            if isinstance(node.func.value, ast.Name) and node.func.value.id in ['torch', 'mlx']:
                location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                self.backend_usage.append({
                    'type': f"{node.func.value.id}.{node.func.attr}",
                    'backend': node.func.value.id,
                    'location': location,
                    'line': node.lineno,
                    'col_offset': node.col_offset,
                    'end_lineno': getattr(node, 'end_lineno', node.lineno),
                    'end_col_offset': getattr(node, 'end_col_offset', 0)
                })
            
            # Check for backend-specific attribute access (nested)
            if isinstance(node.func.value, ast.Attribute):
                if isinstance(node.func.value.value, ast.Name) and node.func.value.value.id in ['torch', 'mlx']:
                    location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
                    self.backend_usage.append({
                        'type': f"{node.func.value.value.id}.{node.func.value.attr}.{node.func.attr}",
                        'backend': node.func.value.value.id,
                        'location': location,
                        'line': node.lineno,
                        'col_offset': node.col_offset,
                        'end_lineno': getattr(node, 'end_lineno', node.lineno),
                        'end_col_offset': getattr(node, 'end_col_offset', 0)
                    })
        
        self.generic_visit(node)

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        # Skip backend directory
        if "/backend/" in root:
            continue
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def check_backend_specific_code(file_path: str) -> Tuple[Set[str], List[Dict]]:
    """Check for backend-specific imports and code in the file."""
    # Skip backend directory
    if "/backend/" in file_path:
        return set(), []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return set(), []
    
    # Check for backend-specific imports and code
    visitor = BackendSpecificVisitor()
    visitor.visit(tree)
    
    return visitor.backend_imports, visitor.backend_usage

def fix_backend_specific_imports(file_path: str, imports: Set[str]) -> bool:
    """Fix backend-specific imports in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    
    # Replace backend-specific imports with ops import
    new_lines = []
    added_ops_import = False
    
    for line in lines:
        skip_line = False
        
        for backend_import in imports:
            if backend_import in line:
                if not added_ops_import:
                    new_lines.append('from ember_ml import ops\n')
                    added_ops_import = True
                    modified = True
                skip_line = True
                break
        
        if not skip_line:
            new_lines.append(line)
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    
    return modified

def fix_backend_specific_code(file_path: str, usages: List[Dict]) -> bool:
    """Fix backend-specific code in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    
    # Map of common backend-specific functions to ops equivalents
    backend_to_ops = {
        # Tensor creation
        'torch.tensor': 'ops.convert_to_tensor',
        'torch.Tensor': 'ops.convert_to_tensor',
        'torch.zeros': 'ops.zeros',
        'torch.ones': 'ops.ones',
        'torch.eye': 'ops.eye',
        'torch.arange': 'ops.arange',
        'torch.linspace': 'ops.linspace',
        'torch.full': 'ops.full',
        'torch.zeros_like': 'ops.zeros_like',
        'torch.ones_like': 'ops.ones_like',
        'torch.randn': 'ops.random_normal',
        'torch.rand': 'ops.random_uniform',
        'torch.randint': 'ops.random_uniform_int',
        'torch.randperm': 'ops.random_permutation',
        
        # Math operations
        'torch.add': 'ops.add',
        'torch.sub': 'ops.subtract',
        'torch.mul': 'ops.multiply',
        'torch.div': 'ops.divide',
        'torch.matmul': 'ops.matmul',
        'torch.mm': 'ops.matmul',
        'torch.bmm': 'ops.batch_matmul',
        'torch.sum': 'ops.sum',
        'torch.mean': 'ops.mean',
        'torch.max': 'ops.max',
        'torch.min': 'ops.min',
        'torch.abs': 'ops.abs',
        'torch.exp': 'ops.exp',
        'torch.log': 'ops.log',
        'torch.sqrt': 'ops.sqrt',
        'torch.sin': 'ops.sin',
        'torch.cos': 'ops.cos',
        'torch.tan': 'ops.tan',
        'torch.tanh': 'ops.tanh',
        'torch.sinh': 'ops.sinh',
        'torch.cosh': 'ops.cosh',
        'torch.clip': 'ops.clip',
        'torch.clamp': 'ops.clip',
        'torch.sign': 'ops.sign',
        'torch.pow': 'ops.pow',
        'torch.square': 'ops.square',
        'torch.sigmoid': 'ops.sigmoid',
        'torch.relu': 'ops.relu',
        
        # Comparison operations
        'torch.eq': 'ops.equal',
        'torch.ne': 'ops.not_equal',
        'torch.gt': 'ops.greater',
        'torch.ge': 'ops.greater_equal',
        'torch.lt': 'ops.less',
        'torch.le': 'ops.less_equal',
        
        # Shape operations
        'torch.reshape': 'ops.reshape',
        'torch.transpose': 'ops.transpose',
        'torch.permute': 'ops.transpose',
        'torch.cat': 'ops.concatenate',
        'torch.concat': 'ops.concatenate',
        'torch.stack': 'ops.stack',
        'torch.split': 'ops.split',
        'torch.squeeze': 'ops.squeeze',
        'torch.unsqueeze': 'ops.expand_dims',
        
        # Type operations
        'torch.cast': 'ops.cast',
        'torch.float32': 'ops.float32',
        'torch.float64': 'ops.float64',
        'torch.int32': 'ops.int32',
        'torch.int64': 'ops.int64',
        'torch.bool': 'ops.bool',
        
        # Device operations
        'torch.device': 'ops.device',
        'torch.cuda.is_available': 'ops.is_gpu_available',
        
        # MLX equivalents
        'mlx.core.zeros': 'ops.zeros',
        'mlx.core.ones': 'ops.ones',
        'mlx.core.array': 'ops.convert_to_tensor',
        'mlx.core.matmul': 'ops.matmul',
    }
    
    # Replace backend-specific code with ops equivalents
    for usage in usages:
        backend_func = usage['type']
        if backend_func in backend_to_ops:
            ops_func = backend_to_ops[backend_func]
            pattern = re.escape(backend_func)
            content = re.sub(pattern, ops_func, content)
            modified = True
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return modified

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix backend-specific code in Ember ML codebase.")
    parser.add_argument("path", help="Directory or file to scan")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files, just report issues")
    args = parser.parse_args()
    
    # Check if the path is a file or directory
    if os.path.isfile(args.path) and args.path.endswith('.py'):
        # Skip backend directory
        if "/backend/" in args.path:
            print(f"Skipping backend file: {args.path}")
            return
        
        # Fix a single file
        imports, usages = check_backend_specific_code(args.path)
        if imports or usages:
            print(f"Found backend-specific code in {args.path}:")
            if imports:
                print(f"  Imports: {', '.join(imports)}")
            if usages:
                for usage in usages:
                    print(f"  {usage['type']} at {usage['location']}")
            
            if not args.dry_run:
                fixed_imports = fix_backend_specific_imports(args.path, imports)
                fixed_code = fix_backend_specific_code(args.path, usages)
                if fixed_imports or fixed_code:
                    print(f"Fixed backend-specific code in {args.path}")
        else:
            print(f"No backend-specific code found in {args.path}")
    else:
        # Fix a directory
        python_files = find_python_files(args.path)
        for file_path in python_files:
            imports, usages = check_backend_specific_code(file_path)
            if imports or usages:
                print(f"Found backend-specific code in {file_path}:")
                if imports:
                    print(f"  Imports: {', '.join(imports)}")
                if usages:
                    for usage in usages:
                        print(f"  {usage['type']} at {usage['location']}")
                
                if not args.dry_run:
                    fixed_imports = fix_backend_specific_imports(file_path, imports)
                    fixed_code = fix_backend_specific_code(file_path, usages)
                    if fixed_imports or fixed_code:
                        print(f"Fixed backend-specific code in {file_path}")
    
    print("\nDone!")
    print("Note: This script performs basic replacements. Manual review is still necessary.")
    print("Run emberlint.py to verify the changes: python utils/emberlint.py path/to/file.py --verbose")

if __name__ == "__main__":
    main()