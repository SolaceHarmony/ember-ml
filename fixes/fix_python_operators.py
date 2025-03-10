#!/usr/bin/env python3
"""
Script to fix direct Python operators on tensors in Ember ML codebase.

This script scans Python files for direct Python operators on tensors
(+, -, *, /, etc.) and replaces them with ops functions.
"""

import os
import re
import ast
import argparse
from typing import List, Dict, Tuple, Set, Optional, Any

class PythonOperatorVisitor(ast.NodeVisitor):
    """AST visitor to find direct Python operators on tensors."""
    
    def __init__(self):
        self.python_operators = []
        self.current_function = None
        self.current_line = 0
    
    def visit_FunctionDef(self, node):
        """Track the current function being visited."""
        old_function = self.current_function
        self.current_function = node.name
        self.current_line = node.lineno
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_BinOp(self, node):
        """Visit binary operations to detect Python operators."""
        # Map AST operator types to their string representations
        op_map = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '//',
            ast.Mod: '%',
            ast.Pow: '**',
            ast.MatMult: '@',
        }
        
        # Check if this is a Python operator we want to detect
        op_type = type(node.op)
        if op_type in op_map:
            location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
            self.python_operators.append({
                'type': op_map[op_type],
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
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def check_python_operators(file_path: str) -> List[Dict]:
    """Check for direct Python operators in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    
    # Check for Python operators
    visitor = PythonOperatorVisitor()
    visitor.visit(tree)
    
    return visitor.python_operators

def fix_python_operators(file_path: str, operators: List[Dict]) -> bool:
    """Fix direct Python operators in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Sort operators by position in reverse order to avoid offset issues
    operators.sort(key=lambda x: (x['line'], x['col_offset']), reverse=True)
    
    # Get the lines of the file
    lines = content.split('\n')
    
    modified = False
    
    # Check if we need to add the import
    if operators and 'from ember_ml import ops' not in content:
        lines.insert(0, 'from ember_ml import ops')
        modified = True
    
    # Map of Python operators to ops functions
    op_map = {
        '+': 'ops.add',
        '-': 'ops.subtract',
        '*': 'ops.multiply',
        '/': 'ops.divide',
        '//': 'ops.floor_divide',
        '%': 'ops.mod',
        '**': 'ops.pow',
        '@': 'ops.matmul',
    }
    
    # Process each operator
    for op in operators:
        line_idx = op['line'] - 1
        line = lines[line_idx]
        
        # Skip if this is a simple numeric operation (e.g., 1 + 2)
        # This is a simple heuristic and may not catch all cases
        if re.search(r'\b\d+\s*' + re.escape(op['type']) + r'\s*\d+\b', line):
            continue
        
        # Skip if this is a string operation (e.g., "a" + "b")
        if re.search(r'["\'][^"\']*["\']' + re.escape(op['type']) + r'["\'][^"\']*["\']', line):
            continue
        
        # Skip if this is a list operation (e.g., [1, 2] + [3, 4])
        if re.search(r'\[[^\]]*\]' + re.escape(op['type']) + r'\[[^\]]*\]', line):
            continue
        
        # Try to find the binary operation in the line
        # This is a simple approach and may not work for complex expressions
        # A more robust approach would be to use the AST to get the exact source code
        pattern = r'([^=+\-*/%@]+)' + re.escape(op['type']) + r'([^=+\-*/%@]+)'
        match = re.search(pattern, line)
        
        if match:
            left = match.group(1).strip()
            right = match.group(2).strip()
            
            # Replace with ops function
            replacement = f"{op_map[op['type']]}({left}, {right})"
            new_line = line.replace(match.group(0), replacement)
            
            if new_line != line:
                lines[line_idx] = new_line
                modified = True
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    return modified

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix direct Python operators in Ember ML codebase.")
    parser.add_argument("path", help="Directory or file to scan")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files, just report issues")
    args = parser.parse_args()
    
    # Check if the path is a file or directory
    if os.path.isfile(args.path) and args.path.endswith('.py'):
        # Fix a single file
        operators = check_python_operators(args.path)
        if operators:
            print(f"Found {len(operators)} direct Python operators in {args.path}:")
            for op in operators:
                print(f"  {op['type']} operator at {op['location']}")
            
            if not args.dry_run:
                fixed = fix_python_operators(args.path, operators)
                if fixed:
                    print(f"Fixed direct Python operators in {args.path}")
        else:
            print(f"No direct Python operators found in {args.path}")
    else:
        # Fix a directory
        python_files = find_python_files(args.path)
        for file_path in python_files:
            operators = check_python_operators(file_path)
            if operators:
                print(f"Found {len(operators)} direct Python operators in {file_path}:")
                for op in operators:
                    print(f"  {op['type']} operator at {op['location']}")
                
                if not args.dry_run:
                    fixed = fix_python_operators(file_path, operators)
                    if fixed:
                        print(f"Fixed direct Python operators in {file_path}")
    
    print("\nDone!")
    print("Note: This script performs basic replacements. Manual review is still necessary.")
    print("Run emberlint.py to verify the changes: python utils/emberlint.py path/to/file.py --verbose")

if __name__ == "__main__":
    main()