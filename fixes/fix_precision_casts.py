#!/usr/bin/env python3
"""
Script to fix precision-reducing casts in Ember ML codebase.

This script scans Python files for direct float() and int() casts,
and replaces them with ops.cast() with appropriate dtype.
"""

import os
import re
import ast
import argparse
from typing import List, Dict, Tuple, Set, Optional, Any

class PrecisionCastVisitor(ast.NodeVisitor):
    """AST visitor to find precision-reducing casts."""
    
    def __init__(self):
        self.precision_casts = []
        self.current_function = None
        self.current_line = 0
    
    def visit_FunctionDef(self, node):
        """Track the current function being visited."""
        old_function = self.current_function
        self.current_function = node.name
        self.current_line = node.lineno
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_Call(self, node):
        """Visit function calls to detect precision-reducing casts."""
        # Check for float(), int() casts
        if isinstance(node.func, ast.Name) and node.func.id in ('float', 'int'):
            location = f"{self.current_function}:{node.lineno}" if self.current_function else f"line {node.lineno}"
            self.precision_casts.append({
                'type': node.func.id,
                'location': location,
                'line': node.lineno,
                'col_offset': node.col_offset
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

def check_precision_casts(file_path: str) -> List[Dict]:
    """Check for precision-reducing casts in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    
    # Check for precision-reducing casts
    visitor = PrecisionCastVisitor()
    visitor.visit(tree)
    
    return visitor.precision_casts
def fix_precision_casts(file_path: str, casts: List[Dict]) -> bool:
    """Fix precision-reducing casts in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    
    # Sort casts by line number in reverse order to avoid offset issues
    casts.sort(key=lambda x: x['line'], reverse=True)
    
    for cast in casts:
        line_idx = cast['line'] - 1
        line = lines[line_idx]
        
        if cast['type'] == 'float':
            # Skip special cases
            if "float('inf')" in line or 'float("inf")' in line:
                continue
                
            # Skip cases where we're converting tensor values to Python floats
            if "K.to_numpy" in line:
                continue
                
            # Replace float() with ops.cast(..., ops.float32)
            pattern = r'float\((.*?)\)'
            replacement = r'ops.cast(\1, ops.float32)'
            new_line = re.sub(pattern, replacement, line)
            
            # Check if we need to add the import
            if 'from ember_ml import ops' not in ''.join(lines):
                lines.insert(0, 'from ember_ml import ops\n')
                # Adjust line index for the cast we're currently processing
                line_idx += 1
            
            if new_line != line:
                lines[line_idx] = new_line
                modified = True
        
        elif cast['type'] == 'int':
            # Skip cases where int() is used for indexing
            if "[int(" in line or "].int(" in line:
                continue
                
            # Replace int() with ops.cast(..., ops.int32)
            pattern = r'int\((.*?)\)'
            replacement = r'ops.cast(\1, ops.int32)'
            new_line = re.sub(pattern, replacement, line)
            
            # Check if we need to add the import
            if 'from ember_ml import ops' not in ''.join(lines):
                lines.insert(0, 'from ember_ml import ops\n')
                # Adjust line index for the cast we're currently processing
                line_idx += 1
            
            if new_line != line:
                lines[line_idx] = new_line
                modified = True
                modified = True
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    return modified

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix precision-reducing casts in Ember ML codebase.")
    parser.add_argument("path", help="Directory or file to scan")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files, just report issues")
    args = parser.parse_args()
    
    # Check if the path is a file or directory
    if os.path.isfile(args.path) and args.path.endswith('.py'):
        # Fix a single file
        casts = check_precision_casts(args.path)
        if casts:
            print(f"Found {len(casts)} precision-reducing casts in {args.path}:")
            for cast in casts:
                print(f"  {cast['type']}() at {cast['location']}")
            
            if not args.dry_run:
                fixed = fix_precision_casts(args.path, casts)
                if fixed:
                    print(f"Fixed precision-reducing casts in {args.path}")
        else:
            print(f"No precision-reducing casts found in {args.path}")
    else:
        # Fix a directory
        python_files = find_python_files(args.path)
        for file_path in python_files:
            casts = check_precision_casts(file_path)
            if casts:
                print(f"Found {len(casts)} precision-reducing casts in {file_path}:")
                for cast in casts:
                    print(f"  {cast['type']}() at {cast['location']}")
                
                if not args.dry_run:
                    fixed = fix_precision_casts(file_path, casts)
                    if fixed:
                        print(f"Fixed precision-reducing casts in {file_path}")
    
    print("\nDone!")
    print("Note: This script performs basic replacements. Manual review is still necessary.")
    print("Run emberlint.py to verify the changes: python utils/emberlint.py path/to/file.py --verbose")

if __name__ == "__main__":
    main()