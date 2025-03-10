#!/usr/bin/env python3
"""
Script to fix missing type annotations in Ember ML codebase.

This script scans Python files for missing type annotations and adds them.
"""

import os
import re
import ast
import argparse
from typing import List, Dict, Tuple, Set, Optional, Any

class MissingTypeAnnotationVisitor(ast.NodeVisitor):
    """AST visitor to find missing type annotations."""
    
    def __init__(self):
        self.missing_annotations = []
        self.current_function = None
        self.current_line = 0
    
    def visit_FunctionDef(self, node):
        """Track the current function being visited."""
        old_function = self.current_function
        self.current_function = node.name
        self.current_line = node.lineno
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_AnnAssign(self, node):
        """Visit annotated assignments."""
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        """Visit assignments to find missing type annotations."""
        # Skip class attributes (handled separately)
        if isinstance(node.targets[0], ast.Attribute):
            self.generic_visit(node)
            return
        
        # Skip tuple unpacking (complex to annotate)
        if not isinstance(node.targets[0], ast.Name):
            self.generic_visit(node)
            return
        
        # Check if this is a variable assignment at module or class level
        if self.current_function is None:
            var_name = node.targets[0].id
            location = f"line {node.lineno}"
            
            # Try to infer the type from the value
            inferred_type = self._infer_type(node.value)
            
            self.missing_annotations.append({
                'name': var_name,
                'location': location,
                'line': node.lineno,
                'col_offset': node.col_offset,
                'inferred_type': inferred_type
            })
        
        self.generic_visit(node)
    
    def _infer_type(self, node):
        """Try to infer the type of a node."""
        if isinstance(node, ast.List):
            return "List"
        elif isinstance(node, ast.Dict):
            return "Dict"
        elif isinstance(node, ast.Set):
            return "Set"
        elif isinstance(node, ast.Tuple):
            return "Tuple"
        elif isinstance(node, ast.Str):
            return "str"
        elif isinstance(node, ast.Num):
            if isinstance(node.n, int):
                return "int"
            elif isinstance(node.n, float):
                return "float"
            else:
                return "Any"
        elif isinstance(node, ast.NameConstant):
            if node.value is None:
                return "None"
            elif isinstance(node.value, bool):
                return "bool"
            else:
                return "Any"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == 'list':
                    return "List"
                elif node.func.id == 'dict':
                    return "Dict"
                elif node.func.id == 'set':
                    return "Set"
                elif node.func.id == 'tuple':
                    return "Tuple"
                elif node.func.id == 'str':
                    return "str"
                elif node.func.id == 'int':
                    return "int"
                elif node.func.id == 'float':
                    return "float"
                elif node.func.id == 'bool':
                    return "bool"
            return "Any"
        else:
            return "Any"

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory and its subdirectories."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def check_missing_type_annotations(file_path: str) -> List[Dict]:
    """Check for missing type annotations in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    
    # Check for missing type annotations
    visitor = MissingTypeAnnotationVisitor()
    visitor.visit(tree)
    
    return visitor.missing_annotations

def fix_missing_type_annotations(file_path: str, annotations: List[Dict]) -> bool:
    """Fix missing type annotations in the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    
    # Sort annotations by line number in reverse order to avoid offset issues
    annotations.sort(key=lambda x: x['line'], reverse=True)
    
    for annotation in annotations:
        line_idx = annotation['line'] - 1
        line = lines[line_idx]
        
        # Skip if the line already has a type annotation
        if ': ' in line and ' = ' in line:
            continue
        
        # Extract variable name and value
        match = re.match(r'(\s*)(\w+)\s*=\s*(.*)', line)
        if match:
            indent, var_name, value = match.groups()
            
            # Add type annotation
            inferred_type = annotation['inferred_type']
            if inferred_type == "List":
                new_line = f"{indent}{var_name}: List[Any] = {value}"
            elif inferred_type == "Dict":
                new_line = f"{indent}{var_name}: Dict[Any, Any] = {value}"
            elif inferred_type == "Set":
                new_line = f"{indent}{var_name}: Set[Any] = {value}"
            elif inferred_type == "Tuple":
                new_line = f"{indent}{var_name}: Tuple[Any, ...] = {value}"
            elif inferred_type == "None":
                new_line = f"{indent}{var_name}: Optional[Any] = {value}"
            else:
                new_line = f"{indent}{var_name}: {inferred_type} = {value}"
            
            # Check if we need to add the import
            if 'List' in new_line or 'Dict' in new_line or 'Set' in new_line or 'Tuple' in new_line or 'Optional' in new_line:
                if 'from typing import ' not in ''.join(lines):
                    imports_to_add = []
                    if 'List' in new_line:
                        imports_to_add.append('List')
                    if 'Dict' in new_line:
                        imports_to_add.append('Dict')
                    if 'Set' in new_line:
                        imports_to_add.append('Set')
                    if 'Tuple' in new_line:
                        imports_to_add.append('Tuple')
                    if 'Optional' in new_line:
                        imports_to_add.append('Optional')
                    if 'Any' in new_line:
                        imports_to_add.append('Any')
                    
                    # Add import at the top of the file
                    import_line = f"from typing import {', '.join(imports_to_add)}\n"
                    lines.insert(0, import_line)
                    # Adjust line index for the annotation we're currently processing
                    line_idx += 1
            
            if new_line != line:
                lines[line_idx] = new_line
                modified = True
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    return modified

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fix missing type annotations in Ember ML codebase.")
    parser.add_argument("path", help="Directory or file to scan")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files, just report issues")
    args = parser.parse_args()
    
    # Check if the path is a file or directory
    if os.path.isfile(args.path) and args.path.endswith('.py'):
        # Fix a single file
        annotations = check_missing_type_annotations(args.path)
        if annotations:
            print(f"Found {len(annotations)} missing type annotations in {args.path}:")
            for annotation in annotations:
                print(f"  {annotation['name']} at {annotation['location']} (inferred type: {annotation['inferred_type']})")
            
            if not args.dry_run:
                fixed = fix_missing_type_annotations(args.path, annotations)
                if fixed:
                    print(f"Fixed missing type annotations in {args.path}")
        else:
            print(f"No missing type annotations found in {args.path}")
    else:
        # Fix a directory
        python_files = find_python_files(args.path)
        for file_path in python_files:
            annotations = check_missing_type_annotations(file_path)
            if annotations:
                print(f"Found {len(annotations)} missing type annotations in {file_path}:")
                for annotation in annotations:
                    print(f"  {annotation['name']} at {annotation['location']} (inferred type: {annotation['inferred_type']})")
                
                if not args.dry_run:
                    fixed = fix_missing_type_annotations(file_path, annotations)
                    if fixed:
                        print(f"Fixed missing type annotations in {file_path}")
    
    print("\nDone!")
    print("Note: This script performs basic type inference. Manual review is still necessary.")
    print("Run emberlint.py to verify the changes: python utils/emberlint.py path/to/file.py --verbose")

if __name__ == "__main__":
    main()