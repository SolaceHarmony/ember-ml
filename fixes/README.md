# Ember ML Backend Purity Fixes

This directory contains scripts to help fix backend purity issues in the Ember ML codebase. These scripts are designed to be used in conjunction with the `emberlint.py` tool to identify and fix common backend purity violations.

## Available Scripts

### 1. `fix_numpy_usage.py`

Fixes direct NumPy imports and usage in frontend code.

```bash
# Dry run (report issues without modifying files)
python fixes/fix_numpy_usage.py path/to/file.py --dry-run

# Fix issues
python fixes/fix_numpy_usage.py path/to/file.py
```

**Note:** This script will skip legitimate uses of NumPy, such as:
- NumPy usage in visualization code
- Converting EmberTensor to NumPy arrays for external libraries

### 2. `fix_precision_casts.py`

Fixes precision-reducing casts (`float()`, `int()`) in frontend code.

```bash
# Dry run (report issues without modifying files)
python fixes/fix_precision_casts.py path/to/file.py --dry-run

# Fix issues
python fixes/fix_precision_casts.py path/to/file.py
```

**Note:** This script will skip special cases, such as:
- `float('inf')` for initializing variables with infinity
- Converting tensor values to Python floats for storing in dictionaries or for printing
- Using `int()` for indexing in Python lists

### 3. `fix_python_operators.py`

Fixes direct Python operators on tensors (`+`, `-`, `*`, `/`, etc.) in frontend code.

```bash
# Dry run (report issues without modifying files)
python fixes/fix_python_operators.py path/to/file.py --dry-run

# Fix issues
python fixes/fix_python_operators.py path/to/file.py
```

**Note:** This script will skip simple numeric operations, string operations, and list operations.

### 4. `fix_backend_specific_code.py`

Fixes backend-specific imports and code (torch, mlx, etc.) in frontend code.

```bash
# Dry run (report issues without modifying files)
python fixes/fix_backend_specific_code.py path/to/file.py --dry-run

# Fix issues
python fixes/fix_backend_specific_code.py path/to/file.py
```

**Note:** This script will skip files in the backend directory.

### 5. `fix_type_annotations.py`

Fixes missing type annotations in the codebase.

```bash
# Dry run (report issues without modifying files)
python fixes/fix_type_annotations.py path/to/file.py --dry-run

# Fix issues
python fixes/fix_type_annotations.py path/to/file.py
```

**Note:** This script performs basic type inference and may not always infer the correct type. Manual review is necessary.

## Recommended Workflow

1. Run `emberlint.py` to identify backend purity issues:
   ```bash
   python utils/emberlint.py path/to/file.py --verbose
   ```

2. Run the appropriate fix script in dry-run mode to see what changes would be made:
   ```bash
   python fixes/fix_numpy_usage.py path/to/file.py --dry-run
   ```

3. If the changes look good, run the fix script without the `--dry-run` flag:
   ```bash
   python fixes/fix_numpy_usage.py path/to/file.py
   ```

4. Run `emberlint.py` again to verify that the issues have been fixed:
   ```bash
   python utils/emberlint.py path/to/file.py --verbose
   ```

5. Manually review the changes to ensure they are correct.

## Important Notes

- These scripts perform basic replacements and may not catch all issues or may make incorrect replacements in some cases. Always manually review the changes.
- Some backend purity issues may require more complex fixes that cannot be automated. In these cases, manual intervention is necessary.
- Always run `emberlint.py` after making changes to verify that the issues have been fixed.
- Consider running the scripts on a small subset of files first to ensure they work as expected before running them on the entire codebase.