# Documentation Reorganization Plan

## Overview

This document outlines a plan for reorganizing the emberharmony documentation to centralize it in a structured docs folder. Currently, documentation is scattered throughout the project, making it difficult to find and maintain.

## Current State

Documentation in the emberharmony project is currently distributed across various locations:

1. **Project Root**: README.md, NOTEBOOK_ISSUE_SUMMARY.md, etc.
2. **Subdirectories**: README.md files in various subdirectories
3. **Code Comments**: Docstrings and inline comments
4. **Jupyter Notebooks**: Documentation embedded in notebooks

This distributed approach has several drawbacks:

1. **Poor Discoverability**: It's difficult to find specific documentation
2. **Inconsistent Formatting**: Different documentation uses different styles
3. **Duplication**: The same information may be repeated in multiple places
4. **Maintenance Challenges**: Updates require modifying files in different locations

## Proposed Structure

We propose reorganizing the documentation into a centralized, structured docs folder:

```
emberharmony/
├── docs/
│   ├── index.md                      # Main documentation entry point
│   ├── architecture/                 # Architectural documentation
│   │   ├── index.md                  # Architecture overview
│   │   ├── backend_system.md         # Backend abstraction system
│   │   ├── data_processing.md        # Data processing architecture
│   │   ├── emberharmony_purification_plan.md
│   │   ├── backend_purification_implementation.md
│   │   ├── bigquery_streaming_implementation.md
│   │   └── notebook_tools_organization.md
│   ├── api/                          # API documentation
│   │   ├── index.md                  # API overview
│   │   ├── ops.md                    # Operations API
│   │   ├── features.md               # Feature extraction API
│   │   └── ...
│   ├── tutorials/                    # Step-by-step tutorials
│   │   ├── index.md                  # Tutorials overview
│   │   ├── getting_started.md        # Getting started guide
│   │   ├── bigquery_integration.md   # BigQuery integration tutorial
│   │   └── feature_extraction.md     # Feature extraction tutorial
│   ├── examples/                     # Code examples
│   │   ├── index.md                  # Examples overview
│   │   ├── basic_usage.md            # Basic usage examples
│   │   └── advanced_usage.md         # Advanced usage examples
│   ├── troubleshooting/              # Troubleshooting guides
│   │   ├── index.md                  # Troubleshooting overview
│   │   ├── common_issues.md          # Common issues and solutions
│   │   └── notebook_issues.md        # Notebook-specific issues
│   └── development/                  # Development guides
│       ├── index.md                  # Development overview
│       ├── contributing.md           # Contribution guidelines
│       ├── code_style.md             # Code style guidelines
│       └── testing.md                # Testing guidelines
```

This structure provides several benefits:

1. **Centralized Location**: All documentation is in one place
2. **Logical Organization**: Documentation is organized by type and purpose
3. **Consistent Formatting**: All documentation follows the same format
4. **Easy Navigation**: Index files provide navigation within sections
5. **Reduced Duplication**: Information is organized to minimize repetition

## Implementation Plan

### Phase 1: Create Directory Structure

1. Create the necessary directories:
   ```bash
   mkdir -p emberharmony/docs/{architecture,api,tutorials,examples,troubleshooting,development}
   ```

2. Create index files for each section:
   ```bash
   for dir in emberharmony/docs/*; do
     if [ -d "$dir" ]; then
       touch "$dir/index.md"
     fi
   done
   ```

### Phase 2: Migrate Existing Documentation

1. Move architecture documentation:
   ```bash
   # Already created in previous steps
   # emberharmony_purification_plan.md
   # backend_purification_implementation.md
   # bigquery_streaming_implementation.md
   # notebook_tools_organization.md
   ```

2. Move notebook-related documentation:
   ```bash
   mv NOTEBOOK_ISSUE_SUMMARY.md emberharmony/docs/troubleshooting/notebook_issues.md
   mv FIXED_NOTEBOOK_CELL.md emberharmony/docs/troubleshooting/fixed_notebook_cell.md
   mv FINAL_SUMMARY.md emberharmony/docs/troubleshooting/final_summary.md
   ```

3. Create API documentation from docstrings:
   ```bash
   # Use a tool like Sphinx or MkDocs to generate API documentation
   # For example, with Sphinx:
   sphinx-apidoc -o emberharmony/docs/api emberharmony
   ```

4. Create tutorial documentation:
   ```bash
   # Create basic tutorials
   touch emberharmony/docs/tutorials/getting_started.md
   touch emberharmony/docs/tutorials/bigquery_integration.md
   touch emberharmony/docs/tutorials/feature_extraction.md
   ```

5. Extract examples from notebooks:
   ```bash
   # Use a tool like nbconvert to extract examples from notebooks
   jupyter nbconvert --to markdown notebooks/*.ipynb --output-dir emberharmony/docs/examples/notebooks
   ```

### Phase 3: Create Navigation and Index Files

1. Create main index file:
   ```markdown
   # emberharmony Documentation

   Welcome to the emberharmony documentation. emberharmony is a library for efficient feature extraction and processing of terabyte-scale datasets.

   ## Documentation Sections

   - [Architecture](architecture/index.md): Architectural documentation
   - [API Reference](api/index.md): API documentation
   - [Tutorials](tutorials/index.md): Step-by-step tutorials
   - [Examples](examples/index.md): Code examples
   - [Troubleshooting](troubleshooting/index.md): Troubleshooting guides
   - [Development](development/index.md): Development guides

   ## Quick Start

   ```python
   import emberharmony as eh
   from emberharmony import ops

   # Create a feature extractor
   extractor = eh.features.TerabyteFeatureExtractor(
       project_id="your-project-id",
       location="US"
   )

   # Extract features
   result = extractor.prepare_data(
       table_id="your-dataset.your-table",
       target_column="your-target-column"
   )
   ```

   For more information, see the [Getting Started](tutorials/getting_started.md) guide.
   ```

2. Create section index files:
   ```markdown
   # Architecture Documentation

   This section contains architectural documentation for emberharmony.

   ## Contents

   - [Backend System](backend_system.md): The backend abstraction system
   - [Data Processing](data_processing.md): Data processing architecture
   - [Purification Plan](emberharmony_purification_plan.md): Plan for purifying emberharmony
   - [Backend Implementation](backend_purification_implementation.md): Implementation guide for backend purification
   - [BigQuery Streaming](bigquery_streaming_implementation.md): Implementation guide for BigQuery streaming
   - [Notebook Tools](notebook_tools_organization.md): Organization of notebook simulation tools
   ```

### Phase 4: Update References and Links

1. Update README.md to point to the new documentation:
   ```markdown
   # emberharmony

   A library for efficient feature extraction and processing of terabyte-scale datasets.

   ## Documentation

   For full documentation, see the [docs](docs/index.md) directory.

   ## Quick Start

   ```python
   import emberharmony as eh
   from emberharmony import ops

   # Create a feature extractor
   extractor = eh.features.TerabyteFeatureExtractor(
       project_id="your-project-id",
       location="US"
   )

   # Extract features
   result = extractor.prepare_data(
       table_id="your-dataset.your-table",
       target_column="your-target-column"
   )
   ```

   For more information, see the [Getting Started](docs/tutorials/getting_started.md) guide.
   ```

2. Update links in code comments:
   ```python
   """
   Feature extractor optimized for terabyte-scale BigQuery tables.
   
   For more information, see the documentation:
   https://github.com/your-org/emberharmony/blob/main/docs/api/features.md
   """
   ```

### Phase 5: Documentation Generation Setup

1. Set up Sphinx for API documentation:
   ```bash
   # Install Sphinx
   pip install sphinx sphinx-rtd-theme

   # Initialize Sphinx
   cd emberharmony
   sphinx-quickstart docs

   # Configure Sphinx to use autodoc
   # Edit docs/conf.py to include:
   extensions = [
       'sphinx.ext.autodoc',
       'sphinx.ext.viewcode',
       'sphinx.ext.napoleon',
   ]
   ```

2. Set up MkDocs for user documentation:
   ```bash
   # Install MkDocs
   pip install mkdocs mkdocs-material

   # Initialize MkDocs
   cd emberharmony
   mkdocs new .

   # Configure MkDocs
   # Edit mkdocs.yml to include:
   site_name: emberharmony
   theme:
     name: material
   nav:
     - Home: index.md
     - Architecture: architecture/index.md
     - API Reference: api/index.md
     - Tutorials: tutorials/index.md
     - Examples: examples/index.md
     - Troubleshooting: troubleshooting/index.md
     - Development: development/index.md
   ```

## Documentation Standards

To ensure consistency across all documentation, we propose the following standards:

### Markdown Formatting

- Use ATX-style headers (`#` for h1, `##` for h2, etc.)
- Use fenced code blocks with language specifiers
- Use reference-style links for better readability
- Use tables for structured information
- Use bullet points for lists
- Use numbered lists for sequential steps

### Content Structure

- Each document should start with a clear title
- Include an overview section at the beginning
- Use descriptive section headers
- Include code examples where appropriate
- End with a conclusion or summary
- Include links to related documentation

### API Documentation

- Use Google-style docstrings
- Document all parameters, return values, and exceptions
- Include type annotations
- Provide usage examples
- Document any side effects or performance considerations

### Tutorials

- Start with clear objectives
- List prerequisites
- Provide step-by-step instructions
- Include complete code examples
- Explain the expected output
- Include troubleshooting tips

## Benefits

Reorganizing the documentation in this way provides several benefits:

1. **Improved Discoverability**: Makes it easier to find specific documentation
2. **Consistent Formatting**: Ensures all documentation follows the same format
3. **Reduced Duplication**: Minimizes repetition of information
4. **Easier Maintenance**: Centralizes documentation for easier updates
5. **Better User Experience**: Provides a clear path through the documentation

## Conclusion

This plan provides a comprehensive approach to reorganizing the emberharmony documentation. By implementing this plan, we will improve the discoverability, consistency, and maintainability of the documentation, making it easier for users to understand and use the library.

The centralized documentation structure will also make it easier to keep the documentation up-to-date as the codebase evolves, ensuring that users always have access to accurate and comprehensive information.