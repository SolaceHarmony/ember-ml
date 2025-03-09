#!/usr/bin/env python3
import os
import re

def update_imports(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace imports from neural_lib to emberharmony
    updated_content = re.sub(r'from\s+neural_lib', 'from emberharmony', content)
    updated_content = re.sub(r'import\s+neural_lib', 'import emberharmony', updated_content)
    
    # Replace imports from notebooks.neural_experiments to emberharmony
    updated_content = re.sub(r'from\s+notebooks\.neural_experiments', 'from emberharmony', updated_content)
    updated_content = re.sub(r'import\s+notebooks\.neural_experiments', 'import emberharmony', updated_content)
    
    # Replace imports from notebooks.binary_wave_neurons to emberharmony
    updated_content = re.sub(r'from\s+notebooks\.binary_wave_neurons', 'from emberharmony.wave', updated_content)
    updated_content = re.sub(r'import\s+notebooks\.binary_wave_neurons', 'import emberharmony.wave', updated_content)
    
    # Replace imports from notebooks.audio_processing to emberharmony.audio
    updated_content = re.sub(r'from\s+notebooks\.audio_processing', 'from emberharmony.audio', updated_content)
    updated_content = re.sub(r'import\s+notebooks\.audio_processing', 'import emberharmony.audio', updated_content)
    
    # Replace imports from notebooks.infinitemath to emberharmony.math.infinite
    updated_content = re.sub(r'from\s+notebooks\.infinitemath', 'from emberharmony.math.infinite', updated_content)
    updated_content = re.sub(r'import\s+notebooks\.infinitemath', 'import emberharmony.math.infinite', updated_content)
    
    if content != updated_content:
        with open(file_path, 'w') as f:
            f.write(updated_content)
        return True
    return False

def main():
    updated_files = 0
    for root, dirs, files in os.walk('emberharmony'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_imports(file_path):
                    updated_files += 1
                    print(f"Updated imports in {file_path}")
    
    print(f"\nUpdated imports in {updated_files} files")

if __name__ == "__main__":
    main()
