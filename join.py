# Define the folder containing your .py files and the output file name
import os
from pathlib import Path

# Configuration
input_folder = './deployment/app'  # Change this to your source folder containing .py files
output_file = 'combined_src_code.py'
# Convert to absolute path to avoid including the output if it's within src
output_path = Path(output_file).absolute()

print(f"Starting merge of all .py files in {input_folder}...")

with open(output_file, 'w', encoding='utf-8') as outfile:
    # rglob('*.py') finds all .py files recursively
    # sorted() ensures files are added in alphabetical order
    for file_path in sorted(Path(input_folder).rglob('*.py')):
        
        # Skip the output file if it's inside the source folder to avoid recursion
        if file_path.absolute() == output_path:
            continue
            
        # Get the relative path for a cleaner header (e.g., agentic/nodes/media.py)
        relative_path = file_path.relative_to(input_folder)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = infile.read()
                
                # Write Header with the relative file path
                outfile.write(f"\n\n{'#' * 60}\n")
                outfile.write(f"# FILE: {relative_path}\n")
                outfile.write(f"{'#' * 60}\n\n")
                
                outfile.write(content)
                
                # Write Footer
                outfile.write(f"\n\n# --- END OF {relative_path} ---\n")
                
            print(f"Successfully added: {relative_path}")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

print(f"\nDone! All code combined into: {output_file}")
