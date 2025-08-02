#!/bin/bash
# Build TikZ diagrams to PDF and SVG

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="$SCRIPT_DIR/../diagrams"

# Function to compile a single TikZ file
compile_tikz() {
    local texfile=$1
    local basename=$(basename "$texfile" .tex)
    
    echo "Compiling $texfile..."
    
    # Compile to PDF in script directory
    pdflatex -interaction=nonstopmode -output-directory="$SCRIPT_DIR" "$texfile"
    
    # Convert to SVG and move to output directory
    if command -v pdf2svg &> /dev/null; then
        pdf2svg "$SCRIPT_DIR/${basename}.pdf" "$OUTPUT_DIR/${basename}.svg"
        echo "Created $OUTPUT_DIR/${basename}.svg"
    else
        echo "pdf2svg not found. Install it to generate SVG files."
    fi
    
    # Move PDF to output directory
    if [ -f "$SCRIPT_DIR/${basename}.pdf" ]; then
        mv "$SCRIPT_DIR/${basename}.pdf" "$OUTPUT_DIR/"
    fi
    
    # Clean up auxiliary files
    rm -f "$SCRIPT_DIR/${basename}.aux" "$SCRIPT_DIR/${basename}.log"
}

# Change to script directory
cd "$SCRIPT_DIR"

# Compile all TikZ files
for texfile in *.tex; do
    if [ -f "$texfile" ]; then
        compile_tikz "$texfile"
    fi
done

echo "Done! Generated PDF and SVG files in $OUTPUT_DIR"