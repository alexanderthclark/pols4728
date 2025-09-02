LaTeX Standalone to Image Conversion
====================================

This directory contains a Makefile system for compiling standalone LaTeX documents 
to high-quality SVG and PNG images. Perfect for creating diagrams, figures, and 
mathematical illustrations for course materials.

REQUIREMENTS
------------
- LaTeX distribution (TeXLive, MiKTeX, etc.)
- pdf2svg (for SVG conversion)
- pdftocairo (for PNG conversion, part of poppler-utils)

On macOS with Homebrew:
  brew install pdf2svg poppler

USAGE
-----
Basic commands:
  make all           - Compile all .tex files to both SVG and PNG
  make example.svg   - Compile example.tex to SVG only  
  make example.png   - Compile example.tex to PNG only (300 DPI)
  make clean         - Remove auxiliary files (.aux, .log, etc.)
  make cleanall      - Remove all generated files
  make help          - Show available targets

WORKFLOW
--------
1. Create your .tex file using \documentclass{standalone}
2. Run make filename.svg or make filename.png
3. Makefile automatically:
   - Compiles LaTeX with pdflatex --shell-escape
   - Converts PDF to SVG/PNG
   - Deletes intermediate PDF file
   - Removes auxiliary files (.aux, .log, etc.)

STANDALONE DOCUMENTCLASS
------------------------
Use this template for your .tex files:

\documentclass[tikz,border=5pt]{standalone}
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}
  % Your TikZ code here
\end{tikzpicture}
\end{document}

The standalone class automatically crops output to the content boundaries,
making it perfect for figures that will be embedded in other documents.

OUTPUT FORMATS
--------------
- SVG: Vector format, scalable, perfect for web display
- PNG: Raster format, 300 DPI, high quality for print/presentations

TIPS
----
- Use TikZ for geometric diagrams and plots
- Include amsmath, amsfonts for mathematical notation
- The border=5pt option adds small padding around your content
- Generated images are automatically cropped to content