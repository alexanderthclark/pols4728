# Makefile for compiling standalone LaTeX documents to PDF, SVG, and PNG
# Usage: make, make all, make clean, make svg, make png, make filename.svg

# Find all .tex files in current directory
SOURCES := $(wildcard *.tex)
PDFS := $(SOURCES:.tex=.pdf)
SVGS := $(SOURCES:.tex=.svg)
PNGS := $(SOURCES:.tex=.png)

# Default target: compile all to PDF, SVG, and PNG
all: pdf svg png

# Compile all to PDF
pdf: $(PDFS)

# Convert all to SVG
svg: $(SVGS)

# Convert all to PNG
png: $(PNGS)

# Rule to compile .tex to .pdf
%.pdf: %.tex
	pdflatex --shell-escape $<
	rm -f $(basename $<).aux $(basename $<).log $(basename $<).fls $(basename $<).fdb_latexmk $(basename $<).synctex.gz
	@echo "Compiled $< to $@"

# Rule to convert .pdf to .svg
%.svg: %.pdf
	pdf2svg $< $@
	rm -f $<
	@echo "Converted $< to $@ and deleted PDF"

# Rule to convert .pdf to .png (high quality 300 DPI)
%.png: %.pdf
	pdftocairo -png -r 300 $< $*
	mv $*-1.png $@
	rm -f $<
	@echo "Converted $< to $@ at 300 DPI and deleted PDF"

# Alternative PNG rule using ImageMagick (if pdftocairo not available)
%.png-alt: %.pdf
	convert -density 300 -quality 100 $< $(@:.png-alt=.png)
	@echo "Converted $< to PNG using ImageMagick"

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.fls *.fdb_latexmk *.synctex.gz *.out *.nav *.snm *.toc *.vrb
	@echo "Cleaned auxiliary files"

# Clean all generated files (including PDFs, SVGs, PNGs)
cleanall: clean
	rm -f *.pdf *.svg *.png
	@echo "Cleaned all generated files"

# Show available targets
help:
	@echo "Available targets:"
	@echo "  all       - Compile all .tex files to PDF, SVG, and PNG"
	@echo "  pdf       - Compile all .tex files to PDF"
	@echo "  svg       - Convert all PDFs to SVG"
	@echo "  png       - Convert all PDFs to PNG (300 DPI)"
	@echo "  clean     - Remove auxiliary files"
	@echo "  cleanall  - Remove all generated files"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Individual file targets:"
	@echo "  filename.pdf - Compile specific .tex file"
	@echo "  filename.svg - Convert specific file to SVG"
	@echo "  filename.png - Convert specific file to PNG"

# Declare phony targets
.PHONY: all pdf svg png clean cleanall help

# Prevent deletion of intermediate PDFs
.PRECIOUS: %.pdf