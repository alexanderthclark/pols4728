# Makefile for LaTeX compilation and cleanup
# Usage: make <filename> (without .tex extension)
# Example: make learning

.PHONY: clean all help

# Default target
help:
	@echo "Usage:"
	@echo "  make <filename>  - Compile <filename>.tex to PDF and clean auxiliary files"
	@echo "  make clean       - Remove all auxiliary files"
	@echo "  make all         - Compile all .tex files"
	@echo ""
	@echo "Available tex files:"
	@ls -1 *.tex | sed 's/\.tex$$//' | sed 's/^/  /'

# Pattern rule for compiling any .tex file
%: %.tex
	@echo "Compiling $<..."
	pdflatex $<
	@if grep -q "\\\\addbibresource" $< || grep -q "\\\\cite" $<; then \
		echo "Running biber..."; \
		biber $@; \
		echo "Recompiling for bibliography..."; \
		pdflatex $<; \
		pdflatex $<; \
	fi
	@echo "Cleaning auxiliary files..."
	@rm -f *.aux *.log *.blg *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz *.bcf *.run.xml
	@echo "Done! Output: $@.pdf"

# Compile all .tex files
all:
	@for file in *.tex; do \
		if [ -f "$$file" ]; then \
			base=$$(basename "$$file" .tex); \
			echo "Compiling $$file..."; \
			$(MAKE) $$base; \
		fi \
	done

# Clean auxiliary files only
clean:
	@echo "Removing auxiliary files..."
	@rm -f *.aux *.log *.blg *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz *.bcf *.run.xml
	@echo "Auxiliary files removed."

# Clean everything including PDFs
distclean: clean
	@echo "Removing PDF files..."
	@rm -f *.pdf
	@echo "All generated files removed."