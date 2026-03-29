.PHONY: book-clean book-build book-serve fidelity-audit

book-clean:
	rm -rf book/_build book/.jupyter_cache
	find book -type d \( -name .ipynb_checkpoints -o -name .virtual_documents \) -prune -exec rm -rf {} +

book-build:
	jupyter-book build book

book-serve:
	python3 -m http.server --directory book/_build/html 8000

fidelity-audit:
	bash ./helpers/chapter_fidelity_audit.sh
