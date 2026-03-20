PYTHON = .venv/Scripts/python.exe

manual:
	$(PYTHON) make_html.py USER_MANUAL.md

claude:
	$(PYTHON) make_html.py CLAUDE.md

readme:
	$(PYTHON) make_html.py README.md

# Convert all .md files in root
all-docs:
	$(foreach f, $(wildcard *.md), $(PYTHON) make_html.py $(f);)

.PHONY: manual claude readme all-docs
