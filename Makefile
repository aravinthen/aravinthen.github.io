PYTHON ?= python3
SITE_DIR ?= _site
JEKYLL ?= bundle exec jekyll

.PHONY: build check serve test test-full test-growth test-pages
build:
	$(JEKYLL) build --destination "$(SITE_DIR)"
check: build
	$(PYTHON) tests/check_site.py "$(SITE_DIR)"
serve:
	$(JEKYLL) serve --destination "$(SITE_DIR)"
test: check
	$(PYTHON) tests/automatic_indexing.py --browser
	$(PYTHON) tests/browser_site.py --site "$(SITE_DIR)"
test-growth:
	$(PYTHON) tests/automatic_indexing.py
test-full: test
	$(PYTHON) tests/render_site.py --site "$(SITE_DIR)"
test-pages:
	bash tests/github_pages_build.sh
