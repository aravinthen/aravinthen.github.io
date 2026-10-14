# Research notebook website

A Jekyll website deployed with GitHub Pages. Articles live in `_posts/`, daily logs
in `_log/`, and shared indexing metadata in `_data/`.

Use Ruby (3.3 is tested), Bundler and Python 3.10 or newer:

```sh
bundle install
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r tests/requirements.txt
make check
make serve
```

`make check` builds the site and validates the current content. See
[MAINTENANCE.md](MAINTENANCE.md) for publishing, browser tests and troubleshooting.
Historical redesign/migration records remain in `reports/`; they are not ongoing
publishing requirements.
