# Maintaining the website

Write Markdown, supply metadata, commit and push. Indexes and navigation are
assembled by Jekyll; do not edit generated `_site/` files. The current templates,
styles, navigation and JavaScript require no changes for ordinary publishing.

## Publish content

**Daily log:** create `_log/YYYY-MM-DD.md` with matching filename/date:

```yaml
---
layout: log
title: "Your entry title"
date: YYYY-MM-DD
tags: [reinforcement-learning]
math: true # optional
---
```

Write your entry beneath the front matter. Choose **one** canonical tag, or **two**
when another distinct subject is substantially discussed. No duplicate tags.
The current choices are `reinforcement-learning`, `deep-learning`, `post-training`,
`ai-safety`, `multi-agent`, `systems`, `mathematics`, and `general`. The authoritative
list is `_data/log_topics.yml`, not a migration report. Legacy aliases work in
filter URLs; do not use aliases in new front matter. Search, month archives,
previous/next links, related entries, timelines and the activity calendar update
automatically. `math: true` enables MathJax; layouts also detect rendered math
boundaries. Keep mathematical expressions in the authored body.

**Technical note:** create `_posts/technical/YYYY-MM-DD-slug.md`:

```yaml
---
title: "Your note title"
category: technical
research_programme: deep-reinforcement-learning # optional
math: true # optional
---
```

The filename supplies the publication date and URL slug. Match the programme
identifier on an existing project exactly. Current identifiers include
`deep-reinforcement-learning` and `foundation-models-post-training`. Omit the field
for an independent note. An unmatched identifier also appears independently, but
usually indicates a spelling error. The Research index and programme breadcrumbs
update automatically; no manually maintained list is needed.

**Research project:** create `_posts/projects/YYYY-MM-DD-slug.md`:

```yaml
---
title: "Your programme title"
category: projects
research_programme: your-unique-programme-id
# Optional author-supplied metadata:
# repository: https://github.com/OWNER/REPOSITORY
# status: "Your status"
# topics: [your-topic]
# summary: "Your own description"
---
```

Use a unique programme identifier on the project and reuse it on associated
technical notes. Programme notes are indexed automatically. Existing authored
links are not repeated in generated project navigation. Without your `summary`,
the Research index uses a visibly truncated, unchanged extract. Never generate
summaries as part of publishing.

**Literature review:** create `_posts/literature/YYYY-MM-DD-slug.md`:

```yaml
---
title: "Your review title"
category: literature
series: "Notes on Technical AI Safety" # optional; exact series name
math: true # optional
---
```

Other existing series include `Weekly literature reviews` and `Multi-agent
Reinforcement Learning`. Use the exact name to join an existing series or supply
your own name for a new series. Ordering is chronological within series;
previous/next links follow that order. Omit `series` for other reading. Historical
entries use series metadata in `_data/writing.yml`; new entries can declare it in
front matter without updating that sidecar. Front matter takes precedence.

For all posts, retain the internal `projects`, `technical`, `literature`, or `misc`
category. The shared article layout is the default. Dates in the future are not
published as posts until that date unless building with `--future`. Use stable
filenames/URL slugs for existing articles; renaming them changes incoming links.
Date overrides and explicit permalinks should be used deliberately. Run
`make check` after writing; it does not require a historical Git baseline.

## Personal information

Edit your existing prose in `index.html` (homepage) or `about/index.md` (About).
Contact links also appear in `_layouts/default.html`; keep them consistent with
About when changing them. The site title and existing description live in
`_config.yml`. You own these words; the tests permit ordinary editorial changes.

## Local development

Prerequisites: Ruby and Bundler, Python **3.10+**, and Make. Ruby **3.3** is tested
against the current Pages action. Native gem installation may need a C/C++ compiler,
Ruby headers and OpenSSL development libraries (Linux), or Xcode command-line tools
(macOS). macOS execution has not been verified in this PR; the lockfile includes
Ruby-platform gems for portability.

The small Gemfile pins **github-pages 232**, matching the deployed build diagnosed
in the earlier CI log, and the lockfile pins its dependencies. This includes
**Jekyll 3.10.0**. Do not substitute unrelated Jekyll 4 builds as a compatibility
check. Do not update the lockfile routinely when publishing.

```sh
bundle install
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r tests/requirements.txt
make check
make serve
```

`make serve` prints the local preview address. Stop it with Ctrl-C. Build output is
ignored by Git and is never content to commit. `bundle exec jekyll build` and
`bundle exec jekyll serve` also work directly.

| Command | Purpose |
| --- | --- |
| `make build` | GitHub Pages-compatible local Jekyll build |
| `make check` | Build plus fast current-content, YAML, tags, indexes, relationships, links, search and navigation checks |
| `make serve` | Local development server with regeneration |
| `make test-growth` | Disposable additions/edits, including expanded taxonomy, built and checked |
| `make test` | Fast checks, growth scenarios and normal desktop/mobile browser regressions |
| `make test-full` | Normal suite plus comprehensive rendering of every page |
| `make test-pages` | Optional exact GitHub Pages action-container build; requires Docker |

Routine publishing: write/edit the content, run `make check`, then commit and push.
`make test` is useful for shared template, JavaScript, CSS or navigation changes.
Full audits are optional; they are not required for every daily entry.

Install optional browser prerequisites once:

```sh
python3 -m pip install -r tests/requirements-browser.txt
python3 -m playwright install chromium
# Linux only, if browser system libraries are missing:
python3 -m playwright install-deps chromium
make test
```

Browser tests start and stop their own server on an available port. Set `SITE_URL`
(or `--base-url` on a browser script) to use an existing server. `BROWSER_EXECUTABLE`
can point to an already installed Chromium; otherwise Playwright uses its managed
browser. `BROWSER_NO_SANDBOX=1` is only for containers that require it, not the normal
default. On a network with a proxy certificate trusted by Python but not Chromium,
`BROWSER_CDN_PROXY=1` loads unchanged CDN assets through Python's verified TLS.
Never disable certificate verification. Standard `SSL_CERT_FILE` trust configuration
may be needed on such a network. External CDN availability is a browser-test
prerequisite; the fast publishing check needs no browser or CDN.

Override `PYTHON`, `SITE_DIR`, or `JEKYLL` in Make if needed. Growth tests accept
`JEKYLL_COMMAND` for a build-command override; default is `bundle exec jekyll`.
Scripts find the checkout relative to themselves and accept source/output paths.
Docker is optional: `bash tests/github_pages_build.sh /absolute/output/path` stores
an exact-image build outside the checkout; without a path it uses a temporary
folder and prints its location. `SOURCE_DIR` and `PAGES_IMAGE` override the Docker
source/image when intentionally validating another checkout or production image.

## Deployment and failures

Push approved content to `main` using your usual GitHub workflow. This repository
contains no custom deployment workflow under `.github/workflows/`; deployment is
managed by the configured GitHub Pages workflow. The previously diagnosed run uses
`actions/jekyll-build-pages@v1` (image v1.0.13). Keep the existing Pages settings.
The live Pages configuration could not be queried from this environment because
GitHub API access was denied; verify the configured source under **Settings → Pages**
if troubleshooting deployment.

In GitHub **Actions**, check that the Pages build and deployment finish successfully
and correspond to your pushed commit. Then check the published page. A local build
or pushed commit alone does not prove deployment completed. If deployment fails,
read the first error in the build log and reproduce with `make check` or the optional
exact-image build. Existing `/projects/` and `/technical/` URLs remain HTML redirects
to `/research/`; individual article URLs remain independent.

| Problem | Check |
| --- | --- |
| Invalid YAML | Opening/closing `---`, spaces instead of tabs, quote titles containing colons, no duplicate keys; error includes the source filename |
| Tag error | Use one or two unique slugs from `_data/log_topics.yml`; new topics go there with matching `slug`/`tags` and non-conflicting optional `aliases` |
| Missing programme association | Project and note need identical `research_programme`; project IDs must be unique |
| Missing series/navigation | Check exact `series` spelling or the existing URL key in `_data/writing.yml` |
| Broken internal link | Match the generated URL and heading fragment; do not rename old slugs casually |
| Failed Jekyll build | Run `bundle check`, confirm the locked Pages versions, then inspect the first Liquid/YAML error; Jekyll 3.10 rejects compound `where_exp` conditions |
| Failed browser setup | Install Chromium/system libraries or supply `BROWSER_EXECUTABLE`; browser dependencies are optional for publishing |
| Failed deployment | Confirm the Actions run/commit and Pages source, then reproduce the build; investigate the actual error rather than upgrading dependencies blindly |

Historical reports in `reports/` preserve migration evidence and are never updated
by publishing tests. For a PR specifically requiring preservation, run
`python3 tests/preserve_changes.py BASE_REVISION --maintenance`; supply that PR's
parent or another explicit baseline. This check is intentionally outside normal
publishing acceptance. Permanent tests validate current content, not immutable prose.
