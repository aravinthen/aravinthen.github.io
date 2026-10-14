# Research section unification

Implemented on the existing `feature/research-notebook-design-tags` branch. No merge or deployment performed. This report describes the follow-up to the earlier design/tag migration reports.

## Structure and relationships

`/research/` is the unified index. All three existing project overviews appear under Research programmes, using their existing titles/dates and clearly truncated extracts from their first authored paragraph. No descriptions were generated. BlueDot's explicitly documented literature links remain available under Associated reading.

The Deep RL project and its three technical notes now share `research_programme: deep-reinforcement-learning`:

- Lessons from Classical RL
- From REINFORCE to Actor-Critic
- Proximal Policy Optimization, layer by layer

Foundation Models and Post-training has `research_programme: foundation-models-post-training`. No technical notes or repository links were invented for it.

Independent technical notes appear in a separate section when present. Currently, all three technical notes belong to Deep RL. A missing or unmatched programme identifier leaves a note independent, rather than hiding or forcing it into a programme. Temporary test fixtures verify both cases.

To add a note to a programme, give its existing `technical` post category the same `research_programme` identifier as the project. Listings and navigation update automatically. No hard-coded note lists remain for Deep RL. On project pages, generated navigation adds new associated notes only when their links are absent from the authored body, avoiding duplicate existing links.

## Files changed

- Created `research/index.html`, `_includes/programme-notes.html`, and `_includes/research-navigation.html`.
- Updated `_data/navigation.yml`, `_layouts/default.html`, `_layouts/article.html`, and `index.html` for Research navigation, programme breadcrumbs, homepage aggregation and secondary Miscellaneous navigation.
- Updated `_data/writing.yml` to remove the redundant manually maintained Deep RL note list; retained its documented repository and all existing literature-series/BlueDot relationships.
- Added one front-matter field to each of the two programme overviews and three RL technical notes. No article moved or renamed.
- Updated `projects/index.html` and `technical/index.html` for compatibility redirects.
- Updated `assets/css/style.css` to use the existing monospace stack consistently across articles, daily logs, About, homepage, and interface text. Log and article body sizes/line heights also agree at desktop and mobile widths. Mathematical glyph rendering remains managed by MathJax.
- Added `tests/research_site.py` and updated the existing build, browser, automatic-indexing and full-render checks for the new navigation and redirects.

## Navigation and URL compatibility

Primary navigation: Research, Literature, Logbook, About. Research is active on both project and technical articles. The site title continues to link home. Miscellaneous remains available in the footer, with its existing pages/content intact.

`/projects/` and `/technical/` issue zero-delay HTML meta-refresh redirects to `/research/`, with a canonical target and a direct fallback link. Their existing authored introductions remain in the fallback HTML/source. No JavaScript, redirect plugin, HTTP server configuration or dependency was added. These are HTML redirects, not HTTP 301 responses. Every individual article URL, category, title, date, permalink and original metadata value is retained.

## Validation

- Jekyll build passes: 324 HTML pages, including Research and all existing article URLs.
- `tests/check_site.py`: zero broken internal links/fragments or duplicate IDs; article heading checks, valid RSS/search JSON, tag counts and index-prose preservation pass. The original articles differ only by the five explicitly permitted front-matter fields.
- `tests/research_site.py`: all three projects and three notes appear exactly once on Research; RL grouping, unassociated Foundation programme, excerpts/repository, active navigation, programme breadcrumbs, non-duplicated existing article links, homepage updates, footer Miscellaneous link and both legacy redirects pass at 1280px and 390px.
- Preservation check against the start of this task: all 316 Markdown files (30 articles, 285 logs, About) are byte-identical after removing only the five added `research_programme` lines. All log files are entirely unchanged in this task.
- `tests/automatic_indexing.py`: newly added projects, linked notes, independent notes and notes with an unmatched programme identifier appear automatically, once each. Programme-to-note and note-to-programme links are automatic. Fixtures exist only in a temporary source copy.
- `tests/browser_site.py`: the existing logbook search/fallback, canonical filters/counts/aliases, monthly archive, timelines, activity calendar, related/next entries, MathJax, images, article series links and keyboard navigation pass on desktop/mobile. No JavaScript page errors.
- `tests/render_site.py`: all 324 pages checked at mobile width; no broken images, MathJax parser errors, JavaScript page errors or page-wide overflow.
- Desktop/mobile Research screenshots inspected visually. Computed font-family assertions confirm matching font stacks across logs, articles, About and homepage at both widths.
- JavaScript syntax checks and `git diff --check` pass. The full implementation diff was reviewed, including each metadata-only article change.

Browser MathJax checks retain the earlier test-harness workaround: verified curl downloads supply unchanged CDN assets because Chromium does not trust the cloud egress proxy's certificate. TLS verification is not disabled. Hosted GitHub Pages deployment and other browser engines were not tested.

## Editorial decisions left to the author

- Optionally supply a concise `summary` for each programme. Until then, the index displays an unchanged, explicitly truncated author-written extract; the Foundation introduction leads into a list and may particularly benefit from a standalone author-written summary.
- Add Foundation technical notes/repository metadata only when those documents and relationships are ready.
- Existing unfinished project/technical sections remain untouched.
- BlueDot remains included because it is an existing project. Its study-note links are preserved; it has no new technical-note programme association.
