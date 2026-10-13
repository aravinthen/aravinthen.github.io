/* Client-side logbook search, topic filtering, history, and activity calendar.
   All entry content is sourced from Jekyll's site.log collection. */
(function () {
  'use strict';

  const root = document.querySelector('[data-log-index]');
  if (!root) return;

  const search = document.getElementById('log-search');
  const monthSelect = document.getElementById('log-month-select');
  const tagContainer = document.getElementById('log-tag-filters');
  const status = document.getElementById('log-result-status');
  const warning = document.getElementById('log-search-warning');
  const empty = document.getElementById('log-empty');
  const monthSections = Array.from(document.querySelectorAll('.log-month'));
  const rows = Array.from(document.querySelectorAll('.log-item')).map(function (element) {
    return {
      element: element,
      url: element.dataset.url,
      date: element.dataset.date,
      month: element.dataset.month,
      title: element.dataset.title,
      tags: element.dataset.tags ? element.dataset.tags.split('|').filter(Boolean) : [],
      excerpt: (element.querySelector('.log-item-excerpt') || {}).textContent || ''
    };
  });

  function normalise(value) {
    return String(value || '').normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
  }

  function displayTag(value) {
    const special = { ai: 'AI', rl: 'RL', llm: 'LLM', marl: 'MARL', ppo: 'PPO', rlhf: 'RLHF' };
    return value.replace(/[_-]/g, ' ').split(' ').map(function (word) {
      const lower = word.toLowerCase();
      return special[lower] || (word.charAt(0).toUpperCase() + word.slice(1));
    }).join(' ');
  }

  // Tags are derived from the current archive, not from site.tags (which
  // represents posts rather than this custom Jekyll collection).
  const counts = new Map();
  rows.forEach(function (row) {
    row.tags.forEach(function (tag) { counts.set(tag, (counts.get(tag) || 0) + 1); });
  });
  const checkboxes = new Map();
  Array.from(counts.keys()).sort(function (a, b) {
    return counts.get(b) - counts.get(a) || a.localeCompare(b);
  }).forEach(function (tag) {
    const label = document.createElement('label');
    label.className = 'log-filter-chip';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.value = tag;
    input.name = 'tag';
    const caption = document.createElement('span');
    caption.textContent = displayTag(tag) + ' (' + counts.get(tag) + ')';
    label.append(input, caption);
    tagContainer.appendChild(label);
    checkboxes.set(tag, input);
    input.addEventListener('change', function () { update(true); });
  });

  function selectedTags() {
    return Array.from(checkboxes).filter(function (pair) { return pair[1].checked; })
      .map(function (pair) { return pair[0]; });
  }

  function readUrl() {
    const params = new URLSearchParams(window.location.search);
    search.value = params.get('q') || '';
    const month = params.get('month') || '';
    monthSelect.value = Array.from(monthSelect.options).some(function (opt) { return opt.value === month; }) ? month : '';
    const requested = new Set(params.getAll('tag'));
    checkboxes.forEach(function (input, tag) { input.checked = requested.has(tag); });
  }

  function writeUrl(push) {
    const url = new URL(window.location.href);
    url.searchParams.delete('q');
    url.searchParams.delete('month');
    url.searchParams.delete('tag');
    if (search.value.trim()) url.searchParams.set('q', search.value.trim());
    if (monthSelect.value) url.searchParams.set('month', monthSelect.value);
    selectedTags().forEach(function (tag) { url.searchParams.append('tag', tag); });
    if (url.href !== window.location.href) {
      window.history[push ? 'pushState' : 'replaceState'](null, '', url.pathname + url.search + url.hash);
    }
  }

  let searchPromise = null;
  let searchFailed = false;
  function loadFullText() {
    if (searchPromise) return searchPromise;
    searchPromise = fetch(root.dataset.searchUrl).then(function (response) {
      if (!response.ok) throw new Error('Search index returned HTTP ' + response.status);
      return response.json();
    }).then(function (entries) {
      const map = new Map();
      entries.forEach(function (entry) { map.set(entry.url, normalise(entry.body)); });
      return map;
    }).catch(function (error) {
      searchFailed = true;
      console.warn('Logbook full-text search unavailable:', error);
      return new Map();
    });
    return searchPromise;
  }

  let revision = 0;
  let wasFiltering = false;
  let expandedBeforeFilter = new Map();
  async function update(changeUrl, push) {
    if (changeUrl) writeUrl(Boolean(push));
    const currentRevision = ++revision;
    const query = normalise(search.value.trim());
    const words = query.split(/\s+/).filter(Boolean);
    const tagSelection = selectedTags();
    const selectedMonth = monthSelect.value;
    const isFiltering = words.length > 0 || tagSelection.length > 0 || Boolean(selectedMonth);
    let bodyIndex = null;

    if (words.length) bodyIndex = await loadFullText();
    if (currentRevision !== revision) return;  // Ignore superseded async searches.
    warning.hidden = !searchFailed || !words.length;

    if (isFiltering && !wasFiltering) {
      expandedBeforeFilter = new Map(monthSections.map(function (section) { return [section, section.open]; }));
      wasFiltering = true;
    }

    let resultCount = 0;
    const visibleByMonth = new Map();
    rows.forEach(function (row) {
      const text = normalise(row.title + ' ' + row.excerpt + ' ' + row.tags.join(' ')) +
        ' ' + (bodyIndex ? (bodyIndex.get(row.url) || '') : '');
      const matches = (!selectedMonth || row.month === selectedMonth) &&
        tagSelection.every(function (tag) { return row.tags.includes(tag); }) &&
        words.every(function (word) { return text.includes(word); });
      row.element.hidden = !matches;
      if (matches) {
        resultCount += 1;
        visibleByMonth.set(row.month, (visibleByMonth.get(row.month) || 0) + 1);
      }
    });

    monthSections.forEach(function (section) {
      const count = visibleByMonth.get(section.dataset.month) || 0;
      section.hidden = count === 0;
      const badge = section.querySelector('.log-month-count');
      badge.textContent = count + (count === 1 ? ' entry' : ' entries');
      if (isFiltering && count > 0) section.open = true;
    });

    if (!isFiltering && wasFiltering) {
      monthSections.forEach(function (section) {
        section.open = expandedBeforeFilter.get(section) || false;
      });
      wasFiltering = false;
    }
    status.textContent = 'Showing ' + resultCount + ' of ' + rows.length + (rows.length === 1 ? ' entry' : ' entries');
    empty.hidden = resultCount !== 0;
  }

  let debounceTimer = null;
  search.addEventListener('input', function () {
    window.clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(function () { update(true, false); }, 160);
  });
  monthSelect.addEventListener('change', function () { update(true, true); });
  document.getElementById('log-clear-filters').addEventListener('click', function () {
    search.value = '';
    monthSelect.value = '';
    checkboxes.forEach(function (input) { input.checked = false; });
    update(true, true);
  });
  document.getElementById('log-expand-all').addEventListener('click', function () {
    monthSections.forEach(function (section) { if (!section.hidden) section.open = true; });
  });
  document.getElementById('log-collapse-all').addEventListener('click', function () {
    monthSections.forEach(function (section) { section.open = false; });
  });
  window.addEventListener('popstate', function () { readUrl(); update(false); });

  // Render a compact, optional calendar. No additional data request is needed.
  const calendar = document.getElementById('log-calendar');
  const calendarYear = document.getElementById('log-calendar-year');
  const byDate = new Map();
  rows.forEach(function (row) {
    const entries = byDate.get(row.date) || [];
    entries.push(row);
    byDate.set(row.date, entries);
  });
  const years = Array.from(new Set(rows.map(function (row) { return row.date.slice(0, 4); })))
    .sort().reverse();
  years.forEach(function (year) {
    const option = document.createElement('option');
    option.value = year;
    option.textContent = year;
    calendarYear.appendChild(option);
  });

  function renderCalendar() {
    const year = Number(calendarYear.value);
    calendar.replaceChildren();
    if (!year) return;
    const day = 86400000;
    const januaryFirst = Date.UTC(year, 0, 1);
    const decemberLast = Date.UTC(year, 11, 31);
    const weekday = (new Date(januaryFirst).getUTCDay() + 6) % 7; // Monday = 0.
    const start = januaryFirst - weekday * day;
    const endingWeekday = (new Date(decemberLast).getUTCDay() + 6) % 7;
    const end = decemberLast + (6 - endingWeekday) * day;
    for (let timestamp = start; timestamp <= end; timestamp += day) {
      const iso = new Date(timestamp).toISOString().slice(0, 10);
      const inYear = iso.slice(0, 4) === String(year);
      const entries = byDate.get(iso) || [];
      const square = document.createElement(entries.length ? 'a' : 'span');
      square.className = 'log-calendar-day';
      if (!inYear) {
        square.classList.add('outside-year');
        square.setAttribute('aria-hidden', 'true');
      } else if (entries.length) {
        square.classList.add('has-entry');
        if (entries.length > 1) square.classList.add('multiple-entries');
        square.href = entries[0].url;
        square.setAttribute('aria-label', iso + ': ' + entries.length + (entries.length === 1 ? ' entry' : ' entries'));
      } else {
        square.setAttribute('aria-label', iso + ': no entries');
      }
      square.title = iso + ': ' + entries.length + (entries.length === 1 ? ' entry' : ' entries');
      calendar.appendChild(square);
    }
  }
  calendarYear.addEventListener('change', renderCalendar);
  if (years.length) calendarYear.value = years.includes(String(new Date().getFullYear())) ? String(new Date().getFullYear()) : years[0];
  renderCalendar();

  readUrl();
  update(false);
}());
