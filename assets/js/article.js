/* Index existing headings without changing the author's wording. */
(function () {
  'use strict';
  const content = document.querySelector('.article-content');
  const toc = document.querySelector('.article-toc');
  if (!content || !toc) return;
  const headings = Array.from(content.querySelectorAll('h2, h3'));
  if (headings.length < 5) return;
  const list = toc.querySelector('ol');
  headings.forEach(function (heading, index) {
    if (!heading.id) heading.id = 'article-heading-' + index;
    const item = document.createElement('li');
    item.className = heading.tagName === 'H3' ? 'toc-subheading' : '';
    const link = document.createElement('a');
    link.href = '#' + heading.id;
    link.textContent = heading.textContent;
    item.appendChild(link);
    list.appendChild(item);
  });
  toc.hidden = false;
}());
