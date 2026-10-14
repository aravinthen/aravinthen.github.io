"""Normal browser regressions, driven by the current source and generated site.
Starts an isolated local server unless --base-url / SITE_URL is supplied.
"""
from collections import Counter
import json
from urllib.parse import urlencode
from support import Inventory, args, server, launch, verified_cdn, tree

def browser_checks(source, site, base_url=None):
    from playwright.sync_api import sync_playwright
    inventory=Inventory(source)
    errors=[]
    with server(site,base_url) as base, sync_playwright() as p:
        browser=launch(p)
        page=browser.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        with verified_cdn(page):
            def visit(url):
                response=page.goto(base+url)
                assert response is None or response.status==200,url
                redirects=page.locator('meta[http-equiv="refresh"]')
                if redirects.count():
                    destination=redirects.get_attribute('content').split('url=',1)[1]
                    page.wait_for_url(base+destination)
                    page.wait_for_load_state('load')
            def visible(count):
                page.wait_for_function('(count)=>document.querySelectorAll(".log-item:not([hidden])").length===count',arg=count)
                assert page.locator('#log-result-status').inner_text().startswith(f'Showing {count} of {len(inventory.logs)}')
            routes=['/', '/research/', '/literature/', '/log/', '/log/timelines/', '/about/', '/misc/', '/projects/', '/technical/']
            representatives=[]
            for docs in [inventory.technical,inventory.projects,inventory.literature,inventory.logs]:
                if docs:representatives.append(docs[-1].url)
            fonts=[]
            for width,height in [(1280,900),(390,844)]:
                page.set_viewport_size({'width':width,'height':height})
                for url in [inventory.baseurl+r for r in routes]+representatives:
                    visit(url)
                    assert page.locator('main').inner_text().strip(),url
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),f'{url}: overflow at {width}px'
                    expected=None
                    for item in inventory.navigation:
                        if url.startswith(inventory.baseurl+item['url']) and item['url']!='/':expected=item['title']
                    doc=next((d for d in inventory.posts if d.url==url),None)
                    if doc:
                        section='research' if doc.category in {'projects','technical'} else doc.category
                        expected=next((n['title'] for n in inventory.navigation if n['section']==section),None)
                    if url.endswith(('/projects/','/technical/')):expected=next(n['title'] for n in inventory.navigation if n['section']=='research')
                    active=page.locator('.section-nav [aria-current=page]')
                    if expected:assert active.inner_text()==expected,(url,expected)
                    else:assert active.count()==0,url
                    assert page.locator('.section-nav a').all_text_contents()==[n['title'] for n in inventory.navigation]
                    body=page.locator('.article-content, .log-content, main .about')
                    if body.count():fonts.append(body.first.evaluate('(e)=>getComputedStyle(e).fontFamily'))
                    if doc:
                        headings=page.locator('.article-content h2, .article-content h3').count()
                        assert page.locator('.article-toc').is_visible()==(headings>=5),url
                visit(inventory.baseurl+'/')
                page.keyboard.press('Tab')
                assert page.locator('.skip-link').evaluate('(e)=>e===document.activeElement')
                page.keyboard.press('Enter')
                assert page.locator('#content').evaluate('(e)=>e===document.activeElement')
            assert len(set(fonts))<=1,fonts
            visit(inventory.baseurl+'/log/')
            assert page.locator('.log-item').count()==len(inventory.logs)
            assert set(page.locator('#log-tag-filters input').evaluate_all('(els)=>els.map(e=>e.value)'))==set(inventory.counts)
            for tag,count in inventory.counts.items():
                visit(inventory.baseurl+'/log/?'+urlencode({'tag':tag}))
                visible(count)
                assert page.locator(f'input[value="{tag}"]').is_checked()
                assert f'({count})' in page.locator(f'label:has(input[value="{tag}"])').inner_text()
                actual=set(page.locator('.log-item:not([hidden])').evaluate_all('(els)=>els.map(e=>e.dataset.url)'))
                assert actual=={d.url for d in inventory.logs if tag in d.meta['tags']},tag
            for alias,tag in inventory.aliases.items():
                if tag not in inventory.counts:continue
                visit(inventory.baseurl+'/log/?'+urlencode({'tag':alias}));visible(inventory.counts[tag])
                assert page.locator(f'input[value="{tag}"]').is_checked(),alias
            dual=next((d for d in inventory.logs if len(d.meta['tags'])==2),None)
            if dual:
                query=urlencode([('tag',t) for t in dual.meta['tags']])
                visit(inventory.baseurl+'/log/?'+query)
                visible(sum(set(dual.meta['tags'])<=set(d.meta['tags']) for d in inventory.logs))
            month_counts=Counter(d.day[:7] for d in inventory.logs)
            visit(inventory.baseurl+'/log/')
            for month,count in month_counts.items():
                page.locator('#log-month-select').select_option(month);visible(count)
                assert page.locator('.log-month:not([hidden])').count()==1
            page.locator('#log-clear-filters').click();visible(len(inventory.logs))
            page.locator('#log-collapse-all').click();assert page.locator('.log-month[open]').count()==0
            page.locator('#log-expand-all').click();assert page.locator('.log-month[open]').count()==len(month_counts)
            # Use a current indexed body word, not a historically unique magic string.
            entries=json.loads((site/'log/search.json').read_text())
            word=next((w.strip('.,:;!?()').lower() for e in entries for w in e['body'].split() if len(w)>8 and w.isascii() and w.isalpha()),None)
            if word:
                page.locator('#log-search').fill(word)
                page.wait_for_function('!document.querySelector("#log-search-warning").hidden || document.querySelectorAll(".log-item:not([hidden])").length < document.querySelectorAll(".log-item").length')
                assert not page.locator('#log-search-warning').is_visible()
                indexed={e['url'] for e in entries if word in e['body'].lower()}
                actual=set(page.locator('.log-item:not([hidden])').evaluate_all('(els)=>els.map(e=>e.dataset.url)'))
                assert indexed<=actual and actual,'full text search omitted body matches'
            page.locator('#log-search').fill('maintenance-no-match-7c98cf204a')
            page.wait_for_selector('#log-empty:not([hidden])')
            page.locator('#log-clear-filters').click()
            page.locator('.log-calendar-panel summary').click()
            years={d.day[:4] for d in inventory.logs}
            for year in years:
                page.locator('#log-calendar-year').select_option(year)
                assert page.locator('#log-calendar a.has-entry').count()==len({d.day for d in inventory.logs if d.day.startswith(year)})
            if inventory.logs:
                page.locator('#log-calendar a.has-entry').first.click()
                assert page.url.startswith(base+inventory.baseurl+'/log/')
                related=page.locator('.log-related a')
                assert all(url!=page.url.removeprefix(base) for url in related.evaluate_all('(els)=>els.map(e=>e.getAttribute("href"))'))
            visit(inventory.baseurl+'/log/timelines/')
            for tag,count in inventory.counts.items():assert page.locator(f'#{tag} .log-timeline-list li').count()==count
            if 'post-training' in inventory.counts:
                visit(inventory.baseurl+'/log/timelines/#post-training-reasoning')
                assert page.locator('#post-training').evaluate('(e)=>e.open')
            # Representative math/images are selected from the current output.
            math_pages=[f for f in site.rglob('*.html') if 'window.MathJax' in f.read_text()]
            image_pages=[f for f in site.rglob('*.html') if '<img' in f.read_text()]
            for file in dict.fromkeys(math_pages[:1]+image_pages[:1]):
                audit_page(page,file,site,base,inventory.baseurl)
            if inventory.logs:
                page.route('**/log/search.json',lambda route:route.abort())
                query=inventory.logs[-1].meta['tags'][0]
                visit(inventory.baseurl+'/log/?'+urlencode({'q':query}))
                page.wait_for_selector('#log-search-warning:not([hidden])')
                assert page.locator('.log-item:not([hidden])').count()>0,'title/tag fallback failed'
            assert not errors,errors
        browser.close()
    print('PASS: current navigation, responsive/focus/font/TOC checks, full text/fallback, all tags and aliases, dual filters, months, calendar, timelines, representative rendering')

def audit_page(page,file,site,base,baseurl):
    relative=file.relative_to(site).as_posix()
    url=baseurl+'/'+(relative.removesuffix('index.html') if file.name=='index.html' else relative)
    assert page.goto(base+url).status==200,url
    redirects=page.locator('meta[http-equiv="refresh"]')
    if redirects.count():
        destination=redirects.get_attribute('content').split('url=',1)[1]
        page.wait_for_url(base+destination);page.wait_for_load_state('load')
    if 'window.MathJax' in file.read_text():
        page.wait_for_function('window.MathJax && MathJax.startup && MathJax.startup.promise')
        page.evaluate('MathJax.startup.promise')
        assert page.locator('mjx-merror').count()==0,url
    assert page.locator('img').evaluate_all('(els)=>els.every(e=>e.complete && e.naturalWidth>0)'),f'{url}: broken image'
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),f'{url}: mobile overflow'

if __name__=='__main__':
    options=args(__doc__);browser_checks(options.source.resolve(),options.site.resolve(),options.base_url)
