"""Optional browser checks: uses an existing Playwright installation and Chromium.
Run after jekyll serve: python tests/browser_site.py
"""
from playwright.sync_api import sync_playwright
import json, pathlib, subprocess, hashlib
root=pathlib.Path(__file__).resolve().parents[1]
report=json.loads((root/'reports/log-tag-migration.json').read_text())
base='http://127.0.0.1:4000'
errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 page=browser.new_page()
 page.on('pageerror',lambda e:errors.append(str(e)))
 # Chromium lacks the egress proxy CA. Fetch CDN assets through curl's
 # verified system trust store and fulfill them unchanged for this test.
 cache=pathlib.Path('/workspace/site-tools/mathjax-cache');cache.mkdir(exist_ok=True)
 def cdn_asset(route):
  url=route.request.url
  asset=cache/hashlib.sha256(url.encode()).hexdigest()
  if not asset.exists():subprocess.run(['curl','-fsS',url,'-o',str(asset)],check=True)
  mime='application/javascript' if '.js' in url else 'font/woff'
  route.fulfill(body=asset.read_bytes(),content_type=mime)
 page.route('https://cdn.jsdelivr.net/**',cdn_asset)
 for width,height in [(1280,900),(390,844)]:
  page.set_viewport_size({'width':width,'height':height})
  for route,active in [('/',None),('/research/','Research'),('/about/','About'),('/projects/','Research'),('/technical/','Research'),('/literature/','Literature'),('/misc/',None),('/log/','Logbook'),('/log/timelines/','Logbook'),('/2026/05/30/ppo/','Research'),('/2026/10/03/bluedot4/','Literature'),('/log/2026-01-21/','Logbook')]:
   assert page.goto(base+route).status==200,route
   if active: assert page.locator('.section-nav [aria-current=page]').inner_text()==active,(route,active)
   else: assert page.locator('.section-nav [aria-current=page]').count()==0
   assert page.locator('main').inner_text().strip(),route
   assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),(width,route,'horizontal overflow')
   if route in ['/2026/05/30/ppo/','/2026/10/03/bluedot4/']:
    assert page.locator('.research-article h1').count()==1
    assert page.locator('.article-toc').is_visible()
   if route=='/2026/05/30/ppo/':
    assert page.locator('a[href="https://github.com/aravinthen/deep_rl_experiments"]').count()>0
   if route=='/log/2026-01-21/':
    page.wait_for_selector('mjx-container',timeout=30000)
    assert page.locator('mjx-merror').count()==0,'MathJax parse error'
   if route=='/':
    page.keyboard.press('Tab');assert page.locator('.skip-link').evaluate('(el)=>el===document.activeElement')
    page.keyboard.press('Enter');assert page.locator('#content').evaluate('(el)=>el===document.activeElement')
  page.goto(base+'/log/')
  assert page.locator('#log-tag-filters input').count()==8
  assert page.locator('.log-item').count()==285
  for tag,count in report['new_frequencies'].items():
   page.goto(base+'/log/?tag='+tag)
   page.wait_for_function('(count)=>document.querySelectorAll(".log-item:not([hidden])").length===count',arg=count)
   assert page.locator(f'input[value="{tag}"]').is_checked()
  page.goto(base+'/log/?tag=game-theory')
  assert page.locator('input[value="multi-agent"]').is_checked()
  page.wait_for_function('document.querySelectorAll(".log-item:not([hidden])").length===49')
  page.goto(base+'/log/?tag=AI%20safety')
  assert page.locator('input[value="ai-safety"]').is_checked()
  page.locator('#log-clear-filters').click()
  page.locator('#log-search').fill('Kolmogorov')
  page.wait_for_function('document.querySelectorAll(".log-item:not([hidden])").length===1')
  assert page.locator('.log-item:not([hidden])').get_attribute('data-date')=='2026-02-18'
  assert not page.locator('#log-search-warning').is_visible()
  page.locator('#log-search').fill('zzzz-nothing-matches-12345')
  page.wait_for_selector('#log-empty:not([hidden])')
  page.locator('#log-clear-filters').click()
  page.locator('#log-month-select').select_option('2026-02')
  page.wait_for_function('document.querySelectorAll(".log-item:not([hidden])").length===28')
  assert page.locator('.log-month:not([hidden])').count()==1
  page.locator('#log-clear-filters').click()
  page.locator('#log-collapse-all').click()
  assert page.locator('.log-month[open]').count()==0
  page.locator('#log-expand-all').click()
  assert page.locator('.log-month[open]').count()==10
  page.locator('.log-calendar-panel summary').click()
  assert page.locator('#log-calendar a.has-entry').count()==285
  page.locator('#log-calendar a.has-entry').first.click()
  assert '/log/2026-01-02/' in page.url
  assert page.locator('.log-pagination-next').count()==1
  assert page.locator('.log-related li').count()>0
  page.goto(base+'/log/timelines/#post-training-reasoning')
  assert page.locator('#post-training').evaluate('(e)=>e.open')
  for tag,count in report['new_frequencies'].items():
   assert page.locator(f'#{tag} .log-timeline-list li').count()==count
  page.goto(base+'/2026/09/16/bluedot2/')
  assert page.locator('.article-pagination a[rel=prev]').get_attribute('href')=='/2026/09/11/bluedot1/'
  assert page.locator('.article-pagination a[rel=next]').get_attribute('href')=='/2026/09/28/bluedot3/'
  page.goto(base+'/2026/04/18/reinforce/')
  images=page.locator('.article-content img')
  assert images.count()>0
  assert all(images.nth(i).evaluate('(e)=>e.complete && e.naturalWidth>0') for i in range(images.count()))
  out=pathlib.Path('/workspace/site-tools/screenshots');out.mkdir(exist_ok=True)
  page.goto(base+'/');page.screenshot(path=str(out/f'home-{width}.png'),full_page=True)
  page.goto(base+'/log/');page.screenshot(path=str(out/f'log-{width}.png'),full_page=True)
 # Existing articles without a math flag must still render their equations.
 for route in ['/2026/03/07/literature_review/', '/2026/09/11/bluedot1/']:
  page.goto(base+route)
  page.wait_for_selector('mjx-container')
  assert page.locator('mjx-merror').count()==0
 # Full-text fetch failure remains usable for title/tag searches.
 page.route('**/log/search.json',lambda route:route.abort())
 page.goto(base+'/log/?q=Phantom')
 page.wait_for_selector('#log-search-warning:not([hidden])')
 assert page.locator('.log-item:not([hidden])').count()>0
 assert not errors,errors
 browser.close()
print('PASS: desktop/mobile pages, navigation, focus, overflow, article titles/TOC, MathJax, images, series links; search, fallback, all tags/counts, aliases, monthly archives, timelines, calendar, related/next entries; no JS page errors')
