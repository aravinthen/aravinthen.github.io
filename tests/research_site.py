"""Research aggregation, front-matter-only preservation, redirects and typography.
Run with Jekyll serving the build on port 4000.
"""
from pathlib import Path
import subprocess, re
from html.parser import HTMLParser
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1]
baseline='dbfc0d85c09898d30c979d7b707b12e16f843926'
changed={
 '_posts/projects/2026-01-25-deep_reinforcement.md':'deep-reinforcement-learning',
 '_posts/projects/2026-07-26-foundation_models.md':'foundation-models-post-training',
 '_posts/technical/2026-02-28-lessons_from_classical_rl.md':'deep-reinforcement-learning',
 '_posts/technical/2026-04-18-reinforce.md':'deep-reinforcement-learning',
 '_posts/technical/2026-05-30-ppo.md':'deep-reinforcement-learning',
}
for path in list(root.glob('_posts/**/*.md'))+list(root.glob('_log/*.md'))+[root/'about/index.md']:
 relative=str(path.relative_to(root));before=subprocess.check_output(['git','show',f'{baseline}:{relative}'],cwd=root);after=path.read_bytes()
 if relative in changed:
  assert re.sub(rb'(?m)^research_programme:.*\n',b'',after,count=1)==before,relative
  assert f'research_programme: {changed[relative]}'.encode() in after
 else:assert after==before,relative
# Individual post output URLs have not changed.
for post in root.glob('_posts/**/*.md'):
 url=post.stem[:10].replace('-','/')+'/'+post.stem[11:]+'/index.html'
 assert (Path('/workspace/site-tools/site')/url).exists(),url
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 page=browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 for width in [1280,390]:
  page.set_viewport_size({'width':width,'height':900})
  page.goto('http://127.0.0.1:4000/research/')
  assert page.locator('.section-nav a').all_text_contents()==['Research','Literature','Logbook','About']
  assert page.locator('.research-programme').count()==3
  for post in list(root.glob('_posts/projects/*.md'))+list(root.glob('_posts/technical/*.md')):
   url='/'+post.stem[:10].replace('-','/')+'/'+post.stem[11:]+'/'
   assert page.locator(f'main a[href="{url}"]').count()==1,url
  rl=page.locator('[data-programme="deep-reinforcement-learning"]')
  assert rl.locator('.programme-notes .writing-list li').count()==3
  foundation=page.locator('[data-programme="foundation-models-post-training"]')
  assert foundation.locator('.programme-notes').count()==0
  assert rl.locator('a[href="https://github.com/aravinthen/deep_rl_experiments"]').count()==1
  assert page.locator('.programme-excerpt').count()==3
  assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
  out=Path('/workspace/site-tools/screenshots');out.mkdir(exist_ok=True)
  page.screenshot(path=str(out/f'research-{width}.png'),full_page=True)
  fonts=[]
  for route,selector in [('/2026/05/30/ppo/','.article-content p'),('/log/2026-10-13/','.log-content li'),('/about/','main p'),('/','main p')]:
   page.goto('http://127.0.0.1:4000'+route,wait_until='domcontentloaded')
   fonts.append(page.locator(selector).first.evaluate('(e)=>getComputedStyle(e).fontFamily'))
  assert len(set(fonts))==1,fonts
  for route in ['projects','technical']:
   page.goto(f'http://127.0.0.1:4000/{route}/')
   page.wait_for_url('**/research/')
   assert page.locator('.section-nav [aria-current=page]').inner_text()=='Research'
  page.goto('http://127.0.0.1:4000/2026/05/30/ppo/',wait_until='domcontentloaded')
  assert page.locator('.research-breadcrumbs a').count()==2
  assert page.locator('.research-breadcrumbs a').last.get_attribute('href')=='/2026/01/25/deep_reinforcement/'
  page.goto('http://127.0.0.1:4000/2026/01/25/deep_reinforcement/')
  # Existing authored navigation links are not repeated by generated navigation.
  for slug,date in [('lessons_from_classical_rl','2026/02/28'),('reinforce','2026/04/18'),('ppo','2026/05/30')]:
   assert page.locator(f'main a[href$="/{date}/{slug}/"]').count()==1
  page.goto('http://127.0.0.1:4000/')
  assert page.locator('main a[href="/research/"]').count()==1
  assert page.locator('main a[href="/projects/"], main a[href="/technical/"]').count()==0
  assert page.locator('.site-footer a[href="/misc/"]').count()==1
  page.goto('http://127.0.0.1:4000/misc/')
  assert page.locator('.site-footer [aria-current=page]').inner_text()=='Miscellaneous'
 assert not errors,errors
 browser.close()
print('PASS: all programmes and notes exactly once, explicit RL relationships, Foundation unassociated; 316 Markdown files unchanged except five metadata additions; original URLs, redirects, nav, breadcrumbs, excerpts, desktop/mobile single font, no JS errors')
