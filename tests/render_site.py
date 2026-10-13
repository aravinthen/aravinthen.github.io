"""Audit every built page on mobile, including MathJax, images and JS errors.
Uses the same verified-CDN workaround as browser_site.py in this cloud machine.
Run after serving the build: python tests/render_site.py
"""
import asyncio, json, pathlib, hashlib
from playwright.async_api import async_playwright
site=pathlib.Path('/workspace/site-tools/site');cache=pathlib.Path('/workspace/site-tools/mathjax-cache');queue=asyncio.Queue();findings=[];errors=[]
files=list(site.rglob('*.html'))
for f in files:queue.put_nowait(f)
async def main():
 async with async_playwright() as p:
  b=await p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
  async def worker():
   page=await b.new_page(viewport={'width':390,'height':844})
   page.on('pageerror',lambda e:errors.append(str(e)))
   async def asset(route):
    u=route.request.url;f=cache/hashlib.sha256(u.encode()).hexdigest()
    if not f.exists():
     proc=await asyncio.create_subprocess_exec('curl','-fsS',u,'-o',str(f));assert await proc.wait()==0
    await route.fulfill(body=f.read_bytes(),content_type='application/javascript' if '.js' in u else 'font/woff')
   await page.route('https://cdn.jsdelivr.net/**',asset)
   while not queue.empty():
    f=queue.get_nowait();route='/'+str(f.relative_to(site));await page.goto('http://127.0.0.1:4000'+route)
    if 'window.MathJax' in f.read_text():
     await page.wait_for_function('window.MathJax && window.MathJax.startup && window.MathJax.startup.promise')
     await page.evaluate('MathJax.startup.promise')
     m=await page.locator('mjx-merror').all_text_contents()
     if m:findings.append({'page':route,'math_errors':m})
    broken=await page.locator('img').evaluate_all('(images)=>images.filter(e=>!e.complete || !e.naturalWidth).map(e=>e.getAttribute("src"))')
    if broken:findings.append({'page':route,'images':broken})
    if not await page.evaluate('document.documentElement.scrollWidth <= innerWidth'):
     findings.append({'page':route,'overflow':True})
    queue.task_done()
   await page.close()
  await asyncio.gather(*(worker() for _ in range(4)))
  await b.close()
asyncio.run(main());pathlib.Path('/workspace/site-tools/render-findings.json').write_text(json.dumps({'pages':len(files),'findings':findings,'js_errors':errors},indent=2));print(json.dumps({'pages':len(files),'findings':findings,'js_errors':errors},indent=2))

assert not findings and not errors, 'Rendering findings; see /workspace/site-tools/render-findings.json'
