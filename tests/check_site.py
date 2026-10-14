"""Build-output and authorship checks; Python standard library only.
Usage: python tests/check_site.py /path/to/_site [baseline-git-ref]
"""
import collections, json, pathlib, re, subprocess, sys
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import xml.etree.ElementTree as ET
root=pathlib.Path(__file__).resolve().parents[1]
site=pathlib.Path(sys.argv[1]);baseline=sys.argv[2] if len(sys.argv)>2 else 'origin/main'
class Page(HTMLParser):
 def __init__(self,text):
  super().__init__();self.refs=[];self.ids=[];self.h1=0;self.rows=[];self.feed(text)
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.append(a['id'])
  if tag=='h1':self.h1+=1
  if tag in ['a','link'] and a.get('href'):self.refs.append(a['href'])
  if tag in ['img','script'] and a.get('src'):self.refs.append(a['src'])
  if 'log-item' in a.get('class','').split():self.rows.append(a)
files=list(site.rglob('*.html'));pages={p:Page(p.read_text()) for p in files}
errors=[]
for path,page in pages.items():
 if len(set(page.ids))!=len(page.ids):errors.append((str(path.relative_to(site)),'duplicate IDs'))
 for ref in page.refs:
  u=urlsplit(ref)
  if u.netloc and u.netloc!='aravinthen.github.io':continue
  if u.scheme and u.scheme not in ['http','https']:continue
  target=site/unquote(u.path).lstrip('/') if u.path.startswith('/') or u.netloc else path.parent/unquote(u.path)
  if not u.path:target=path
  if target.is_dir():target=target/'index.html'
  if not target.exists():errors.append((str(path.relative_to(site)),ref));continue
  if u.fragment and target.suffix=='.html':
   parsed=pages.get(target) or Page(target.read_text())
   if unquote(u.fragment) not in parsed.ids:errors.append((str(path.relative_to(site)),ref+' [missing fragment]'))
for route in ['','about','projects','technical','literature','misc','log','log/timelines']:
 assert (site/route/'index.html').exists(),route
posts=list(root.glob('_posts/**/*.md'));logs=list(root.glob('_log/*.md'))
for p in posts+[root/'about/index.md']:
 old=subprocess.check_output(['git','show',f'{baseline}:{p.relative_to(root)}'],cwd=root)
 assert p.read_bytes()==old,p
for p in logs:
 old=subprocess.check_output(['git','show',f'{baseline}:{p.relative_to(root)}'],cwd=root)
 assert re.sub(rb'(?m)^tags:.*$',b'tags: <tags>',old,count=1)==re.sub(rb'(?m)^tags:.*$',b'tags: <tags>',p.read_bytes(),count=1),p
for filename in ['index.html','projects/index.html','technical/index.html','literature/index.html','misc/index.html']:
 old=subprocess.check_output(['git','show',f'{baseline}:{filename}'],cwd=root).decode()
 assert all(p in (root/filename).read_text() for p in re.findall(r'<p>.*?</p>',old,re.S)),filename
archive=pages[site/'log/index.html'];assert len(archive.rows)==len(logs)==285
index=json.loads((site/'log/search.json').read_text());assert len(index)==285
assert {e['url'] for e in index}=={r['data-url'] for r in archive.rows}
report=json.loads((root/'reports/log-tag-migration.json').read_text());counts=collections.Counter(t for r in archive.rows for t in r['data-tags'].split('|'))
assert dict(counts)==report['new_frequencies']
for p in posts:
 slug=p.stem[11:];url=p.stem[:10].replace('-','/')+'/'+slug+'/index.html'
 assert pages[site/url].h1==1,(url,pages[site/url].h1)
ET.parse(site/'feed.xml')
print(f'PASS: {len(files)} HTML pages, {len(posts)} articles, 285 log bodies/metadata, search JSON, taxonomy counts, RSS, index prose')
print(f'Internal-link findings: {len(errors)}')
(site.parent/'link-findings.json').write_text(json.dumps(errors,indent=2))
for finding in errors[:25]:print(finding)

assert not errors, 'Broken internal links or duplicate IDs; see link-findings.json'
