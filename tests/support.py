"""Shared portable content inventory, DOM parsing, build and browser utilities."""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import hashlib
import mimetypes
import os
import shlex
import subprocess
import tempfile
import threading
from urllib.parse import urlsplit, unquote
import urllib.request
import yaml

ROOT = Path(__file__).resolve().parents[1]

class UniqueLoader(yaml.SafeLoader):
    pass

def unique_mapping(loader, node, deep=False):
    explicit = set()
    for key, value in node.value:
        if key.tag == 'tag:yaml.org,2002:merge':
            continue
        name = loader.construct_object(key, deep=deep)
        if name in explicit:
            raise ValueError(f'duplicate YAML key: {name}')
        explicit.add(name)
    loader.flatten_mapping(node)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)

def load_yaml(text, context):
    try:
        return yaml.load(text, Loader=UniqueLoader)
    except (yaml.YAMLError, ValueError) as error:
        raise AssertionError(f'{context}: {error}') from error

def front_matter(path):
    text = path.read_text()
    parts = text.split('---', 2)
    assert text.startswith('---\n') and len(parts) == 3, f'{path}: missing YAML front matter'
    data = load_yaml(parts[1], path)
    assert isinstance(data, dict), f'{path}: front matter must be a mapping'
    return data, parts[2]

@dataclass
class Document:
    path: Path
    meta: dict
    body: str
    url: str
    day: str
    category: str = ''
    series: str = ''

class Inventory:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.config = load_yaml((self.root / '_config.yml').read_text(), '_config.yml')
        self.baseurl = self.config.get('baseurl', '').rstrip('/')
        self.topics = load_yaml((self.root / '_data/log_topics.yml').read_text(), 'log_topics.yml')
        self.writing = load_yaml((self.root / '_data/writing.yml').read_text(), 'writing.yml') or {}
        self.navigation = load_yaml((self.root / '_data/navigation.yml').read_text(), 'navigation.yml')
        self.canonical = {t['slug'] for t in self.topics}
        assert len(self.canonical) == len(self.topics), 'duplicate canonical topic'
        self.aliases = {}
        for topic in self.topics:
            assert topic['tags'] == [topic['slug']], 'topic tags must agree with its canonical slug'
            for alias in topic.get('aliases', []):
                assert alias not in self.aliases and alias not in self.canonical, f'ambiguous alias {alias}'
                self.aliases[alias] = topic['slug']
        self.posts = self.documents('_posts', posts=True)
        self.logs = self.documents('_log')
        self.counts = Counter(tag for doc in self.logs for tag in doc.meta['tags'])
        self.projects = [p for p in self.posts if p.category == 'projects']
        self.technical = [p for p in self.posts if p.category == 'technical']
        self.literature = [p for p in self.posts if p.category == 'literature']
        ids = [p.meta['research_programme'] for p in self.projects if p.meta.get('research_programme')]
        assert len(ids) == len(set(ids)), 'research programme IDs must be unique on projects'

    def documents(self, folder, posts=False):
        import re
        result = []
        for path in sorted((self.root / folder).rglob('*.md')):
            meta, body = front_matter(path)
            match = re.fullmatch(r'(\d{4}-\d{2}-\d{2})(?:-(.+))?', path.stem)
            assert match, f'{path}: filename must start with YYYY-MM-DD'
            date.fromisoformat(match[1])
            assert isinstance(meta.get('title'), str) and meta['title'].strip(), f'{path}: title required'
            raw_date = meta.get('date', match[1])
            day = str(raw_date)[:10]
            date.fromisoformat(day)
            if not posts:
                assert meta.get('layout') == 'log' and 'date' in meta, f'{path}: layout: log and date required'
                assert day == match[1], f'{path}: date must agree with filename'
                tags = meta.get('tags')
                assert isinstance(tags, list) and 1 <= len(tags) <= 2, f'{path}: choose one or two tags'
                assert all(isinstance(t, str) and t in self.canonical for t in tags), f'{path}: noncanonical tags {tags}'
                assert len(set(tags)) == len(tags), f'{path}: duplicate tags'
            if meta.get('published') is False:
                continue
            if posts and not self.config.get('future') and day > datetime.now(timezone.utc).date().isoformat():
                continue
            category = meta.get('category', '')
            if posts:
                assert category in {'projects', 'technical', 'literature', 'misc'}, f'{path}: valid category required'
                assert match[2], f'{path}: article filename needs a slug'
            slug = meta.get('slug', match[2] or '')
            permalink = meta.get('permalink') or (self.config['permalink'] if posts else self.config['collections']['log']['permalink'])
            for key, value in {'year':day[:4], 'month':day[5:7], 'day':day[8:10], 'title':slug, 'name':path.stem, 'categories':category}.items():
                permalink = permalink.replace(':' + key, value)
            assert ':' not in permalink, f'{path}: unsupported permalink token {permalink}'
            url = self.baseurl + permalink
            series = meta.get('series') or self.writing.get(permalink, {}).get('series', '')
            for key in ['research_programme', 'series']:
                assert not meta.get(key) or isinstance(meta[key], str), f'{path}: {key} must be a string'
            result.append(Document(path, meta, body, url, day, category, series))
        assert len({d.url for d in result}) == len(result), f'duplicate URLs in {folder}'
        return sorted(result, key=lambda d:(d.day, d.path.name))

@dataclass
class Node:
    tag: str
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)
    text: str = ''
    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()
    def has(self, cls):
        return cls in self.attrs.get('class', '').split()
    def links(self):
        return [n.attrs['href'] for n in self.walk() if n.tag == 'a' and 'href' in n.attrs]

class DOM(HTMLParser):
    VOID = {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}
    def __init__(self, text):
        super().__init__(); self.root=Node('document'); self.stack=[self.root]; self.feed(text)
    def handle_starttag(self, tag, attrs):
        node=Node(tag,dict(attrs)); self.stack[-1].children.append(node)
        if tag not in self.VOID:self.stack.append(node)
    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag,attrs)
        if tag not in self.VOID:self.stack.pop()
    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.stack[i].tag == tag:
                self.stack=self.stack[:i]; break
    def handle_data(self, data):
        for node in self.stack:node.text+=data

def output_file(site, url, baseurl=''):
    path=unquote(urlsplit(url).path)
    if baseurl and path.startswith(baseurl+'/'):path=path[len(baseurl):]
    target=Path(site)/path.lstrip('/')
    return target/'index.html' if path.endswith('/') or target.is_dir() else target

def tree(site, url, baseurl=''):
    path=output_file(site,url,baseurl)
    assert path.exists(), f'missing output {url}'
    return DOM(path.read_text()).root

def build(source, destination):
    command=shlex.split(os.environ.get('JEKYLL_COMMAND','bundle exec jekyll'))
    subprocess.run(command+['build','--source',str(source),'--config',str(Path(source)/'_config.yml'),'--destination',str(destination)],cwd=source,check=True)

def args(description):
    parser=argparse.ArgumentParser(description=description)
    parser.add_argument('--source',type=Path,default=ROOT)
    parser.add_argument('--site',type=Path,default=ROOT/'_site')
    parser.add_argument('--base-url',default=os.environ.get('SITE_URL'))
    return parser.parse_args()

@contextmanager
def server(site, base_url=None):
    if base_url:
        yield base_url.rstrip('/'); return
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    httpd=ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(Path(site).resolve())))
    thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{httpd.server_port}'
    finally:httpd.shutdown();httpd.server_close();thread.join()

def launch(playwright):
    options={}
    if os.environ.get('BROWSER_EXECUTABLE'):options['executable_path']=os.environ['BROWSER_EXECUTABLE']
    if os.environ.get('BROWSER_NO_SANDBOX')=='1':options['args']=['--no-sandbox']
    return playwright.chromium.launch(**options)

@contextmanager
def verified_cdn(page):
    """Optional proxy-CA workaround: Python's verified TLS, never ignore browser TLS errors."""
    with tempfile.TemporaryDirectory(prefix='notebook-cdn-') as directory:
        def intercept(route):
            url=route.request.url
            path=Path(directory)/hashlib.sha256(url.encode()).hexdigest()
            if not path.exists():
                with urllib.request.urlopen(url,timeout=30) as response:path.write_bytes(response.read())
            route.fulfill(body=path.read_bytes(),content_type=mimetypes.guess_type(url)[0] or 'application/octet-stream')
        if os.environ.get('BROWSER_CDN_PROXY')=='1':page.route('https://cdn.jsdelivr.net/**',intercept)
        yield
