"""Fast current-content validation. No Git baselines or migration-report counts.
Usage: python3 tests/check_site.py [SITE] [--source CHECKOUT]
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from urllib.parse import urlsplit, unquote
import xml.etree.ElementTree as ET
from support import ROOT, Inventory, DOM, tree, output_file

def validate(source, site):
    inventory=Inventory(source);site=Path(site).resolve();errors=[]
    pages={path:DOM(path.read_text()).root for path in site.rglob('*.html')}
    assert pages, 'build the site before running validation'
    host=urlsplit(inventory.config['url']).netloc
    for path, page in pages.items():
        ids=[n.attrs['id'] for n in page.walk() if 'id' in n.attrs]
        assert len(ids)==len(set(ids)), f'{path}: duplicate IDs'
        for node in page.walk():
            ref=node.attrs.get('href') if node.tag in {'a','link'} else node.attrs.get('src') if node.tag in {'img','script'} else None
            if not ref:continue
            url=urlsplit(ref)
            if (url.netloc and url.netloc!=host) or (url.scheme and url.scheme not in {'http','https'}):continue
            if not url.path:target=path
            elif url.path.startswith('/') or url.netloc:target=output_file(site,ref,inventory.baseurl)
            else:
                target=(path.parent/unquote(url.path)).resolve()
                if target.is_dir():target=target/'index.html'
            if not target.exists():errors.append(f'{path.relative_to(site)}: {ref}');continue
            if url.fragment and target.suffix=='.html':
                linked=pages.get(target) or DOM(target.read_text()).root
                if unquote(url.fragment) not in {n.attrs.get('id') for n in linked.walk()}:errors.append(f'{path.relative_to(site)}: missing fragment {ref}')
    for route in ['/', '/about/', '/research/', '/literature/', '/misc/', '/log/', '/log/timelines/', '/projects/', '/technical/']:
        tree(site,inventory.baseurl+route,inventory.baseurl)
    for item in inventory.navigation:
        assert output_file(site,inventory.baseurl+item['url'],inventory.baseurl).exists(),item
    archive=tree(site,inventory.baseurl+'/log/',inventory.baseurl)
    rows=[n for n in archive.walk() if n.has('log-item')]
    expected={d.url:d for d in inventory.logs}
    assert Counter(r.attrs['data-url'] for r in rows)==Counter(expected.keys()), 'log index must contain every current log once'
    for row in rows:
        doc=expected[row.attrs['data-url']]
        assert row.attrs['data-tags'].split('|')==doc.meta['tags'],doc.path
        assert row.attrs['data-date']==doc.day,doc.path
    actual_counts=Counter(tag for row in rows for tag in row.attrs['data-tags'].split('|'))
    assert actual_counts==inventory.counts,'tag counts do not match current logs'
    taxonomy=next(n for n in archive.walk() if n.attrs.get('id')=='log-taxonomy')
    assert json.loads(taxonomy.text)==inventory.topics,'client taxonomy differs from YAML'
    search=json.loads(output_file(site,inventory.baseurl+'/log/search.json',inventory.baseurl).read_text())
    assert Counter(e['url'] for e in search)==Counter(expected.keys()),'full-text search index must contain every log once'
    assert all(isinstance(e.get('body'),str) for e in search),'invalid full-text body'
    timelines=tree(site,inventory.baseurl+'/log/timelines/',inventory.baseurl)
    for topic in inventory.topics:
        docs=[d for d in inventory.logs if topic['slug'] in d.meta['tags']]
        sections=[n for n in timelines.walk() if n.attrs.get('id')==topic['slug']]
        if not docs:assert not sections;continue
        assert len(sections)==1,topic['slug']
        listing=next(n for n in sections[0].walk() if n.has('log-timeline-list'))
        assert listing.links()==[d.url for d in docs],f"timeline ordering/count: {topic['slug']}"
    research=tree(site,inventory.baseurl+'/research/',inventory.baseurl)
    main=next(n for n in research.walk() if n.tag=='main')
    for doc in inventory.projects+inventory.technical:
        assert main.links().count(doc.url)==1,f'{doc.path}: research listing must appear once'
    for project in inventory.projects:
        key=project.meta.get('research_programme')
        if not key:continue
        section=next(n for n in main.walk() if n.attrs.get('data-programme')==key)
        associated=[d for d in inventory.technical if d.meta.get('research_programme')==key]
        assert [u for u in section.links() if u in {d.url for d in inventory.technical}]==[d.url for d in associated],f'programme {key}: associations/order'
    independent=[d for d in inventory.technical if not any(d.meta.get('research_programme') and d.meta.get('research_programme')==p.meta.get('research_programme') for p in inventory.projects)]
    if independent:
        section=next(n for n in main.walk() if any(c.attrs.get('id')=='independent-heading' for c in n.children))
        assert set(section.links())=={d.url for d in independent},'independent notes missing'
    literature=tree(site,inventory.baseurl+'/literature/',inventory.baseurl)
    main_lit=next(n for n in literature.walk() if n.tag=='main')
    for doc in inventory.literature:assert main_lit.links().count(doc.url)==1,f'{doc.path}: missing/duplicated literature listing'
    series=defaultdict(list)
    for doc in inventory.posts:
        if doc.series:series[doc.series].append(doc)
    for index,doc in enumerate(inventory.logs):
        page=tree(site,doc.url,inventory.baseurl)
        prev=[n.attrs['href'] for n in page.walk() if n.has('log-pagination-prev')]
        nxt=[n.attrs['href'] for n in page.walk() if n.has('log-pagination-next')]
        assert prev==([inventory.logs[index-1].url] if index else []),f'{doc.path}: previous log'
        assert nxt==([inventory.logs[index+1].url] if index+1<len(inventory.logs) else []),f'{doc.path}: next log'
    for doc in inventory.posts:
        page=tree(site,doc.url,inventory.baseurl)
        assert sum(n.tag=='h1' for n in page.walk())==1,f'{doc.path}: article heading'
        if doc.category in {'projects','technical'}:
            breadcrumb=next(n for n in page.walk() if n.has('research-breadcrumbs'))
            assert inventory.baseurl+'/research/' in breadcrumb.links()
            if doc.category=='technical':
                parent=next((p for p in inventory.projects if doc.meta.get('research_programme') and p.meta.get('research_programme')==doc.meta.get('research_programme')),None)
                if parent:assert parent.url in breadcrumb.links(),f'{doc.path}: missing programme breadcrumb'
        if doc.series:
            siblings=series[doc.series];i=siblings.index(doc)
            nav=next(n for n in page.walk() if n.has('article-pagination'))
            prev=[n.attrs['href'] for n in nav.walk() if n.attrs.get('rel')=='prev'];nxt=[n.attrs['href'] for n in nav.walk() if n.attrs.get('rel')=='next']
            assert prev==([siblings[i-1].url] if i else []),f'{doc.path}: series previous'
            assert nxt==([siblings[i+1].url] if i+1<len(siblings) else []),f'{doc.path}: series next'
    ET.parse(site/'feed.xml')
    assert not errors,'Broken internal links:\n'+'\n'.join(errors)
    print(f'PASS: {len(pages)} pages; {len(inventory.posts)} articles; {len(inventory.logs)} logs; current taxonomy, indexing, relationships, navigation, RSS and internal links')
    return inventory

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site',nargs='?',type=Path,default=ROOT/'_site')
    parser.add_argument('--source',type=Path,default=ROOT)
    options=parser.parse_args();validate(options.source,options.site)
