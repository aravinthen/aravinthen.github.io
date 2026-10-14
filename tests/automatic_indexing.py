"""Simulate ordinary publishing in a disposable source copy; no permanent fixtures.
JEKYLL_COMMAND optionally overrides the default 'bundle exec jekyll'.
Pass --browser to also exercise the expanded taxonomy in the browser.
"""
import argparse
from datetime import date, timedelta
from pathlib import Path
import shutil
import tempfile
import yaml
from support import ROOT, Inventory, build
from check_site import validate

def growth(root=ROOT,browser=False):
    root=Path(root).resolve();inventory=Inventory(root)
    with tempfile.TemporaryDirectory(prefix='notebook-growth-') as folder:
        source=Path(folder)/'source';output=Path(folder)/'output'
        shutil.copytree(root,source,ignore=shutil.ignore_patterns('.git','_site','.venv','.bundle','.jekyll-cache','.sass-cache','__pycache__'))
        tag=next(iter(sorted(inventory.canonical)))
        second=next(t for t in sorted(inventory.canonical) if t!=tag)
        # Pick unused, past publication dates to work regardless of today's date.
        first=date.fromisoformat(min(d.day for d in inventory.logs))-timedelta(days=3)
        def publish(path,metadata,body='Fixture text.\n'):
            path=source/path;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('---\n'+yaml.safe_dump(metadata,sort_keys=False)+'---\n'+body)
        for offset,tags in [(0,[tag]),(1,[tag,second])]:
            day=(first+timedelta(days=offset)).isoformat()
            publish(f'_log/{day}.md',{'layout':'log','title':'Maintenance fixture','date':day,'tags':tags})
        programme=next(p.meta['research_programme'] for p in inventory.projects if p.meta.get('research_programme'))
        series=next(d.series for d in inventory.literature if d.series)
        first_post=date.fromisoformat(min(p.day for p in inventory.posts))-timedelta(days=20)
        def post(offset,category,slug,**extra):
            day=(first_post+timedelta(days=offset)).isoformat()
            publish(f'_posts/{category}/{day}-{slug}.md',{'title':slug,'category':category,**extra})
        post(0,'technical','fixture-associated',research_programme=programme)
        post(1,'technical','fixture-independent')
        post(2,'projects','fixture-programme',research_programme='maintenance-fixture')
        post(3,'literature','fixture-series',series=series)
        post(4,'literature','fixture-unaffiliated')
        # An unmatched programme is intentionally indexed as independent.
        post(5,'technical','fixture-orphan',research_programme='unmatched-fixture')
        edited=source/inventory.technical[0].path.relative_to(root)
        with edited.open('a') as handle:handle.write('\nFixture paragraph added in the temporary copy.\n')
        topics_path=source/'_data/log_topics.yml'
        topics=yaml.safe_load(topics_path.read_text())
        topics.append({'title':'Maintenance fixture topic','slug':'maintenance-fixture-topic','tags':['maintenance-fixture-topic'],'aliases':['maintenance-old-topic']})
        topics_path.write_text(yaml.safe_dump(topics,sort_keys=False))
        day=(first+timedelta(days=2)).isoformat()
        publish(f'_log/{day}.md',{'layout':'log','title':'New taxonomy fixture','date':day,'tags':['maintenance-fixture-topic']})
        build(source,output)
        grown=validate(source,output)
        assert len(grown.logs)==len(inventory.logs)+3
        assert len(grown.posts)==len(inventory.posts)+6
        if browser:
            from browser_site import browser_checks
            browser_checks(source,output)
    print('PASS: new daily/dual-tag logs, programme note, independent/orphan notes, project, series/unaffiliated reviews, edited article and expanded canonical taxonomy; fixtures removed')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT)
    parser.add_argument('--browser',action='store_true')
    options=parser.parse_args();growth(options.source,options.browser)
