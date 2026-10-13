"""Check new documents and optional metadata using a temporary source copy.
Run with the local Jekyll environment activated. No fixture touches the checkout.
"""
from pathlib import Path
import shutil, subprocess, tempfile
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='notebook-indexing-',dir='/tmp') as folder:
 source=Path(folder)/'source';output=Path(folder)/'site'
 shutil.copytree(root,source,ignore=shutil.ignore_patterns('.git','tests','reports'))
 (source/'_posts/technical/2026-10-12-index-fixture.md').write_text('---\ntitle: Index fixture\ncategory: technical\ntopic: Fixture group\nstatus: Fixture status\n---\n')
 (source/'_posts/projects/2026-10-12-project-fixture.md').write_text('---\ntitle: Project fixture\ncategory: projects\nstatus: Fixture status\nrepository: https://github.com/aravinthen/deep_rl_experiments\ntopics: [fixture]\n---\n')
 (source/'_posts/literature/2026-10-12-series-fixture.md').write_text('---\ntitle: Series fixture\ncategory: literature\nseries: Fixture series\n---\n')
 (source/'_log/2026-10-14.md').write_text('---\nlayout: log\ntitle: Activity fixture\ndate: 2026-10-14\ntags: [post-training]\n---\n')
 subprocess.run(['ruby','/workspace/site-tools/root/usr/bin/jekyll','build','--config',str(source/'_config.yml'),'--source',str(source),'--destination',str(output),'--disable-disk-cache'],check=True,cwd=source)
 for path,text in [('index.html','Index fixture'),('index.html','Project fixture'),('index.html','Activity fixture'),('technical/index.html','Fixture group'),('projects/index.html','Fixture status'),('literature/index.html','Fixture series'),('log/index.html','Activity fixture'),('log/timelines/index.html','Activity fixture')]:
  assert text in (output/path).read_text(),(path,text)
print('PASS: newly added technical/project/literature/log documents index automatically; optional grouping/status/repository metadata renders')
