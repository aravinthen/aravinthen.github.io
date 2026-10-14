"""Opt-in PR preservation check; never part of routine publishing acceptance.
Usage: python3 tests/preserve_changes.py BASELINE [--maintenance]
The maintenance option allows only development config, tests and docs changes.
"""
import argparse
import subprocess
from pathlib import Path
from support import ROOT
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('baseline')
parser.add_argument('--maintenance',action='store_true')
options=parser.parse_args()
changed=subprocess.check_output(['git','diff','--name-only',options.baseline,'--'],cwd=ROOT,text=True).splitlines()
for name in changed:
    protected=name.startswith(('_posts/','_log/','assets/','_layouts/','_includes/','_data/','reports/')) or name in {'index.html','about/index.md','research/index.html','log/index.html','log/timelines/index.html','feed.xml','literature/index.html','misc/index.html','projects/index.html','technical/index.html'}
    if options.maintenance:
        assert not protected,f'Unexpected production/historical change: {name}'
    elif name.startswith(('_posts/','_log/')) or name=='about/index.md':
        # General authorship-preserving tasks can supply their own parent revision.
        before=subprocess.check_output(['git','show',f'{options.baseline}:{name}'],cwd=ROOT)
        after=(ROOT/name).read_bytes()
        assert before.split(b'---',2)[2]==after.split(b'---',2)[2],f'Body changed: {name}'
print(f'PASS: change-specific preservation against {options.baseline}')
