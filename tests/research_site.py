"""Research relationships are part of the shared fast site validation.
Retained command for developers using the former research-only check.
"""
from support import args
from check_site import validate
if __name__=='__main__':
    options=args(__doc__);validate(options.source,options.site)
