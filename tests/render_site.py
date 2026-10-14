"""Optional full-site rendering audit, sharing browser/server and page checks.
Normal logbook regressions live in browser_site.py and are not repeated here.
"""
from support import Inventory, args, server, launch, verified_cdn
from browser_site import audit_page

def audit(source,site,base_url=None):
    from playwright.sync_api import sync_playwright
    inventory=Inventory(source);errors=[];files=list(site.rglob('*.html'))
    with server(site,base_url) as base, sync_playwright() as p:
        browser=launch(p);page=browser.new_page(viewport={'width':390,'height':844})
        page.on('pageerror',lambda e:errors.append(str(e)))
        with verified_cdn(page):
            for file in files:audit_page(page,file,site,base,inventory.baseurl)
        assert not errors,errors
        browser.close()
    print(f'PASS: all {len(files)} pages; MathJax, images, JavaScript and mobile overflow')

if __name__=='__main__':
    options=args(__doc__);audit(options.source.resolve(),options.site.resolve(),options.base_url)
