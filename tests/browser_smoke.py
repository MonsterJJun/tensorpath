"""Optional real-browser regression checks; run with playwright + Chromium."""
import argparse
import json
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright, Error, expect
from tensorpath.visualizer import export_html

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / 'examples/sample_traces'


def main(browser_path=None, screenshot=None):
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        tmp = Path(tmp)
        view = export_html(SAMPLES, 'healthy', output=tmp/'view.html')
        browser = p.chromium.launch(executable_path=browser_path, args=['--no-sandbox'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        requests = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda req: requests.append(req.url))
        try:
            page.goto(view.as_uri())
        except Error as exc:
            if 'ERR_BLOCKED_BY_ADMINISTRATOR' not in str(exc):
                raise
            print('Browser policy blocked file navigation; testing exported HTML via set_content instead')
            page.close()
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda req: requests.append(req.url))
            page.set_content(view.read_text(encoding='utf-8'))
        page.get_by_role('button', name='Event 2: embedding', exact=True).click()
        assert '[2, 8, 16]' in page.locator('#detail').inner_text()
        page.get_by_text('Call hierarchy diagram (visible page)', exact=True).click()
        assert page.locator('svg path').count() == 11
        if screenshot:
            page.screenshot(path=screenshot, full_page=True)
        page.get_by_role('button', name='Play', exact=True).click()
        page.wait_for_timeout(800)
        assert page.locator('#seek').input_value() == '2'
        page.get_by_role('button', name='Pause', exact=True).click()
        page.get_by_role('textbox', name='Filter module calls').fill('head')
        assert page.locator('.event').count() == 1
        page.get_by_role('button', name='Gradients', exact=True).click()
        assert '0.14681' in page.locator('#main').inner_text()
        page.locator('#upload').set_input_files(str(SAMPLES/'detached.json'))
        expect(page.locator("#runs")).to_have_value("detached")
        assert 'unused' in page.locator('#main').inner_text()
        page.get_by_role('button', name='Parameters', exact=True).click()
        assert 'embedding.weight' in page.locator('#main').inner_text()
        page.get_by_role('button', name='Compare', exact=True).click()
        page.get_by_label('Comparison section').select_option('gradients')
        assert 'modified' in page.locator('#main').inner_text()
        # Untrusted trace strings are text only, never executed.
        d = json.loads((SAMPLES/'healthy.json').read_text())
        d['trace_id'] = 'hostile'
        d['events'][0]['module'] = '<img src=x onerror=alert(1)>'
        bad = tmp/'hostile.json'; bad.write_text(json.dumps(d))
        page.locator('#upload').set_input_files(str(bad))
        expect(page.locator("#runs")).to_have_value("hostile")
        page.get_by_role('button', name='Execution', exact=True).click()
        assert '<img src=x onerror=alert(1)>' in page.locator('#main').inner_text()
        assert page.locator('img').count() == 0
        invalid = tmp/'bad.json'; invalid.write_text('{}')
        page.locator('#upload').set_input_files(str(invalid))
        expect(page.locator("#error")).to_contain_text("Invalid")
        assert not errors, errors
        assert not [x for x in requests if x.startswith(('http:', 'https:'))], requests
        # Small viewport should not produce horizontal page overflow.
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        browser.close()
    print('Browser smoke: PASS (selection, graph, playback, filter, gradients, compare, input validation, XSS, offline, mobile)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser-path')
    parser.add_argument('--screenshot')
    args = parser.parse_args()
    main(args.browser_path, args.screenshot)
