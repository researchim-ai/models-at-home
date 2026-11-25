"""Execute the paper studies, not merely compile their Python source."""
import json
import time
from pathlib import Path

import pytest

from backend.services import notebook_runtime as runtime
from scripts.build_paper_notebooks import ROOT, PAPERS


def test_paper_catalog_and_bilingual_contracts():
    import nbformat
    files = sorted(ROOT.glob('*/*.ipynb'))
    assert len(files) == len(PAPERS) * 2 == 20
    for item in PAPERS:
        documents = [json.loads((ROOT / language / f"{item['slug']}.ipynb").read_text()) for language in ('ru', 'en')]
        for doc in documents:
            nbformat.validate(nbformat.from_dict(doc))
            info = doc['metadata']['studio']
            assert info['track'] == 'papers' and info['paper']['url'].startswith('https://arxiv.org/abs/')
            assert len(doc['cells']) >= 16
            codes = [cell['source'] for cell in doc['cells'] if cell['cell_type'] == 'code']
            assert len(codes) >= 4 and sum('assert ' in source for source in codes) >= 2
            assert all(not cell.get('outputs') for cell in doc['cells'])
        assert [c['source'] for c in documents[0]['cells'] if c['cell_type'] == 'code'] == [c['source'] for c in documents[1]['cells'] if c['cell_type'] == 'code']


@pytest.mark.parametrize('slug', [paper['slug'] for paper in PAPERS])
def test_paper_actual_kernel(slug, tmp_path):
    pytest.importorskip('ipykernel')
    pytest.importorskip('torch')
    doc = json.loads((ROOT / 'en' / f'{slug}.ipynb').read_text())
    session = runtime.start(tmp_path, slug)
    try:
        for cell in doc['cells']:
            if cell['cell_type'] != 'code':
                continue
            runtime.get(session['id']).execute(cell['source'], cell['id'])
            deadline = time.monotonic() + 40
            while True:
                status = runtime.get(session['id']).snapshot()
                if status['execution']['status'] != 'running':
                    break
                assert time.monotonic() < deadline, (slug, cell['id'])
                time.sleep(0.05)
            assert status['execution']['status'] == 'ok', (slug, cell['id'], status)
            if 'plt.show()' in cell['source']:
                assert any('image/png' in output.get('data', {}) for output in status['execution']['outputs']), (slug, cell['id'])
    finally:
        runtime.stop(session['id'])


def test_paper_library_metadata(client, workspace, monkeypatch):
    from backend.routes import notebooks
    monkeypatch.setattr(notebooks, 'NOTEBOOKS_DIR', ROOT)
    for language in ('en', 'ru'):
        response = client.get(f'/api/notebooks/library?lang={language}')
        assert response.status_code == 200
        entries = response.json()['notebooks']
        assert len(entries) == 10
        assert all(entry['track'] == 'papers' and entry['paper']['year'] >= 2017 for entry in entries)
