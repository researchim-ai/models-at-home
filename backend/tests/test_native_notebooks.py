import json
import time
from pathlib import Path

import pytest
from backend.routes import notebooks
from backend.services import notebook_runtime as runtime


@pytest.fixture
def notebook_client(client, workspace, monkeypatch):
    directory = workspace / 'notebooks'
    directory.mkdir()
    monkeypatch.setattr(notebooks, 'NOTEBOOKS_DIR', directory)
    yield client, directory
    runtime.shutdown()


def document(code='x = 41\nprint(x)'):
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}, 'cells': [
        {'id': 'test-cell', 'cell_type': 'code', 'source': code, 'metadata': {}, 'execution_count': None, 'outputs': []}]}


def wait(client, session):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = client.get(f"/api/notebooks/sessions/{session['id']}")
        assert response.status_code == 200, response.text
        session = response.json()
        if session['execution']['status'] != 'running':
            return session
        time.sleep(0.05)
    pytest.fail('Execution did not finish')


def execute(client, session, code):
    response = client.post(f"/api/notebooks/sessions/{session['id']}/execute", json={'code': code, 'cell_id': 'test-cell'})
    assert response.status_code == 200, response.text
    return wait(client, response.json())


def test_document_conflicts_validation_and_paths(notebook_client, tmp_path):
    client, directory = notebook_client
    saved = client.put('/api/notebooks/document', json={'path': 'mine.ipynb', 'notebook': document()})
    assert saved.status_code == 200
    loaded = client.get('/api/notebooks/document', params={'path': 'mine.ipynb'}).json()
    assert loaded['notebook']['cells'][0]['source'].startswith('x =')
    assert client.put('/api/notebooks/document', json={'path': 'mine.ipynb', 'notebook': document()}).status_code == 409
    assert client.put('/api/notebooks/document', json={'path': 'mine.ipynb', 'notebook': document('x = 2'), 'revision': loaded['revision']}).status_code == 200
    assert client.get('/api/notebooks/document', params={'path': '../secret.ipynb'}).status_code == 400
    (directory / 'escape.ipynb').symlink_to(tmp_path / 'secret.ipynb')
    assert client.put('/api/notebooks/document', json={'path': 'escape.ipynb', 'notebook': document()}).status_code == 400
    assert client.put('/api/notebooks/document', json={'path': 'bad.ipynb', 'notebook': {'nbformat': 3}}).status_code == 422
    assert client.get('/api/notebooks/document', params={'path': 'missing.ipynb'}).status_code == 404


def test_library_languages(notebook_client):
    client, directory = notebook_client
    for language in ('ru', 'en'):
        data = document()
        data['metadata']['studio'] = {'title': language, 'language': language, 'lesson': True, 'topic': 'llm'}
        (directory / f'{language}.ipynb').write_text(json.dumps(data))
    data = client.get('/api/notebooks/library?lang=en').json()
    assert [entry['title'] for entry in data['notebooks']] == ['en']


def test_real_kernel_state_errors_interrupt_and_isolation(notebook_client):
    pytest.importorskip('ipykernel')
    client, directory = notebook_client
    (directory / 'mine.ipynb').write_text(json.dumps(document()))
    response = client.post('/api/notebooks/sessions', json={'path': 'mine.ipynb'})
    assert response.status_code == 200, response.text
    session = response.json()
    session = execute(client, session, 'x = 41\nprint(x)')
    assert session['execution']['status'] == 'ok'
    session = execute(client, session, 'print(x + 1)')
    assert '42' in ''.join(output.get('text', '') for output in session['execution']['outputs'])
    session = execute(client, session, 'raise ValueError("intentional")')
    assert session['execution']['status'] == 'error' and session['state'] == 'idle'
    session = execute(client, session, 'from IPython.display import display, HTML\ndisplay(HTML("<script>alert(1)</script>"))')
    assert all('text/html' not in output.get('data', {}) for output in session['execution']['outputs'])
    running = client.post(f"/api/notebooks/sessions/{session['id']}/execute", json={'code': 'import time\ntime.sleep(30)', 'cell_id': 'test-cell'})
    assert running.status_code == 200
    assert client.post(f"/api/notebooks/sessions/{session['id']}/execute", json={'code': '1', 'cell_id': 'test-cell'}).status_code == 409
    time.sleep(0.5)
    assert client.post(f"/api/notebooks/sessions/{session['id']}/interrupt").status_code == 200
    session = wait(client, running.json())
    assert session['execution']['status'] == 'error'
    assert execute(client, session, 'print(x)')['execution']['status'] == 'ok'
    second = client.post('/api/notebooks/sessions', json={'path': 'mine.ipynb'}).json()
    assert execute(client, second, 'print(x)')['execution']['status'] == 'error'
    assert client.delete(f"/api/notebooks/sessions/{session['id']}").status_code == 200
    assert client.get(f"/api/notebooks/sessions/{session['id']}").status_code == 404


def test_curriculum_real_execution(tmp_path):
    """Every RU/EN lesson must execute, including actual backward and plots."""
    pytest.importorskip('ipykernel')
    pytest.importorskip('torch')
    from scripts.build_notebook_lessons import ROOT, LESSONS
    files = sorted(ROOT.glob('*/*.ipynb'))
    assert len(files) == len(LESSONS) * 2 == 32
    import nbformat
    # Both languages have the same executable code. Validate every notebook,
    # execute one copy of every unique lesson in a fresh kernel.
    code_sets = {}
    for file in files:
        doc = json.loads(file.read_text())
        nbformat.validate(nbformat.from_dict(doc))
        codes = tuple(c['source'] for c in doc['cells'] if c['cell_type'] == 'code')
        code_sets.setdefault(codes, file)
    assert len(code_sets) == 16
    try:
        for codes, file in code_sets.items():
            session = runtime.start(tmp_path, file.name)
            for idx, code in enumerate(codes):
                runtime.get(session['id']).execute(code, str(idx))
                deadline = time.monotonic() + 35
                while True:
                    status = runtime.get(session['id']).snapshot()
                    if status['execution']['status'] != 'running':
                        break
                    assert time.monotonic() < deadline, file
                    time.sleep(0.05)
                assert status['execution']['status'] == 'ok', (file.name, status)
                if 'plt.plot' in code or 'plt.bar' in code:
                    assert any('image/png' in output.get('data', {}) for output in status['execution']['outputs']), file
            runtime.stop(session['id'])
    finally:
        runtime.shutdown()
