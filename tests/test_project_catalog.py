"""The organization tool must inventory existing artifacts without training."""

import json
from pathlib import Path

import pytest


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_catalog_keeps_failed_runs_and_does_not_invent_missing_conditions(tmp_path):
    from scripts.catalog_results import collect_experiments
    root = tmp_path / 'results'
    write_json(root / 'trial/experiment.json', {'config': {'epochs': 5000, 'models': ['tanh']}})
    write_json(root / 'trial/runs/good/metrics.json', {'model': 'tanh', 'seed': 0, 'status': 'completed', 'relative_l2': .1})
    write_json(root / 'trial/runs/fail/metrics.json', {'model': 'tanh', 'seed': 1, 'status': 'diverged', 'epochs_completed': 9})
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    experiments = collect_experiments(root, tmp_path)
    assert len(experiments) == 1
    record = experiments[0]
    assert record['category'] == 'experiment'
    assert len(record['runs']) == 2
    assert record['completed_runs'] == record['failed_runs'] == 1
    assert record['config'].get('optimizer') is None
    assert next(r for r in record['runs'] if r['seed'] == 1)['relative_l2'] is None
    assert {p: p.read_bytes() for p in before} == before


def test_reference_bundle_is_a_copy_not_an_additional_training_run(tmp_path):
    from scripts.catalog_results import collect_experiments
    origin = tmp_path / 'results/trial/runs/tanh_seed_0'
    origin.mkdir(parents=True)
    (origin / 'checkpoint.pt').write_bytes(b'same-checkpoint')
    reference = tmp_path / 'results/reference'
    write_json(reference / 'reference.json', {'source_run': str(origin)})
    write_json(reference / 'metrics.json', {'model': 'tanh', 'seed': 0, 'relative_l2': .002})
    (reference / 'checkpoint.pt').write_bytes(b'same-checkpoint')
    records = collect_experiments(tmp_path / 'results', tmp_path)
    assert len(records) == 1
    assert records[0]['category'] == 'reference_bundle'
    assert records[0]['reference_checkpoint_matches_source'] is True


def test_source_hash_mismatch_is_reported(tmp_path):
    import hashlib
    from scripts.catalog_results import collect_experiments
    exp = tmp_path / 'results/trial'
    write_json(exp / 'experiment.json', {'config': {}, 'source_sha256': {'train.py': hashlib.sha256(b'original').hexdigest()}})
    (exp / 'source').mkdir()
    (exp / 'source/train.py').write_bytes(b'changed')
    assert collect_experiments(tmp_path / 'results', tmp_path)[0]['source_integrity'] == 'mismatch'


def test_duplicates_use_file_contents_and_ignore_environment_and_caches(tmp_path):
    from scripts.catalog_results import inventory_files, duplicate_groups
    for name, content in [('a.py', b'123'), ('b.py', b'123'), ('c.py', b'456'),
                          ('.venv/copy', b'123'), ('__pycache__/copy', b'123')]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    files = inventory_files(tmp_path)
    assert {r['path'] for r in files} == {'a.py', 'b.py', 'c.py'}
    assert duplicate_groups(files) == [{'bytes_each': 3, 'sha256': files[0]['sha256'], 'paths': ['a.py', 'b.py']}]


def test_export_refuses_existing_output_and_preserves_it(tmp_path):
    from scripts.catalog_results import export_catalog
    output = tmp_path / 'reports'
    output.mkdir()
    existing = output / 'important.txt'
    existing.write_text('keep')
    with pytest.raises(FileExistsError):
        export_catalog(output, [], [], [])
    assert existing.read_text() == 'keep'


def test_export_has_run_level_rows_with_no_fake_zero_values(tmp_path):
    import csv
    from scripts.catalog_results import collect_experiments, export_catalog
    root = tmp_path / 'results'
    write_json(root / 'trial/experiment.json', {'config': {'epochs': 10}})
    write_json(root / 'trial/runs/one/metrics.json', {'model': 'tanh', 'seed': 0, 'relative_l2': 0.0})
    records = collect_experiments(root, tmp_path)
    output = tmp_path / 'reports'
    export_catalog(output, records, [], [])
    with (output / 'runs.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['relative_l2'] == '0.0'
    assert rows[0]['pde_loss'] == '不明'
    assert rows[0]['seed'] == '0'
