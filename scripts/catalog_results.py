"""List saved experiments without training, changing data, or importing PyTorch.

The default command is read-only. --export-dir writes a catalog into a new,
empty directory. --duplicates additionally hashes non-cache project files.
Absent metadata remains null in JSON and '不明' in CSV; it is never guessed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path

IGNORED_DIRECTORIES = {'.venv', '.git', '__pycache__', '.pytest_cache'}
RUN_FIELDS = ('model', 'gamma', 'seed', 'optimizer', 'status', 'epochs_requested',
              'epochs_completed', 'lbfgs_steps_completed', 'lbfgs_closure_evaluations',
              'relative_l2', 'residual_rms', 'total_loss', 'pde_loss', 'ic_loss',
              'ic_value_loss', 'ic_velocity_loss', 'bc_loss', 'ic_velocity_relative_l2',
              'train_seconds', 'has_nonfinite', 'divergence_detected')


def _read_json(path):
    """Require an object and retain the source path in malformed-data errors."""
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return data


def _relative(path, project):
    """Use portable project-relative links where a saved path permits it."""
    try:
        return path.resolve().relative_to(project.resolve()).as_posix()
    except ValueError:
        return str(path)


def _hash(path):
    """Hash complete file bytes with bounded memory, including large weights."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_integrity(directory, declared):
    """Compare archived source to recorded hashes without changing either."""
    if not declared:
        return '不明', []
    differences = []
    for name, expected in declared.items():
        path = directory / 'source' / name
        if not path.is_file() or _hash(path) != expected:
            differences.append(name)
    return ('mismatch' if differences else 'verified'), differences


def collect_experiments(results_root, project_root):
    """Collect experiment runs and explicitly distinguish reference copies."""
    results_root, project_root = Path(results_root), Path(project_root)
    records = []
    for directory in sorted(results_root.iterdir()):
        if not directory.is_dir() or directory.name in IGNORED_DIRECTORIES:
            continue
        manifest = directory / 'experiment.json'
        reference_path = directory / 'reference.json'
        reference = {}
        if manifest.is_file():
            metadata = _read_json(manifest)
            config = metadata.get('config', {})
            category = 'experiment'
            metrics_paths = sorted((directory / 'runs').glob('*/metrics.json'))
            declared = metadata.get('source_sha256', {})
            configuration_path = manifest
        elif reference_path.is_file() and (directory / 'metrics.json').is_file():
            reference = _read_json(reference_path)
            metadata = reference
            configuration_path = directory / 'config.json'
            config = _read_json(configuration_path) if configuration_path.is_file() else {}
            category = 'reference_bundle'
            metrics_paths = [directory / 'metrics.json']
            declared = reference.get('runtime_source_sha256', {})
        else:
            continue
        if not isinstance(config, dict):
            raise ValueError(f'Expected config object in {configuration_path}')
        runs = []
        for metrics_path in metrics_paths:
            metrics = _read_json(metrics_path)
            run = {key: metrics.get(key) for key in RUN_FIELDS}
            if run['optimizer'] is None:
                run['optimizer'] = config.get('optimizer')
            if run['epochs_requested'] is None:
                run['epochs_requested'] = config.get('epochs')
            run.update(run_id=metrics_path.parent.name, metrics_path=_relative(metrics_path, project_root),
                       checkpoint_path=_relative(metrics_path.parent / 'checkpoint.pt', project_root)
                       if (metrics_path.parent / 'checkpoint.pt').is_file() else None)
            runs.append(run)
        integrity, differences = _source_integrity(directory, declared)
        log = directory.parent / (directory.name + '.log')
        record = dict(name=directory.name, category=category, path=_relative(directory, project_root),
                      config=config, configuration_path=_relative(configuration_path, project_root),
                      status=metadata.get('status'), parent=config.get('resume'),
                      runs=runs, completed_runs=sum(r['status'] == 'completed' for r in runs),
                      failed_runs=sum(r['status'] in ('diverged', 'nonfinite', 'failed') for r in runs),
                      source_path=_relative(directory / 'source', project_root), source_integrity=integrity,
                      source_differences=differences, log_path=_relative(log, project_root) if log.is_file() else None,
                      figures_path=_relative(directory / 'figures', project_root),
                      dense_validation=reference.get('dense_validation'),
                      random_validation=reference.get('random_validation'))
        if reference:
            saved_source = Path(reference.get('source_run', ''))
            if not saved_source.is_absolute():
                saved_source = project_root / saved_source
            original, copied = saved_source / 'checkpoint.pt', directory / 'checkpoint.pt'
            record.update(reference_source_run=_relative(saved_source, project_root),
                          reference_checkpoint_matches_source=(_hash(original) == _hash(copied))
                          if original.is_file() and copied.is_file() else None)
        records.append(record)
    return sorted(records, key=lambda record: (record['category'] != 'reference_bundle', record['name']))


def inventory_files(project_root):
    """Hash research files while excluding environments, caches and Finder data."""
    root, files = Path(project_root), []
    for directory, subdirectories, names in os.walk(root):
        subdirectories[:] = sorted(name for name in subdirectories
                                   if name not in IGNORED_DIRECTORIES and not (Path(directory) / name).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            if name == '.DS_Store' or path.is_symlink() or not path.is_file():
                continue
            files.append(dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=_hash(path)))
    return sorted(files, key=lambda row: row['path'])


def duplicate_groups(files):
    """Group nonempty files only after exact SHA-256/content-size matching."""
    groups = defaultdict(list)
    for row in files:
        if row['bytes']:
            groups[(row['bytes'], row['sha256'])].append(row['path'])
    return [dict(bytes_each=size, sha256=digest, paths=sorted(paths))
            for (size, digest), paths in sorted(groups.items(), key=lambda item: item[1][0]) if len(paths) > 1]


def _safe_json(value):
    """Keep nonfinite recorded values unknown without producing invalid JSON."""
    if isinstance(value, dict):
        return {key: _safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def export_catalog(destination, experiments, files, duplicates):
    """Write a new report set, refusing to overwrite any nonempty directory."""
    destination = Path(destination)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f'Export directory must be empty: {destination}')
    destination.mkdir(parents=True, exist_ok=True)
    payloads = {'experiments.json': experiments, 'files.json': files, 'duplicates.json': duplicates}
    for name, data in payloads.items():
        (destination / name).write_text(json.dumps(_safe_json(data), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (destination / 'runs.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['experiment', 'category', 'run_id', *RUN_FIELDS, 'metrics_path', 'checkpoint_path']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for experiment in experiments:
            for run in experiment['runs']:
                row = dict(experiment=experiment['name'], category=experiment['category'], **run)
                cells = {key: '不明' if value is None else value for key, value in _safe_json(row).items()}
                if row['model'] in ('tanh', 'siren') and row['gamma'] is None:
                    cells['gamma'] = '対象外'
                writer.writerow(cells)
    with (destination / 'duplicates.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['group', 'bytes_each', 'sha256', 'path'])
        writer.writeheader()
        for number, group in enumerate(duplicates, start=1):
            for path in group['paths']:
                writer.writerow(dict(group=number, bytes_each=group['bytes_each'], sha256=group['sha256'], path=path))


def main(argv=None):
    """List saved outcomes or export new metadata, without executing training."""
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=project)
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--export-dir', type=Path)
    parser.add_argument('--duplicates', action='store_true')
    args = parser.parse_args(argv)
    results = args.results_root or args.project_root / 'pinn_activation_comparison/results'
    try:
        experiments = collect_experiments(results, args.project_root)
        for experiment in experiments:
            config = experiment['config']
            print(f"{experiment['name']:24} {experiment['category']:17} "
                  f"completed={experiment['completed_runs']}/{len(experiment['runs'])} "
                  f"failed={experiment['failed_runs']} Adam={config.get('epochs', '不明')} "
                  f"source={experiment['source_integrity']}")
        if args.export_dir:
            files = inventory_files(args.project_root) if args.duplicates else []
            duplicates = duplicate_groups(files)
            export_catalog(args.export_dir, experiments, files, duplicates)
            print(f'Catalog: {args.export_dir.resolve()}')
        elif args.duplicates:
            duplicates = duplicate_groups(inventory_files(args.project_root))
            for group in duplicates:
                print(f"SHA256 {group['sha256']} ({group['bytes_each']} bytes each):")
                for path in group['paths']:
                    print(f'  {path}')
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
