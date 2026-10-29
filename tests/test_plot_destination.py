"""Plot regeneration can write elsewhere while preserving archived results."""

import json
from pathlib import Path

import pytest

from pinn_activation_comparison import plot_results


def make_source(root):
    run = root / 'runs/tanh_seed_0'
    run.mkdir(parents=True)
    (run / 'metrics.json').write_text(json.dumps(dict(model='tanh', gamma=None, seed=0,
                                                    status='completed', relative_l2=.1)))
    (run / 'history.csv').write_text('epoch,total_loss,pde_loss,grad_norm\n1,1,.5,2\n')
    (root / 'figures').mkdir()
    (root / 'figures/original.png').write_bytes(b'keep-original')


def test_custom_destination_keeps_all_input_files_unchanged(tmp_path):
    source, destination = tmp_path / 'source', tmp_path / 'new_figures'
    make_source(source)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    paths = plot_results.plot_all(source, figures_dir=destination)
    assert paths and all(p.parent == destination and p.is_file() for p in paths)
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()} == before


def test_existing_custom_destination_is_rejected_without_writes(tmp_path):
    source, destination = tmp_path / 'source', tmp_path / 'existing'
    make_source(source)
    destination.mkdir()
    (destination / 'keep.txt').write_text('preserved')
    with pytest.raises(FileExistsError):
        plot_results.plot_all(source, figures_dir=destination)
    assert (destination / 'keep.txt').read_text() == 'preserved'


def test_cli_accepts_separate_output_directory(tmp_path, monkeypatch):
    source, destination = tmp_path / 'source', tmp_path / 'new'
    seen = {}
    def fake_plot(path, **kwargs):
        seen.update(source=path, **kwargs)
        return []
    monkeypatch.setattr(plot_results, 'plot_all', fake_plot)
    plot_results.main(['--results', str(source), '--output-dir', str(destination)])
    assert seen['source'] == source
    assert seen['figures_dir'] == destination
