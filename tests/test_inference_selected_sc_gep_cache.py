from pathlib import Path

from vaedecon.workflow.inference import _selected_sc_gep_cache_complete


def test_selected_sc_gep_cache_complete_requires_manifest_and_all_cell_type_files(tmp_path: Path):
    sc_gep_result_dir = tmp_path / "sc_gep"
    sc_gep_result_dir.mkdir(parents=True)
    selected_fp = sc_gep_result_dir / "selected_100_samples2sct_ids.csv"
    selected_fp.write_text("sample_id,selected_cell_id,cell_type\n", encoding="utf-8")

    cell_types = ["A", "B"]
    n_samples = 100

    assert not _selected_sc_gep_cache_complete(
        sc_gep_result_dir=sc_gep_result_dir,
        cell_types=cell_types,
        n_samples=n_samples,
        selected_sample2cell_id_file_path=selected_fp,
    )

    for cell_type in cell_types:
        (sc_gep_result_dir / f"sct_gep_{cell_type}_from_{n_samples}_bulksamples.csv").write_text(
            "gene,sample\n",
            encoding="utf-8",
        )

    assert _selected_sc_gep_cache_complete(
        sc_gep_result_dir=sc_gep_result_dir,
        cell_types=cell_types,
        n_samples=n_samples,
        selected_sample2cell_id_file_path=selected_fp,
    )
