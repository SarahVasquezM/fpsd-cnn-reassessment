from fpsd_reassessment.reproducibility import git_commit, source_tree_sha256


def test_git_commit_reads_packaged_source_commit(tmp_path):
    expected = "a" * 40
    (tmp_path / "SOURCE_COMMIT").write_text(expected + "\n", encoding="utf-8")
    assert git_commit(tmp_path) == expected


def test_source_tree_hash_changes_with_scientific_code(tmp_path):
    source = tmp_path / "src" / "package"
    source.mkdir(parents=True)
    file = source / "module.py"
    file.write_text("VALUE = 1\n", encoding="utf-8")
    first = source_tree_sha256(tmp_path)
    assert len(first) == 64
    file.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_tree_sha256(tmp_path) != first
