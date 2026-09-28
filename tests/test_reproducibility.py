from fpsd_reassessment.reproducibility import git_commit


def test_git_commit_reads_packaged_source_commit(tmp_path):
    expected = "a" * 40
    (tmp_path / "SOURCE_COMMIT").write_text(expected + "\n", encoding="utf-8")
    assert git_commit(tmp_path) == expected
