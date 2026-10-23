from fpsd_reassessment.manifests import fold_index


def test_fold_index_reads_structured_fold() -> None:
    assert fold_index({"fold": {"fold": 17}}) == 17


def test_fold_index_remains_compatible_with_flat_fold() -> None:
    assert fold_index({"fold": 4}) == 4
