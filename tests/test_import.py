def test_import():
    import data_repo_gen
    assert hasattr(data_repo_gen, '__version__')
