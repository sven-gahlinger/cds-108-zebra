from crossing_detector.classmate_dataset import split_for_id


def test_split_for_id_is_stable():
    assert split_for_id("example-tile") == split_for_id("example-tile")
    assert split_for_id("example-tile") in {"train", "val", "test"}
