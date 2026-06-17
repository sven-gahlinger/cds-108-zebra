from pathlib import Path

from PIL import Image

from crossing_detector.predict import find_image_paths, unique_output_path


def test_find_image_paths_recurses_supported_images(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    Image.new("RGB", (4, 4), color="white").save(tmp_path / "a.png")
    Image.new("RGB", (4, 4), color="black").save(nested / "b.jpg")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    paths = find_image_paths(tmp_path)

    assert paths == [tmp_path / "a.png", nested / "b.jpg"]


def test_unique_output_path_adds_suffix_when_name_exists(tmp_path):
    existing = tmp_path / "image.png"
    existing.write_bytes(b"already here")

    assert unique_output_path(tmp_path, "image.png") == tmp_path / "image_001.png"
