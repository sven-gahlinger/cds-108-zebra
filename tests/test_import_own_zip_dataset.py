from crossing_detector.import_own_zip_dataset import ZipImage, split_images


def make_image(index: int, source_class: str) -> ZipImage:
    return ZipImage(
        zip_path=f"zebra/{source_class}/{index}.png",
        source_class=source_class,
        target_class="crossing" if source_class == "y" else "no_crossing",
        filename=f"{index}.png",
        length=10,
    )


def test_split_images_keeps_small_positive_class_in_all_splits() -> None:
    images = [make_image(index, "y") for index in range(17)]
    images.extend(make_image(index, "n") for index in range(100))

    split_map = split_images(
        images=images,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=108,
    )

    positive_counts = {
        split: sum(image.target_class == "crossing" for image in split_images)
        for split, split_images in split_map.items()
    }

    assert positive_counts == {"train": 14, "val": 2, "test": 1}
    assert sum(len(split_images) for split_images in split_map.values()) == 117
