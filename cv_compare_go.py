from group_project.cv_compare import compare_images
import os
import cv2
import random

test_files_dir = "cv_compare_test_files"

test_files = {}

for root, dirs, files in os.walk(test_files_dir):
    for file in files:
        if file.endswith(".png"):
            _, category = root.split("/")
            if category not in test_files:
                test_files[category] = []
            test_files[category].append(os.path.join(root, file))


def get_random_file():
    category = random.choice(list(test_files.keys()))
    return category, random.choice(test_files[category])


def test_compare_images():
    for _ in range(20):
        category1, file1 = get_random_file()
        category2, file2 = get_random_file()
        img1 = cv2.imread(file1)
        img2 = cv2.imread(file2)

        cv2.imshow("img1", img1)
        cv2.imshow("img2", img2)

        similar = compare_images(img1, img2)
        print()
        print(f"{category1} and {category2} are similar: {similar}")
        print()

        cv2.waitKey(0)


if __name__ == "__main__":
    test_compare_images()
