import unittest

import cv2
import numpy as np

from utils.manas_preprocessing import find_candidate_regions, subtraction_morphology


class ManasPreprocessingTests(unittest.TestCase):
    def test_identical_images_produce_an_empty_mask(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)

        _, _, _, morphology, _, _ = subtraction_morphology(image, image.copy())

        self.assertEqual(np.count_nonzero(morphology), 0)

    def test_a_clear_difference_is_kept_as_a_candidate(self):
        reference = np.zeros((120, 160, 3), dtype=np.uint8)
        defective = reference.copy()
        cv2.rectangle(defective, (60, 40), (90, 70), (255, 255, 255), -1)

        _, _, _, morphology, _, _ = subtraction_morphology(
            reference,
            defective,
            kernel_size=3,
            iterations=1,
        )
        regions = find_candidate_regions(morphology)

        self.assertEqual(len(regions), 1)
        self.assertGreater(regions[0]["area"], 800)

    def test_even_kernel_size_is_rejected(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            subtraction_morphology(image, image, kernel_size=4)


if __name__ == "__main__":
    unittest.main()
