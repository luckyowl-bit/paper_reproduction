import os
import logging
import cv2
import numpy as np
from agents import (
    FeatureMatchingAgent,
    PatchSamplingAgent,
    MatrixBuilderAgent,
    LowRankSolverAgent,
    ColorCorrectionAgent,
    OutlierDetectionAgent,
)


def main(image_dir: str, output_dir: str) -> None:
    logging.info("Loading images from %s", image_dir)
    fm = FeatureMatchingAgent(image_dir)
    fm.load_images()
    fm.extract()
    matches = fm.match()

    logging.info("Sampling patches")
    ps = PatchSamplingAgent(fm.images, fm.kps, matches)
    samples = ps.sample()

    logging.info("Building observation matrix")
    mb = MatrixBuilderAgent(samples, len(fm.images))
    I, W = mb.build()

    logging.info("Solving low-rank approximation")
    solver = LowRankSolverAgent(I, W)
    P, Q = solver.solve()
    c, gamma = solver.extract_params(P)

    logging.info("Applying color correction")
    corrector = ColorCorrectionAgent(fm.images, c, gamma)
    corrected = corrector.correct()

    os.makedirs(output_dir, exist_ok=True)
    for idx, img in enumerate(corrected):
        path = os.path.join(output_dir, f"corrected_{idx}.jpg")
        cv2.imwrite(path, img)
        logging.info("Saved %s", path)

    logging.info("Detecting outliers")
    outlier = OutlierDetectionAgent(I, P, Q)
    mask = outlier.detect()
    mask_path = os.path.join(output_dir, "outlier_mask.npy")
    np.save(mask_path, mask)
    logging.info("Outlier mask saved to %s", mask_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="../datasets")
    parser.add_argument("--output", default="../output")
    args = parser.parse_args()
    main(args.images, args.output)
