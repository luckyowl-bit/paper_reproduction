import os
from agents import (FeatureMatchingAgent, PatchSamplingAgent,
                    MatrixBuilderAgent, LowRankSolverAgent,
                    ColorCorrectionAgent, OutlierDetectionAgent)


def main(image_dir, output_dir):
    fm = FeatureMatchingAgent(image_dir)
    fm.load_images()
    fm.extract()
    matches = fm.match()

    ps = PatchSamplingAgent(fm.images, fm.kps, matches)
    samples = ps.sample()

    mb = MatrixBuilderAgent(samples, len(fm.images))
    I, W = mb.build()

    solver = LowRankSolverAgent(I, W)
    P, Q = solver.solve()
    c, gamma = solver.extract_params(P)

    corrector = ColorCorrectionAgent(fm.images, c, gamma)
    corrected = corrector.correct()

    os.makedirs(output_dir, exist_ok=True)
    for idx, img in enumerate(corrected):
        path = os.path.join(output_dir, f"corrected_{idx}.jpg")
        cv2.imwrite(path, img)

    outlier = OutlierDetectionAgent(I, P, Q)
    mask = outlier.detect()
    np.save(os.path.join(output_dir, "outlier_mask.npy"), mask)


if __name__ == "__main__":
    import argparse
    import cv2
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="../datasets")
    parser.add_argument("--output", default="../output")
    args = parser.parse_args()
    main(args.images, args.output)
