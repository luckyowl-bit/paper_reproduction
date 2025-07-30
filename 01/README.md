# Efficient and Robust Color Consistency for Community Photo Collections

This folder contains an initial implementation for reproducing the algorithm from Park et al. (2013). The code relies on open source tools such as OpenCV, NumPy and SciPy. The dataset used for quick experiments is provided in `datasets/`.

The main entry point is `src/pipeline.py` which demonstrates how to execute the individual agents defined in the project. The pipeline now performs
ratio-test feature matching, patch alignment using keypoint orientation, builds sparse matrices and solves for low-rank factors with a weighted ALS
routine.
