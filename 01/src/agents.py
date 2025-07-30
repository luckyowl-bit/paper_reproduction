import os
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

import cv2
import numpy as np
from scipy import sparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


@dataclass
class PatchPair:
    """Store a pair of aligned patches from two images."""
    idx_i: int
    idx_j: int
    patch_i: np.ndarray
    patch_j: np.ndarray


class FeatureMatchingAgent:
    """Extract SIFT features and compute pairwise matches."""

    def __init__(self, image_dir: str, max_features: int = 500):
        self.image_dir = image_dir
        self.max_features = max_features
        self.images = []
        self.gray = []
        self.kps = []
        self.descs = []

    def load_images(self) -> List[np.ndarray]:
        paths = [p for p in sorted(os.listdir(self.image_dir)) if p.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for name in paths:
            path = os.path.join(self.image_dir, name)
            img = cv2.imread(path)
            if img is None:
                logging.warning("Failed to load %s", name)
                continue
            self.images.append(img)
            self.gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        return self.images

    def extract(self) -> Tuple[List[List[cv2.KeyPoint]], List[np.ndarray]]:
        for g in self.gray:
            h, w = g.shape
            nfeat = int(self.max_features * max(h, w) / 500)
            sift = cv2.SIFT_create(nfeatures=max(nfeat, self.max_features))
            kp, des = sift.detectAndCompute(g, None)
            self.kps.append(kp)
            self.descs.append(des)
        return self.kps, self.descs

    def match(self) -> Dict[Tuple[int, int], List[cv2.DMatch]]:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = {}
        for i in range(len(self.images)):
            for j in range(i + 1, len(self.images)):
                if self.descs[i] is None or self.descs[j] is None:
                    continue
                knn = bf.knnMatch(self.descs[i], self.descs[j], k=2)
                good = []
                for m, n in knn:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                matches[(i, j)] = good
        return matches


class PatchSamplingAgent:
    """Sample aligned patches around matched keypoints."""

    def __init__(self, images: List[np.ndarray], keypoints: List[List[cv2.KeyPoint]],
                 matches: Dict[Tuple[int, int], List[cv2.DMatch]], patch_size: int = 30):
        self.images = images
        self.keypoints = keypoints
        self.matches = matches
        self.patch_size = patch_size

    def sample(self) -> List[PatchPair]:
        samples: List[PatchPair] = []
        for (i, j), mlist in self.matches.items():
            for m in mlist:
                kp_i = self.keypoints[i][m.queryIdx]
                kp_j = self.keypoints[j][m.trainIdx]
                patch_i = self._extract_patch(self.images[i], kp_i)
                patch_j = self._extract_patch(self.images[j], kp_j)
                if patch_i is not None and patch_j is not None:
                    samples.append(PatchPair(i, j, patch_i, patch_j))
        return samples

    def _extract_patch(self, img: np.ndarray, kp: cv2.KeyPoint) -> np.ndarray:
        half = self.patch_size // 2
        M = cv2.getRotationMatrix2D(kp.pt, kp.angle, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        patch = cv2.getRectSubPix(rotated, (self.patch_size, self.patch_size), kp.pt)
        return patch


class MatrixBuilderAgent:
    """Build observation matrix ``I`` and mask ``W`` in log space."""

    def __init__(self, samples: List[PatchPair], num_images: int):
        self.samples = samples
        self.num_images = num_images

    def build(self) -> Tuple['scipy.sparse.csr_matrix', 'scipy.sparse.csr_matrix']:
        from scipy import sparse

        n = len(self.samples)
        data_I: List[float] = []
        rows: List[int] = []
        cols: List[int] = []
        for j, sample in enumerate(self.samples):
            for idx, patch in ((sample.idx_i, sample.patch_i), (sample.idx_j, sample.patch_j)):
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                val = np.clip(np.mean(gray) / 255.0, 1e-6, 1.0)
                data_I.append(np.log(val))
                rows.append(idx)
                cols.append(j)
        I = sparse.csr_matrix((data_I, (rows, cols)), shape=(self.num_images, n), dtype=np.float32)
        W = sparse.csr_matrix((np.ones(len(data_I)), (rows, cols)), shape=(self.num_images, n), dtype=np.float32)
        return I, W


class LowRankSolverAgent:
    """Compute rank-2 approximation of ``I`` using weighted ALS."""

    def __init__(self, I: 'scipy.sparse.csr_matrix', W: 'scipy.sparse.csr_matrix'):
        self.I = I
        self.W = W

    def solve(self, max_iter: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        m, n = self.I.shape
        P = np.random.randn(m, 2)
        Q = np.random.randn(n, 2)
        I = self.I.tocsr()
        W = self.W.tocsr()
        for _ in range(max_iter):
            # update P
            for i in range(m):
                idx = W[i].indices
                if len(idx) == 0:
                    continue
                Q_i = Q[idx]
                b = I[i].data
                A = Q_i.T @ Q_i + 1e-6 * np.eye(2)
                P[i] = np.linalg.solve(A, Q_i.T @ b)
            # update Q
            for j in range(n):
                idx = W[:, j].indices
                if len(idx) == 0:
                    continue
                P_j = P[idx]
                b = I[:, j].data
                A = P_j.T @ P_j + 1e-6 * np.eye(2)
                Q[j] = np.linalg.solve(A, P_j.T @ b)
        return P, Q

    def extract_params(self, P):
        gamma = P[:, 1].copy()
        gamma = np.clip(gamma, 1e-3, None)
        log_c = P[:, 0] / gamma
        c = np.exp(log_c)
        return c, gamma


class ColorCorrectionAgent:
    """Apply color correction to images."""

    def __init__(self, images: List[np.ndarray], c: np.ndarray, gamma: np.ndarray):
        self.images = images
        self.c = c
        self.gamma = gamma

    def correct(self):
        out = []
        for img, ci, gi in zip(self.images, self.c, self.gamma):
            img_f = img.astype(np.float32)
            img_corr = (img_f / ci) ** (1.0 / gi)
            img_corr = np.clip(img_corr, 0, 255).astype(np.uint8)
            out.append(img_corr)
        return out


class OutlierDetectionAgent:
    """Detect high residual patches."""

    def __init__(self, I, P, Q):
        self.I = I
        self.P = P
        self.Q = Q

    def detect(self) -> np.ndarray:
        recon = self.P @ self.Q.T
        I_dense = self.I.toarray() if hasattr(self.I, "toarray") else self.I
        E = I_dense - recon
        med = np.median(E)
        mad = np.median(np.abs(E - med))
        thresh = med + 3 * 1.4826 * mad
        mask = np.abs(E - med) >= thresh
        return mask

