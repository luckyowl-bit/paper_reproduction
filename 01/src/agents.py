import os
import cv2
import numpy as np
import networkx as nx
from PIL import Image
from tqdm import tqdm


class FeatureMatchingAgent:
    """Extract SIFT features and compute pairwise matches."""

    def __init__(self, image_dir, max_features=500):
        self.image_dir = image_dir
        self.max_features = max_features
        self.images = []
        self.gray = []
        self.kps = []
        self.descs = []

    def load_images(self):
        paths = [os.path.join(self.image_dir, f) for f in sorted(os.listdir(self.image_dir))
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            self.images.append(img)
            self.gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        return self.images

    def extract(self):
        sift = cv2.SIFT_create(nfeatures=self.max_features)
        for g in self.gray:
            kp, des = sift.detectAndCompute(g, None)
            self.kps.append(kp)
            self.descs.append(des)
        return self.kps, self.descs

    def match(self):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = {}
        for i in range(len(self.images)):
            for j in range(i + 1, len(self.images)):
                if self.descs[i] is None or self.descs[j] is None:
                    continue
                m = bf.match(self.descs[i], self.descs[j])
                m = sorted(m, key=lambda x: x.distance)[:50]  # take best 50
                matches[(i, j)] = m
        return matches


class PatchSamplingAgent:
    """Sample patches around matched keypoints."""

    def __init__(self, images, keypoints, matches, patch_size=30):
        self.images = images
        self.keypoints = keypoints
        self.matches = matches
        self.patch_size = patch_size

    def sample(self):
        half = self.patch_size // 2
        samples = []
        for (i, j), mlist in self.matches.items():
            for m in mlist:
                pt_i = self.keypoints[i][m.queryIdx].pt
                pt_j = self.keypoints[j][m.trainIdx].pt
                patch_i = self._extract_patch(self.images[i], pt_i, half)
                patch_j = self._extract_patch(self.images[j], pt_j, half)
                if patch_i is not None and patch_j is not None:
                    samples.append({i: patch_i, j: patch_j})
        return samples

    def _extract_patch(self, img, pt, half):
        x, y = int(pt[0]), int(pt[1])
        h, w = img.shape[:2]
        if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h:
            return None
        patch = img[y - half:y + half, x - half:x + half]
        return patch


class MatrixBuilderAgent:
    """Build observation matrix I and mask W."""

    def __init__(self, samples, num_images):
        self.samples = samples
        self.num_images = num_images

    def build(self):
        n = len(self.samples)
        I = np.zeros((self.num_images, n), dtype=np.float32)
        W = np.zeros_like(I)
        for j, sample in enumerate(self.samples):
            for i, patch in sample.items():
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                val = np.mean(gray) + 1e-6  # avoid log(0)
                I[i, j] = np.log(val)
                W[i, j] = 1
        return I, W


class LowRankSolverAgent:
    """Compute rank-2 approximation of I."""

    def __init__(self, I, W):
        self.I = I
        self.W = W

    def solve(self):
        # naive: fill missing values with column mean
        I_filled = self.I.copy()
        for j in range(I_filled.shape[1]):
            col = I_filled[:, j]
            mask = self.W[:, j] == 1
            if np.any(mask):
                mean = np.mean(col[mask])
                col[~mask] = mean
                I_filled[:, j] = col
        U, S, Vt = np.linalg.svd(I_filled, full_matrices=False)
        P = U[:, :2] * S[:2]
        Q = Vt[:2, :].T
        return P, Q

    def extract_params(self, P):
        gamma = P[:, 1].copy()
        gamma[gamma == 0] = 1e-6
        log_c = P[:, 0] / gamma
        c = np.exp(log_c)
        return c, gamma


class ColorCorrectionAgent:
    """Apply color correction to images."""

    def __init__(self, images, c, gamma):
        self.images = images
        self.c = c
        self.gamma = gamma

    def correct(self):
        out = []
        for img, ci, gi in zip(self.images, self.c, self.gamma):
            img_corr = ((img.astype(np.float32) / ci) ** (1.0 / gi))
            img_corr = np.clip(img_corr, 0, 255).astype(np.uint8)
            out.append(img_corr)
        return out


class OutlierDetectionAgent:
    """Detect high residual patches."""

    def __init__(self, I, P, Q):
        self.I = I
        self.P = P
        self.Q = Q

    def detect(self):
        recon = self.P @ self.Q.T
        E = self.I - recon
        flat = np.abs(E).flatten()
        thresh = np.quantile(flat, 0.9)
        mask = np.abs(E) >= thresh
        return mask

