import cv2
import numpy as np
import logging
from kalman import KalmanFilter
from utils import get_camera_matrix, get_distortion_coeffs, validate_pose

class OptimizedTrajectoryTracker:
    def __init__(self, use_kalman=True, camera_matrix=None, dist_coeffs=None):
        self.positions = []
        self.current_pos = np.array([0., 0., 0.])
        self.current_rot = np.eye(3)
        self.scale = 1.0
        self.use_kalman = use_kalman
        
        # Kamera parametreleri
        self.camera_matrix = camera_matrix if camera_matrix is not None else get_camera_matrix()
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else get_distortion_coeffs()
        
        if self.use_kalman:
            self.kalman = KalmanFilter(dt=1.0)
        
        # Feature extraction parametreleri
        self.feature_params = {
            'SIFT': {'nfeatures': 2000, 'contrastThreshold': 0.04, 'edgeThreshold': 10},
            'ORB': {'nfeatures': 2000, 'scaleFactor': 1.2, 'nlevels': 8},
            'AKAZE': {'threshold': 0.003, 'nOctaves': 4, 'nOctaveLayers': 4}
        }

    def extract_features_optimized(self, img, method='SIFT', max_features=2000):
        """Optimize edilmiş feature extraction"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Histogram eşitleme (düşük kontrast için)
        gray = cv2.equalizeHist(gray)
        
        # Gaussian blur (noise azaltma)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        if method == 'SIFT':
            params = self.feature_params['SIFT']
            detector = cv2.SIFT_create(
                nfeatures=min(max_features, params['nfeatures']),
                contrastThreshold=params['contrastThreshold'],
                edgeThreshold=params['edgeThreshold']
            )
        elif method == 'ORB':
            params = self.feature_params['ORB']
            detector = cv2.ORB_create(
                nfeatures=min(max_features, params['nfeatures']),
                scaleFactor=params['scaleFactor'],
                nlevels=params['nlevels']
            )
        elif method == 'AKAZE':
            params = self.feature_params['AKAZE']
            detector = cv2.AKAZE_create(
                threshold=params['threshold'],
                nOctaves=params['nOctaves'],
                nOctaveLayers=params['nOctaveLayers']
            )
        else:
            raise ValueError(f"Desteklenmeyen method: {method}")
        
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        
        # Feature quality filtering
        if keypoints and len(keypoints) > max_features:
            # Response'a göre sırala ve en iyileri al
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
            keypoints = keypoints[:max_features]
            if descriptors is not None:
                # Descriptor'ları da aynı sırayla al
                indices = [kp.class_id for kp in keypoints if hasattr(kp, 'class_id')]
                if len(indices) == len(keypoints):
                    descriptors = descriptors[indices]
                else:
                    descriptors = descriptors[:max_features]
        
        return keypoints, descriptors

    def match_features_robust(self, desc1, desc2, method='SIFT', ratio_threshold=0.7):
        """Robust feature matching"""
        if desc1 is None or desc2 is None:
            return []
        
        if method in ['SIFT', 'AKAZE']:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:  # ORB
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        try:
            matches = matcher.knnMatch(desc1, desc2, k=2)
            
            # Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
            
            # Distance'a göre sırala
            good_matches = sorted(good_matches, key=lambda x: x.distance)
            
            return good_matches
            
        except Exception as e:
            logging.warning(f"Feature matching hatası: {e}")
            return []

    def estimate_camera_motion_robust(self, prev_img, curr_img, method='SIFT'):
        """Robust camera motion estimation"""
        # Feature extraction
        kp1, desc1 = self.extract_features_optimized(prev_img, method)
        kp2, desc2 = self.extract_features_optimized(curr_img, method)
        
        if desc1 is None or desc2 is None:
            logging.warning("Descriptor bulunamadı")
            return None, None
        
        # Feature matching
        matches = self.match_features_robust(desc1, desc2, method)
        
        if len(matches) < 8:  # Minimum 8 nokta gerekli
            logging.warning(f"Yetersiz match: {len(matches)}")
            return None, None
        
        # Matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Essential matrix estimation
        E, mask = cv2.findEssentialMat(
            pts1, pts2, 
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if E is None:
            logging.warning("Essential matrix hesaplanamadı")
            return None, None
        
        # Pose recovery
        _, R, t, mask_pose = cv2.recoverPose(
            E, pts1, pts2, self.camera_matrix, mask=mask
        )
        
        # Pose validation
        if not validate_pose(R, t, mask_pose):
            logging.warning("Pose validation başarısız")
            return None, None
        
        return R, t

    def update_trajectory(self, R, t):
        """Trajektori güncelleme"""
        # Global koordinatlara dönüştür
        self.current_pos = self.current_pos + self.current_rot @ t.flatten()
        self.current_rot = self.current_rot @ R
        
        # Kalman filter
        if self.use_kalman:
            filtered_pos = self.kalman.predict_and_update(self.current_pos)
            self.current_pos = filtered_pos[:3]  # İlk 3 eleman pozisyon
        
        self.positions.append(self.current_pos.copy())

    def get_trajectory(self):
        """Trajektori döndür"""
        return np.array(self.positions) if self.positions else np.array([])

# Eski TrajectoryTracker sınıfı (geriye uyumluluk için)
class TrajectoryTracker:
    def __init__(self, use_kalman=True):
        self.positions = []
        self.current_pos = np.array([0., 0., 0.])
        self.current_rot = np.eye(3)
        self.scale = 1.0
        self.use_kalman = use_kalman
        
        if self.use_kalman:
            self.kalman = KalmanFilter(dt=1.0)

    def extract_features(self, img, method='SIFT', max_features=1000):
        """Feature extraction (eski versiyon)"""
        tracker = OptimizedTrajectoryTracker(use_kalman=False)
        return tracker.extract_features_optimized(img, method, max_features)

    def match_features(self, desc1, desc2, method='SIFT'):
        """Feature matching (eski versiyon)"""
        tracker = OptimizedTrajectoryTracker(use_kalman=False)
        return tracker.match_features_robust(desc1, desc2, method)

    def estimate_camera_motion(self, prev_img, curr_img, method='SIFT'):
        """Camera motion estimation (eski versiyon)"""
        tracker = OptimizedTrajectoryTracker(use_kalman=False)
        return tracker.estimate_camera_motion_robust(prev_img, curr_img, method)

    def update_trajectory(self, R, t):
        """Trajektori güncelleme (eski versiyon)"""
        self.current_pos = self.current_pos + self.current_rot @ t.flatten()
        self.current_rot = self.current_rot @ R
        
        if self.use_kalman:
            filtered_pos = self.kalman.predict_and_update(self.current_pos)
            self.current_pos = filtered_pos[:3]
        
        self.positions.append(self.current_pos.copy())

    def get_trajectory(self):
        """Trajektori döndür (eski versiyon)"""
        return np.array(self.positions) if self.positions else np.array([])