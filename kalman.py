import numpy as np
import logging

class KalmanFilter:
    """
    3D pozisyon tahmini için Kalman Filter
    State: [x, y, z, vx, vy, vz] (pozisyon ve hız)
    """
    
    def __init__(self, dt=1.0, process_noise=1e-4, measurement_noise=1e-2):
        """
        Kalman Filter başlatma
        
        Args:
            dt: Zaman adımı
            process_noise: Süreç gürültüsü varyansı
            measurement_noise: Ölçüm gürültüsü varyansı
        """
        self.dt = dt
        self.dim_state = 6  # [x, y, z, vx, vy, vz]
        self.dim_measurement = 3  # [x, y, z]
        
        # State vector [x, y, z, vx, vy, vz]
        self.x = np.zeros((self.dim_state, 1))
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance
        q = process_noise
        self.Q = np.array([
            [dt**4/4*q, 0, 0, dt**3/2*q, 0, 0],
            [0, dt**4/4*q, 0, 0, dt**3/2*q, 0],
            [0, 0, dt**4/4*q, 0, 0, dt**3/2*q],
            [dt**3/2*q, 0, 0, dt**2*q, 0, 0],
            [0, dt**3/2*q, 0, 0, dt**2*q, 0],
            [0, 0, dt**3/2*q, 0, 0, dt**2*q]
        ])
        
        # Measurement noise covariance
        r = measurement_noise
        self.R = np.eye(self.dim_measurement) * r
        
        # Error covariance matrix
        self.P = np.eye(self.dim_state) * 1000  # Başlangıçta yüksek belirsizlik
        
        # İlk ölçüm flag'i
        self.initialized = False
        
        # Outlier detection için
        self.max_innovation = 5.0  # Maksimum innovation (Mahalanobis distance)
        
    def initialize(self, measurement):
        """İlk ölçümle filter'ı başlat"""
        if len(measurement) != 3:
            raise ValueError("Measurement 3D pozisyon olmalı [x, y, z]")
        
        # İlk pozisyonu ayarla, hızı sıfır yap
        self.x[:3, 0] = measurement
        self.x[3:, 0] = 0  # İlk hız sıfır
        
        self.initialized = True
        logging.debug(f"Kalman Filter başlatıldı: {measurement}")
    
    def predict(self):
        """Prediction step"""
        # State prediction
        self.x = self.F @ self.x
        
        # Error covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[:3, 0]  # Sadece pozisyon döndür
    
    def update(self, measurement):
        """Update step"""
        if len(measurement) != 3:
            raise ValueError("Measurement 3D pozisyon olmalı [x, y, z]")
        
        z = np.array(measurement).reshape(-1, 1)
        
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Outlier detection (Mahalanobis distance)
        try:
            mahalanobis_dist = np.sqrt(y.T @ np.linalg.inv(S) @ y)[0, 0]
            if mahalanobis_dist > self.max_innovation:
                logging.warning(f"Outlier detected (Mahalanobis: {mahalanobis_dist:.2f}), skipping update")
                return self.x[:3, 0]
        except np.linalg.LinAlgError:
            logging.warning("Singular matrix in outlier detection, skipping check")
        
        # Kalman gain
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logging.warning("Singular matrix in Kalman gain calculation")
            return self.x[:3, 0]
        
        # State update
        self.x = self.x + K @ y
        
        # Error covariance update
        I = np.eye(self.dim_state)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[:3, 0]  # Sadece pozisyon döndür
    
    def predict_and_update(self, measurement):
        """Predict ve update'i birlikte yap"""
        if not self.initialized:
            self.initialize(measurement)
            return np.array(measurement)
        
        # Prediction
        predicted_pos = self.predict()
        
        # Update
        updated_pos = self.update(measurement)
        
        return updated_pos
    
    def get_position(self):
        """Mevcut pozisyonu döndür"""
        return self.x[:3, 0]
    
    def get_velocity(self):
        """Mevcut hızı döndür"""
        return self.x[3:, 0]
    
    def get_covariance(self):
        """Mevcut covariance matrisini döndür"""
        return self.P
    
    def reset(self):
        """Filter'ı sıfırla"""
        self.x = np.zeros((self.dim_state, 1))
        self.P = np.eye(self.dim_state) * 1000
        self.initialized = False
        logging.debug("Kalman Filter sıfırlandı")

class AdaptiveKalmanFilter(KalmanFilter):
    """
    Adaptive Kalman Filter - noise parametrelerini otomatik ayarlar
    """
    
    def __init__(self, dt=1.0, initial_process_noise=1e-4, initial_measurement_noise=1e-2):
        super().__init__(dt, initial_process_noise, initial_measurement_noise)
        
        # Adaptive parameters
        self.innovation_history = []
        self.history_size = 10
        self.adaptation_rate = 0.1
        
        # Initial noise values
        self.base_process_noise = initial_process_noise
        self.base_measurement_noise = initial_measurement_noise
    
    def adapt_noise_parameters(self):
        """Innovation geçmişine göre noise parametrelerini ayarla"""
        if len(self.innovation_history) < 3:
            return
        
        # Son innovation'ların varyansını hesapla
        recent_innovations = np.array(self.innovation_history[-self.history_size:])
        innovation_var = np.var(recent_innovations, axis=0)
        
        # Measurement noise'ı innovation varyansına göre ayarla
        avg_innovation_var = np.mean(innovation_var)
        
        if avg_innovation_var > 0:
            # Yüksek innovation -> yüksek measurement noise
            adaptive_factor = min(avg_innovation_var / self.base_measurement_noise, 10.0)
            new_measurement_noise = self.base_measurement_noise * (1 + adaptive_factor * self.adaptation_rate)
            
            # R matrisini güncelle
            self.R = np.eye(self.dim_measurement) * new_measurement_noise
            
            logging.debug(f"Adaptive noise: {new_measurement_noise:.6f} (factor: {adaptive_factor:.2f})")
    
    def update(self, measurement):
        """Update step with adaptation"""
        if len(measurement) != 3:
            raise ValueError("Measurement 3D pozisyon olmalı [x, y, z]")
        
        z = np.array(measurement).reshape(-1, 1)
        
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation'ı kaydet
        self.innovation_history.append(y.flatten())
        if len(self.innovation_history) > self.history_size:
            self.innovation_history.pop(0)
        
        # Noise parametrelerini uyarla
        self.adapt_noise_parameters()
        
        # Normal update işlemi
        return super().update(measurement)