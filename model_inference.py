import cv2
import numpy as np
import os
import re
import logging
import matplotlib
matplotlib.use('Agg')  # GUI olmayan ortamlar için
import matplotlib.pyplot as plt
from trajectory import OptimizedTrajectoryTracker
from kalman import KalmanFilter
from utils import (plot_trajectory, plot_trajectory_comparison_2d, print_trajectory_stats,
                   load_ground_truth_csv, calibrate_scale_3d, get_camera_matrix, get_distortion_coeffs,
                   plot_3d_trajectory_comparison)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def frame_number(filename):
    """Dosya adından frame numarasını çıkar"""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

def load_ground_truth(file_path):
    """Ground truth dosyasını yükle"""
    if not file_path or not os.path.exists(file_path):
        logging.warning(f"Ground truth dosyası bulunamadı: {file_path}")
        return None
    
    try:
        gt = np.genfromtxt(file_path, delimiter=',', skip_header=1, usecols=(0,1,2), names=['x', 'y', 'z'])
        logging.info(f"Ground truth başarıyla yüklendi: {file_path}")
        return np.column_stack((gt['x'], gt['y'], gt['z']))
    except Exception as e:
        logging.warning(f"Ground truth yüklenemedi: {e}")
        return None

# --- Kamera parametreleri ---
CAMERA_INTRINSICS = {
    'K': np.array([[2792.2, 0, 1988.0],
                  [0, 2795.2, 1562.2],
                  [0, 0, 1]]),
    'dist': np.array([0.0798, -0.1867, 0, 0])
}

def extract_features(img, method='SIFT'):
    """
    Görüntüden özellikleri çıkartır (SIFT veya ORB)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Görüntü iyileştirme
    gray = cv2.equalizeHist(gray)  # Kontrast iyileştirme
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Gürültü azaltma
    
    if method == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
    elif method == 'ORB':
        detector = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
    else:
        raise ValueError(f"Desteklenmeyen özellik çıkarım yöntemi: {method}")
    
    return keypoints, descriptors

def match_features(desc1, desc2, method='FLANN', detector='SIFT'):
    """
    İki görüntü arasında özellik eşleştirme yapar (FLANN veya BFMatcher)
    """
    if desc1 is None or desc2 is None:
        return []
    
    if method == 'BF':
        # Brute Force eşleştirici
        if detector == 'ORB':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # kNN eşleştirme ile ratio test
        matches = matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                good_matches.append(m)
    else:
        # FLANN eşleştirici
        if detector == 'ORB':
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        else:
            index_params = dict(algorithm=1, trees=5)
        
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m_n in matches:
            if len(m_n) >= 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
    
    return good_matches

def filter_matches_ransac(kp1, kp2, matches, K):
    """
    RANSAC ile eşleşmeleri filtrele ve Essential Matrix hesapla
    """
    if len(matches) < 8:  # Essential matrix için en az 8 nokta gerekir
        return None, None, None
    
    # Eşleşen noktaları al
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # RANSAC ile Essential Matrix hesapla
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    if E is None or mask is None:
        return None, None, None
    
    # Sadece inlier noktaları seç
    pts1_inliers = pts1[mask.ravel() == 1]
    pts2_inliers = pts2[mask.ravel() == 1]
    
    return E, pts1_inliers, pts2_inliers

def recover_pose_from_essential(E, pts1, pts2, K):
    """
    Essential matrixten kamera pozunu hesapla (R, t)
    """
    # recoverPose ile rotasyon ve translasyon hesapla
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    
    return R, t, mask

def visualize_matches(img1, img2, kp1, kp2, matches, title="Feature Matches"):
    """
    Eşleşen özellikleri görselleştir
    """
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imshow(title, img_matches)
    cv2.waitKey(1)

def process_video_sequence(image_folder, ground_truth_file=None, 
                           output_folder="results", feature_method='SIFT', 
                           matching_method='FLANN', use_kalman=True):
    """
    Görüntü dizisini işleyerek kamera hareketini takip eder
    
    1. Görüntüleri Oku ve Ön İşlem
    2. Özellik Çıkarımı (ORB/SIFT)
    3. Özellik Eşleştirme (BFMatcher/FLANN + RANSAC)
    4. Essential Matrix Hesapla
    5. Kamera Hareketini (Pose) Bul
    6. Kalman Filtresi ile Trajektori Düzeltme
    7. Sonuçları Kaydet/Göster
    """
    logging.info("🚀 Visual Odometry başlatılıyor...")
    
    # Çıktı klasörünü oluştur
    os.makedirs(output_folder, exist_ok=True)
    
    # Kamera parametreleri
    K = CAMERA_INTRINSICS['K']
    dist = CAMERA_INTRINSICS['dist']
    
    # Kalman filtresi için
    kalman = KalmanFilter(dt=1.0, process_noise=1e-4, measurement_noise=1e-2)
    
    # Görüntüleri yükle
    frames = sorted([f for f in os.listdir(image_folder) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))],
                   key=frame_number)
    
    if len(frames) < 2:
        logging.error("En az 2 görüntü gerekli!")
        return None
    
    logging.info(f"📁 Toplam {len(frames)} görüntü yüklendi.")
    logging.info(f"🔧 Özellik çıkarım yöntemi: {feature_method}")
    logging.info(f"🔧 Eşleştirme yöntemi: {matching_method}")
    logging.info(f"🔧 Kalman filtresi: {'Aktif' if use_kalman else 'Pasif'}")
    
    # Trajektori için
    positions = []  # Kamera pozisyonları
    rotations = []  # Kamera rotasyonları
    
    # Başlangıç pozisyon ve rotasyon
    current_R = np.eye(3)  # Başlangıç rotasyonu (identity matrix)
    current_t = np.zeros((3, 1))  # Başlangıç konumu (sıfır vektör)
    
    # İlk pozisyonu ekle
    positions.append(current_t.flatten())
    rotations.append(current_R.copy())
    
    prev_img = None
    prev_kp = None
    prev_desc = None
    
    successful_poses = 0
    failed_poses = 0
    
    for idx, frame_name in enumerate(frames):
        path = os.path.join(image_folder, frame_name)
        curr_img = cv2.imread(path)
        
        if curr_img is None:
            logging.warning(f"Görüntü açılamadı: {path}, atlanıyor.")
            continue
        
        # Distorsiyon düzeltme
        curr_img = cv2.undistort(curr_img, K, dist)
        
        # 1. Özellik Çıkarımı
        curr_kp, curr_desc = extract_features(curr_img, method=feature_method)
        
        # İlk kare için sadece özellikleri sakla ve devam et
        if prev_img is None:
            prev_img = curr_img
            prev_kp = curr_kp
            prev_desc = curr_desc
            logging.info(f"İlk kare hazırlandı: {frame_name}")
            continue
        
        # 2. Özellik Eşleştirme
        if prev_desc is not None and curr_desc is not None and len(prev_kp) > 10 and len(curr_kp) > 10:
            matches = match_features(prev_desc, curr_desc, method=matching_method, detector=feature_method)
            
            # Yeterli eşleşme varsa devam et
            if len(matches) >= 8:
                # 3. RANSAC ile eşleşmeleri filtrele ve Essential Matrix hesapla
                E, pts1, pts2 = filter_matches_ransac(prev_kp, curr_kp, matches, K)
                
                if E is not None and pts1 is not None and pts2 is not None:
                    # 4. Essential Matrix'ten kamera hareketini (R, t) bul
                    R, t, _ = recover_pose_from_essential(E, pts1, pts2, K)
                    
                    # 5. Hareketi kümülatif olarak hesapla
                    current_t = current_t + current_R @ t
                    current_R = R @ current_R
                    
                    # 6. Kalman Filtresi ile düzeltme (opsiyonel)
                    if use_kalman:
                        current_t_filtered = kalman.predict_and_update(current_t.flatten())
                        current_t = current_t_filtered.reshape(3, 1)
                    
                    # Yeni pozisyonu ve rotasyonu kaydet
                    positions.append(current_t.flatten())
                    rotations.append(current_R.copy())
                    
                    successful_poses += 1
                    
                    # Her 10 karede bir görselleştirme
                    if idx % 10 == 0:
                        visualize_matches(prev_img, curr_img, prev_kp, curr_kp, matches, 
                                         title=f"Frame {idx}: {len(matches)} matches")
                else:
                    logging.warning(f"Frame {idx}: Essential Matrix hesaplanamadı")
                    positions.append(current_t.flatten())  # Son pozisyonu tekrar ekle
                    rotations.append(current_R.copy())
                    failed_poses += 1
            else:
                logging.warning(f"Frame {idx}: Yeterli eşleşme bulunamadı ({len(matches)} < 8)")
                positions.append(current_t.flatten())  # Son pozisyonu tekrar ekle
                rotations.append(current_R.copy())
                failed_poses += 1
        else:
            logging.warning(f"Frame {idx}: Yeterli özellik bulunamadı")
            positions.append(current_t.flatten())  # Son pozisyonu tekrar ekle
            rotations.append(current_R.copy())
            failed_poses += 1
        
        # Sonraki iterasyon için
        prev_img = curr_img
        prev_kp = curr_kp
        prev_desc = curr_desc
        
        # İlerleme durumu
        if idx % 100 == 0 and idx > 0:
            success_rate = (successful_poses / (successful_poses + failed_poses)) * 100
            logging.info(f"İşlenen: {idx+1}/{len(frames)} - Başarı oranı: {success_rate:.1f}%")
    
    cv2.destroyAllWindows()
    
    # Sonuçları numpy array'e dönüştür
    trajectory = np.array(positions)
    
    if len(trajectory) == 0:
        logging.error("Hiç poz tahmini yapılmadı.")
        return None
    
    # Ground truth ile karşılaştırma ve ölçeklendirme
    if ground_truth_file and os.path.exists(ground_truth_file):
        logging.info("Ground truth ile karşılaştırma yapılıyor...")
        
        # Ground truth verisini yükle - doğrudan CSV'den
        gt_positions, gt_frame_indices = load_ground_truth_csv(ground_truth_file)
        
        if gt_positions is not None and len(gt_positions) > 0:
            # Ölçeklendirme
            trajectory_scaled, scale = calibrate_scale_3d(trajectory, gt_positions, gt_frame_indices)
            logging.info(f"Ölçeklendirme faktörü: {scale:.4f}")
            
            # Hata hesapla
            from utils import calculate_trajectory_errors
            error_stats = calculate_trajectory_errors(trajectory_scaled, gt_positions, gt_frame_indices)
            
            if error_stats:
                logging.info(f"3D ortalama hata: {error_stats['3d_mean']:.4f}m")
                logging.info(f"2D ortalama hata: {error_stats['2d_mean']:.4f}m")
            
            # Görselleştirme
            plot_3d_trajectory_comparison(
                trajectory_scaled[:min(len(trajectory_scaled), len(gt_positions))],
                gt_positions[:min(len(trajectory_scaled), len(gt_positions))],
                title="Trajectory Comparison",
                save_path=os.path.join(output_folder, "trajectory_comparison.png")
            )
        else:
            trajectory_scaled = trajectory
            scale = 1.0
            logging.warning("Ground truth verisi yüklenemedi veya karşılaştırma yapılamadı.")
    else:
        trajectory_scaled = trajectory
        scale = 1.0
        logging.info("Ground truth verisi bulunamadı, sadece tahmin edilen trajektori kaydediliyor.")
    
    # Sonuçları kaydet
    np.savetxt(os.path.join(output_folder, "estimated_trajectory.txt"), trajectory_scaled, fmt="%.6f")
    logging.info(f"Trajektori kaydedildi: {os.path.join(output_folder, 'estimated_trajectory.txt')}")
    
    # İstatistikler
    total_distance = 0
    for i in range(1, len(trajectory_scaled)):
        total_distance += np.linalg.norm(trajectory_scaled[i] - trajectory_scaled[i-1])
    
    success_rate = (successful_poses / (successful_poses + failed_poses)) * 100 if (successful_poses + failed_poses) > 0 else 0
    
    logging.info(f"İşlem tamamlandı:")
    logging.info(f"  - Toplam kare: {len(frames)}")
    logging.info(f"  - Başarılı pozlar: {successful_poses}")
    logging.info(f"  - Başarısız pozlar: {failed_poses}")
    logging.info(f"  - Başarı oranı: {success_rate:.1f}%")
    logging.info(f"  - Toplam mesafe: {total_distance:.4f}m")
    
    # Sonuçları görselleştir
    plot_trajectory(trajectory_scaled, 
                   title=f"Estimated Trajectory (Total distance: {total_distance:.2f}m)", 
                   save_path=os.path.join(output_folder, "trajectory_3d.png"))
    
    return trajectory_scaled

def main():
    """
    Ana fonksiyon - Tüm parametreler burada tanımlı, terminal parametresi gerektirmez
    """
    # Parametreler - Bu değerleri kendi projenize göre değiştirin
    image_folder = "C:\\Users\\zeyne\\Desktop\\termal calısmaları\\termal1_frames"
    ground_truth_file = "C:\\Users\\zeyne\\Desktop\\termal calısmaları\\termal1.csv"
    output_folder = "C:\\Users\\zeyne\\Desktop\\termal calısmaları\\results"
    
    # Diğer parametreler
    feature_method = 'SIFT'  # 'SIFT' veya 'ORB'
    matching_method = 'FLANN'  # 'FLANN' veya 'BF'
    use_kalman = True
    
    print("🚀 Visual Odometry Sistemi")
    print("=" * 50)
    print(f"📁 Görüntü klasörü: {image_folder}")
    print(f"📊 Ground truth: {ground_truth_file}")
    print(f"📁 Sonuç klasörü: {output_folder}")
    print(f"🔧 Özellik çıkarımı: {feature_method}")
    print(f"🔧 Eşleştirme yöntemi: {matching_method}")
    print(f"� Kalman filtresi: {'Aktif' if use_kalman else 'Pasif'}")
    print("=" * 50)
    
    try:
        # İşlemi başlat
        trajectory = process_video_sequence(
            image_folder=image_folder,
            ground_truth_file=ground_truth_file,
            output_folder=output_folder,
            feature_method=feature_method,
            matching_method=matching_method,
            use_kalman=use_kalman
        )
        
        if trajectory is not None:
            print("\n✅ İşlem başarıyla tamamlandı!")
            print(f"📈 Toplam {len(trajectory)} trajektori noktası")
            print(f"📂 Sonuçlar: {output_folder}/")
            
            return 0
        else:
            print("\n⚠️ İşlem tamamlandı ancak trajektori oluşturulamadı.")
            return 1
            
    except Exception as e:
        logging.error(f"Ana hata: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())