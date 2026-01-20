import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import logging

def get_camera_matrix():
    """Senin kamera parametrelerin ile güncellenmiş"""
    # FocalLength: [2.7922e+03 2.7952e+03] -> pixel cinsinden
    # PrincipalPoint: [1.9880e+03 1.5622e+03]
    fx, fy = 2792.2, 2795.2
    cx, cy = 1988.0, 1562.2
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)

def get_distortion_coeffs():
    """Senin distortion parametrelerin"""
    # RadialDistortion: [0.0798 -0.1867]
    # TangentialDistortion: [0 0]
    return np.array([0.0798, -0.1867, 0, 0], dtype=np.float32)

def load_ground_truth_csv(file_path, max_frames=450):
    """
    CSV formatını doğru şekilde yükle
    Format: translation_x,translation_y,translation_z,frame_numbers
    """
    if not file_path or not os.path.exists(file_path):
        logging.warning(f"Ground truth dosyası bulunamadı: {file_path}")
        return None, None
    
    try:
        # CSV'yi pandas ile yükle
        df = pd.read_csv(file_path)
        
        # Kolon kontrolü
        required_cols = ['translation_x', 'translation_y', 'translation_z', 'frame_numbers']
        if not all(col in df.columns for col in required_cols):
            logging.error(f"CSV formatı hatalı. Gerekli kolonlar: {required_cols}")
            return None, None
        
        # İlk max_frames kadar al
        if max_frames:
            df = df.head(max_frames)
        
        # Frame numaralarını çıkar (frame_000001 -> 1)
        frame_indices = []
        for frame_str in df['frame_numbers']:
            try:
                frame_num = int(frame_str.split('_')[-1])
                frame_indices.append(frame_num)
            except:
                logging.warning(f"Frame numarası çıkarılamadı: {frame_str}")
                continue
        
        # 3D pozisyonları al
        positions = df[['translation_x', 'translation_y', 'translation_z']].values
        
        logging.info(f"Ground truth yüklendi: {len(positions)} nokta")
        logging.info(f"Frame aralığı: {min(frame_indices)} - {max(frame_indices)}")
        
        return positions, frame_indices
        
    except Exception as e:
        logging.error(f"Ground truth yüklenemedi: {e}")
        return None, None

def calibrate_scale_3d(trajectory, gt_positions, gt_frame_indices):
    """
    3D trajektori için gelişmiş ölçeklendirme
    """
    if gt_positions is None or len(gt_positions) < 2:
        return trajectory, 1.0
    
    # GT olan frame'lerdeki trajektori noktalarını al
    valid_traj_points = []
    valid_gt_points = []
    
    for i, frame_idx in enumerate(gt_frame_indices):
        if frame_idx < len(trajectory):
            valid_traj_points.append(trajectory[frame_idx])
            valid_gt_points.append(gt_positions[i])
    
    if len(valid_traj_points) < 2:
        return trajectory, 1.0
    
    valid_traj_points = np.array(valid_traj_points)
    valid_gt_points = np.array(valid_gt_points)
    
    # Merkezi sıfırla
    traj_mean = np.mean(valid_traj_points, axis=0)
    gt_mean = np.mean(valid_gt_points, axis=0)
    
    traj_centered = valid_traj_points - traj_mean
    gt_centered = valid_gt_points - gt_mean
    
    # Ölçek hesapla - RMS mesafe oranı
    traj_scale = np.sqrt(np.mean(np.sum(traj_centered**2, axis=1)))
    gt_scale = np.sqrt(np.mean(np.sum(gt_centered**2, axis=1)))
    
    if traj_scale > 1e-6:
        scale = gt_scale / traj_scale
    else:
        scale = 1.0
    
    # Tüm trajektoriyi ölçeklendir ve hizala
    scaled_trajectory = (trajectory - traj_mean) * scale + gt_mean
    
    return scaled_trajectory, scale

def calculate_trajectory_errors(trajectory, gt_positions, gt_frame_indices):
    """
    Detaylı hata analizi
    """
    errors_3d = []
    errors_2d = []
    errors_z = []
    
    for i, frame_idx in enumerate(gt_frame_indices):
        if frame_idx < len(trajectory):
            traj_point = trajectory[frame_idx]
            gt_point = gt_positions[i]
            
            # 3D Euclidean hata
            error_3d = np.linalg.norm(traj_point - gt_point)
            errors_3d.append(error_3d)
            
            # 2D (XY düzlemi) hata
            error_2d = np.linalg.norm(traj_point[:2] - gt_point[:2])
            errors_2d.append(error_2d)
            
            # Z ekseni hata
            error_z = abs(traj_point[2] - gt_point[2])
            errors_z.append(error_z)
    
    if not errors_3d:
        return None
    
    return {
        '3d_mean': np.mean(errors_3d),
        '3d_std': np.std(errors_3d),
        '3d_max': np.max(errors_3d),
        '3d_min': np.min(errors_3d),
        '2d_mean': np.mean(errors_2d),
        '2d_std': np.std(errors_2d),
        'z_mean': np.mean(errors_z),
        'z_std': np.std(errors_z),
        'evaluated_points': len(errors_3d),
        'total_trajectory_points': len(trajectory)
    }

def is_valid_rotation_matrix(R):
    """Rotasyon matrisinin geçerli olup olmadığını kontrol et"""
    if R is None or R.shape != (3, 3):
        return False
    
    # Determinant 1'e yakın olmalı
    det = np.linalg.det(R)
    if abs(det - 1.0) > 1e-6:
        return False
    
    # R @ R.T = I olmalı
    should_be_identity = R @ R.T
    identity = np.eye(3)
    if not np.allclose(should_be_identity, identity, atol=1e-6):
        return False
    
    return True

def validate_pose(R, t, mask=None):
    """Pose validation"""
    if R is None or t is None:
        return False
    
    # Rotation matrix kontrolü
    if not is_valid_rotation_matrix(R):
        return False
    
    # Translation magnitude kontrolü
    t_norm = np.linalg.norm(t)
    if t_norm < 1e-6 or t_norm > 10.0:  # Çok küçük veya çok büyük hareket
        return False
    
    # Inlier oranı kontrolü
    if mask is not None:
        inlier_ratio = np.sum(mask) / len(mask)
        if inlier_ratio < 0.3:  # %30'dan az inlier
            return False
    
    return True

def plot_trajectory(trajectory, title="3D Trajectory", save_path=None):
    """3D trajektori çizimi"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                'b-', linewidth=2, alpha=0.8)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                  color='green', s=100, label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                  color='red', s=100, label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_trajectory_comparison_2d(estimated, ground_truth, title="2D Trajectory Comparison", save_path=None):
    """2D trajektori karşılaştırması"""
    plt.figure(figsize=(12, 8))
    
    plt.plot(estimated[:, 0], estimated[:, 1], 'b-', linewidth=2, label='Estimated', alpha=0.8)
    plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'r--', linewidth=2, label='Ground Truth', alpha=0.8)
    
    plt.scatter(estimated[0, 0], estimated[0, 1], color='green', s=100, label='Start')
    plt.scatter(estimated[-1, 0], estimated[-1, 1], color='blue', s=100, label='End (Est.)')
    plt.scatter(ground_truth[-1, 0], ground_truth[-1, 1], color='red', s=100, label='End (GT)')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_3d_trajectory_comparison(estimated, ground_truth, title="3D Trajectory Comparison", save_path=None):
    """3D trajektori karşılaştırma grafiği"""
    fig = plt.figure(figsize=(15, 10))
    
    # 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(estimated[:, 0], estimated[:, 1], estimated[:, 2], 
             'b-', linewidth=2, label='Estimated', alpha=0.8)
    ax1.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], 
             'r--', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    
    # XY düzlemi
    ax2 = fig.add_subplot(222)
    ax2.plot(estimated[:, 0], estimated[:, 1], 'b-', linewidth=2, label='Estimated')
    ax2.plot(ground_truth[:, 0], ground_truth[:, 1], 'r--', linewidth=2, label='Ground Truth')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane View')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # Z ekseni zaman serisi
    ax3 = fig.add_subplot(223)
    frames = range(len(estimated))
    ax3.plot(frames, estimated[:, 2], 'b-', linewidth=2, label='Estimated Z')
    ax3.plot(frames, ground_truth[:, 2], 'r--', linewidth=2, label='Ground Truth Z')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Z Coordinate Over Time')
    ax3.legend()
    ax3.grid(True)
    
    # Hata grafiği
    ax4 = fig.add_subplot(224)
    errors = np.linalg.norm(estimated - ground_truth, axis=1)
    ax4.plot(frames, errors, 'g-', linewidth=2)
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('3D Error (m)')
    ax4.set_title(f'3D Error Over Time (Mean: {np.mean(errors):.4f}m)')
    ax4.grid(True)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_trajectory_stats(trajectory, title="Trajectory Statistics"):
    """Trajektori istatistikleri yazdır"""
    if len(trajectory) == 0:
        print(f"{title}: Boş trajektori!")
        return
    
    total_distance = 0
    for i in range(1, len(trajectory)):
        total_distance += np.linalg.norm(trajectory[i] - trajectory[i-1])
    
    print(f"\\n{title}:")
    print(f"  Toplam nokta: {len(trajectory)}")
    print(f"  Toplam mesafe: {total_distance:.4f}m")
    print(f"  Ortalama adım: {total_distance/(len(trajectory)-1):.4f}m" if len(trajectory) > 1 else "  Ortalama adım: N/A")
    print(f"  Başlangıç: [{trajectory[0, 0]:.4f}, {trajectory[0, 1]:.4f}, {trajectory[0, 2]:.4f}]")
    print(f"  Bitiş: [{trajectory[-1, 0]:.4f}, {trajectory[-1, 1]:.4f}, {trajectory[-1, 2]:.4f}]")

# Eski fonksiyonlar (geriye uyumluluk için)
def calibrate_scale_with_ground_truth(trajectory, ground_truth):
    """Eski fonksiyon - geriye uyumluluk için"""
    if ground_truth is None or len(ground_truth) < 2:
        return trajectory, 1.0
    
    # Frame indices'i otomatik oluştur
    gt_frame_indices = list(range(len(ground_truth)))
    
    # 3D formatına dönüştür (eğer 2D ise)
    if ground_truth.shape[1] == 2:
        gt_3d = np.column_stack([ground_truth, np.zeros(len(ground_truth))])
    else:
        gt_3d = ground_truth
    
    return calibrate_scale_3d(trajectory, gt_3d, gt_frame_indices)