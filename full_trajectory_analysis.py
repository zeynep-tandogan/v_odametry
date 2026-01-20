#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rpg_trajectory_evaluation - Tam Analiz Scripti
Doc'taki örnekler gibi yörünge karşılaştırması ve hatalar
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # LaTeX olmadan
import matplotlib.pyplot as plt
from pathlib import Path

# Çalışma dizini
PROJECT_DIR = r'C:\Users\Cyber\Desktop\tezz\test\rpg_trajectory_evaluation-master\rpg_trajectory_evaluation-master'
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))
sys.path.insert(0, os.path.join(PROJECT_DIR, 'scripts'))

from rpg_trajectory_evaluation.trajectory import Trajectory
from rpg_trajectory_evaluation import plot_utils as pu

def analyze_and_plot_trajectory(result_dir, est_type='ba_estimate', trial_idx=0):
    """
    Yörüngeyi analiz et ve doc'ta gösterilenlere benzer grafikler oluştur
    """
    
    plots_dir = os.path.join(result_dir, 'plots_analysis')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    print(f"✓ Analiz klasörü: {plots_dir}")
    print(f"\n📊 Trajectory analiz ediliyor: {result_dir}")
    
    # ========================================================================
    # TRAJECTORY OBJESI OLUŞTUR VE YÜKLEYİŞ
    # ========================================================================
    print("\n1️⃣ Trajectory nesnesi oluşturuluyor...")
    
    try:
        # Suffix belirle
        if trial_idx == 0:
            suffix = ''
        else:
            suffix = str(trial_idx)
        
        traj = Trajectory(
            result_dir,
            est_type=est_type,
            suffix=suffix,
            nm_est=f'stamped_traj_estimate{suffix}.txt' if suffix else 'stamped_traj_estimate.txt',
            nm_gt='stamped_groundtruth.txt'
        )
        
        print(f"✓ Trajectory nesnesi oluşturuldu")
        print(f"  - Pozisyon (estimate): {len(traj.p_es)} poz")
        print(f"  - Pozisyon (groundtruth): {len(traj.p_gt)} poz")
        print(f"  - Alignment type: {traj.align_type}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False
    
    # ========================================================================
    # 1. YÖRÜNGE KARŞILAŞTIRMASI - ÜST GÖRÜNÜŞ
    # ========================================================================
    print("\n2️⃣ Yörünge Üst Görünüş (Top View) grafiği oluşturuluyor...")
    
    try:
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, aspect='equal', xlabel='x [m]', ylabel='y [m]')
        
        # Hizalı tahmin edilen yörünge (mavi)
        pu.plot_trajectory_top(ax, traj.p_es_aligned, 'b', 'Estimate (aligned)')
        
        # Gerçek yörünge (macenta/pembe)
        pu.plot_trajectory_top(ax, traj.p_gt, 'm', 'Groundtruth')
        
        # Hizalama çizgileri (başlangıçtan itibaren)
        pu.plot_aligned_top(ax, traj.p_es_aligned, traj.p_gt, traj.align_num_frames)
        
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(plots_dir, f'trajectory_top_view_{est_type}_{trial_idx}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Kaydedildi: {output_file}")
        plt.close()
    except Exception as e:
        print(f"❌ Hata: {e}")
    
    # ========================================================================
    # 2. YÖRÜNGE KARŞILAŞTIRMASI - YAN GÖRÜNÜŞ
    # ========================================================================
    print("\n3️⃣ Yörünge Yan Görünüş (Side View) grafiği oluşturuluyor...")
    
    try:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, aspect='equal', xlabel='x [m]', ylabel='z [m]')
        
        # Yan görünüş (x-z düzlemi)
        pu.plot_trajectory_side(ax, traj.p_es_aligned, 'b', 'Estimate (aligned)')
        pu.plot_trajectory_side(ax, traj.p_gt, 'm', 'Groundtruth')
        
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(plots_dir, f'trajectory_side_view_{est_type}_{trial_idx}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Kaydedildi: {output_file}")
        plt.close()
    except Exception as e:
        print(f"❌ Hata: {e}")
    
    # ========================================================================
    # 3. MUTLAK HATA (ATE) - TRANSLATION
    # ========================================================================
    print("\n4️⃣ Mutlak Hata (ATE) - Translation grafiği oluşturuluyor...")
    
    try:
        # Mutlak hatayı hesapla
        traj.compute_absolute_error()
        
        abs_errors = traj.abs_errors['abs_err']
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Hata grafiği
        ax.plot(range(len(abs_errors)), abs_errors, 'b-', linewidth=1.5, label='Absolute Error')
        ax.fill_between(range(len(abs_errors)), abs_errors, alpha=0.3, color='blue')
        
        # İstatistikler
        mean_err = np.mean(abs_errors)
        median_err = np.median(abs_errors)
        max_err = np.max(abs_errors)
        
        ax.axhline(mean_err, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_err:.4f}')
        ax.axhline(median_err, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_err:.4f}')
        ax.axhline(max_err, color='red', linestyle='--', linewidth=2, label=f'Max: {max_err:.4f}')
        
        ax.set_xlabel('Frame Index', fontsize=11)
        ax.set_ylabel('Absolute Error [m]', fontsize=11)
        ax.set_title(f'Absolute Trajectory Error (ATE) - {est_type} Trial {trial_idx}', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(plots_dir, f'absolute_error_{est_type}_{trial_idx}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Kaydedildi: {output_file}")
        print(f"  - Mean: {mean_err:.6f} m")
        print(f"  - Median: {median_err:.6f} m")
        print(f"  - Max: {max_err:.6f} m")
        plt.close()
    except Exception as e:
        print(f"❌ Hata: {e}")
    
    # ========================================================================
    # 4. GÖRECELI HATA (RE) - SUB-TRAJECTORY
    # ========================================================================
    print("\n5️⃣ Göreceli Hata (RE) - Sub-trajectory grafiği oluşturuluyor...")
    
    try:
        # Göreceli hataları hesapla (varsayılan sub-trajectory uzunlukları)
        traj.compute_relative_errors()
        
        # Boxplot için veriler hazırla
        rel_errors = traj.rel_errors
        
        if rel_errors:
            # Üst 5 uzunluğu seç
            sorted_lengths = sorted(list(rel_errors.keys()))[:5]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            
            metrics_to_plot = ['rel_trans', 'rel_trans_perc', 'rel_yaw', 'rel_rot', 'rel_gravity', 'rel_rot_deg_per_m']
            metric_titles = ['Translation (m)', 'Translation %', 'Yaw (rad)', 
                           'Rotation (rad)', 'Gravity (rad)', 'Rotation (deg/m)']
            
            for idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
                ax = axes[idx]
                
                # Her uzunluk için hata değerleri topla
                data_to_plot = []
                labels = []
                
                for length in sorted_lengths:
                    if metric in rel_errors[length]:
                        errors = rel_errors[length][metric]
                        data_to_plot.append(errors)
                        labels.append(f'{length:.1f}m')
                
                if data_to_plot:
                    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                    
                    # Renklendirme
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightblue')
                    
                    ax.set_title(title, fontsize=11, fontweight='bold')
                    ax.set_xlabel('Sub-trajectory Length', fontsize=10)
                    ax.set_ylabel('Error', fontsize=10)
                    ax.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle(f'Relative Trajectory Error (RTE) - {est_type} Trial {trial_idx}', 
                        fontsize=13, fontweight='bold')
            plt.tight_layout()
            output_file = os.path.join(plots_dir, f'relative_error_{est_type}_{trial_idx}.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"✓ Kaydedildi: {output_file}")
            plt.close()
    except Exception as e:
        print(f"❌ Hata: {e}")
    
    # ========================================================================
    # 5. HATA İSTATİSTİKLERİ
    # ========================================================================
    print("\n6️⃣ Hata İstatistikleri tablosu oluşturuluyor...")
    
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        abs_err_stat = traj.abs_errors
        
        table_data = [
            ['Metric', 'Value'],
            ['Number of Poses', str(len(traj.p_es))],
            ['Alignment Type', traj.align_type],
            ['', ''],
            ['ABSOLUTE ERROR (ATE)', ''],
            ['Mean Translation [m]', f"{abs_err_stat['mean'][0]:.6f}"],
            ['Median Translation [m]', f"{abs_err_stat['median'][0]:.6f}"],
            ['Max Translation [m]', f"{abs_err_stat['max'][0]:.6f}"],
            ['RMSE Translation [m]', f"{abs_err_stat['rmse'][0]:.6f}"],
            ['', ''],
            ['Mean Rotation [rad]', f"{abs_err_stat['mean'][1]:.6f}"],
            ['Median Rotation [rad]', f"{abs_err_stat['median'][1]:.6f}"],
            ['Max Rotation [rad]', f"{abs_err_stat['max'][1]:.6f}"],
            ['RMSE Rotation [rad]', f"{abs_err_stat['rmse'][1]:.6f}"],
        ]
        
        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                        colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Header styling
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Section styling
        for row in [4, 9]:
            table[(row, 0)].set_facecolor('#e0e0e0')
            table[(row, 0)].set_text_props(weight='bold')
        
        plt.title(f'Trajectory Error Statistics - {est_type} Trial {trial_idx}', 
                 fontsize=13, fontweight='bold', pad=20)
        output_file = os.path.join(plots_dir, f'error_statistics_{est_type}_{trial_idx}.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Kaydedildi: {output_file}")
        plt.close()
    except Exception as e:
        print(f"❌ Hata: {e}")
    
    # ========================================================================
    # ÖZET
    # ========================================================================
    print("\n" + "="*70)
    print("✅ YÖRÜNGE ANALİZİ TAMAMLANDI!")
    print("="*70)
    print(f"\nGrafikler klasörü: {plots_dir}")
    print("\nOluşturulan dosyalar:")
    for fname in sorted(os.listdir(plots_dir)):
        if fname.endswith('.png'):
            fpath = os.path.join(plots_dir, fname)
            fsize = os.path.getsize(fpath) / 1024  # KB
            print(f"  ✓ {fname} ({fsize:.1f} KB)")
    
    return True


if __name__ == '__main__':
    # Örnek veri klasörü
    result_dir = os.path.join(
        PROJECT_DIR,
        'results/euroc_vislam_mono/laptop/vislam_ba/laptop_vislam_ba_MH_01'
    )
    
    print("🚀 rpg_trajectory_evaluation - Tam Analiz Başlatılıyor\n")
    
    # Trial 0 için analiz yap
    success = analyze_and_plot_trajectory(result_dir, est_type='ba_estimate', trial_idx=0)
    
    if success:
        print("\n" + "="*70)
        print("📊 Tüm analizler başarıyla tamamlandı!")
        print("="*70)
