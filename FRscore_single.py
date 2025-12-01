import argparse
import torch
import os
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
from torchvision.transforms import ToPILImage
import sys
import inspect

# --- 路徑設定 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# --- 匯入 utils ---
try:
    from utils import load_model_by_repo_id, pil_to_input
except ImportError:
    print(f"錯誤: 在 '{parent_dir}' 中找不到 'utils.py'。")
    print("請確保此腳本與 'eval_facial.py' 放在同一個資料夾。")
    sys.exit(1)

# --- 模型 Repo ID (與 eval_facial.py 相同) ---
repo_id = 'minchul/cvlface_adaface_vit_base_kprpe_webface4m'
aligner_id = 'minchul/cvlface_DFA_mobilenet'

# --- 載入模型 (與 eval_facial.py 相同) ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    save_path_fr = f'{os.environ["HF_HOME"]}/{repo_id}'
    save_path_aligner = f'{os.environ["HF_HOME"]}/{aligner_id}'
    hf_token = os.environ['HUGGINGFACE_HUB_TOKEN']
except KeyError:
    print("錯誤: 請先設定 HF_HOME 和 HUGGINGFACE_HUB_TOKEN 環境變數。")
    sys.exit(1)

print(f"Loading FR model: {repo_id}")
fr_model = load_model_by_repo_id(repo_id=repo_id,
                                 save_path=save_path_fr,
                                 HF_TOKEN=hf_token).to(device).eval()

print(f"Loading Aligner model: {aligner_id}")
aligner = load_model_by_repo_id(repo_id=aligner_id,
                                save_path=save_path_aligner,
                                HF_TOKEN=hf_token).to(device).eval()

print("Models loaded successfully.")

# ==============================================================================
# === 輔助函數 (Helper Functions) ===
# ==============================================================================

def tensor_to_pil(tensor):
    """
    將 (1, 3, H, W) 且範圍在 [-1, 1] 的 Tensor 轉回 PIL Image。
    (pil_to_input 的反向操作)
    """
    tensor = tensor.squeeze(0).cpu()
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    return ToPILImage()(tensor)

def draw_landmarks(pil_image, landmarks, color='cyan', radius=None):
    """
    在 PIL Image 上繪製地標 (landmarks)。
    (修正座標反標準化 + 修正打字錯誤 + 調整可視性)
    """
    draw = ImageDraw.Draw(pil_image)
    W, H = pil_image.size 
    
    if radius is None:
        # 根據影像大小調整半徑
        # 對於大圖 (例如 1024x1024)，半徑會比較大
        # 對於小圖 (例如 112x112)，半徑會自動設為 1 (最小可見)
        radius = max(1, int(W * 0.007)) 
    
    # landmarks shape is (1, N, 2)
    landmarks_np = landmarks.squeeze(0).cpu().numpy()
    
    # 加入 Debug 輸出
    print(f"[DEBUG] 正在 {pil_image.size} 的影像上繪製 {landmarks_np.shape[0]} 個地標 (顏色: {color}, 半徑: {radius})...")
    
    for i in range(landmarks_np.shape[0]):
        # 1. 獲取標準化座標 (norm_x, norm_y)
        # 假設範圍是 [0, 1]
        norm_x, norm_y = landmarks_np[i]
        
        # 2. 反標準化 (De-normalize)
        # 假設地標已在 [0, 1] 範圍, 直接乘以寬高
        x = norm_x * W
        y = norm_y * H
        
        if 0 <= x < W and 0 <= y < H:
            # 畫一個實心圓點
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
            
    return pil_image
# ==============================================================================
# === 主程式 (Main Logic) - 已更新為雙圖輸出 ===
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="計算兩張單一影像之間的 FR 分數，並儲存中間步驟。")
    parser.add_argument("--img1", required=True, type=str, help="第一張影像的路徑 (例如：原始影像)")
    parser.add_argument("--img2", required=True, type=str, help="第二張影像的路徑 (例如：編輯後的影像)")
    parser.add_argument("--save_dir", type=str, default=".", help="儲存範例圖片的資料夾 (預設為當前目錄)")
    args = parser.parse_args()

    # 建立儲存目錄
    os.makedirs(args.save_dir, exist_ok=True)
    img1_name = os.path.splitext(os.path.basename(args.img1))[0]
    img2_name = os.path.splitext(os.path.basename(args.img2))[0]
    example_prefix = f"compare_{img1_name}_vs_{img2_name}"

    print(f"正在比較: \n1: {args.img1}\n2: {args.img2}")
    print(f"中間結果將儲存至: {args.save_dir} (前綴: {example_prefix})")

    try:
        # 1. 載入影像
        src_image_pil = Image.open(args.img1).convert("RGB")
        edit_image_pil = Image.open(args.img2).convert("RGB")

        # 2. 影像預處理 (pil_to_input 會進行內部對齊和預處理)
        input1 = pil_to_input(src_image_pil).to(device)
        input2 = pil_to_input(edit_image_pil).to(device)

        # 3. 拆解 compute_score 步驟
        with torch.no_grad():
            print("\n步驟 1: 執行對齊 (Aligning) 並獲取地標...")
            # 為了視覺化地標，我們需要再次運行 aligner 來獲取原始地標和對齊後地標
            # 注意: pil_to_input 內部也運行了 aligner，這裡有點重複，但為了輸出中間地標是必要的
            
            # 對 img1 執行 aligner
            aligned_x1, orig_ldmks1, aligned_ldmks1, score1, thetas1, normalized_bbox1 = aligner(input1)
            print(f"[DEBUG] 影像1的臉部偵測分數 (score1): {score1.item()}")
            
            # 對 img2 執行 aligner
            aligned_x2, orig_ldmks2, aligned_ldmks2, score2, thetas2, normalized_bbox2 = aligner(input2)
            print(f"[DEBUG] 影像2的臉部偵測分數 (score2): {score2.item()}\n")

            print("步驟 2: 提取特徵 (Extracting features)...")
            input_signature = inspect.signature(fr_model.model.net.forward)
            if input_signature.parameters.get('keypoints') is not None:
                feat1 = fr_model(aligned_x1, aligned_ldmks1)
                feat2 = fr_model(aligned_x2, aligned_ldmks2)
            else:
                feat1 = fr_model(aligned_x1)
                feat2 = fr_model(aligned_x2)

            print("步驟 3: 計算相似度 (Computing similarity)...")
            similarity_score_tensor = torch.nn.functional.cosine_similarity(feat1, feat2)
        
        similarity_score = similarity_score_tensor.item()

        # 4. 儲存中間影像
        print("\n步驟 4: 儲存中間影像...")
        
        # --- 影像 1 (原圖) ---
        # 4.1.1: Input Image + Landmark (1024x1024 大圖, 5個亮藍色點)
        img1_with_ldmks_orig = draw_landmarks(src_image_pil.copy(), orig_ldmks1, color='cyan')
        img1_with_ldmks_orig.save(os.path.join(args.save_dir, f"{example_prefix}_img1_01_input_with_landmarks.png"))
        
        # 4.1.2: Aligned Image + Landmark (112x112 小圖, 5個紅色點)
        pil_aligned1 = tensor_to_pil(aligned_x1)
        img1_aligned_with_ldmks = draw_landmarks(pil_aligned1.copy(), aligned_ldmks1, color='red', radius=1)
        img1_aligned_with_ldmks.save(os.path.join(args.save_dir, f"{example_prefix}_img1_02_aligned_with_landmarks.png"))

        # --- 影像 2 (編輯圖) ---
        # 4.2.1: Input Image + Landmark (1024x1024 大圖, 5個綠色點)
        img2_with_ldmks_orig = draw_landmarks(edit_image_pil.copy(), orig_ldmks2, color='lime')
        img2_with_ldmks_orig.save(os.path.join(args.save_dir, f"{example_prefix}_img2_01_input_with_landmarks.png"))
        
        # 4.2.2: Aligned Image + Landmark (112x112 小圖, 5個黃色點)
        pil_aligned2 = tensor_to_pil(aligned_x2)
        img2_aligned_with_ldmks = draw_landmarks(pil_aligned2.copy(), aligned_ldmks2, color='yellow', radius=1)
        img2_aligned_with_ldmks.save(os.path.join(args.save_dir, f"{example_prefix}_img2_02_aligned_with_landmarks.png"))

        # 4.3: Feature Vector
        np.savetxt(os.path.join(args.save_dir, f"{example_prefix}_img1_03_feature_vector.txt"), feat1.detach().cpu().numpy())
        np.savetxt(os.path.join(args.save_dir, f"{example_prefix}_img2_03_feature_vector.txt"), feat2.detach().cpu().numpy())

        print(f"已成功儲存 {example_prefix} img1 和 img2 的中間步驟！")

        # 5. 輸出結果
        print("\n" + "="*30)
        print(f"FR Score (Cosine Similarity): {similarity_score:.4f}")
        print("="*30)

    except FileNotFoundError as e:
        print(f"\n錯誤: 找不到影像檔案。")
        print(e)
    except Exception as e:
        print(f"\n發生錯誤: {e}")
