import torch
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- 1. 設定 SDA 攻擊參數 ---
DEVICE = "cuda"
MODEL_ID = "runwayml/stable-diffusion-v1-5" # 攻擊目標通常是這個通用底模
STEPS = 50           # 攻擊迭代次數
EPS = 0.05           # 擾動預算 (可調整，越大越亂但也越有效)
ALPHA = 0.01         # 攻擊步長 (Step Size)
ATTACK_TIMESTEP = 981 # 論文提到攻擊"初始去噪步驟"。SD通常T=1000，選一個靠近1000的數字

# --- 2. 載入模型 ---
print(f"Loading Stable Diffusion: {MODEL_ID}...")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
vae = pipe.vae
unet = pipe.unet

# 凍結參數，我們只攻擊圖片
vae.requires_grad_(False)
unet.requires_grad_(False)

# --- 3. 定義 Hook (攔截 Self-Attention 的 Query) ---
# SDA 論文核心：干擾 Self-Attention 的 Query 向量
clean_queries = {}
adv_queries = {}

def get_query_hook(layer_name, save_dict):
    def hook(module, input, output):
        # input[0] 通常是 hidden_states
        # 對於 Linear 層 (to_q)，output 就是 Query
        save_dict[layer_name] = output
    return hook

# 註冊 Hook 到 U-Net 的所有 Attention 層
# 我們主要攻擊 "attn1" (Self-Attention) 的 "to_q" (Query Projection)
for name, module in unet.named_modules():
    if "attn1" in name and "to_q" in name:
        module.register_forward_hook(get_query_hook(name, adv_queries))

# 輔助函式：影像前處理
def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8
    image = image.resize((w, h), resample=Image.BICUBIC)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def postprocess(image_tensor):
    image = (image_tensor / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

# --- 4. SDA 攻擊主函式 ---
def sda_attack(image_path, output_path):
    # 讀取圖片
    img_pil = Image.open(image_path).convert("RGB")
    x_clean = preprocess(img_pil).to(DEVICE).half()
    
    # 初始化擾動圖片
    x_adv = x_clean.clone().detach()
    x_adv.requires_grad = True
    
    # 準備雜訊 (固定雜訊以確保公平比較)
    latents_clean = vae.encode(x_clean).latent_dist.sample() * 0.18215
    noise = torch.randn_like(latents_clean)
    timesteps = torch.tensor([ATTACK_TIMESTEP], device=DEVICE, dtype=torch.long)
    
    # (A) 取得乾淨圖片的 Query (Ground Truth)
    # 我們需要先解除 hook 紀錄到 adv_queries，改存到 clean_queries
    # 這裡簡化處理：直接再跑一次 forward pass 並手動切換儲存目標，
    # 或者我們直接用兩個 dict，上面 hook 寫死了 adv_queries，
    # 這裡我們用一個 trick：先跑一次 adv (其實是 clean)，把結果存到 clean_queries
    
    # 暫時重新註冊 hook 到 clean_queries
    hooks = []
    for name, module in unet.named_modules():
        if "attn1" in name and "to_q" in name:
            # 移除舊 hook (如果有)
            module._forward_hooks.clear() 
            hooks.append(module.register_forward_hook(get_query_hook(name, clean_queries)))
            
    with torch.no_grad():
        noisy_latents = pipe.scheduler.add_noise(latents_clean, noise, timesteps)
        unet(noisy_latents, timesteps, encoder_hidden_states=torch.zeros((1, 77, 768), device=DEVICE, dtype=torch.float16)) # dummy prompt
    
    # 清除 hook 並註冊回 adv_queries
    for h in hooks: h.remove()
    for name, module in unet.named_modules():
        if "attn1" in name and "to_q" in name:
            module.register_forward_hook(get_query_hook(name, adv_queries))

    print(f"Start SDA Attack... Target: Maximize Query Difference")
    optimizer = torch.optim.Adam([x_adv], lr=ALPHA)

    for i in tqdm(range(STEPS)):
        # 1. 前向傳播 (Forward)
        latents_adv = vae.encode(x_adv).latent_dist.sample() * 0.18215
        noisy_latents_adv = pipe.scheduler.add_noise(latents_adv, noise, timesteps)
        
        adv_queries.clear()
        unet(noisy_latents_adv, timesteps, encoder_hidden_states=torch.zeros((1, 77, 768), device=DEVICE, dtype=torch.float16))
        
        # 2. 計算 Loss: Query 差異最大化
        # Loss = - sum(|| Q_adv - Q_clean ||)  (因為 optimizer 是 minimize，所以加負號變成 maximize 距離)
        loss = 0
        for name, q_adv in adv_queries.items():
            q_clean = clean_queries[name].detach()
            # 計算 MSE 距離
            dist = F.mse_loss(q_adv, q_clean)
            loss -= dist # 我們要讓距離越大越好
            
        # 3. 更新圖片
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 4. PGD 限制 (Projected Gradient Descent)
        # 確保擾動不超過 EPS
        diff = x_adv - x_clean
        diff = torch.clamp(diff, -EPS, EPS)
        x_adv.data = torch.clamp(x_clean + diff, -1, 1)
        
    # 儲存結果
    final_img = postprocess(x_adv.detach())
    final_img.save(output_path)
    print(f"SDA Attack Complete. Saved to {output_path}")

# --- 5. 執行 ---
if __name__ == "__main__":
    # 使用範例：把 FaceLock 跑完的圖當作輸入
    input_image = "facelock_result.png" 
    sda_attack(input_image, "final_protected_image.png")
