import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap


def layer_average(original_dict):
    layer_values = defaultdict(list)
    for key, value in original_dict.items():
        # ['model', 'layers', '17', 'self_attn', 'q_proj', 'weight']
        parts = key.split('.')
        layer_num = parts[2] 
        layer_values[layer_num].append(value)
    new_dict = {}
    for layer_num, values_list in layer_values.items():
        avg_value = sum(values_list) / len(values_list)
        new_dict[f"model.layers.{layer_num}"] = avg_value
    for k, v in new_dict.items():
        print(f"{k}: {v}")
    return new_dict


def svd_weight(param, r):
    U, S, Vh = torch.svd(param)
    U_r = U[:, :r]          
    S_r = torch.diag(S[:r]) 
    Vh_r = Vh[:, :r]        
    return U_r @ S_r, Vh_r


def trans_weight(slm, llm, save_path, r=8, key_parameters=["q_proj","v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj"]):
    layer_site = len(llm.model.layers) - len(slm.model.layers)
    params_llm = llm.state_dict()
    norms = {}
    for layer_site_temp in range(19, layer_site):
        norm_temp = {}
        for name, param in slm.named_parameters():
            if "layer" in name and any(name_lora in name for name_lora in key_parameters) and "weight" in name:
                param_slm = param
                layer_num = int(re.search(r'layers\.(\d+)\.', name).group(1))
                name_llm = name.replace(str(layer_num), str(layer_num+layer_site_temp), 1)
                param_llm = params_llm[name_llm]
                slm_U, slm_V = svd_weight(param_slm, r)  ## m,n to mxr, nxr
                llm_U, llm_V = svd_weight(param_llm, r)
                Wu = llm_U @ torch.linalg.pinv(slm_U)  
                Wv = llm_V @ torch.linalg.pinv(slm_V)  
                error_temp = torch.linalg.norm((Wu @ param_slm @ Wv.T) - param_llm)
                norm_temp[name] = error_temp
        norm_avgs_layer = layer_average(norm_temp)
        norms[layer_site_temp] = norm_avgs_layer
    print(norms)
    torch.save(norms, save_path)


def dynamic_value(data_np):
    R, C = data_np.shape

    # ==========================================
    # Dynamic Programming
    # ==========================================
    # dp[r, c]: for the first c columns, if the r-th row is selected in the c-th column, the maximum path sum that can be obtained.
    dp = np.zeros((R, C))
    # choice[r, c]: to get the maximum sum by choosing row r in column c, which row was chosen in column c-1 (used for backtracking).
    choice = np.zeros((R, C), dtype=int)

    # Boundary initialization: the maximum sum of the first column (c=0) is the original data itself
    for r in range(R):
        dp[r, 0] = data_np[r, 0]

    # State trans: per-column
    for c in range(1, C):
        for r in range(R):
            max_val_prev_col = -1e9
            best_r_prev = -1
            
            # Constraint: The row number r_prev selected in the previous column must be <= the row number r in the current column (to ensure non-decreasing order)
            for r_prev in range(r + 1):
                if dp[r_prev, c-1] > max_val_prev_col:
                    max_val_prev_col = dp[r_prev, c-1]
                    best_r_prev = r_prev
                    
            # Update dp 
            dp[r, c] = data_np[r, c] + max_val_prev_col
            choice[r, c] = best_r_prev

    # ==========================================
    # 3. Traceback for optimal path
    # ==========================================
    # Find the maximum sum in the last column and the corresponding row number
    best_last_r = np.argmax(dp[:, C-1])
    max_sum = dp[best_last_r, C-1]

    # Find the specific path from back to front
    path = []
    curr_r = best_last_r
    for c in range(C-1, -1, -1):
        path.append(curr_r)
        if c > 0: # jump forward
            curr_r = choice[curr_r, c]

    # Reverse for optimal selection [c=0, c=1, ..., c=23]
    path = path[::-1]

    print(f"Max Sum: {max_sum:.4f}")
    print(f"Row indices for each column: \n{path}")
    return path


path_slm = '/models/Qwen2.5-0.5B-Instruct/'  
path_llm = '/models/Qwen2.5-3B-Instruct/'  ## Qwen2.5-0.5B: 24 layers, Qwen2.5-3B: 36 layers, Qwen2.5-7B: 28 layers, Qwen2.5-1.5B: 28 layers, Qwen2.5-14B: 48 layers
model_llm = AutoModelForCausalLM.from_pretrained(path_llm, trust_remote_code=True)
model_slm = AutoModelForCausalLM.from_pretrained(path_slm, trust_remote_code=True)
trans_weight(model_slm, model_llm, 'qwen250523_c.pt')


sims = torch.load('qwen250523_c.pt')
sims_list = []
for idx in sims:
    sims_list.append([i.item() for i in sims[idx].values()])
data_np = np.array(sims_list)

path = dynamic_value(1/data_np)
R, C = data_np.shape


## 0-1 norm
min_vals = np.min(data_np, axis=0)
max_vals = np.max(data_np, axis=0)
ptp = max_vals - min_vals
ptp[ptp == 0] = 1.0 
data_np = (data_np - min_vals) / ptp
quit()


plt.figure(figsize=(12, 6))

color_list = ['#7eb0d5', '#ffffff', '#d9a5b3']
custom_cmap = LinearSegmentedColormap.from_list('blue_pink', color_list)


# plt.imshow(data_np, cmap='viridis', aspect='auto')

plt.imshow(data_np, cmap=custom_cmap, aspect='auto')

plt.colorbar()
x_coords = np.arange(C)
y_coords = path
plt.plot(x_coords, y_coords, color='#4a4a4a', marker='o', markersize=8, 
         markerfacecolor='none', markeredgewidth=2, linewidth=2, label='Optimal monotonic projection')


# plt.title('Projection Error (Column-wise Normization)', fontsize=20, fontweight='bold')
plt.xlabel('Layer Index of Qwen2.5-0.5B ($L_{0.5B}$)', fontsize=20, fontweight='bold')
plt.ylabel('Layer Offset ($\Delta L$) to Qwen2.5-3B' + '\n' + r'($L_{3B} = L_{0.5B} + \Delta L$)', fontsize=16, fontweight='bold')


plt.xticks(np.arange(0, 24, 1))
plt.yticks(np.arange(0, 12, 1))
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')
plt.legend(fontsize=20)

plt.tight_layout()
plt.savefig('heatmap_norm.pdf', dpi=1024,bbox_inches='tight')
plt.show()

