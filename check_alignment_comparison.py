
import torch
import numpy as np
from pathlib import Path
from models.sae_model import JumpReLUSAE

def load_sae(model_name, layer_idx, device='cpu'):
    path = Path(f"./models/checkpoints/{model_name}/sae_layer_{layer_idx}_best.pt")
    if not path.exists():
        # Try alternate path
        path = Path(f"./results_downloaded/models/saes/sae_layer_{layer_idx}_best.pt")
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    d_hidden = state_dict['W_enc.weight'].shape[0]
    d_model = state_dict['W_enc.weight'].shape[1]
    
    sae = JumpReLUSAE(d_model=d_model, d_hidden=d_hidden)
    sae.load_state_dict(state_dict)
    sae.to(device)
    return sae

def check_alignment_for_model(model_name, layer1=5, layer2=6):
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"Checking alignment between Layer {layer1} and Layer {layer2}")
    print(f"{'='*50}")
    
    try:
        sae1 = load_sae(model_name, layer1)
        sae2 = load_sae(model_name, layer2)
    except Exception as e:
        print(f"Could not load SAEs: {e}")
        return None
    
    W1 = sae1.W_dec.weight.detach() # [d_model, d_hidden]
    W2 = sae2.W_dec.weight.detach() # [d_model, d_hidden]
    
    # Normalize
    W1 = torch.nn.functional.normalize(W1, dim=0)
    W2 = torch.nn.functional.normalize(W2, dim=0)
    
    print(f"W_dec shape: {W1.shape}")
    
    # Compute max similarities for first 200 features
    max_sims = []
    n_features = min(200, W1.shape[1])
    
    for i in range(n_features):
        feat = W1[:, i] # [d_model]
        sims = torch.matmul(feat, W2) # [d_hidden]
        max_sim = sims.max().item()
        max_sims.append(max_sim)
        
    avg_max_sim = np.mean(max_sims)
    min_max_sim = np.min(max_sims)
    max_max_sim = np.max(max_sims)
    
    print(f"\nAlignment Statistics:")
    print(f"  Average Max Similarity: {avg_max_sim:.4f}")
    print(f"  Min Max Similarity: {min_max_sim:.4f}")
    print(f"  Max Max Similarity: {max_max_sim:.4f}")
    
    # Distribution
    print(f"\nSimilarity Distribution:")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(max_sims, bins=bins)
    for i in range(len(bins)-1):
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}]: {hist[i]} features")
    
    return avg_max_sim

if __name__ == "__main__":
    # Check both models
    small_align = check_alignment_for_model("gpt2", layer1=5, layer2=6)
    medium_align = check_alignment_for_model("gpt2-medium", layer1=10, layer2=11)
    
    if small_align and medium_align:
        print(f"\n{'='*50}")
        print("COMPARISON")
        print(f"{'='*50}")
        print(f"GPT-2 Small Alignment:  {small_align:.4f}")
        print(f"GPT-2 Medium Alignment: {medium_align:.4f}")
        print(f"Ratio: {small_align/medium_align:.2f}x")
