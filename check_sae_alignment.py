
import torch
import numpy as np
from pathlib import Path
from models.sae_model import JumpReLUSAE
import matplotlib.pyplot as plt

def load_sae(layer_idx, device='cuda'):
    path = Path(f"./models/checkpoints/gpt2-medium/sae_layer_{layer_idx}_best.pt")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    d_hidden = state_dict['W_enc.weight'].shape[0]
    d_model = state_dict['W_enc.weight'].shape[1]
    
    sae = JumpReLUSAE(d_model=d_model, d_hidden=d_hidden)
    sae.load_state_dict(state_dict)
    sae.to(device)
    return sae

def check_alignment():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading SAE Layer 10...")
    sae1 = load_sae(10, device)
    print("Loading SAE Layer 11...")
    sae2 = load_sae(11, device)
    
    W1 = sae1.W_dec.weight.detach() # [d_model, d_hidden]
    W2 = sae2.W_dec.weight.detach() # [d_model, d_hidden]
    
    # Normalize
    W1 = torch.nn.functional.normalize(W1, dim=0)
    W2 = torch.nn.functional.normalize(W2, dim=0)
    
    print(f"Shape: {W1.shape}")
    
    # Compute similarity matrix [d_hidden, d_hidden]
    # This is huge (5120x5120), might OOM if done naively
    # Let's do it in chunks or just sample random features
    
    print("Computing max similarities for first 100 features of Layer 10...")
    
    max_sims = []
    for i in range(100):
        feat = W1[:, i] # [d_model]
        sims = torch.matmul(feat, W2) # [d_hidden]
        max_sim = sims.max().item()
        max_sims.append(max_sim)
        
    avg_max_sim = np.mean(max_sims)
    print(f"Average Max Similarity: {avg_max_sim:.4f}")
    print(f"Min Max Similarity: {np.min(max_sims):.4f}")
    print(f"Max Max Similarity: {np.max(max_sims):.4f}")
    
    print("\nDistribution of best matches:")
    hist, bins = np.histogram(max_sims, bins=10)
    for h, b in zip(hist, bins):
        print(f"{b:.2f}: {h}")

if __name__ == "__main__":
    check_alignment()
