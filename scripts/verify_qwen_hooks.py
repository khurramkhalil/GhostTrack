
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models import get_model_wrapper

def test_hooks():
    print("Initializing Qwen wrapper...")
    # Use cpu for login node
    model = get_model_wrapper('Qwen/Qwen2.5-1.5B', device='cpu')
    
    text = "The quick brown fox jumps over the lazy dog."
    print(f"\nProcessing text: '{text}'")
    
    try:
        results = model.process_text(text)
        
        print("\nChecking captured activations:")
        
        resid = results.get('residual_stream', [])
        print(f"Residual Stream: {len(resid)} layers")
        if resid:
            print(f"  Shape: {resid[0].shape}")
            
        mlp = results.get('mlp_outputs', [])
        print(f"MLP Outputs: {len(mlp)} layers")
        if mlp:
            print(f"  Shape: {mlp[0].shape}")
            
        attn = results.get('attn_outputs', [])
        print(f"Attn Outputs: {len(attn)} layers")
        if attn:
            print(f"  Shape: {attn[0].shape}")
            
        if len(resid) == 0:
            print("❌ FAILURE: No residual stream activations captured!")
        else:
            print("✅ SUCCESS: Activations captured.")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hooks()
