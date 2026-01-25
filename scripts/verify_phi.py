
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from models import get_model_wrapper
from config import load_config

def test_phi_loading():
    print("Testing Phi-2 Model Wrapper Loading...")
    try:
        # Reduced precision/cpu to avoid OOM on local test if necessary
        # but using 'cuda' if available for realistic test
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Test 1: Load by short name
        print("\n1. Loading by short name 'phi'...")
        # Note: 'phi' isn't a direct model name in wrapper, usually 'microsoft/phi-2'
        # But let's test the factory resolution via HF path
        
        # We need to test the actual model name used in config
        model_name = "microsoft/phi-2"
        print(f"Loading {model_name}...")
        
        model = get_model_wrapper(model_name, device=device)
        print("✓ Model loaded successfully")
        
        print("\n2. Verifying Attributes...")
        print(f"  n_layers: {model.n_layers}")
        print(f"  d_model: {model.d_model}")
        
        assert model.n_layers == 32
        assert model.d_model == 2560
        print("✓ Attributes correct")
        
        print("\n3. Testing Hooks...")
        model.register_hooks()
        print(f"✓ Registered {len(model.hooks)} hooks (Expected ~96)")
        
        # Basic check
        assert len(model.hooks) > 0
        
        print("\n4. Testing Forward Pass (Mock)...")
        text = "Hello, world!"
        res = model.process_text(text)
        
        print("  Logits shape:", res['logits'].shape)
        print("  Residual stream layers:", len(res['residual_stream']))
        print("  MLP output layers:", len(res['mlp_outputs']))
        
        assert len(res['residual_stream']) == 32
        
        # Cleanup
        model.remove_hooks()
        print("✓ Forward pass successful")
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_phi_loading()
