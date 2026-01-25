import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
import sys

print("=== Debugging Phi-2 Mode Loading ===")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Cuda Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

model_name = "microsoft/phi-2"
cache_dir = "./data/cache/phi-2"
os.makedirs(cache_dir, exist_ok=True)

print(f"\nTarget Model: {model_name}")
print(f"Cache Dir: {cache_dir}")
print(f"HF_HOME: {os.environ.get('HF_HOME')}")

try:
    print("\n1. Loading Tokenizer...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    print(f"✓ Tokenizer loaded in {time.time() - start:.2f}s")
    
    print("\n2. Loading Model Config...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir=cache_dir,
        device_map="auto" # Let HF handle device placement to see if it works
    )
    print(f"✓ Model loaded in {time.time() - start:.2f}s")
    
    print("\n3. Testing Forward Pass...")
    input_text = "Hello, world!"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    print("✓ Forward pass successful")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n=== End Debug ===")
