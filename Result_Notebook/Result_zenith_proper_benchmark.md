ini adalah hasil dari test di Notebook : 
https://colab.research.google.com/github/vibeswithkk/ZENITH/blob/main/notebooks/zenith_proper_benchmark.ipynb 

Zenith Performance Benchmark - Proper Backend
This notebook tests the real Zenith optimization pipeline, not a pass-through.

Hardware: NVIDIA Tesla T4 (Google Colab) Model: TinyLlama 1.1B with LoRA fine-tuning

==================================================================
# Install dependencies
!pip install -q transformers peft datasets trl accelerate bitsandbytes
# Install Zenith from GitHub (latest with integrations module)
!pip install -q git+https://github.com/vibeswithkk/ZENITH.git
--- 
Cell output : 
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 518.9/518.9 kB 13.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 59.1/59.1 MB 17.0 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Building wheel for pyzenith (pyproject.toml) ... done


=================================================================
# Verify GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
--- 
Cell output : 
PyTorch: 2.9.0+cu126
CUDA Available: True
GPU: Tesla T4
VRAM: 15.8 GB


==================================================================
# Import Zenith - this auto-registers the 'zenith' backend!
import zenith
print(f"Zenith Version: {zenith.__version__}")

# Check if backend is registered
from zenith.integrations.torch_dynamo import is_registered
print(f"Zenith Backend Registered: {is_registered()}")

# List available backends
if hasattr(torch, '_dynamo'):
    backends = torch._dynamo.list_backends()
    print(f"Available backends: {backends}")
    assert 'zenith' in backends, "Zenith backend not registered!"
---
Cell output : 
Zenith Version: 0.2.10
Zenith Backend Registered: True
Available backends: ['cudagraphs', 'inductor', 'openxla', 'tvm', 'zenith']

==================================================================
# Setup imports
import gc
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
==================================================================
# Benchmark function
def run_benchmark(use_zenith, steps=50, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    mode_name = "ZENITH (Real Backend)" if use_zenith else "PYTORCH (Baseline)"
    print(f"\n{'='*20} {mode_name} {'='*20}")
    
    clean_memory()
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    
    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    
    # Apply Zenith optimization (REAL backend, not pass-through!)
    if use_zenith:
        print("Applying Zenith optimization via torch.compile...")
        # This uses auto-registered 'zenith' backend from zenith.integrations
        model.model = torch.compile(model.model, backend="zenith")
        print("Zenith compilation complete!")
    
    # Dataset
    dataset = load_dataset("tatsu-lab/alpaca", split=f"train[:{steps*2}]")
    def format_prompt(sample):
        return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"
    
    # Trainer
    args = SFTConfig(
        output_dir=f"./results_{mode_name.replace(' ', '_')}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=steps,
        fp16=(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        packing=False
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=args,
        processing_class=tokenizer,
        formatting_func=format_prompt
    )
    
    # Train and time
    start = time.time()
    trainer.train()
    end = time.time()
    
    # Results
    total_time = end - start
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    
    results = {
        "mode": mode_name,
        "total_time": total_time,
        "peak_vram_gb": peak_mem,
        "steps": steps
    }
    
    print(f"Total Time: {total_time:.2f}s")
    print(f"Peak VRAM: {peak_mem:.2f} GB")
    
    del model, trainer, dataset
    clean_memory()
    
    return results
==================================================================

# Run benchmarks
print("="*60)
print("  ZENITH PERFORMANCE BENCHMARK - PROPER BACKEND")
print("="*60)

# Run PyTorch baseline first
pytorch_results = run_benchmark(use_zenith=False, steps=50)

# Run Zenith optimized
zenith_results = run_benchmark(use_zenith=True, steps=50)
---
Cell output : 
  ============================================================
  ZENITH PERFORMANCE BENCHMARK - PROPER BACKEND
============================================================

==================== PYTORCH (Baseline) ====================
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
tokenizer_config.json:  1.29k/? [00:00<00:00, 135kB/s]tokenizer.model: 100% 500k/500k [00:01<00:00, 418kB/s]tokenizer.json:  1.84M/? [00:00<00:00, 57.6MB/s]special_tokens_map.json: 100% 551/551 [00:00<00:00, 56.7kB/s]config.json: 100% 608/608 [00:00<00:00, 70.5kB/s]`torch_dtype` is deprecated! Use `dtype` instead!
model.safetensors: 100% 2.20G/2.20G [00:25<00:00, 109MB/s]generation_config.json: 100% 124/124 [00:00<00:00, 4.62kB/s]README.md:  7.47k/? [00:00<00:00, 156kB/s]data/train-00000-of-00001-a09b74b3ef9c3b(…): 100% 24.2M/24.2M [00:00<00:00, 33.0MB/s]Generating train split: 100% 52002/52002 [00:00<00:00, 107092.74 examples/s]/usr/local/lib/python3.12/dist-packages/peft/tuners/tuners_utils.py:282: UserWarning: Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!
  warnings.warn(
Applying formatting function to train dataset: 100% 100/100 [00:00<00:00, 1062.93 examples/s]Adding EOS to train dataset: 100% 100/100 [00:00<00:00, 1467.62 examples/s]Tokenizing train dataset: 100% 100/100 [00:00<00:00, 342.68 examples/s]Truncating train dataset: 100% 100/100 [00:00<00:00, 2506.04 examples/s]The model is already on multiple devices. Skipping the move to device specified in `args`.
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'pad_token_id': 2}.

    
      
      
      [50/50 01:31, Epoch 2/2]
    
    
  
 
      Step
      Training Loss
    
  
  
    
      10
      1.802200
    
    
      20
      1.551600
    
    
      30
      1.373400
    
    
      40
      1.333400
    
    
      50
      1.338100
    
  
Total Time: 96.28s
Peak VRAM: 2.25 GB

==================== ZENITH (Real Backend) ====================
Applying Zenith optimization via torch.compile...
Zenith compilation complete!
/usr/local/lib/python3.12/dist-packages/peft/tuners/tuners_utils.py:282: UserWarning: Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!
  warnings.warn(
The model is already on multiple devices. Skipping the move to device specified in `args`.
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'pad_token_id': 2}.

    
      
      
      [50/50 01:27, Epoch 2/2]
    
    
  
 
      Step
      Training Loss
    
  
  
    
      10
      1.809200
    
    
      20
      1.553600
    
    
      30
      1.378000
    
    
      40
      1.335600
    
    
      50
      1.340600
    
  
Total Time: 89.09s
Peak VRAM: 2.25 GB

==================================================================
# Calculate and display results
print("\n" + "="*60)
print("  BENCHMARK RESULTS")
print("="*60)

speedup = (pytorch_results['total_time'] - zenith_results['total_time']) / pytorch_results['total_time'] * 100

print(f"\nPyTorch Baseline: {pytorch_results['total_time']:.2f}s")
print(f"Zenith Optimized: {zenith_results['total_time']:.2f}s")
print(f"\nSpeedup: {speedup:+.2f}%")
print(f"\nPeak VRAM (PyTorch): {pytorch_results['peak_vram_gb']:.2f} GB")
print(f"Peak VRAM (Zenith):  {zenith_results['peak_vram_gb']:.2f} GB")
---
Cell output : 

============================================================
  BENCHMARK RESULTS
============================================================

PyTorch Baseline: 96.28s
Zenith Optimized: 89.09s

Speedup: +7.47%

Peak VRAM (PyTorch): 2.25 GB
Peak VRAM (Zenith):  2.25 GB

==============================================================

Expected Results
With the proper Zenith backend (not pass-through):

Training speedup: +5-15%
Same or lower VRAM usage
Numerical accuracy preserved (MSE ~ 0)
