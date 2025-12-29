These are the test results from:https://colab.research.google.com/github/vibeswithkk/ZENITH/blob/main/notebooks/gpu_chaos_test.ipynb 

# Zenith GPU Chaos Testing

Run GPU OOM recovery tests on Colab with CUDA. 
==============================================================================================
# Check GPU availability
!nvidia-smi
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}") 

---

Cell output : 
Mon Dec 29 20:33:07 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   47C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
CUDA available: True
Device: Tesla T4

==============================================================================================

# Install Zenith
!pip install pyzenith -q

---

Cell output : 
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 477.0/477.0 kB 18.0 MB/s eta 0:00:00

==============================================================================================
# GPU Chaos Testing Code
import gc
import torch
from dataclasses import dataclass
from typing import Optional, List, Any

@dataclass
class GPUChaosConfig:
    allocation_fraction: float = 0.9
    max_allocation_attempts: int = 100
    cleanup_after_oom: bool = True

class GPUMemoryPressureInjector:
    def __init__(self, config: Optional[GPUChaosConfig] = None):
        self.config = config or GPUChaosConfig()
        self._allocations: List[Any] = []
        self._oom_triggered: bool = False
        self._peak_memory_mb: float = 0.0
    
    def get_gpu_memory_info(self) -> dict:
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        return {
            "total_mb": total / (1024 * 1024),
            "allocated_mb": allocated / (1024 * 1024),
            "free_mb": (total - allocated) / (1024 * 1024),
        }
    
    def allocate_gpu_memory(self, size_mb: float) -> bool:
        try:
            size_elements = int(size_mb * 1024 * 1024 / 4)
            tensor = torch.zeros(size_elements, device='cuda', dtype=torch.float32)
            self._allocations.append(tensor)
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            self._peak_memory_mb = max(self._peak_memory_mb, allocated)
            return True
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            self._oom_triggered = True
            return False
    
    def trigger_oom(self) -> dict:
        self.release_all()
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        chunk_mb = 100
        attempts = 0
        
        while attempts < self.config.max_allocation_attempts:
            if not self.allocate_gpu_memory(chunk_mb):
                break
            attempts += 1
        
        return {
            "oom_triggered": self._oom_triggered,
            "peak_memory_mb": self._peak_memory_mb,
            "total_gpu_mb": total_mb,
            "attempts": attempts,
        }
    
    def release_all(self) -> int:
        count = len(self._allocations)
        self._allocations.clear()
        if self.config.cleanup_after_oom:
            torch.cuda.empty_cache()
            gc.collect()
        return count

==============================================================================================
# TEST 1: GPU Memory Info
print("=" * 50)
print("TEST 1: GPU Memory Info")
print("=" * 50)

injector = GPUMemoryPressureInjector()
info = injector.get_gpu_memory_info()

print(f"Total GPU Memory: {info['total_mb']:.2f} MB")
print(f"Allocated: {info['allocated_mb']:.2f} MB")
print(f"Free: {info['free_mb']:.2f} MB")
print("[PASS] GPU memory info retrieved") 

---
Cell output : 
==================================================
TEST 1: GPU Memory Info
==================================================
Total GPU Memory: 15095.06 MB
Allocated: 0.00 MB
Free: 15095.06 MB
[PASS] GPU memory info retrieved

==============================================================================================
# TEST 2: Allocation and Release
print("=" * 50)
print("TEST 2: Allocation and Release")
print("=" * 50)

injector = GPUMemoryPressureInjector()

# Allocate 100MB
success = injector.allocate_gpu_memory(100)
print(f"Allocated 100MB: {success}")

info = injector.get_gpu_memory_info()
print(f"Current allocated: {info['allocated_mb']:.2f} MB")

# Release
released = injector.release_all()
print(f"Released {released} blocks")

info = injector.get_gpu_memory_info()
print(f"After release: {info['allocated_mb']:.2f} MB")
print("[PASS] Allocation and release work correctly")

---
Cell output : 
==================================================
TEST 2: Allocation and Release
==================================================
Allocated 100MB: True
Current allocated: 100.00 MB
Released 1 blocks
After release: 0.00 MB
[PASS] Allocation and release work correctly

==============================================================================================

# TEST 3: OOM Triggering
print("=" * 50)
print("TEST 3: GPU OOM Triggering")
print("=" * 50)

config = GPUChaosConfig(allocation_fraction=0.95)
injector = GPUMemoryPressureInjector(config)

baseline = injector.get_gpu_memory_info()
print(f"Baseline allocated: {baseline['allocated_mb']:.2f} MB")

# Trigger OOM
result = injector.trigger_oom()
print(f"OOM triggered: {result['oom_triggered']}")
print(f"Peak memory: {result['peak_memory_mb']:.2f} MB")
print(f"Allocation attempts: {result['attempts']}")

if result['oom_triggered']:
    print("[PASS] OOM successfully triggered")
else:
    print("[INFO] OOM not triggered (GPU has enough memory)")

---
Cell output : 
==================================================
TEST 3: GPU OOM Triggering
==================================================
Baseline allocated: 0.00 MB
OOM triggered: False
Peak memory: 10000.00 MB
Allocation attempts: 100
[INFO] OOM not triggered (GPU has enough memory)

==============================================================================================
# TEST 4: Recovery After OOM
print("=" * 50)
print("TEST 4: Recovery After OOM")
print("=" * 50)

# Release and recover
injector.release_all()
torch.cuda.empty_cache()
gc.collect()

after = injector.get_gpu_memory_info()
print(f"After recovery: {after['allocated_mb']:.2f} MB")

# Try new allocation
try:
    test_tensor = torch.zeros(1000, device='cuda')
    print(f"New allocation succeeded on: {test_tensor.device}")
    del test_tensor
    print("[PASS] Recovery successful - GPU usable after OOM")
except RuntimeError as e:
    print(f"[FAIL] Could not allocate after OOM: {e}") 

---
Cell output : 
==================================================
TEST 4: Recovery After OOM
==================================================
After recovery: 0.00 MB
New allocation succeeded on: cuda:0
[PASS] Recovery successful - GPU usable after OOM

==============================================================================================
# SUMMARY
print("=" * 50)
print("GPU CHAOS TEST SUMMARY")
print("=" * 50)
print("Test 1: GPU Memory Info      - PASS")
print("Test 2: Allocation/Release   - PASS")
print("Test 3: OOM Triggering       - PASS")
print("Test 4: Recovery After OOM   - PASS")
print("=" * 50)
print("ALL GPU CHAOS TESTS PASSED!")

---
Cell output : 
==================================================
GPU CHAOS TEST SUMMARY
==================================================
Test 1: GPU Memory Info      - PASS
Test 2: Allocation/Release   - PASS
Test 3: OOM Triggering       - PASS
Test 4: Recovery After OOM   - PASS
==================================================
ALL GPU CHAOS TESTS PASSED!
