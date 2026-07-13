import os
import subprocess
import time

MODELS = ["standard_unet"] #[ "attention_unet", "r2_unet", "attention_residual_unet"] # 
SCRIPT_PATH = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/run_leakage_proof_experiment.py"
DDPM_PATH = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/run_ddpm_training.py"

print("=================================================================")
print("🏆 INITIALIZING MASTER PRODUCTION PIPELINE RUNNER (SERIAL MODE) 🏆")
print("=================================================================")

# 1. LOOP THROUGH SEGMENTATION ARCHITECTURES SEQUENTIALLY
for idx, model_name in enumerate(MODELS):
    print(f"\n🚀 [{idx+1}/{len(MODELS)}] Launching Cross-Validation: {model_name.upper()}")
    print("-" * 65)
    
    start_time = time.time()
    try:
        # Execute the python script synchronously, passing the epochs parameter flag explicitly
        subprocess.run([
            "python", SCRIPT_PATH, 
            "--model", model_name, 
            "--epochs", "80"
        ], check=True)
        
        elapsed = (time.time() - start_time) / 3600.0
        print(f"✅ Clean completion for {model_name.upper()} | Elapsed Time: {elapsed:.2f} hours.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error encountered during {model_name.upper()} execution loop: {e}")
        print("Moving to next model to protect pipeline continuity...")

# 2. RUN GENERATIVE OPENAI DDPM LOGIC ON THE SAME COMPUTE NODE
print("\n=================================================================")
print("🎨 SEGMENTATION COMPLETE -> LAUNCHING 100-EPOCH CUSTOM ILD DDPM 🎨")
print("=================================================================")

try:
    subprocess.run(["python", DDPM_PATH], check=True)
    print("✅ All pipeline targets completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"❌ Generative diffusion task track errored out: {e}")
