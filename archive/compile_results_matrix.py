import os
import glob
import pandas as pd
import numpy as np

OUTPUT_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/benchmarks_cv"
MODELS = ["standard_unet", "attention_unet", "r2_unet", "attention_residual_unet"]

# Updated clean mapping strictly matching your 8 consolidated channels
CLASS_MAPPING = {
    1: "healthy", 
    2: "emphysema", 
    3: "ground_glass", 
    4: "fibrosis", 
    5: "micronodules", 
    6: "consolidation", 
    7: "other_rare_pathologies"
}

# Isolate your core foreground pathologies for pure diagnostic evaluation averaging
FOREGROUND_DISEASES = ["emphysema", "ground_glass", "fibrosis", "micronodules", "consolidation", "other_rare_pathologies"]

print("=== 🏆 COMPILING CONSOLIDATED 8-CLASS MASTER THESIS RESULTS MATRIX ===")

master_summary_rows = []

for model_name in MODELS:
    history_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, model_name, "epoch_history_fold_*.csv")))
    if not history_files:
        print(f"  • {model_name.upper():<25} | ⏳ Awaiting active serial pipeline completion...")
        continue
        
    print(f"  • {model_name.upper():<25} | Found {len(history_files)} completed fold logs. Processing...")
    
    fold_global_dices = []
    fold_foreground_dices = []
    class_dices_accumulator = {cls_name: [] for cls_name in CLASS_MAPPING.values()}
    
    for f_path in history_files:
        df = pd.read_csv(f_path)
        final_row = df.iloc[-1]  # Safely extracts the final completed epoch row
        
        # Guard check: handles both historical 18-class and new 8-class file lengths safely
        if "Val_Mean_Dice" in final_row:
            fold_global_dices.append(final_row["Val_Mean_Dice"])
        elif "Val_Global_Dice" in final_row:
            fold_global_dices.append(final_row["Val_Global_Dice"])
            
        fg_values = []
        for cls_name in CLASS_MAPPING.values():
            col_name = f"Dice_{cls_name}"
            if col_name in final_row:
                val = final_row[col_name]
                class_dices_accumulator[cls_name].append(val)
                if cls_name in FOREGROUND_DISEASES:
                    fg_values.append(val)
                    
        if fg_values:
            fold_foreground_dices.append(np.mean(fg_values))
            
    report_row = {
        "Architecture": model_name.upper(),
        "Global_Mean_Dice (Base)": np.mean(fold_global_dices) if fold_global_dices else 0.0,
        "Foreground_Mean_Dice (ILD)": np.mean(fold_foreground_dices) if fold_foreground_dices else 0.0
    }
    
    for cls_name in CLASS_MAPPING.values():
        report_row[f"Dice_{cls_name}"] = np.mean(class_dices_accumulator[cls_name]) if class_dices_accumulator[cls_name] else 0.0
        
    master_summary_rows.append(report_row)

if master_summary_rows:
    df_master = pd.DataFrame(master_summary_rows)
    df_thesis_ready = df_master.set_index("Architecture").T
    
    print("\n" + "="*80)
    print("📈 FINAL EXPERIMENTAL MODEL PERFORMANCE COMPARISON SHEET")
    print("="*80)
    print(df_thesis_ready.to_string())
    print("="*80)
    
    output_matrix_csv = os.path.join(OUTPUT_DIR, "master_thesis_comparison_matrix.csv")
    df_thesis_ready.to_csv(output_matrix_csv)
    print(f"\n✅ Definitive comparison sheet updated and saved to disk:\n👉 {output_matrix_csv}")
