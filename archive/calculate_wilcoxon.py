import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/benchmarks_cv"
# Track the upcoming advanced architectures running in your 23-hour pipeline
MODELS = ["attention_unet", "r2_unet", "attention_residual_unet"]

# Load your completed standard baseline scores matrix
base_path = os.path.join(BASE_DIR, "patient_scores_standard_unet.csv")
if not os.path.exists(base_path):
    print("⏳ Baseline standard_unet patient scores sheet not detected yet.")
    print("   (This script will run automatically once the baseline cross-validation concludes.)")
    # We create a dummy baseline dataframe placeholder if running a pre-flight test pass
    base_df = pd.DataFrame(columns=["Patient_ID", "Achieved_Dice"])
else:
    base_df = pd.read_csv(base_path)

def calculate_hodges_lehmann_ci(x, y, alpha=0.05):
    """Computes non-parametric 95% confidence intervals for paired differences."""
    differences = y - x
    paired_averages = (differences[:, None] + differences[None, :]) / 2.0
    flat_averages = sorted(paired_averages[np.triu_indices(len(differences))])
    
    n = len(differences)
    w_crit = n * (n + 1) / 4.0
    std_dev = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = 1.96  
    
    lower_idx = int(round(w_crit - z * std_dev))
    upper_idx = int(round(w_crit + z * std_dev))
    
    lower_idx = max(0, min(lower_idx, len(flat_averages) - 1))
    upper_idx = max(0, min(upper_idx, len(flat_averages) - 1))
    
    median_diff = np.median(differences)
    return median_diff, flat_averages[lower_idx], flat_averages[upper_idx]

print("\n=== 📊 MULTI-MODEL PATIENT-WISE 8-CLASS WILCOXON REPORT ===")
print(f"Total Patient Cohorts Logged in Baseline: {len(base_df)}")

stats_summary = []
for m_name in MODELS:
    m_path = os.path.join(BASE_DIR, f"patient_scores_{m_name}.csv")
    if not os.path.exists(m_path):
        print(f"  • {m_name.upper():<25} | ⏳ Awaiting production pipeline execution...")
        continue
        
    comp_df = pd.read_csv(m_path)
    merged = pd.merge(base_df, comp_df, on="Patient_ID", suffixes=("_base", "_var")).dropna()
    
    if len(merged) == 0:
        print(f"  • {m_name.upper():<25} | ❌ Error: Merged array contains zero overlapping patient IDs.")
        continue

    v_base = merged["Achieved_Dice_base"].values
    v_var = merged["Achieved_Dice_var"].values
    
    if len(v_base) < 2 or np.all(v_base == v_var):
        print(f"  • {m_name.upper():<25} | ⚠️ Insufficient sample variance to compute statistical ranks.")
        continue
        
    _, p_val = wilcoxon(v_base, v_var)
    med_diff, lower_ci, upper_ci = calculate_hodges_lehmann_ci(v_base, v_var)
    
    stats_summary.append({
        "Comparison (vs Base)": m_name.upper(),
        "Base Median Dice": np.median(v_base),
        "Variant Median Dice": np.median(v_var),
        "Median Delta": med_diff,
        "95% CI Lower": lower_ci,
        "95% CI Upper": upper_ci,
        "p-value": p_val,
        "Significant (p<0.05)": "✅ YES" if p_val < 0.05 else "❌ NO"
    })

if stats_summary:
    df_stats = pd.DataFrame(stats_summary)
    print("\n", df_stats.to_string(index=False))
    df_stats.to_csv(os.path.join(BASE_DIR, "statistical_significance_report.csv"), index=False)
    print(f"\n🏆 Statistical evaluation saved to: {BASE_DIR}/statistical_significance_report.csv")
