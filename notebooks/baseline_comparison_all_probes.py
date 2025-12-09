"""
Baseline Comparison for ALL Probes
Compares all probe architectures to naive baselines to demonstrate their utility.
"""

print("="*70)
print("Baseline Comparisons: Testing ALL Probes vs Naive Approaches")
print("="*70)

# Get baseline results (same as before)
baseline_results = {}

# Baseline 1: Random predictions
print("\n1. Random Baseline...")
np.random.seed(42)
random_conf = np.random.uniform(0, 1, size=len(y_test))
random_auroc = roc_auc_score(y_test, random_conf)
random_brier = brier_score_loss(y_test, random_conf)
random_ece = compute_ece(random_conf, y_test)

baseline_results['Random'] = {
    'auroc': random_auroc,
    'brier': random_brier,
    'ece': random_ece,
}

print(f"  AUROC: {random_auroc:.3f}")
print(f"  Brier: {random_brier:.4f}")
print(f"  ECE:   {random_ece:.4f}")

# Baseline 2: Constant (always predict model's overall accuracy)
print("\n2. Constant Baseline (predict overall accuracy)...")
overall_accuracy = y_test.mean()
constant_conf = np.full(len(y_test), overall_accuracy)
constant_brier = brier_score_loss(y_test, constant_conf)
constant_ece = compute_ece(constant_conf, y_test)

baseline_results['Constant'] = {
    'auroc': np.nan,  # Can't compute AUROC for constant
    'brier': constant_brier,
    'ece': constant_ece,
}

print(f"  Constant confidence: {overall_accuracy:.3f}")
print(f"  AUROC: N/A (constant predictions)")
print(f"  Brier: {constant_brier:.4f}")
print(f"  ECE:   {constant_ece:.4f}")

# Baseline 3: Sequence length heuristic
print("\n3. Sequence Length Heuristic...")
test_prompts = [prompts[i] for i in test_idx_final]
seq_lengths = np.array([len(tokenizer.encode(p)) for p in test_prompts])

min_len, max_len = seq_lengths.min(), seq_lengths.max()
if max_len > min_len:
    length_conf = 1.0 - (seq_lengths - min_len) / (max_len - min_len)
else:
    length_conf = np.full(len(seq_lengths), 0.5)

length_auroc = roc_auc_score(y_test, length_conf)
length_brier = brier_score_loss(y_test, length_conf)
length_ece = compute_ece(length_conf, y_test)

baseline_results['Seq Length'] = {
    'auroc': length_auroc,
    'brier': length_brier,
    'ece': length_ece,
}

print(f"  AUROC: {length_auroc:.3f}")
print(f"  Brier: {length_brier:.4f}")
print(f"  ECE:   {length_ece:.4f}")

# ============================================================================
# COMPARE ALL PROBES TO BASELINES
# ============================================================================

print("\n" + "="*70)
print("ALL PROBES vs BASELINES COMPARISON")
print("="*70)

# Combine all probes and baselines
all_methods = {}
all_methods.update(baseline_results)

# Add all probes from results
for probe_name, probe_metrics in results.items():
    all_methods[probe_name] = {
        'auroc': probe_metrics['auroc'],
        'brier': probe_metrics['brier'],
        'ece': probe_metrics['ece'],
    }

# Print comparison table
print(f"\n{'Method':<30} {'AUROC':<12} {'Brier':<12} {'ECE':<12}")
print("-"*70)

# Sort by AUROC (best first)
sorted_methods = sorted(
    all_methods.items(),
    key=lambda x: x[1]['auroc'] if not np.isnan(x[1]['auroc']) else -1,
    reverse=True
)

for name, metrics in sorted_methods:
    auroc_str = f"{metrics['auroc']:.4f}" if not np.isnan(metrics['auroc']) else "N/A"
    # Highlight probes vs baselines
    if name in results:
        name_display = f"✓ {name}"
    else:
        name_display = f"  {name}"
    print(f"{name_display:<30} {auroc_str:<12} {metrics['brier']:<12.4f} {metrics['ece']:<12.4f}")

# ============================================================================
# IMPROVEMENT ANALYSIS FOR EACH PROBE
# ============================================================================

print("\n" + "="*70)
print("IMPROVEMENT ANALYSIS: Each Probe vs Baselines")
print("="*70)

for probe_name in sorted(results.keys()):
    probe_metrics = results[probe_name]
    probe_auroc = probe_metrics['auroc']
    probe_brier = probe_metrics['brier']
    probe_ece = probe_metrics['ece']
    
    print(f"\n{probe_name}:")
    print(f"  AUROC: {probe_auroc:.4f}, Brier: {probe_brier:.4f}, ECE: {probe_ece:.4f}")
    
    # vs Random
    if not np.isnan(random_auroc):
        auroc_improvement = (probe_auroc - random_auroc) / random_auroc * 100
        brier_improvement = (random_brier - probe_brier) / random_brier * 100
        print(f"  vs Random:      AUROC +{auroc_improvement:+.1f}%, Brier {brier_improvement:+.1f}%")
    
    # vs Seq Length
    if not np.isnan(length_auroc):
        auroc_improvement = (probe_auroc - length_auroc) / length_auroc * 100
        brier_improvement = (length_brier - probe_brier) / length_brier * 100
        print(f"  vs Seq Length:  AUROC +{auroc_improvement:+.1f}%, Brier {brier_improvement:+.1f}%")

# ============================================================================
# VISUALIZATION: All Probes vs Baselines
# ============================================================================

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 10))

# 1. AUROC Comparison (horizontal bar chart)
ax1 = plt.subplot(2, 3, 1)
probe_names = list(results.keys())
probe_aurocs = [results[p]['auroc'] for p in probe_names]
baseline_names = [k for k in baseline_results.keys() if not np.isnan(baseline_results[k]['auroc'])]
baseline_aurocs = [baseline_results[k]['auroc'] for k in baseline_names]

# Combine and sort
all_names = probe_names + baseline_names
all_aurocs = probe_aurocs + baseline_aurocs
colors = ['green' if n in results else 'lightblue' for n in all_names]

# Sort by AUROC
sorted_idx = np.argsort(all_aurocs)
sorted_names = [all_names[i] for i in sorted_idx]
sorted_aurocs = [all_aurocs[i] for i in sorted_idx]
sorted_colors = [colors[i] for i in sorted_idx]

bars = ax1.barh(range(len(sorted_names)), sorted_aurocs, color=sorted_colors, edgecolor='black', linewidth=1.5)
ax1.set_yticks(range(len(sorted_names)))
ax1.set_yticklabels(sorted_names)
ax1.set_xlabel('AUROC', fontsize=11)
ax1.set_title('Discrimination: AUROC Comparison\n(All Probes vs Baselines)', fontsize=12, fontweight='bold')
ax1.set_xlim([0.4, 1.0])
ax1.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Random chance')
ax1.grid(axis='x', alpha=0.3)
ax1.legend()

for i, (bar, val) in enumerate(zip(bars, sorted_aurocs)):
    ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontweight='bold', fontsize=9)

# 2. Brier Score Comparison
ax2 = plt.subplot(2, 3, 2)
probe_briers = [results[p]['brier'] for p in probe_names]
# Include ALL baselines for Brier (including Constant)
all_baseline_names = list(baseline_results.keys())
baseline_briers = [baseline_results[k]['brier'] for k in all_baseline_names]

all_names_brier = probe_names + all_baseline_names
all_briers = probe_briers + baseline_briers
colors_brier = ['green' if n in results else 'lightblue' for n in all_names_brier]

sorted_idx = np.argsort(all_briers)
sorted_names_brier = [all_names_brier[i] for i in sorted_idx]
sorted_briers = [all_briers[i] for i in sorted_idx]
sorted_colors_brier = [colors_brier[i] for i in sorted_idx]

bars = ax2.barh(range(len(sorted_names_brier)), sorted_briers, color=sorted_colors_brier, edgecolor='black', linewidth=1.5)
ax2.set_yticks(range(len(sorted_names_brier)))
ax2.set_yticklabels(sorted_names_brier)
ax2.set_xlabel('Brier Score (lower is better)', fontsize=11)
ax2.set_title('Calibration: Brier Score Comparison\n(All Probes vs Baselines)', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, sorted_briers)):
    ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontweight='bold', fontsize=9)

# 3. ECE Comparison
ax3 = plt.subplot(2, 3, 3)
probe_eces = [results[p]['ece'] for p in probe_names]
# Include ALL baselines for ECE (including Constant)
baseline_eces = [baseline_results[k]['ece'] for k in all_baseline_names]

all_names_ece = probe_names + all_baseline_names
all_eces = probe_eces + baseline_eces
colors_ece = ['green' if n in results else 'lightblue' for n in all_names_ece]

sorted_idx = np.argsort(all_eces)
sorted_names_ece = [all_names_ece[i] for i in sorted_idx]
sorted_eces = [all_eces[i] for i in sorted_idx]
sorted_colors_ece = [colors_ece[i] for i in sorted_idx]

bars = ax3.barh(range(len(sorted_names_ece)), sorted_eces, color=sorted_colors_ece, edgecolor='black', linewidth=1.5)
ax3.set_yticks(range(len(sorted_names_ece)))
ax3.set_yticklabels(sorted_names_ece)
ax3.set_xlabel('ECE (lower is better)', fontsize=11)
ax3.set_title('Calibration: ECE Comparison\n(All Probes vs Baselines)', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, sorted_eces)):
    ax3.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontweight='bold', fontsize=9)

# 4. Improvement over Random (AUROC)
ax4 = plt.subplot(2, 3, 4)
improvements_random = [(results[p]['auroc'] - random_auroc) / random_auroc * 100 for p in probe_names]
sorted_idx = np.argsort(improvements_random)
sorted_probes_rand = [probe_names[i] for i in sorted_idx]
sorted_improvements_rand = [improvements_random[i] for i in sorted_idx]

bars = ax4.barh(range(len(sorted_probes_rand)), sorted_improvements_rand, color='green', edgecolor='black', linewidth=1.5)
ax4.set_yticks(range(len(sorted_probes_rand)))
ax4.set_yticklabels(sorted_probes_rand)
ax4.set_xlabel('AUROC Improvement (%)', fontsize=11)
ax4.set_title('Improvement over Random Baseline\n(AUROC)', fontsize=12, fontweight='bold')
ax4.axvline(0, color='red', linestyle='--', alpha=0.5)
ax4.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, sorted_improvements_rand)):
    ax4.text(val + 2, bar.get_y() + bar.get_height()/2,
             f'+{val:.1f}%', va='center', fontweight='bold', fontsize=9)

# 5. Improvement over Seq Length (AUROC)
ax5 = plt.subplot(2, 3, 5)
improvements_seq = [(results[p]['auroc'] - length_auroc) / length_auroc * 100 for p in probe_names]
sorted_idx = np.argsort(improvements_seq)
sorted_probes_seq = [probe_names[i] for i in sorted_idx]
sorted_improvements_seq = [improvements_seq[i] for i in sorted_idx]

bars = ax5.barh(range(len(sorted_probes_seq)), sorted_improvements_seq, color='blue', edgecolor='black', linewidth=1.5)
ax5.set_yticks(range(len(sorted_probes_seq)))
ax5.set_yticklabels(sorted_probes_seq)
ax5.set_xlabel('AUROC Improvement (%)', fontsize=11)
ax5.set_title('Improvement over Seq Length Baseline\n(AUROC)', fontsize=12, fontweight='bold')
ax5.axvline(0, color='red', linestyle='--', alpha=0.5)
ax5.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, sorted_improvements_seq)):
    ax5.text(val + 1, bar.get_y() + bar.get_height()/2,
             f'+{val:.1f}%', va='center', fontweight='bold', fontsize=9)

# 6. Scatter: AUROC vs Brier (all methods)
ax6 = plt.subplot(2, 3, 6)
probe_x = [results[p]['auroc'] for p in probe_names]
probe_y = [results[p]['brier'] for p in probe_names]
baseline_x = [baseline_results[k]['auroc'] for k in baseline_names if not np.isnan(baseline_results[k]['auroc'])]
baseline_y = [baseline_results[k]['brier'] for k in baseline_names if not np.isnan(baseline_results[k]['auroc'])]

ax6.scatter(probe_x, probe_y, s=150, c='green', edgecolors='black', linewidth=2, 
            label='Probes', alpha=0.7, zorder=3)
ax6.scatter(baseline_x, baseline_y, s=150, c='lightcoral', edgecolors='black', linewidth=2,
            label='Baselines', alpha=0.7, zorder=2, marker='s')

# Add labels for probes
for i, name in enumerate(probe_names):
    ax6.annotate(name, (probe_x[i], probe_y[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

ax6.set_xlabel('AUROC (higher is better)', fontsize=11)
ax6.set_ylabel('Brier Score (lower is better)', fontsize=11)
ax6.set_title('Performance Trade-off\n(AUROC vs Brier)', fontsize=12, fontweight='bold')
ax6.grid(alpha=0.3)
ax6.legend()

plt.tight_layout()
plt.savefig('baseline_comparison_all_probes.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Saved: baseline_comparison_all_probes.png")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

# Best probe
best_probe_auroc = max(probe_aurocs)
best_probe_name = probe_names[probe_aurocs.index(best_probe_auroc)]
print(f"\nBest Probe: {best_probe_name}")
print(f"  AUROC: {best_probe_auroc:.4f}")
print(f"  Brier: {results[best_probe_name]['brier']:.4f}")
print(f"  ECE:   {results[best_probe_name]['ece']:.4f}")

# Average improvement over baselines
avg_improvement_random = np.mean(improvements_random)
avg_improvement_seq = np.mean(improvements_seq)

print(f"\nAverage Probe Improvement:")
print(f"  vs Random:     {avg_improvement_random:+.1f}% AUROC")
print(f"  vs Seq Length: {avg_improvement_seq:+.1f}% AUROC")

# How many probes beat each baseline
probes_beating_random = sum(1 for p in probe_names if results[p]['auroc'] > random_auroc)
probes_beating_seq = sum(1 for p in probe_names if results[p]['auroc'] > length_auroc)

print(f"\nProbes Outperforming Baselines:")
print(f"  vs Random:     {probes_beating_random}/{len(probe_names)} probes")
print(f"  vs Seq Length: {probes_beating_seq}/{len(probe_names)} probes")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if avg_improvement_random > 20:
    print("✓ All probes significantly outperform random baseline")
    print("  → Hidden states contain learnable uncertainty signals")
else:
    print("⚠ Probes show limited improvement over random")
    print("  → Uncertainty signals may be weak")

if avg_improvement_seq > 10:
    print("\n✓ Probes outperform sequence length heuristic")
    print("  → Probes capture more than just question difficulty")
else:
    print("\n⚠ Sequence length is a strong baseline")
    print("  → Question length may be a confounding factor")

print("\n" + "="*70)


