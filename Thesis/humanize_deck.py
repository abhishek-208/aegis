import sys, io

with open(r'd:\IITD\MTP 2\FL_Project\Thesis\build_deck.py', 'r', encoding='utf-8') as f:
    code = f.read()

# ==============================================================================
# 1. TEXT HUMANIZATION REPLACEMENTS (Crisp and natural)
# ==============================================================================
replacements = [
    # Slide 2: Agenda
    ('"FL on heterogeneous edge networks; the Byzantine threat under pathological Non-IID data"',
     '"Federated Learning on diverse edge networks; the Byzantine threat under highly skewed data"'),
    ('"Omniscient adaptive adversary; sign-flip, label-flip, IPM, ALIE, Sybil, volume spam"',
     '"Our omniscient, adaptive attacker using six distinct attack strategies"'),
    ('"Evolution from the interim design; the six mathematical stages; O(kd) complexity"',
     '"How Aegis evolved, its six mathematical stages, and achieving O(kd) efficiency"'),
    ('"CIFAR-10, 4-shard Non-IID, ImprovedCNN, five robust-aggregator baselines"',
     '"Testing with CIFAR-10, highly Non-IID data, and five baseline defenses"'),
    ('"Per-attack robustness from the finalised experiments; the IPM/ALIE breaches; sweeps"',
     '"Final experimental results, including successes, the ALIE/IPM breaches, and sweeps"'),
    ('"Achievements, honest boundaries, four research directions, references"',
     '"Summary of achievements, known limitations, and four future research directions"'),

    # Slide 4: Motivation
    ('"Edge devices — drone swarms, hospitals, vehicles — collaboratively train a shared model. "',
     '"Edge devices like drone swarms and hospitals collaborate to train a shared model. "'),
    ('"Only gradient vectors are exchanged; raw data never leaves the node."',
     '"They exchange only gradients, keeping raw data securely on-device."'),
    ('"One gradient per round, cost scales with model size d — not dataset size."',
     '"Costs scale with model size, not dataset size."'),
    ('"4.5 MB / round vs ~21 GB to ship raw data: a 4–5 order-of-magnitude saving."',
     '"Sending just 4.5 MB per round instead of 21 GB saves immense bandwidth."'),
    ('"Server observes only gradients — never raw data, labels, or local loss."',
     '"The server only sees gradient updates, never raw data or labels."'),
    ('"Compatible with encryption & integrity checks, no protocol change."',
     '"Compatible with encryption without altering the core protocol."'),
    ('"Server has zero visibility into any client\'s data or training."',
     '"The server\'s lack of visibility creates a critical blind spot."'),
    ('"A compromised node can submit a poisoned gradient — undetectable by inspection."',
     '"Attackers can submit poisoned gradients that look completely normal."'),

    # Slide 5: Non-IID Crisis
    ('"The Compounding Crisis: Pathological Non-IID Data"',
     '"The Compounding Challenge of Highly Skewed Data"'),
    ('"Real federated nodes are specialists."',
     '"Real-world federated nodes are highly specialized."'),
    ('"4-shard partition: each client sees only ~40% of classes (α = 0.5)."',
     '"In our setup, each client sees only ~40% of the classes."'),
    ('"Honest gradient dissimilarity Γ is 10–50× larger than IID."',
     '"This makes honest updates 10–50× more dissimilar than standard IID data."'),
    ('"Every geometric defence filters by distance from a centroid."',
     '"Most defenses filter out updates far from the center."'),
    ('"Under Non-IID, honest specialist updates ARE centroid-distant."',
     '"But highly specialized honest updates naturally sit far from the center."'),
    ('"Filters discard the signal and admit stealthy near-centroid poison."',
     '"Filters discard this honest signal while admitting stealthy, centered poison."'),
    ('"Accuracy lost to rejecting honest clients, even with zero attackers."',
     '"The accuracy lost just by running the defense under normal conditions."'),
    ('"Multi-Krum tax ≈ 0.3–0.5; Bulyan loses 8.5 pp under no attack."',
     '"For example, Bulyan loses 8.5 percentage points with zero attackers."'),
    ('R("Honest specialist", 13.5, WHITE, True, F_BODY), R("  ≈ geometric outlier", 13, RGBColor(0xC6,0xD2,0xDE), False, F_BODY)',
     'R("Honest specialists", 13.5, WHITE, True, F_BODY), R("  look like outliers", 13, RGBColor(0xC6,0xD2,0xDE), False, F_BODY)'),
    ('R("Stealthy Byzantine", 13.5, WHITE, True, F_BODY), R("  ≈ near the centroid", 13, RGBColor(0xC6,0xD2,0xDE), False, F_BODY)',
     'R("Stealthy attackers", 13.5, WHITE, True, F_BODY), R("  hide near the center", 13, RGBColor(0xC6,0xD2,0xDE), False, F_BODY)'),
    ('R("Centroid filters reject the signal", 13, WHITE, False, F_BODY)',
     'R("Standard filters reject good data", 13, WHITE, False, F_BODY)'),
    ('R("and admit the poison.", 13, WHITE, False, F_BODY)',
     'R("and accept the poison.", 13, WHITE, False, F_BODY)'),

    # Slide 6: Research Gap
    ('"What Is Missing — and the Question"',
     '"What Is Missing — and Our Research Question"'),
    ('R("Research question:  ", 16, ACCENT2, True, F_HEAD)',
     'R("Our Research Question:  ", 16, ACCENT2, True, F_HEAD)'),
    ('"Can one linear-time aggregator protect a highly heterogeneous federated "',
     '"Can a linear-time aggregator protect a highly diverse federated "'),
    ('"network against coordinated Byzantine attacks"',
     '"network against coordinated attacks "'),
    ('"without a crippling accuracy penalty on honest specialist clients?"',
     '"without punishing honest specialists?"'),

    # Slide 8: Threat Model
    ('"Know the aggregation rule and global model."',
     '"Perfectly understand the aggregation rule and global model."'),
    ('"Estimate the honest mean gradient (omniscient)."',
     '"Estimate the honest network\'s mean gradient."'),
    ('"Fully coordinate all Byzantine clients."',
     '"Fully coordinate all compromised clients."'),
    ('"Adapt every round; attack probability = 1.0."',
     '"Persistently attack in every single round."'),
    ('"Intercept honest updates before aggregation"',
     '"Intercept honest updates before they reach the server"'),
    ('"Corrupt the server or the model broadcast"',
     '"Corrupt the central server itself"'),
    ('"Break the honest majority (f < K/2)"',
     '"Take over the majority of the network (f < K/2)"'),

    # Slide 13: Evolution
    ('"Hard cosine gate was a trap"',
     '"A hard cosine gate is a trap"'),
    ('"cos < 0 either rubber-stamps orthogonal attacks (cos ≈ 0) or amputates honest Non-IID specialists (disjoint data → naturally orthogonal). → continuous soft penalty α·Pₖ."',
     '"It allows orthogonal attacks while amputating honest specialists. Our fix: a continuous soft penalty α·Pₖ."'),
    ('"Per-round scoring is blind to stealth"',
     '"Round-by-round scoring misses stealth"'),
    ('"ALIE and low-ε IPM hide inside the round\'s envelope. → cross-round EMA reputation λRₖ accumulates a temporal signal."',
     '"Attacks like ALIE hide within normal variance. Our fix: cross-round EMA reputation λRₖ catches them over time."'),
    ('"Merging the two risked a death spiral"',
     '"Coupling them caused a death spiral"'),
    ('"Reputation cleans the pool → variance drops → threshold tightens → honest clients cut. → decouple: compute K on the full unfiltered set."',
     '"Cleaning the pool tightens thresholds, hurting honest clients. Our fix: decouple threshold calculations from the filtered pool."'),

    # Slide 23: Setup
    ('"Oracle ceiling (f=0, 3 seeds):  75.53 ± 0.22%  —  the hard limit set by the "',
     '"Our oracle ceiling is 75.53 ± 0.22% — the hard accuracy limit set by this "'),
    ('"Non-IID partition itself."',
     '"extreme Non-IID partition."'),

    # Slide 25: Byzantine Tax
    ('"An effectively free defence in the benign case."',
     '"Aegis is effectively free when there are no attacks."'),
    ('"Bulyan loses 8.47 pp with zero attackers — Krum selection collapses to a single Non-IID sector."',
     '"Bulyan loses 8.47 pp because Krum selection collapses to a single sector."'),
    ('"Validates τ = −0.3 + adaptive k_floor: honest specialists preserved."',
     '"This proves our adaptive floor successfully preserves honest specialists."')
]

for old, new in replacements:
    code = code.replace(old, new)

# ==============================================================================
# 2. INTEGRATE EXPR 33 DATA DIRECTLY INTO build_deck_humanized.py
# ==============================================================================

# Add Vol Spam and Sybil to RES dict
res_patch = """
 "alie_cmp":       RESDIR + r"\\Expr 33 - @Aegis, Fedavg, CWmed, Krum, Foolsgold, Bulyan, ALIE\\fed_avg+aegis+cw_med+multi_krum+fools_gold+bulyan_alie_manual_accuracy_line.png",
 "vol_cmp":        RESDIR + r"\\Expr 33 - @Aegis, Fedavg, CWmed, Krum, Foolsgold, Bulyan, Volume Spam\\fed_avg+aegis+cw_med+multi_krum+fools_gold+bulyan_volume_spam_manual_accuracy_line.png",
 "syb_cmp":        RESDIR + r"\\Expr 33 - @Aegis, Fedavg, CWmed, Krum, Foolsgold, Bulyan, Sybil\\fed_avg+aegis+cw_med+multi_krum+fools_gold+bulyan_sybil_manual_accuracy_line.png",
"""
code = code.replace(' "alie_cmp":       RESDIR + r"\\Expr 33 - @Aegis, Fedavg, CWmed, Krum, Foolsgold, Bulyan, ALIE\\fed_avg+aegis+cw_med+multi_krum+fools_gold+bulyan_alie_manual_accuracy_line.png",', res_patch)

# Insert the two new slides right after ALIE
new_slides_patch = """
img_slide("Results · ALIE", "A Little Is Enough — The Breach", "alie_cmp",
          [R("Aegis collapses to 10% (chance) — ALIE hijacks the median; ", 14, TEXT, False, F_BODY),
           R("FoolsGold's history tracking is the only survivor (71%)", 14, CRIMSON, True, F_BODY),
           R(".", 14, TEXT, False, F_BODY)], tsize=27)

img_slide("Results · Volume Spam", "Volume Spam — All Aggregators (f = 0.30)", "vol_cmp",
          [R("Aegis #1 at ", 13.5, TEXT, False, F_BODY),
           R("73.77%", 13.5, GREEN, True, F_BODY),
           R(" — median volume clipping neutralises the inflation attack; ", 13.5, TEXT, False, F_BODY),
           R("FedAvg collapses to 14%", 13.5, CRIMSON, True, F_BODY),
           R(".", 13.5, TEXT, False, F_BODY)], tsize=26)

img_slide("Results · Sybil", "Sybil — All Aggregators (f = 0.30, k = 2)", "syb_cmp",
          [R("FoolsGold #1 (68.76%) via full-history tracking; Aegis #2 at ", 13.5, TEXT, False, F_BODY),
           R("63.17%", 13.5, GREEN, True, F_BODY),
           R(". ", 13.5, TEXT, False, F_BODY),
           R("Geometric baselines destroyed", 13.5, CRIMSON, True, F_BODY),
           R(" (Krum 14%, CWMed 22%).", 13.5, TEXT, False, F_BODY)], tsize=26)
"""

# Replace the ALIE slide with ALIE + Vol Spam + Sybil
alie_old = """img_slide("Results · ALIE", "A Little Is Enough — The Breach", "alie_cmp",
          [R("Aegis collapses to 10% (chance) — ALIE hijacks the median; ", 14, TEXT, False, F_BODY),
           R("FoolsGold's history tracking is the only survivor (71%)", 14, CRIMSON, True, F_BODY),
           R(".", 14, TEXT, False, F_BODY)], tsize=27)"""

code = code.replace(alie_old, new_slides_patch.strip())

# Relocate the Head-to-Head table slide (which was right after Baseline)
# First, remove it from its old location.
head_to_head_code = """# --- 28. HEAD-TO-HEAD TABLE ------------------------------------------------
s = slide(); header(s, "Results", "Head-to-Head Under Active Attack  (f = 0.30)")
data = [
    ["Aggregator", "No Atk", "Label Flip", "Sign Flip", "IPM", "ALIE"],
    ["FedAvg", "76.16", "—", "40.04", "70.25", "66.78"],
    ["CWMed", "72.31", "62.07", "49.64", "10.43", "23.21"],
    ["Multi-Krum", "71.30", "72.85", "70.05", "17.30", "34.18"],
    ["FoolsGold", "73.80", "67.06", "34.20", "71.07", "71.15"],
    ["Bulyan †", "67.69", "68.51", "62.58", "10.00", "20.39"],
    ["Aegis", "76.08", "74.34", "66.61", "63.32", "10.00"],
]
styles = {}
for r in range(1, 7):
    for c in range(1, 6):
        styles[(r, c)] = {"color": TEXT, "bold": False}
styles[(6,1)]["fill"] = GRNTNT; styles[(6,1)]["bold"] = True
styles[(6,2)]["fill"] = GRNTNT; styles[(6,2)]["bold"] = True
styles[(3,3)]["fill"] = GRNTNT; styles[(3,3)]["bold"] = True
styles[(4,4)]["fill"] = GRNTNT; styles[(4,4)]["bold"] = True
styles[(6,5)]["fill"] = CRIMTNT; styles[(6,5)]["bold"] = True; styles[(6,5)]["color"] = CRIMSON
table(s, 0.8, 1.7, 11.75, 7, [2.35, 1.88, 1.88, 1.88, 1.88, 1.88], data, fsize=12.5, hsize=12.5, row_h=0.52, cell_styles=styles)
bullets(s, 0.8, 5.5, 11.8, 1.5, [
    (0, "Aegis beats every geometric baseline (CWMed, Krum, Bulyan) on every attack; best overall on label-flip (+1.49 pp).", NAVY, True),
    (0, "Three regimes: dominant (label-flip), competitive (sign-flip, IPM, Sybil), failed (ALIE). Per-attack figures follow.", TEXT, False),
    (0, "† Bulyan evaluated at f = 0.20 (its n ≥ 4f+3 constraint is violated at f = 0.30).", MUTED, False),
], size=12.5, gap=6)
footer(s, pg())

"""
code = code.replace(head_to_head_code, "")

# The NEW Head-to-Head code (with 8 columns and humanized bullets)
new_head_to_head_code = """
# --- HEAD-TO-HEAD TABLE (Now moved before Summary) --------------------------
s = slide(); header(s, "Results", "Head-to-Head Under Active Attack  (f = 0.30)")
data = [
    ["Aggregator", "No Atk", "Label Flip", "Sign Flip", "IPM", "Vol Spam", "Sybil", "ALIE"],
    ["FedAvg", "76.16", "—", "40.04", "70.25", "14.44", "62.48", "66.78"],
    ["CWMed", "72.31", "62.07", "49.64", "10.43", "63.82", "22.05", "23.21"],
    ["Multi-Krum", "71.30", "72.85", "70.05", "17.30", "72.45", "14.43", "34.18"],
    ["FoolsGold", "73.80", "67.06", "34.20", "71.07", "66.96", "68.76", "71.15"],
    ["Bulyan †", "67.69", "68.51", "62.58", "10.00", "69.21", "46.45", "20.39"],
    ["Aegis (ours)", "76.08", "74.34", "66.61", "63.32", "73.77", "63.17", "10.00"],
]
styles = {}
for r in range(1, 7):
    for c in range(1, 8):
        styles[(r, c)] = {"color": TEXT, "bold": False}
# Highlight best value per attack column
styles[(6,1)]["fill"] = GRNTNT; styles[(6,1)]["bold"] = True
styles[(6,2)]["fill"] = GRNTNT; styles[(6,2)]["bold"] = True
styles[(3,3)]["fill"] = GRNTNT; styles[(3,3)]["bold"] = True
styles[(4,4)]["fill"] = GRNTNT; styles[(4,4)]["bold"] = True
styles[(6,5)]["fill"] = GRNTNT; styles[(6,5)]["bold"] = True
styles[(4,6)]["fill"] = GRNTNT; styles[(4,6)]["bold"] = True
styles[(4,7)]["fill"] = GRNTNT; styles[(4,7)]["bold"] = True
# Failures
styles[(6,7)]["fill"] = CRIMTNT; styles[(6,7)]["bold"] = True; styles[(6,7)]["color"] = CRIMSON # Aegis ALIE
styles[(1,5)]["fill"] = CRIMTNT; styles[(1,5)]["bold"] = True; styles[(1,5)]["color"] = CRIMSON # FedAvg Vol Spam

table(s, 0.55, 1.70, 12.2, 7, [1.60, 1.10, 1.30, 1.30, 1.05, 1.30, 1.05, 1.05], data, fsize=10.5, hsize=10.5, row_h=0.40, cell_styles=styles)
bullets(s, 0.8, 5.60, 11.8, 1.5, [
    (0, "Aegis is #1 on Label-Flip (+1.49 pp) and Vol Spam (+1.32 pp over Krum). Geometric baselines collapse on Sybil (Krum 14%, CWMed 22%).", NAVY, True),
    (0, "Three regimes: dominant (label-flip, vol-spam), competitive (sign-flip, IPM, Sybil), failed (ALIE).", TEXT, False),
    (0, "† Bulyan evaluated at f = 0.20 (its n ≥ 4f+3 constraint is violated at f = 0.30).", MUTED, False),
], size=12.5, gap=6)
footer(s, pg())

"""

# Insert new head-to-head immediately before SUMMARY ACROSS ALL ATTACKS
summary_start = """# --- 36. SUMMARY ACROSS ALL ATTACKS ----------------------------------------
s = slide(); header(s, "Results", "Summary Across All Attacks")"""
code = code.replace(summary_start, new_head_to_head_code + summary_start)

# Update the Summary Table itself
old_summary = """data = [
    ["Attack", "Aegis Acc", "DR %", "Best Baseline", "Gap (pp)", "Regime"],
    ["No Attack", "76.08", "N/A", "FedAvg  76.16", "−0.08", "I"],
    ["Label Flip", "74.34", "76.5", "Krum  72.85", "+1.49", "I"],
    ["Sign Flip", "66.61", "60.8", "Krum  70.05", "−3.44", "II"],
    ["IPM", "63.32", "0.0", "FoolsGold  71.07", "−7.75", "II"],
    ["Sybil", "64.22", "0.2", "FoolsGold  68.49", "−4.27", "II/III"],
    ["ALIE", "10.00", "0.0", "FoolsGold  71.15", "−61.15", "III"],
]
styles = {
    (2, 4): {"fill": GRNTNT, "bold": True, "color": GREEN},
    (6, 1): {"fill": CRIMTNT, "bold": True, "color": CRIMSON},
    (6, 4): {"fill": CRIMTNT, "bold": True, "color": CRIMSON},
}
table(s, 0.8, 1.7, 11.75, 7, [2.05, 1.6, 1.3, 3.8, 1.5, 1.5], data, fsize=12.5, hsize=12.5, row_h=0.48, cell_styles=styles)
bullets(s, 0.8, 5.2, 11.8, 1.5, [
    (0, "Aegis is strictly superior to all geometric baselines (CWMed, Krum, Bulyan) on Label-Flip, IPM, and Sybil, but lags Multi-Krum on Sign-Flip.", NAVY, True),
    (0, "FoolsGold wins ALIE, IPM & Sybil — its full-history tracking outresolves the 20-round EMA. Aegis’s limitation on median-poisoning attacks is a known tradeoff.", TEXT, False),
], size=13, gap=6)"""

new_summary = """data = [
    ["Attack", "Aegis Acc", "DR %", "Best Baseline", "Gap (pp)", "Regime"],
    ["No Attack", "76.08", "N/A", "FedAvg  76.16", "−0.08", "I"],
    ["Label Flip", "74.34", "76.5", "Krum  72.85", "+1.49", "I"],
    ["Vol Spam", "73.77", "56.7", "Krum  72.45", "+1.32", "I"],
    ["Sign Flip", "66.61", "60.8", "Krum  70.05", "−3.44", "II"],
    ["IPM", "63.32", "0.0", "FoolsGold  71.07", "−7.75", "II"],
    ["Sybil", "63.17", "0.2", "FoolsGold  68.76", "−5.59", "II/III"],
    ["ALIE", "10.00", "0.0", "FoolsGold  71.15", "−61.15", "III"],
]
styles = {
    (2, 4): {"fill": GRNTNT, "bold": True, "color": GREEN},
    (3, 4): {"fill": GRNTNT, "bold": True, "color": GREEN},
    (7, 1): {"fill": CRIMTNT, "bold": True, "color": CRIMSON},
    (7, 4): {"fill": CRIMTNT, "bold": True, "color": CRIMSON},
}
table(s, 0.55, 1.70, 7.95, 8, [1.35, 1.15, 0.85, 2.50, 1.10, 1.00], data, fsize=10.5, hsize=10.5, row_h=0.38, cell_styles=styles)
bullets(s, 0.8, 5.10, 11.8, 1.8, [
    (0, "Aegis is strictly superior to geometric baselines on Label-Flip, Vol Spam, IPM, and Sybil, but lags Multi-Krum on Sign-Flip.", NAVY, True),
    (0, "FoolsGold wins ALIE, IPM & Sybil — its full-history tracking outresolves the 20-round EMA. Aegis’s limitation on median-poisoning attacks is a known tradeoff.", TEXT, False),
], size=12.5, gap=6)"""

code = code.replace(old_summary, new_summary)

# Update OUT filename so we don't accidentally wipe out the old deck if something goes wrong,
# but we actually want to overwrite Aegis_Defense.pptx so the user can easily view it.
code = code.replace('OUT = os.path.join(HERE, "Aegis_Defense.pptx")', 'OUT = os.path.join(HERE, "Aegis_Defense.pptx")')

with open(r'd:\IITD\MTP 2\FL_Project\Thesis\build_deck_humanized.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Created build_deck_humanized.py successfully.")
