import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
ARCHIVE_DIR = DATA_DIR / "archived"

# Filename forms:
#   session_<character>_<YYYYMMDD>_<HHMMSS>.csv  (current)
#   session_<YYYYMMDD>_<HHMMSS>.csv              (legacy → grouped under "unlabeled")
_LABELED_RE = re.compile(r'^session_(.+?)_(\d{8}_\d{6})\.csv$')


def _parse_character(path):
    m = _LABELED_RE.match(path.name)
    if m and any(c.isalpha() for c in m.group(1)):
        return m.group(1).lower()
    return None


# ── Session loading ────────────────────────────────────────────────────────────
# Default: combine ALL sessions, grouped by character.
# Flags:
#   --latest N      use only the N most recent sessions (across all characters)
#   --archive       after analysis, move analyzed sessions to data/archived/
#   --archive-old   after analysis, move sessions NOT analyzed to data/archived/
# Examples:
#   python analyzer.py --latest 3 --archive-old
#   python analyzer.py --archive

all_files = sorted(
    [f for f in DATA_DIR.iterdir() if f.name.startswith('session_') and f.suffix == '.csv'],
    key=lambda f: f.name,
)
if not all_files:
    print("No sessions found in data/")
    exit()

n_sessions = len(all_files)
if '--latest' in sys.argv:
    try:
        n_sessions = int(sys.argv[sys.argv.index('--latest') + 1])
    except (IndexError, ValueError):
        print("Usage: python analyzer.py --latest N")
        exit(1)

selected = all_files[-n_sessions:]

character_groups = {}
for f in selected:
    key = _parse_character(f) or 'unlabeled'
    character_groups.setdefault(key, []).append(f)

print(f"Loaded {len(selected)} session(s) across {len(character_groups)} character group(s):")
for char, files in character_groups.items():
    print(f"  {char}: {len(files)} session(s)")
print()


# ── Curve model ───────────────────────────────────────────────────────────────

def _dual_zone_curve(x, bp, static_ratio):
    """
    Dual-Zone S-Curve matching Marvel Rivals' model.
      Inner zone [0, bp]  : linear ramp 0 → static_ratio  (precision)
      Outer zone [bp, 1]  : power ramp  static_ratio → 1.0 (speed)
    All values in normalized [0, 1] space.
    """
    inner = (x / bp) * static_ratio
    outer = static_ratio + (1.0 - static_ratio) * ((x - bp) / (1.0 - bp)) ** 1.5
    return np.where(x < bp, inner, outer)


def _find_zone_breakpoint_estimate(x_norm, dz_norm):
    """
    Find the valley between the inner precision cluster and outer speed cluster.
    Falls back to 0.55 when the distribution isn't clearly bimodal.
    """
    region = x_norm[(x_norm > dz_norm + 0.05) & (x_norm < 0.92)]
    if len(region) < 30:
        return 0.55
    counts, edges = np.histogram(region, bins=25)
    centers = (edges[:-1] + edges[1:]) / 2
    smoothed = uniform_filter1d(counts.astype(float), size=3)
    mid = (centers > 0.15) & (centers < 0.85)
    if mid.sum() < 3:
        return 0.55
    return float(centers[mid][np.argmin(smoothed[mid])])


def _extract_hit_events(df):
    """
    Cluster consecutive hit=1 rows (each marker spans ~50 ms) into single events
    and return their mean timestamps.
    """
    if 'hit' not in df.columns or df['hit'].sum() == 0:
        return np.array([])
    hit_ts = df.loc[df['hit'] == 1, 'timestamp'].values
    clusters, cluster = [], [hit_ts[0]]
    for t in hit_ts[1:]:
        if t - cluster[-1] > 0.15:
            clusters.append(np.mean(cluster))
            cluster = []
        cluster.append(t)
    clusters.append(np.mean(cluster))
    return np.array(clusters)


# ── Per-axis analysis ─────────────────────────────────────────────────────────

def analyze_axis(col, values, timestamps, session_groups, hit_events, output_dir, character):
    abs_values = np.abs(values)

    # Deadzone — noise floor when stick rests near zero
    idle_mask = abs_values < 0.08
    if idle_mask.sum() >= 20:
        deadzone = round(float(np.percentile(abs_values[idle_mask], 99)) * 1.2, 4)
    else:
        deadzone = 0.05

    active = abs_values[abs_values > deadzone]
    if len(active) < 30:
        print(f"  {col}: not enough active samples (deadzone={deadzone})")
        return

    # Max used range — outer clamp
    max_range = float(np.percentile(active, 98))
    if max_range < 0.01:
        print(f"  {col}: stick barely moved, skipping")
        return

    x_norm = np.clip(active / max_range, 0, 1)
    dz_norm = deadzone / max_range

    # Fit dual-zone curve to empirical CDF
    sorted_x = np.sort(x_norm)
    empirical_cdf = np.arange(1, len(sorted_x) + 1) / len(sorted_x)

    bp_est = _find_zone_breakpoint_estimate(x_norm, dz_norm)
    inner_frac = float(np.mean(x_norm < bp_est))
    static_est = max(0.15, min(0.55, inner_frac * 0.6))

    try:
        popt, _ = curve_fit(
            _dual_zone_curve,
            sorted_x,
            empirical_cdf,
            p0=[bp_est, static_est],
            bounds=([dz_norm + 0.05, 0.10], [0.90, 0.65]),
        )
        bp_norm, static_ratio = float(popt[0]), float(popt[1])
    except RuntimeError:
        bp_norm, static_ratio = bp_est, static_est

    bp_norm = round(bp_norm, 3)
    static_ratio = round(static_ratio, 3)

    # Marvel Rivals: Advanced Aim Sensitivity Curve Settings
    mr_custom_min_range     = max(1,  round(deadzone * 100))
    mr_custom_max_range     = min(99, round(max_range * 100))
    mr_min_curve_statics    = round(static_ratio * 100)
    mr_custom_max_dualzone  = round(bp_norm * 100)

    p25, p50, p75, p95 = (round(float(np.percentile(active, p)), 3) for p in (25, 50, 75, 95))
    inner_usage = round(float(np.mean(x_norm < bp_norm)) * 100, 1)

    # Hit-moment stick positions — 50–300 ms before each hit event
    effective_abs = np.array([])
    if len(hit_events) > 0:
        ev_vals = []
        for ht in hit_events:
            mask = (timestamps >= ht - 0.30) & (timestamps < ht - 0.05)
            ev_vals.extend(np.abs(values[mask]))
        if ev_vals:
            effective_abs = np.array(ev_vals)

    print(f"  {col}  ({len(values):,} samples across {len(session_groups)} session(s)):")
    print(f"    ── Advanced Aim Sensitivity Curve Settings (Dual-Zone S-Curve) ──")
    print(f"    Custom Minimum Range          : {mr_custom_min_range}")
    print(f"    Custom Maximum Range          : {mr_custom_max_range}")
    print(f"    Minimum Curve Statics         : {mr_min_curve_statics}")
    print(f"    Custom Maximum Dual-zone Curve: {mr_custom_max_dualzone}")
    print(f"    ── Data summary ──")
    print(f"    Inner zone usage   : {inner_usage}% of active samples")
    print(f"    Active percentiles : p25={p25}  p50={p50}  p75={p75}  p95={p95}")
    if len(effective_abs) >= 5:
        ep25, ep50, ep75 = (round(float(np.percentile(effective_abs, p)), 3) for p in (25, 50, 75))
        eff_max = round(float(np.percentile(effective_abs, 95)), 3)
        print(f"    ── Hit-moment aim analysis ({len(hit_events)} events, {len(effective_abs)} samples) ──")
        print(f"    Effective percentiles : p25={ep25}  p50={ep50}  p75={ep75}")
        if eff_max < max_range * 0.85:
            suggested_max = round(eff_max * 100)
            print(f"    Note: effective max ({eff_max}) is well below overall max ({max_range:.3f})")
            print(f"          Consider lowering Custom Maximum Range to ~{suggested_max}")
    print()

    # Time-series — one continuous line per session with gap + boundary markers
    plt.figure(figsize=(12, 4))
    offset = 0.0
    for i, (_, grp) in enumerate(session_groups):
        t = grp['timestamp'].values - grp['timestamp'].values[0] + offset
        plt.plot(t, grp[col].values, linewidth=0.5, color='steelblue')
        offset = t[-1] + 3.0
        if i < len(session_groups) - 1:
            plt.axvline(offset - 1.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.6,
                        label='Session boundary' if i == 0 else None)
    plt.title(f'[{character}] {col} – Input over time ({len(session_groups)} session(s))')
    plt.xlabel('Time (s, sessions separated by gaps)')
    plt.ylabel('Axis value')
    if len(session_groups) > 1:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f'{col}_plot.png')
    plt.close()

    # Histogram with deadzone and zone boundary markers
    bp_abs = bp_norm * max_range
    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=60, color='steelblue', alpha=0.7, label='All inputs')
    if len(effective_abs) >= 5:
        plt.hist(np.concatenate([effective_abs, -effective_abs]), bins=40,
                 color='limegreen', alpha=0.55, label=f'Hit moments (n={len(hit_events)})')
    plt.axvline( deadzone, color='red',    linestyle='--', label=f'Custom Minimum Range (±{mr_custom_min_range})')
    plt.axvline(-deadzone, color='red',    linestyle='--')
    plt.axvline( bp_abs,   color='orange', linestyle='--', label=f'Custom Max Dual-zone Curve (±{mr_custom_max_dualzone})')
    plt.axvline(-bp_abs,   color='orange', linestyle='--')
    plt.title(f'[{character}] {col} – Input distribution ({len(session_groups)} session(s))')
    plt.xlabel('Axis value')
    plt.ylabel('Sample count')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f'{col}_hist.png')
    plt.close()

    # Dual-zone response curve
    x_plot = np.linspace(0, 1, 300)
    plt.figure(figsize=(7, 6))
    plt.plot(x_plot, x_plot, '--', color='gray', label='Linear reference')
    plt.plot(x_plot, _dual_zone_curve(x_plot, bp_norm, static_ratio), color='royalblue', linewidth=2,
             label=f'Dual-Zone S-Curve\nMin Curve Statics={mr_min_curve_statics}  Max Dual-zone={mr_custom_max_dualzone}')
    plt.axvline(dz_norm,  color='red',    linestyle=':', linewidth=1.2, label=f'Custom Minimum Range ({mr_custom_min_range})')
    plt.axvline(bp_norm,  color='orange', linestyle=':', linewidth=1.2, label=f'Custom Max Dual-zone Curve ({mr_custom_max_dualzone})')
    plt.axhline(static_ratio, color='orange', linestyle=':', linewidth=0.8, alpha=0.5)
    if len(effective_abs) >= 5:
        eff_p25_norm = float(np.percentile(effective_abs, 25)) / max_range
        eff_p75_norm = float(np.percentile(effective_abs, 75)) / max_range
        plt.axvspan(eff_p25_norm, min(eff_p75_norm, 1.0), alpha=0.12, color='limegreen',
                    label='Effective aim range (hit p25–p75)')
    plt.xlabel('Normalized physical input')
    plt.ylabel('Suggested output')
    plt.title(f'[{character}] {col} – Dual-Zone S-Curve (Marvel Rivals)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{col}_curve.png')
    plt.close()


# ── Per-character analysis ────────────────────────────────────────────────────

def analyze_character(character, files):
    print("=" * 60)
    print(f"  Character: {character}")
    print("=" * 60)

    frames = []
    for f in files:
        s = pd.read_csv(f)
        s['_session'] = f.name
        frames.append(s)
        duration = s['timestamp'].iloc[-1] - s['timestamp'].iloc[0]
        print(f"  {f.name}  —  {len(s):,} samples  ({duration:.0f}s)")

    df = pd.concat(frames, ignore_index=True)
    total_duration = sum(
        g['timestamp'].iloc[-1] - g['timestamp'].iloc[0]
        for _, g in df.groupby('_session')
    )
    print(f"  Combined: {len(df):,} samples across {len(files)} session(s) ({total_duration:.0f}s active)")

    hit_events = _extract_hit_events(df)
    if len(hit_events) > 0:
        print(f"  Hit events: {len(hit_events)}")
    print()

    output_dir = DATA_DIR / character
    output_dir.mkdir(exist_ok=True)

    session_groups = list(df.groupby('_session', sort=True))
    timestamps = df['timestamp'].values
    axis_cols = [c for c in df.columns if c.startswith('axis')]

    for col in axis_cols:
        analyze_axis(col, df[col].values, timestamps,
                     session_groups, hit_events, output_dir, character)


# ── Run ───────────────────────────────────────────────────────────────────────

for character, files in character_groups.items():
    analyze_character(character, files)

print("Analysis complete. Plots saved per character under data/<character>/")


# ── Archive ───────────────────────────────────────────────────────────────────

do_archive     = '--archive'     in sys.argv
do_archive_old = '--archive-old' in sys.argv

if do_archive or do_archive_old:
    to_archive = selected if do_archive else list(set(all_files) - set(selected))

    if not to_archive:
        print("\nNothing to archive.")
    else:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        label = "analyzed" if do_archive else "older"
        print(f"\nArchiving {len(to_archive)} {label} session(s) → data/archived/")
        for f in to_archive:
            dest = ARCHIVE_DIR / f.name
            f.rename(dest)
            print(f"  {f.name}")
