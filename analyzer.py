import json
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
    # np.maximum keeps the outer branch from producing NaN when x < bp
    # (np.where evaluates both branches even though it discards one).
    outer = static_ratio + (1.0 - static_ratio) * (np.maximum(x - bp, 0.0) / (1.0 - bp)) ** 1.5
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


def _load_current_curve(output_dir):
    """
    Read data/<character>/current.json if it exists. Expected schema:
      {
        "custom_minimum_range":           1,
        "custom_maximum_range":           2,
        "minimum_curve_statics":          15,
        "custom_maximum_dual_zone_curve": 73
      }
    Returns the dict on success, None otherwise.
    """
    path = output_dir / "current.json"
    if not path.exists():
        return None
    try:
        with open(path) as fp:
            cur = json.load(fp)
        for key in ("custom_minimum_range", "custom_maximum_range",
                    "minimum_curve_statics", "custom_maximum_dual_zone_curve"):
            if key not in cur:
                print(f"  WARNING: {path} missing '{key}', ignoring current curve")
                return None
        return cur
    except (OSError, json.JSONDecodeError) as e:
        print(f"  WARNING: could not read {path}: {e}")
        return None


def _apply_game_curve(stick_abs, curve):
    """
    Vectorized: apply Marvel Rivals dual-zone curve to absolute stick values.
    Returns camera-motion magnitude in [0, 1] (a fraction of max camera speed).
    """
    dz       = curve["custom_minimum_range"]           / 100.0
    mr       = curve["custom_maximum_range"]           / 100.0
    statics  = curve["minimum_curve_statics"]          / 100.0
    bp       = curve["custom_maximum_dual_zone_curve"] / 100.0

    if mr <= dz:
        return np.zeros_like(stick_abs, dtype=float)

    x_norm = np.clip((stick_abs - dz) / (mr - dz), 0.0, 1.0)
    saturated = stick_abs >= mr

    inner_safe_bp = max(bp, 1e-6)
    outer_safe_bp = min(bp, 1.0 - 1e-6)
    inner = (x_norm / inner_safe_bp) * statics
    outer = statics + (1.0 - statics) * (np.maximum(x_norm - outer_safe_bp, 0.0) / (1.0 - outer_safe_bp)) ** 1.5
    out = np.where(x_norm < bp, inner, outer)

    out = np.where(stick_abs < dz, 0.0, out)
    out = np.where(saturated, 1.0, out)
    return out


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

    # Deadzone -noise floor when stick rests near zero
    idle_mask = abs_values < 0.08
    if idle_mask.sum() >= 20:
        deadzone = round(float(np.percentile(abs_values[idle_mask], 99)) * 1.2, 4)
    else:
        deadzone = 0.05

    active = abs_values[abs_values > deadzone]
    if len(active) < 30:
        print(f"  {col}: not enough active samples (deadzone={deadzone})")
        return None

    # Max used range -outer clamp
    max_range = float(np.percentile(active, 98))
    if max_range < 0.01:
        print(f"  {col}: stick barely moved, skipping")
        return None

    x_norm = np.clip(active / max_range, 0, 1)
    dz_norm = deadzone / max_range

    # Fit dual-zone curve to empirical CDF
    sorted_x = np.sort(x_norm)
    empirical_cdf = np.arange(1, len(sorted_x) + 1) / len(sorted_x)

    bp_est = _find_zone_breakpoint_estimate(x_norm, dz_norm)
    inner_frac = float(np.mean(x_norm < bp_est))
    static_est = max(0.15, min(0.55, inner_frac * 0.6))

    # Bounds for curve_fit. When the stick barely moved, dz_norm is large and
    # the lower bound on bp can swallow / exceed the upper bound -clamp p0
    # into the valid interior, and skip the fit entirely if the interval
    # collapses.
    bp_low,  bp_high     = dz_norm + 0.05, 0.90
    stat_low, stat_high  = 0.10, 0.65
    if bp_low >= bp_high - 1e-3:
        bp_norm, static_ratio = bp_est, static_est
    else:
        bp_p0   = min(max(bp_est,    bp_low + 1e-3),   bp_high - 1e-3)
        stat_p0 = min(max(static_est, stat_low + 1e-3), stat_high - 1e-3)
        try:
            popt, _ = curve_fit(
                _dual_zone_curve,
                sorted_x,
                empirical_cdf,
                p0=[bp_p0, stat_p0],
                bounds=([bp_low, stat_low], [bp_high, stat_high]),
            )
            bp_norm, static_ratio = float(popt[0]), float(popt[1])
        except (RuntimeError, ValueError):
            bp_norm, static_ratio = bp_p0, stat_p0

    bp_norm = round(bp_norm, 3)
    static_ratio = round(static_ratio, 3)

    # Marvel Rivals: Advanced Aim Sensitivity Curve Settings
    mr_custom_min_range     = max(1,  round(deadzone * 100))
    mr_custom_max_range     = min(99, round(max_range * 100))
    mr_min_curve_statics    = round(static_ratio * 100)
    mr_custom_max_dualzone  = round(bp_norm * 100)

    p25, p50, p75, p95 = (round(float(np.percentile(active, p)), 3) for p in (25, 50, 75, 95))
    inner_usage = round(float(np.mean(x_norm < bp_norm)) * 100, 1)

    # Hit-moment stick positions -50–300 ms before each hit event
    effective_abs = np.array([])
    if len(hit_events) > 0:
        ev_vals = []
        for ht in hit_events:
            mask = (timestamps >= ht - 0.30) & (timestamps < ht - 0.05)
            ev_vals.extend(np.abs(values[mask]))
        if ev_vals:
            effective_abs = np.array(ev_vals)

    print(f"  {col}  ({len(values):,} samples across {len(session_groups)} session(s)):")
    print(f"    -- Advanced Aim Sensitivity Curve Settings (Dual-Zone S-Curve) --")
    print(f"    Custom Minimum Range          : {mr_custom_min_range}")
    print(f"    Custom Maximum Range          : {mr_custom_max_range}")
    print(f"    Minimum Curve Statics         : {mr_min_curve_statics}")
    print(f"    Custom Maximum Dual-zone Curve: {mr_custom_max_dualzone}")
    print(f"    -- Data summary --")
    print(f"    Inner zone usage   : {inner_usage}% of active samples")
    print(f"    Active percentiles : p25={p25}  p50={p50}  p75={p75}  p95={p95}")
    if len(effective_abs) >= 5:
        ep25, ep50, ep75 = (round(float(np.percentile(effective_abs, p)), 3) for p in (25, 50, 75))
        eff_max = round(float(np.percentile(effective_abs, 95)), 3)
        print(f"    -- Hit-moment aim analysis ({len(hit_events)} events, {len(effective_abs)} samples) --")
        print(f"    Effective percentiles : p25={ep25}  p50={ep50}  p75={ep75}")
        if eff_max < max_range * 0.85:
            suggested_max = round(eff_max * 100)
            print(f"    Note: effective max ({eff_max}) is well below overall max ({max_range:.3f})")
            print(f"          Consider lowering Custom Maximum Range to ~{suggested_max}")
    print()

    # Time-series -one continuous line per session with gap + boundary markers
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

    # Saturation / pin-time: fraction of active time the stick was at the rail.
    # High pin time on the aim stick suggests sensitivity is too low -the camera
    # can't keep up with how far the user is pushing.
    pin_time_pct = round(float(np.mean(active >= 0.95) * 100), 2)
    hit_p50 = (round(float(np.percentile(effective_abs, 50)), 3)
               if len(effective_abs) >= 5 else None)

    return {
        "custom_minimum_range":           mr_custom_min_range,
        "custom_maximum_range":           mr_custom_max_range,
        "minimum_curve_statics":          mr_min_curve_statics,
        "custom_maximum_dual_zone_curve": mr_custom_max_dualzone,
        "samples":                        int(len(values)),
        "pin_time_pct":                   pin_time_pct,
        "hit_moment_p50":                 hit_p50,
    }


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
        print(f"  {f.name}  -  {len(s):,} samples  ({duration:.0f}s)")

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

    current_curve = _load_current_curve(output_dir)
    if current_curve is None:
        print(f"  (tip: create {output_dir / 'current.json'} with your in-game settings"
              f" for current-vs-recommended diff)")
    else:
        print(f"  Current curve loaded from current.json: "
              f"min={current_curve['custom_minimum_range']} "
              f"max={current_curve['custom_maximum_range']} "
              f"statics={current_curve['minimum_curve_statics']} "
              f"max_dz={current_curve['custom_maximum_dual_zone_curve']}")

    session_groups = list(df.groupby('_session', sort=True))
    timestamps = df['timestamp'].values
    axis_cols = [c for c in df.columns if c.startswith('axis')]

    per_axis = {}
    for col in axis_cols:
        result = analyze_axis(col, df[col].values, timestamps,
                              session_groups, hit_events, output_dir, character)
        if result is not None:
            per_axis[col] = result

    # Right stick on this controller (confirmed via isolated stick test, May 2026)
    # is axis_2 (X) and axis_3 (Y). Average those two for the Marvel Rivals
    # recommendation; fall back to the widest-range axis if neither is present.
    recommended = None
    aim_axes = [a for a in ("axis_2", "axis_3") if a in per_axis]
    if not aim_axes and per_axis:
        aim_axes = [max(per_axis, key=lambda c: per_axis[c]["custom_maximum_range"])]

    if aim_axes:
        cmin   = round(np.mean([per_axis[a]["custom_minimum_range"]           for a in aim_axes]))
        cmax   = round(np.mean([per_axis[a]["custom_maximum_range"]           for a in aim_axes]))
        statics = round(np.mean([per_axis[a]["minimum_curve_statics"]          for a in aim_axes]))
        maxdz  = round(np.mean([per_axis[a]["custom_maximum_dual_zone_curve"] for a in aim_axes]))

        # In-game constraint: each "max" slider must sit at least 1 above its
        # paired "min". Bump the max up rather than the min down so the curve
        # keeps the deadzone the user actually plays with.
        cmax  = max(cmax,  cmin    + 1)
        maxdz = max(maxdz, statics + 1)
        cmax  = min(cmax,  100)
        maxdz = min(maxdz, 100)

        # Sensitivity hint -heuristic based on saturation and hit-moment
        # deflection. Useful as a directional nudge, not a precise number.
        avg_pin = float(np.mean([per_axis[a]["pin_time_pct"] for a in aim_axes]))
        hit_p50s = [per_axis[a]["hit_moment_p50"] for a in aim_axes
                    if per_axis[a]["hit_moment_p50"] is not None]
        avg_hit_p50 = round(float(np.mean(hit_p50s)), 3) if hit_p50s else None

        if avg_pin > 10:
            suggestion = f"increase - right stick pinned {avg_pin:.1f}% of active time"
        elif avg_hit_p50 is not None and avg_hit_p50 < 0.20:
            suggestion = (f"consider decrease - hits cluster at low deflection "
                          f"(p50={avg_hit_p50})")
        elif avg_pin < 1 and (avg_hit_p50 is None or avg_hit_p50 < 0.40):
            suggestion = "looks balanced - small motions dominate"
        else:
            suggestion = "looks balanced for current curve"

        recommended = {
            "source_axes":                    aim_axes,
            "aim_curve_type":                 "Dual Zone S Curve",
            "custom_minimum_range":           int(cmin),
            "custom_maximum_range":           int(cmax),
            "minimum_curve_statics":          int(statics),
            "custom_maximum_dual_zone_curve": int(maxdz),
            "sensitivity_hint": {
                "stick_pin_time_pct": round(avg_pin, 1),
                "hit_moment_p50":     avg_hit_p50,
                "suggestion":         suggestion,
            },
        }

    # Camera-motion stats and current-vs-recommended diff (only if current.json exists)
    camera_motion = None
    diff = None
    if current_curve is not None and aim_axes:
        # Combine absolute stick values across the aim axes — that's the
        # magnitude of right-stick deflection at each sample.
        aim_abs = np.concatenate([np.abs(df[a].values) for a in aim_axes])
        cam = _apply_game_curve(aim_abs, current_curve)
        active_cam = cam[cam > 0.001]
        if len(active_cam) >= 30:
            camera_motion = {
                "p50":            round(float(np.percentile(active_cam, 50)), 3),
                "p95":            round(float(np.percentile(active_cam, 95)), 3),
                "saturated_pct":  round(float(np.mean(cam >= 0.95) * 100), 2),
                "active_samples": int(len(active_cam)),
            }

        # Low-confidence guard: if the aim stick barely moved, the fit is fragile
        # and the diff is misleading. Use the recommended Custom Maximum Range
        # as a proxy for "did the stick actually move?".
        low_confidence = recommended["custom_maximum_range"] < 20

        diff = {"low_confidence": low_confidence}
        flags = []
        for key in ("custom_minimum_range", "custom_maximum_range",
                    "minimum_curve_statics", "custom_maximum_dual_zone_curve"):
            cur = current_curve[key]
            rec = recommended[key]
            delta = rec - cur
            pct = round(abs(delta) / max(cur, 1) * 100, 1)
            diff[key] = {"current": cur, "recommended": rec, "delta": delta, "abs_pct": pct}
            if pct > 20 and not low_confidence:
                flags.append(f"{key} {cur} -> {rec} ({delta:+d})")
        diff["material_changes"] = flags

        # Sharper sensitivity hint using camera-motion data.
        if camera_motion:
            sat = camera_motion["saturated_pct"]
            cam_p50 = camera_motion["p50"]
            if sat > 8:
                suggestion = (f"increase - camera saturated {sat}% of active time "
                              f"(can't keep up with stick)")
            elif cam_p50 < 0.15:
                suggestion = (f"consider decrease - typical camera motion is slow "
                              f"(p50={cam_p50}); precision room available")
            else:
                suggestion = (f"looks balanced - camera motion p50={cam_p50}, "
                              f"saturated {sat}% of active time")
            recommended["sensitivity_hint"] = {
                "stick_pin_time_pct": recommended["sensitivity_hint"]["stick_pin_time_pct"],
                "hit_moment_p50":     recommended["sensitivity_hint"]["hit_moment_p50"],
                "camera_motion_p50":  cam_p50,
                "camera_saturated_pct": sat,
                "suggestion":         suggestion,
            }

    settings = {
        "character":     character,
        "sessions":      [f.name for f in files],
        "current_curve": current_curve,
        "camera_motion": camera_motion,
        "recommended":   recommended,
        "diff":          diff,
        "per_axis":      per_axis,
    }
    settings_path = output_dir / "settings.json"
    with open(settings_path, "w") as fp:
        json.dump(settings, fp, indent=2)

    if recommended:
        print(f"  --> Recommended Marvel Rivals settings (from {', '.join(recommended['source_axes'])}):")
        if diff is not None:
            for key, label in (("custom_minimum_range",           "Custom Minimum Range          "),
                               ("custom_maximum_range",           "Custom Maximum Range          "),
                               ("minimum_curve_statics",          "Minimum Curve Statics         "),
                               ("custom_maximum_dual_zone_curve", "Custom Maximum Dual-zone Curve")):
                d = diff[key]
                arrow = "->" if d["delta"] != 0 else "=="
                print(f"        {label}: {d['current']:>3} {arrow} {d['recommended']:<3}  ({d['delta']:+d})")
            if diff.get("low_confidence"):
                print(f"        !! LOW CONFIDENCE: aim stick barely moved this session, "
                      f"recommendation unreliable")
            elif diff["material_changes"]:
                print(f"        Material changes (>20%): {', '.join(diff['material_changes'])}")
        else:
            print(f"        Custom Minimum Range          : {recommended['custom_minimum_range']}")
            print(f"        Custom Maximum Range          : {recommended['custom_maximum_range']}")
            print(f"        Minimum Curve Statics         : {recommended['minimum_curve_statics']}")
            print(f"        Custom Maximum Dual-zone Curve: {recommended['custom_maximum_dual_zone_curve']}")
        if camera_motion:
            print(f"        Camera motion (current curve) : p50={camera_motion['p50']} "
                  f"p95={camera_motion['p95']} saturated={camera_motion['saturated_pct']}%")
        hint = recommended["sensitivity_hint"]
        print(f"        Sensitivity hint              : {hint['suggestion']}")
    print(f"  --> Wrote {settings_path}\n")


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
        print(f"\nArchiving {len(to_archive)} {label} session(s) -> data/archived/")
        for f in to_archive:
            dest = ARCHIVE_DIR / f.name
            f.rename(dest)
            print(f"  {f.name}")
