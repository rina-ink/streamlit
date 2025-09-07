# Instrument Timbre – Interactive Audio Demo (public, no-derivatives, safe)
# - No audio downloads, in-memory playback only
# - Loads audio arrays directly from the HF dataset (decode=True)
# - “License-safe mode” ON by default: no resample/normalize, no custom waveform

from pathlib import Path
import io, wave, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datasets import load_dataset, Audio, Image as HFImage, ClassLabel

# Page setup
st.set_page_config(page_title="Instrument Timbre – Interactive Audio Demo", layout="wide")
st.title("Instrument Timbre – Interactive Audio Demo")
st.caption("Dataset: ccmusic-database/instrument_timbre (Hugging Face)")

# Constants
TIMBRE_COLS = [
    "slim","bright","dark","sharp","thick","thin","vigorous","silvery",
    "raspy","full","coarse","pure","hoarse","consonant","mellow","muddy",
]

# Palette (your original muted look)
COLOR_PRIMARY   = "#1386bc"
COLOR_SECONDARY = "#ac62bb"
COLOR_TEXT      = "#e5e7eb"
COLOR_TEXT2     = "#cbd5e1"
COLOR_TICKS     = "#94a3b8"
COLOR_GRID      = "#334155"
COLOR_FRAME     = "#475569"

# Dataset ---------------------------
@st.cache_resource(show_spinner=True)
def get_dataset():
    # Decode to arrays in RAM; do not rely on file paths on disk
    ds = load_dataset("ccmusic-database/instrument_timbre")
    for split in ("Chinese", "Western"):
        ds[split] = ds[split].cast_column("audio", Audio(decode=True))
        ds[split] = ds[split].cast_column("mel", HFImage(decode=True))
    return ds

def instrument_name(split_features, raw_value):
    feat = split_features.get("instrument")
    if isinstance(feat, ClassLabel):
        try:
            return feat.int2str(int(raw_value))
        except Exception:
            pass
    return str(raw_value)

@st.cache_data(show_spinner=True)
def make_dataframe():
    ds = get_dataset()
    rows = []
    for split in ("Chinese", "Western"):
        feats = ds[split].features
        for i, r in enumerate(ds[split]):
            entry = {"split": split, "idx": i, "instrument": instrument_name(feats, r.get("instrument", "unknown"))}
            for c in TIMBRE_COLS:
                entry[c] = float(r.get(c, np.nan))
            rows.append(entry)
    return pd.DataFrame(rows)

# Audio ----------------------------
def get_audio_np(rec):
    """
    Return mono or multi-channel float32 array in [-1,1] and sr, using the decoded array.
    Do not rely on any file path on disk.
    """
    a = rec["audio"]
    # datasets.Audio(decode=True) provides {"array": np.ndarray, "sampling_rate": int}
    if isinstance(a, dict) and "array" in a and "sampling_rate" in a:
        y = a["array"].astype(np.float32)
        sr = int(a["sampling_rate"])
        return y, sr
    raise RuntimeError("Audio array missing from record (ensure decode=True).")

def _resample_numpy(y, sr, target_sr):
    if target_sr is None or target_sr == sr:
        return y, sr
    # simple linear resample (keeps it dependency-free)
    x = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
    xp = np.linspace(0.0, 1.0, num=int(round(len(y) * target_sr / sr)), endpoint=False)
    if y.ndim == 1:
        y = np.interp(xp, x, y).astype(np.float32)
    else:
        y = np.stack([np.interp(xp, x, y[:, c]) for c in range(y.shape[1])], axis=1).astype(np.float32)
    return y, int(target_sr)

def _normalize_rms(y, target_rms=0.1):
    if y.ndim > 1:
        rms = np.sqrt(np.mean(np.mean(y**2, axis=1)))
        peak = float(np.max(np.abs(y)))
    else:
        rms = float(np.sqrt(np.mean(y**2)))
        peak = float(np.max(np.abs(y)))
    if rms < 1e-9:
        return y
    scale_rms = target_rms / rms
    scale_peak = 0.999 / peak if peak > 0 else scale_rms
    return (y * min(scale_rms, scale_peak)).astype(np.float32)

def wav_bytes_from_record(rec, resample_sr=None, normalize=False, target_rms=0.1) -> bytes:
    """
    Convert in-memory audio to 16-bit WAV bytes.
    Do not cache bytes to avoid Streamlit media-store ID issues.
    """
    y, sr = get_audio_np(rec)
    y, sr = _resample_numpy(y, sr, resample_sr)
    if normalize:
        y = _normalize_rms(y, target_rms=target_rms)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    if y.ndim == 1:
        y16 = (y * 32767.0).astype("<i2")
        nch = 1
    else:
        y16 = (y * 32767.0).astype("<i2")
        nch = y.shape[1]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y16.tobytes(order="C"))
    buf.seek(0)
    return buf.read()

# Chart ----------------------------
def radar_chart(values, labels, title: str = "", color: str = COLOR_PRIMARY):
    vals = np.array(values, dtype=float)
    n = len(vals)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    theta = np.concatenate([theta, theta[:1]])
    vals  = np.concatenate([vals,  vals[:1]])

    fig = plt.figure(figsize=(5.4, 5.4), facecolor="none")
    ax = plt.subplot(111, polar=True, facecolor="none")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(theta, vals, linewidth=2.0, marker="o", markersize=4, color=color)
    ax.fill(theta, vals, alpha=0.16, color=color)

    ax.set_xticks(np.linspace(0, 2*np.pi, n, endpoint=False))
    ax.set_xticklabels(labels, fontsize=9, color=COLOR_TEXT2)
    ax.tick_params(axis="x", pad=8)

    ax.set_ylim(1, 9)
    ax.set_yticks([1, 3, 5, 7, 9])
    ax.set_yticklabels(["1", "3", "5", "7", "9"], color=COLOR_TICKS, fontsize=8)

    ax.grid(color=COLOR_GRID, linewidth=0.6, alpha=0.7)
    ax.spines["polar"].set_color(COLOR_FRAME)
    ax.spines["polar"].set_alpha(0.65)

    ax.set_title(title, color=COLOR_TEXT, pad=12, fontsize=12, weight="bold")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

def radar_compare(vals_a, vals_b, labels, name_a="A", name_b="B",
                  colors=(COLOR_PRIMARY, COLOR_SECONDARY),
                  title="A vs B – Timbre (1–9)"):
    vals_a = np.array(vals_a, dtype=float)
    vals_b = np.array(vals_b, dtype=float)
    n = len(labels)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    theta_c = np.concatenate([theta, theta[:1]])
    close = lambda v: np.concatenate([v, v[:1]])

    fig = plt.figure(figsize=(6.1, 6.1), facecolor="none")
    ax = plt.subplot(111, polar=True, facecolor="none")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(theta)
    ax.set_xticklabels(labels, fontsize=9, color=COLOR_TEXT2)
    ax.tick_params(axis="x", pad=8)
    ax.set_ylim(1, 9)
    ax.set_yticks([1,3,5,7,9])
    ax.set_yticklabels(["1","3","5","7","9"], color=COLOR_TICKS, fontsize=8)
    ax.grid(color=COLOR_GRID, linewidth=0.6, alpha=0.7)
    ax.spines["polar"].set_color(COLOR_FRAME)
    ax.spines["polar"].set_alpha(0.65)

    ax.fill(theta_c, close(vals_a), color=colors[0], alpha=0.14, zorder=1)
    ax.fill(theta_c, close(vals_b), color=colors[1], alpha=0.12, zorder=1)

    line_a, = ax.plot(theta_c, close(vals_a),
                      color=colors[0], linewidth=2.6, marker="o",
                      markersize=5, mfc=colors[0], mec=colors[0], mew=0.5, zorder=3)
    line_b, = ax.plot(theta_c, close(vals_b),
                      color=colors[1], linewidth=2.6, marker="o",
                      markersize=5, mfc=colors[1], mec=colors[1], mew=0.5, zorder=4)

    leg = ax.legend([line_a, line_b], [name_a, name_b],
                    loc="upper right", bbox_to_anchor=(1.20, 1.10),
                    frameon=False, handlelength=2.8)
    for t in leg.get_texts():
        t.set_color(COLOR_TEXT)

    ax.set_title(title, color=COLOR_TEXT, pad=12, fontsize=13, weight="bold")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

# Sidebar ---------------------------
df = make_dataframe()
with st.sidebar:
    st.header("Filters")
    sel_splits = st.multiselect("Split", ["Chinese", "Western"], default=["Chinese", "Western"])
    inst_source = df[df["split"].isin(sel_splits)]
    inst_options = sorted(inst_source["instrument"].unique().tolist())
    sel_instruments = st.multiselect("Instrument", inst_options, default=inst_options)

    st.divider()
    st.header("License safety")
    safe_mode = st.toggle("License-safe mode (recommended)", value=True,
                          help="Keeps playback to original samples only (no transforms), "
                               "and disables local waveform generation.")

    st.divider()
    st.header("Audio options (local preview only)")
    if safe_mode:
        st.caption("Transforms are disabled in license-safe mode.")
        normalize_audio = False
        resample_sr = None
        target_rms = 0.10
    else:
        normalize_audio = st.checkbox("Normalize loudness (RMS)", value=False)
        resample_on = st.checkbox("Resample audio", value=False)
        resample_sr = st.selectbox("Target sample rate", [8000, 16000, 22050, 44100], index=1, disabled=not resample_on)
        resample_sr = int(resample_sr) if resample_on else None
        target_rms = st.slider("Target RMS", 0.05, 0.25, 0.10, step=0.01, disabled=not normalize_audio)

# Filtered table --------------------
filtered = df[df["split"].isin(sel_splits)]
filtered = filtered[filtered["instrument"].isin(sel_instruments)]

# Controls --------------------------
if "shuffle_key" not in st.session_state:
    st.session_state.shuffle_key = 0
st.session_state.setdefault("page_control", 1)
st.session_state.setdefault("page_size", 8)

ctrl_cols = st.columns([1,1,1,1,3,3])
with ctrl_cols[0]:
    if st.button(" Shuffle page"):
        st.session_state.shuffle_key += 1
with ctrl_cols[1]:
    if st.button("↩ Reset order"):
        st.session_state.shuffle_key = 0
with ctrl_cols[2]:
    if st.button(" Random (Western)"):
        pool = filtered[filtered["split"] == "Western"]
        if not pool.empty:
            pos = random.choice(pool.index.tolist())
            st.session_state._jump_to_index = pos
with ctrl_cols[3]:
    if st.button("Random (Chinese)"):
        pool = filtered[filtered["split"] == "Chinese"]
        if not pool.empty:
            pos = random.choice(pool.index.tolist())
            st.session_state._jump_to_index = pos
with ctrl_cols[4]:
    st.number_input("Page", min_value=1, step=1, key="page_control")
with ctrl_cols[5]:
    st.slider("Items per page", 4, 12, key="page_size")

# Shuffle BEFORE pagination
if st.session_state.shuffle_key:
    filtered_view = filtered.sample(frac=1, random_state=st.session_state.shuffle_key).reset_index(drop=True)
else:
    filtered_view = filtered.sort_values(["split", "idx"]).reset_index(drop=True)

# Jump-to
if "_jump_to_index" in st.session_state:
    try:
        target_row = filtered.loc[st.session_state._jump_to_index]
        pos_in_view = filtered_view[
            (filtered_view["split"] == target_row["split"]) &
            (filtered_view["idx"] == target_row["idx"])
        ].index[0]
        st.session_state.page_control = int(pos_in_view // st.session_state.page_size) + 1
    except Exception:
        pass
    finally:
        del st.session_state._jump_to_index

# Grid of cards --------------------
st.subheader("Browse samples")
st.caption("Click ▶ to play.")

page = st.session_state.page_control
PAGE_SIZE = st.session_state.page_size
start, end = (page - 1) * PAGE_SIZE, (page - 1) * PAGE_SIZE + PAGE_SIZE
sub = filtered_view.iloc[start:end]

if sub.empty:
    st.info("No items match your filters.")
else:
    ds = get_dataset()
    cols = st.columns(2)
    for j, (_, row) in enumerate(sub.iterrows()):
        with cols[j % 2]:
            st.markdown(f"### {row['instrument']}")
            st.caption(f"{row['split']} • idx {int(row['idx'])}")

            # In-memory WAV (no caching)
            try:
                rec = ds[row["split"]][int(row["idx"])]
                wav_bytes = wav_bytes_from_record(
                    rec,
                    resample_sr=None if safe_mode else resample_sr,
                    normalize=False if safe_mode else normalize_audio,
                    target_rms=0.10
                )
                st.audio(wav_bytes, format="audio/wav")  # <- Do not store or reuse this outside the block
            except Exception as e:
                st.error(f"Audio unavailable: {e}")

            # Mel image
            mel = rec.get("mel")
            if mel is not None:
                st.image(mel, caption="Mel spectrogram", use_container_width=True)

            # Radar
            radar_chart([row[c] for c in TIMBRE_COLS], TIMBRE_COLS, title="Timbre ratings (1–9)")

# Contrast Check (A vs B) -----------
st.divider()
st.header(" Contrast Check (A vs B)")

st.session_state.setdefault("contrast_a", None)
st.session_state.setdefault("contrast_b", None)
overlay_radar = st.checkbox("Overlay radar for A vs B", value=True)

ds = get_dataset()

def pick_random_from(split: str, instrument: str | None):
    pool = df[df["split"] == split]
    if instrument and instrument != "Any":
        pool = pool[pool["instrument"] == instrument]
    if pool.empty:
        return None
    r = pool.sample(1).iloc[0]
    return (r["split"], int(r["idx"]), r["instrument"])

opt_cols = st.columns(3)
with opt_cols[0]:
    show_wave = st.checkbox("Show waveform (local view only)", value=False if safe_mode else True,
                            disabled=safe_mode)
with opt_cols[1]:
    show_env  = st.checkbox("Show smoothed envelope", value=True if not safe_mode else False,
                            disabled=(safe_mode or not show_wave))
with opt_cols[2]:
    env_ms    = st.slider("Envelope window (ms)", 5, 100, 25, step=5, disabled=(not show_env))

cA, cB = st.columns(2)
with cA:
    st.subheader("A")
    split_a = st.selectbox("Split A", ["Western", "Chinese"], index=0, key="split_a")
    inst_opts_a = ["Any"] + sorted(df[df["split"] == split_a]["instrument"].unique().tolist())
    inst_a = st.selectbox("Instrument A", inst_opts_a, key="inst_a")
    if st.button(" Random A"):
        pick = pick_random_from(split_a, inst_a)
        if pick: st.session_state.contrast_a = pick

with cB:
    st.subheader("B")
    split_b = st.selectbox("Split B", ["Chinese", "Western"], index=1, key="split_b")
    inst_opts_b = ["Any"] + sorted(df[df["split"] == split_b]["instrument"].unique().tolist())
    inst_b = st.selectbox("Instrument B", inst_opts_b, key="inst_b")
    if st.button(" Random B"):
        pick = pick_random_from(split_b, inst_b)
        if pick: st.session_state.contrast_b = pick

def plot_waveform(rec: dict, title: str = "", show_envelope: bool = True, env_ms: int = 25,
                  resample_sr: int | None = None, normalize: bool = False, target_rms: float = 0.1):
    y, sr = get_audio_np(rec)
    y, sr = _resample_numpy(y, sr, resample_sr)
    if normalize:
        y = _normalize_rms(y, target_rms)
    if y.ndim > 1:
        y = y.mean(axis=-1)

    target_pts = 2000
    step = max(1, len(y) // target_pts)
    y_ds = y[::step]
    t = np.arange(len(y_ds)) * step / sr

    fig = plt.figure(figsize=(6, 2.2), facecolor="none")
    ax = plt.subplot(111, facecolor="none")
    ax.plot(t, y_ds, linewidth=0.9)
    ax.set_xlabel("Time (s)", color=COLOR_TEXT2)
    ax.set_ylabel("Amplitude", color=COLOR_TEXT2)
    ax.set_title(title, color=COLOR_TEXT)
    ax.margins(x=0)
    ax.tick_params(colors=COLOR_TICKS, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(COLOR_FRAME); spine.set_alpha(0.65)
    ax.grid(color=COLOR_GRID, linewidth=0.6, alpha=0.5)

    if show_envelope:
        win = max(1, int(env_ms / 1000 * sr))
        env = np.convolve(np.abs(y), np.ones(win, dtype=np.float32) / win, mode="same")
        env_ds = env[::step]
        ax.plot(t, env_ds)
        leg = ax.legend(["waveform", f"envelope ~{env_ms} ms"], loc="upper right", frameon=False, fontsize=8)
        for text in leg.get_texts():
            text.set_color(COLOR_TEXT)

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

def show_pick(label: str, pick):
    if not pick:
        st.info(f"Click **Random {label}** to choose a sample.")
        return None, None
    spl, idx, inst = pick
    rec = ds[spl][idx]
    st.markdown(f"**{inst}**  ·  {spl} • idx {idx}")

    # In-memory WAV each rerun (no caching)
    wav_bytes = wav_bytes_from_record(
        rec,
        resample_sr=None if safe_mode else resample_sr,
        normalize=False if safe_mode else normalize_audio,
        target_rms=0.10
    )
    st.audio(wav_bytes, format="audio/wav")

    if (not safe_mode) and show_wave:
        y, sr = get_audio_np(rec)
        dur = len(y) / sr
        plot_waveform(rec, title=f"{label} waveform (~{dur:.2f}s)",
                      show_envelope=show_env, env_ms=env_ms,
                      resample_sr=resample_sr if not safe_mode else None,
                      normalize=normalize_audio if not safe_mode else False, target_rms=0.10)

    mel = rec.get("mel")
    if mel is not None:
        st.image(mel, caption="Mel spectrogram", use_container_width=True)

    vals = [float(rec.get(t, np.nan)) for t in TIMBRE_COLS]
    radar_chart(vals, TIMBRE_COLS, title=f"{label} – Timbre (1–9)")
    return np.array(vals, dtype=float), inst

st.write("")
c1, c2 = st.columns(2)
with c1:
    vals_a, name_a = show_pick("A", st.session_state.contrast_a)
with c2:
    vals_b, name_b = show_pick("B", st.session_state.contrast_b)

if vals_a is not None and vals_b is not None:
    if overlay_radar:
        radar_compare(vals_a, vals_b, TIMBRE_COLS, name_a=name_a or "A", name_b=name_b or "B")

    st.subheader("Δ (A − B) across timbre traits")
    diff = vals_a - vals_b
    bar_colors = [COLOR_PRIMARY if v >= 0 else COLOR_SECONDARY for v in diff]

    fig = plt.figure(figsize=(10, 3), facecolor="none")
    ax = plt.subplot(111, facecolor="none")
    ax.bar(range(len(TIMBRE_COLS)), diff, color=bar_colors)
    ax.axhline(0, linewidth=1, color=COLOR_FRAME, alpha=0.85)
    ax.set_xticks(range(len(TIMBRE_COLS)))
    ax.set_xticklabels(TIMBRE_COLS, rotation=40, ha="right", fontsize=9, color=COLOR_TEXT2)
    ax.set_ylabel("Δ (A−B)", color=COLOR_TEXT2)
    ax.tick_params(colors=COLOR_TICKS)
    for spine in ax.spines.values():
        spine.set_color(COLOR_FRAME); spine.set_alpha(0.65)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.6, alpha=0.5)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

st.info(" Non-commercial use only; no derivatives.")