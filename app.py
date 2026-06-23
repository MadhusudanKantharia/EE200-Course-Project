"""
app.py - "Zapp-tain America": a Shazam-style song identifier (Streamlit).

Two modes:
  * Single clip - identify one uploaded clip and show the intermediate steps
    (spectrogram, constellation of peaks, offset histogram that decides the match).
  * Batch - identify a set of clips and download results.csv with columns
    exactly: filename,prediction.

The song database is indexed once into database.pkl, which ships with the app so
it works immediately on deploy.

Performance notes
-----------------
  * Only the first fp.MAX_QUERY_SECONDS of each query are analysed. Shazam-style
    matching needs only a few seconds; decoding and processing a full multi-minute
    clip is what previously blew past Streamlit Community Cloud's 1 GB RAM limit.
  * The spectrogram and peaks are computed once per clip and reused for matching
    and for every plot (previously they were computed three times per clip).
  * Matplotlib figures are closed after rendering so they don't accumulate across
    reruns, and the spectrogram is drawn with imshow on a frequency-cropped array
    instead of a gouraud-shaded pcolormesh over the full mesh.
  * Per-file decoding is wrapped in try/except so one unreadable upload can't take
    the whole app down (matters for batch / many songs).
"""

import os
import tempfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import fingerprint as fp

st.set_page_config(page_title="Zapp-tain America", page_icon="🎵", layout="wide")

DB_PATH = "database.pkl"
MAX_FREQ_HZ = 4000          # only frequencies up to here are plotted


# ----------------------------------------------------------------- load the indexed database
def _load_pickle(path):
    """Load a pickle that may be gzip-compressed or plain."""
    import gzip
    import pickle
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(path, "rb") as f:
            return pickle.load(f)


@st.cache_resource
def load_db(path=DB_PATH):
    if not os.path.exists(path):
        return None
    data = _load_pickle(path)
    db = fp.FingerprintDB()
    db.index = data["index"]; db.songs = data["songs"]
    return db


def read_upload(uploaded, max_seconds=fp.MAX_QUERY_SECONDS):
    """Save an uploaded file to disk and load up to `max_seconds` of mono audio."""
    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer()); path = tmp.name
    try:
        y = fp.load_audio(path, duration=max_seconds)
    finally:
        os.unlink(path)
    return y


# ----------------------------------------------------------------- plotting helpers
def plot_spectrogram(f, t, S):
    fmask = f <= MAX_FREQ_HZ
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.imshow(S[fmask], origin="lower", aspect="auto",
              extent=[float(t[0]), float(t[-1]), 0.0, MAX_FREQ_HZ],
              cmap="magma", vmin=-60, vmax=0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectrogram")
    fig.tight_layout(); return fig


def plot_constellation(f, t, peaks):
    fig, ax = plt.subplots(figsize=(7, 3))
    if len(peaks):
        ax.scatter(t[peaks[:, 1]], f[peaks[:, 0]], s=8, c="k")
    ax.set_ylim(0, MAX_FREQ_HZ); ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"Constellation map ({len(peaks)} peaks)")
    fig.tight_layout(); return fig


def plot_histogram(votes, db, best_label):
    fig, ax = plt.subplots(figsize=(7, 3))
    if votes and best_label in db.songs:
        sid = db.songs.index(best_label)
        h = votes.get(sid, {})
        if h:
            offs = np.array(list(h.keys())) * fp.HOP / fp.SR
            ax.bar(offs, list(h.values()), width=0.05, color="tab:green")
    ax.set_xlabel("offset = t_db - t_query (s)"); ax.set_ylabel("aligned hashes")
    ax.set_title(f"Offset histogram for best match: {best_label}")
    fig.tight_layout(); return fig


def show_fig(col, fig):
    """Render a figure into a column, then close it to free memory."""
    col.pyplot(fig); plt.close(fig)


# ----------------------------------------------------------------- UI
st.title("🎵 Zapp-tain America - Audio Identifier")
st.caption("Constellation peaks + combinatorial hashing + offset-histogram voting "
           "(Wang 2003; Ellis 2009).")

db = load_db()
if db is None:
    st.error("database.pkl not found. Build the song library into database.pkl first.")
    st.stop()

st.success(f"Database loaded: {len(db.songs)} songs indexed, {len(db.index)} distinct hashes.")
with st.expander("Songs in the database"):
    st.write(", ".join(db.songs))

st.caption(f"Only the first {fp.MAX_QUERY_SECONDS}s of each clip are analysed "
           "(enough to identify a song; keeps memory and CPU bounded).")

mode = st.radio("Mode", ["Single clip", "Batch"], horizontal=True)
threshold = st.slider(
    "Match threshold (minimum aligned hashes to count as a match)", 1, 50, 10,
    help="A genuine match scores in the hundreds/thousands; an unknown song scores only "
         "a few. Anything below this is reported as 'not in database'. Raise it if "
         "unknown songs slip through, lower it if real matches are missed.")

# ================================================================ SINGLE CLIP
if mode == "Single clip":
    up = st.file_uploader("Upload a query clip", type=["wav", "mp3", "m4a", "flac", "ogg"])

    # Reset stored results whenever a new file is uploaded
    if "single_file_name" not in st.session_state:
        st.session_state.single_file_name = None
        st.session_state.single_result = None

    if up is not None:
        # Clear cached result if user swapped the file
        if st.session_state.single_file_name != up.name:
            st.session_state.single_file_name = up.name
            st.session_state.single_result = None

        st.audio(up)

        # ---- Run button ----
        run = st.button("🎵 Identify Song", type="primary", use_container_width=False)

        if run:
            with st.spinner("Analysing clip…"):
                try:
                    y = read_upload(up)
                    f, t, S = fp.compute_spectrogram(y)
                    peaks = fp.find_peaks(S)
                    label, score, results, votes = db.match_hashes(fp.make_hashes(peaks))
                    st.session_state.single_result = dict(
                        f=f, t=t, S=S, peaks=peaks,
                        label=label, score=score,
                        results=results, votes=votes,
                    )
                except Exception as e:
                    st.error(f"Could not read this file: {e}")

        # ---- Display results (persists until a new file is uploaded) ----
        res = st.session_state.single_result
        if res is not None:
            f, t, S = res["f"], res["t"], res["S"]
            peaks   = res["peaks"]
            label, score, results, votes = (
                res["label"], res["score"], res["results"], res["votes"]
            )

            c1, c2, c3 = st.columns(3)
            show_fig(c1, plot_spectrogram(f, t, S))
            show_fig(c2, plot_constellation(f, t, peaks))
            if label is not None:
                show_fig(c3, plot_histogram(votes, db, label))

            if label is None or score < threshold:
                best = f" (best guess '{label}' scored only {score})" if label else ""
                st.error(f"### ❌ Not in database{best}")
                st.caption(f"No song cleared the threshold of {threshold} aligned hashes - "
                           f"this clip does not appear to match any indexed song.")
            else:
                st.markdown(f"## ✅ Prediction: **{label}** &nbsp; (score = {score} aligned hashes)")

            if results:
                st.subheader("Top candidates")
                st.dataframe(
                    pd.DataFrame(results, columns=["song", "score", "best_offset(frames)"]).head(5),
                    hide_index=True, use_container_width=True)

# ================================================================ BATCH
else:
    ups = st.file_uploader("Upload one or more query clips",
                           type=["wav", "mp3", "m4a", "flac", "ogg"],
                           accept_multiple_files=True)

    # Reset stored results whenever the file set changes
    if "batch_file_names" not in st.session_state:
        st.session_state.batch_file_names = []
        st.session_state.batch_result_df  = None

    if ups:
        current_names = [u.name for u in ups]
        if current_names != st.session_state.batch_file_names:
            st.session_state.batch_file_names = current_names
            st.session_state.batch_result_df  = None

        st.info(f"{len(ups)} file(s) ready to process.")

        # ---- Run button ----
        run_batch = st.button("🚀 Run Batch", type="primary", use_container_width=False)

        if run_batch:
            rows = []
            prog = st.progress(0.0)
            with st.spinner("Processing clips…"):
                for i, up in enumerate(ups):
                    try:
                        y = read_upload(up)
                        label, score, *_ = db.match(y)
                        pred = label if (label and score >= threshold) else "not_in_database"
                    except Exception as e:
                        pred = f"error: {e}"
                    rows.append({"filename": up.name, "prediction": pred})
                    prog.progress((i + 1) / len(ups))
            st.session_state.batch_result_df = pd.DataFrame(rows, columns=["filename", "prediction"])

        # ---- Display results ----
        df = st.session_state.batch_result_df
        if df is not None:
            st.dataframe(df, hide_index=True, use_container_width=True)
            csv = df.to_csv(index=False).encode()
            st.download_button("Download results.csv", csv,
                               file_name="results.csv", mime="text/csv")
