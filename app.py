"""
app.py  -  Q3B "Zapp-tain America": a Shazam-style song identifier (Streamlit).

Two modes:
  * Single clip - identify one uploaded clip and show the intermediate steps
    (spectrogram, constellation of peaks, offset histogram that decides the match).
  * Batch       - identify a set of clips and download results.csv with columns
    exactly:  filename,prediction   (prediction = matched song's filename, no extension).

The song database is indexed once by build_database.py into database.pkl, which
ships with the app so it works immediately on deploy.
"""
import os, io, pickle, tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import fingerprint as fp

st.set_page_config(page_title="Zapp-tain America", page_icon="🎵", layout="wide")
DB_PATH = "database.pkl"


# ----------------------------------------------------------------- load the indexed database
def _load_pickle(path):
    """Load a pickle that may be gzip-compressed or plain."""
    import gzip
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


def read_upload(uploaded):
    """Save an uploaded file to disk and load it as mono audio at fp.SR."""
    suffix = os.path.splitext(uploaded.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer()); path = tmp.name
    try:
        y = fp.load_audio(path)
    finally:
        os.unlink(path)
    return y


# ----------------------------------------------------------------- plotting helpers
def plot_spectrogram(y):
    f, t, S = fp.compute_spectrogram(y)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.pcolormesh(t, f, S, shading="gouraud", cmap="magma", vmin=-60)
    ax.set_ylim(0, 4000); ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectrogram")
    fig.tight_layout(); return fig

def plot_constellation(y):
    f, t, S = fp.compute_spectrogram(y); pk = fp.find_peaks(S)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.scatter(t[pk[:, 1]], f[pk[:, 0]], s=8, c="k")
    ax.set_ylim(0, 4000); ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"Constellation map ({len(pk)} peaks)")
    fig.tight_layout(); return fig

def plot_histogram(votes, db, best_label):
    fig, ax = plt.subplots(figsize=(7, 3))
    if votes:
        sid = db.songs.index(best_label)
        h = votes.get(sid, {})
        if h:
            offs = np.array(list(h.keys())) * fp.HOP / fp.SR
            ax.bar(offs, list(h.values()), width=0.05, color="tab:green")
    ax.set_xlabel("offset = t_db - t_query (s)"); ax.set_ylabel("aligned hashes")
    ax.set_title(f"Offset histogram for best match: {best_label}")
    fig.tight_layout(); return fig


# ----------------------------------------------------------------- UI
st.title("🎵 Zapp-tain America - Audio Identifier")
st.caption("Constellation peaks + combinatorial hashing + offset-histogram voting "
           "(Wang 2003; Ellis 2009).")

db = load_db()
if db is None:
    st.error("database.pkl not found. Run `python build_database.py songs/` first to index the song library.")
    st.stop()
st.success(f"Database loaded: {len(db.songs)} songs indexed, {len(db.index)} distinct hashes.")
with st.expander("Songs in the database"):
    st.write(", ".join(db.songs))

mode = st.radio("Mode", ["Single clip", "Batch"], horizontal=True)
threshold = st.slider("Match threshold (minimum aligned hashes to count as a match)", 1, 50, 10,
                      help="A genuine match scores in the hundreds/thousands; an unknown song scores only "
                           "a few. Anything below this is reported as 'not in database'. Raise it if "
                           "unknown songs slip through, lower it if real matches are missed.")

# ---------------- single clip ----------------
if mode == "Single clip":
    up = st.file_uploader("Upload a query clip", type=["wav", "mp3", "m4a", "flac", "ogg"])
    if up is not None:
        y = read_upload(up)
        st.audio(up)
        label, score, results, scatter, votes = db.match(y)
        # always show the analysis plots
        c1, c2, c3 = st.columns(3)
        c1.pyplot(plot_spectrogram(y))
        c2.pyplot(plot_constellation(y))
        if label is not None:
            c3.pyplot(plot_histogram(votes, db, label))

        if label is None or score < threshold:
            best = f" (best guess '{label}' scored only {score})" if label else ""
            st.error(f"### ❌ Not in database{best}")
            st.caption(f"No song cleared the threshold of {threshold} aligned hashes — "
                       f"this clip does not appear to match any indexed song.")
        else:
            st.markdown(f"## ✅ Prediction: **{label}**  &nbsp; (score = {score} aligned hashes)")

        if results:
            st.subheader("Top candidates")
            st.dataframe(pd.DataFrame(results, columns=["song", "score", "best_offset(frames)"]).head(5),
                         hide_index=True, use_container_width=True)

# ---------------- batch ----------------
else:
    ups = st.file_uploader("Upload one or more query clips", type=["wav", "mp3", "m4a", "flac", "ogg"],
                           accept_multiple_files=True)
    if ups:
        rows = []
        prog = st.progress(0.0)
        for i, up in enumerate(ups):
            y = read_upload(up)
            label, score, *_ = db.match(y)
            pred = label if (label and score >= threshold) else "not_in_database"
            rows.append({"filename": up.name, "prediction": pred})
            prog.progress((i + 1) / len(ups))
        df = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.dataframe(df, hide_index=True, use_container_width=True)
        csv = df.to_csv(index=False).encode()
        st.download_button("Download results.csv", csv, file_name="results.csv", mime="text/csv")
