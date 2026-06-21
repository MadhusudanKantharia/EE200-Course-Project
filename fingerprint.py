"""
fingerprint.py
==============
A compact, Shazam-style audio identifier (constellation peaks + combinatorial
hashing + offset-histogram voting).

Pipeline (Wang 2003; implementation style after Ellis' landmark fingerprinter):

    audio -> spectrogram -> peak constellation -> paired hashes
    query hashes -> matched DB hashes -> per-song offset histogram -> winner

COMPATIBILITY NOTE
------------------
The spectrogram / peak-picking / hashing math below is kept OUTPUT-IDENTICAL to
whatever built `database.pkl`. If it changed, query fingerprints would no longer
line up with the stored index and matching would silently fail. The two
optimisations here are deliberately behaviour-preserving:
  * find_peaks() uses `size=` instead of an explicit all-ones `footprint=`. For a
    rectangular all-True window these give identical results, but `size=` lets
    SciPy use its fast separable maximum-filter path.
  * match() no longer accumulates the (unused) per-pair `scatter` structure that
    was the actual cause of the out-of-memory crash on long clips.
Neither change alters which peaks are found or which song wins.

References
----------
[1] A. L.-C. Wang, "An Industrial-Strength Audio Search Algorithm," ISMIR 2003.
[2] D. P. W. Ellis, "Robust Landmark-Based Audio Fingerprinting," LabROSA,
    Columbia University, 2009.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter

# ----------------------------------------------------------------------------- config
SR     = 11025    # all audio is resampled to this rate (mono)
N_FFT  = 1024     # STFT window length (samples)
HOP    = 256      # STFT hop (samples) -> ~23 ms frames

# peak picking
PEAK_NEIGH_T = 19      # local-max neighbourhood in time frames
PEAK_NEIGH_F = 19      # local-max neighbourhood in frequency bins
PEAK_MIN_DB  = -55.0   # ignore peaks quieter than this (relative to max = 0 dB)

# combinatorial hashing (anchor -> target zone)
FAN_OUT = 8    # max points paired with each anchor
DT_MIN  = 1    # min time gap (frames) between anchor and target
DT_MAX  = 40   # max time gap (frames)
DF_MAX  = 80   # max |freq-bin| difference inside the target zone

# query policy (used by the app, NOT by database building)
MAX_QUERY_SECONDS = 30   # only this many seconds of a query clip are analysed


# ----------------------------------------------------------------------------- spectrogram
def compute_spectrogram(y, sr=SR, n_fft=N_FFT, hop=HOP):
    """Return (freqs, times, S_db): magnitude spectrogram in dB (max = 0 dB)."""
    f, t, Z = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    S = np.abs(Z)
    S_db = 20.0 * np.log10(S + 1e-10)
    S_db -= S_db.max()           # normalise so the loudest bin is 0 dB
    return f, t, S_db


# ----------------------------------------------------------------------------- constellation
def find_peaks(S_db, neigh_t=PEAK_NEIGH_T, neigh_f=PEAK_NEIGH_F, min_db=PEAK_MIN_DB):
    """Return peaks as an (n, 2) int array of [freq_bin, time_frame] coordinates.

    A bin is a peak if it is the maximum in a (neigh_f x neigh_t) window and is
    louder than `min_db`. Amplitude is then discarded (only coordinates remain).

    `size=` is used instead of an explicit all-ones `footprint=`: identical result
    for a rectangular window, but it triggers SciPy's much faster separable path.
    """
    if S_db.size == 0:
        return np.empty((0, 2), dtype=int)
    local_max = (S_db == maximum_filter(S_db, size=(neigh_f, neigh_t)))
    peaks_mask = local_max & (S_db > min_db)
    fb, tf = np.where(peaks_mask)
    return np.stack([fb, tf], axis=1)


# ----------------------------------------------------------------------------- hashing
def _pack(f1, f2, dt):
    """Pack a hash into 32 bits: 10 bits f1 | 10 bits f2 | 10 bits dt (Wang 2003)."""
    return ((int(f1) & 0x3FF) << 20) | ((int(f2) & 0x3FF) << 10) | (int(dt) & 0x3FF)


def make_hashes(peaks, fan_out=FAN_OUT, dt_min=DT_MIN, dt_max=DT_MAX, df_max=DF_MAX):
    """Combinatorial hashing: pair each anchor peak with peaks in its target zone.

    Returns a list of (hash_int, anchor_time_frame) tuples.
    """
    if len(peaks) == 0:
        return []
    pk = peaks[np.argsort(peaks[:, 1])]      # sort by time frame
    f = pk[:, 0]; t = pk[:, 1]
    hashes = []
    n = len(pk)
    for i in range(n):
        f1, t1 = f[i], t[i]
        paired = 0
        j = i + 1
        while j < n and paired < fan_out:
            dt = t[j] - t1
            if dt < dt_min:
                j += 1; continue
            if dt > dt_max:
                break                         # peaks are time-sorted: no farther targets
            if abs(int(f[j]) - int(f1)) <= df_max:
                hashes.append((_pack(f1, f[j], dt), int(t1)))
                paired += 1
            j += 1
    return hashes


def single_peak_tokens(peaks):
    """Degenerate 'fingerprint' using individual peaks (frequency only) as tokens.
    Kept as an optional utility for single-peak vs paired-hash comparisons.
    """
    return [(int(fb), int(tf)) for fb, tf in peaks]


# ----------------------------------------------------------------------------- fingerprint one clip
def fingerprint(y, sr=SR, **kw):
    """audio -> (peaks, hashes). Convenience wrapper around the steps above."""
    _, _, S_db = compute_spectrogram(y, sr=sr)
    peaks = find_peaks(S_db)
    hashes = make_hashes(peaks, **kw)
    return peaks, hashes


# ----------------------------------------------------------------------------- database
class FingerprintDB:
    """Inverted index: hash -> list of (song_id, anchor_time_frame)."""

    def __init__(self):
        self.index: dict[int, list[tuple[int, int]]] = {}
        self.songs: list[str] = []            # song_id -> label (filename w/o ext)

    def add_song(self, label, y, sr=SR, **kw):
        sid = len(self.songs)
        self.songs.append(label)
        _, hashes = fingerprint(y, sr=sr, **kw)
        for h, t in hashes:
            self.index.setdefault(h, []).append((sid, t))
        return sid, len(hashes)

    # ---- matching ----
    def match_hashes(self, q_hashes):
        """Identify from precomputed query hashes via offset-histogram voting.

        Returns (best_label, best_score, results, votes), where `results` is a
        list of (label, score, best_offset) sorted by score and `votes` maps
        song_id -> {offset: count}. Only the compact per-song offset histograms
        are kept (no per-pair scatter), so memory stays bounded even for long
        clips that match a very large number of hashes.
        """
        votes: dict[int, dict[int, int]] = {}      # song_id -> {offset: count}
        for h, q_t in q_hashes:
            posting = self.index.get(h)
            if not posting:
                continue
            for sid, db_t in posting:
                off = db_t - q_t
                d = votes.setdefault(sid, {})
                d[off] = d.get(off, 0) + 1

        results = []
        for sid, hist in votes.items():
            best_off = max(hist, key=hist.get)
            results.append((self.songs[sid], hist[best_off], best_off))
        results.sort(key=lambda r: r[1], reverse=True)

        if not results:
            return None, 0, [], votes
        best_label, best_score, _ = results[0]
        return best_label, best_score, results, votes

    def match(self, y, sr=SR, **kw):
        """Fingerprint a query clip and identify it. See match_hashes()."""
        _, q_hashes = fingerprint(y, sr=sr, **kw)
        return self.match_hashes(q_hashes)


# ----------------------------------------------------------------------------- I/O helper
def load_audio(path, sr=SR, duration=None):
    """Load an audio file as mono float32 at `sr` (needs librosa + ffmpeg).

    `duration` (seconds) caps how much is decoded: pass None to load the whole
    file (database building) or a number to limit a query clip. Capping also
    makes decoding faster, since librosa stops reading early.
    """
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    y = y.astype(np.float32)
    if y.size == 0:
        raise ValueError("decoded audio is empty")
    return y
