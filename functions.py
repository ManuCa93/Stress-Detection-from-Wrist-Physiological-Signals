import pickle 
import numpy as np
from scipy.stats import skew, kurtosis
from scipy import signal as scipy_signal

# === Load all subjects ===
def load_subject_data(DATA_PATH, subject_id):
    """Load a single subject's .pkl file."""
    filepath = DATA_PATH / f'S{subject_id}' / f'S{subject_id}.pkl'
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

# === Windowing function ===
# def create_windows(subject_data, window_size_sec, window_shift_sec, LABEL_SAMPLING_RATE, WRIST_SAMPLING_RATES, VALID_LABELS):
#     """Segment wrist signals into sliding time windows."""
#     labels = subject_data['label'].flatten()
#     wrist = subject_data['signal']['wrist']

#     total_duration = len(labels) / LABEL_SAMPLING_RATE
#     n_windows = int((total_duration - window_size_sec) // window_shift_sec) + 1

#     windows = []
#     window_labels = []

#     for i in range(n_windows):
#         # Majority vote for window label
#         l_start = int(i * window_shift_sec * LABEL_SAMPLING_RATE)
#         l_end = int((i * window_shift_sec + window_size_sec) * LABEL_SAMPLING_RATE)
#         seg = labels[l_start:l_end]
#         if len(seg) == 0:
#             continue
            
#         unique_l, counts_l = np.unique(seg, return_counts=True)
#         majority = int(unique_l[np.argmax(counts_l)])

#         if majority not in VALID_LABELS:
#             continue
#         if np.max(counts_l) / len(seg) < 0.8:
#             continue

#         # Extract signals at native rates
#         window_data = {}
#         valid = True
#         for sig_name, sr in WRIST_SAMPLING_RATES.items():
#             s_start = int(i * window_shift_sec * sr)
#             s_end = int((i * window_shift_sec + window_size_sec) * sr)
#             if s_end <= len(wrist[sig_name]):
#                 window_data[sig_name] = wrist[sig_name][s_start:s_end]
#             else:
#                 valid = False
#                 break

#         if valid:
#             windows.append(window_data)
#             window_labels.append(majority)

#     return windows, np.array(window_labels)

# Modifica alla funzione create_windows per includere il preprocessing
def create_windows(subject_data, window_size_sec, window_shift_sec, LABEL_SAMPLING_RATE, WRIST_SAMPLING_RATES, VALID_LABELS, apply_filters=True):
    """Segment wrist signals into sliding time windows and apply preprocessing."""
    labels = subject_data['label'].flatten()
    wrist = subject_data['signal']['wrist']

    total_duration = len(labels) / LABEL_SAMPLING_RATE
    n_windows = int((total_duration - window_size_sec) // window_shift_sec) + 1

    windows = []
    window_labels = []

    for i in range(n_windows):
        l_start = int(i * window_shift_sec * LABEL_SAMPLING_RATE)
        l_end = int((i * window_shift_sec + window_size_sec) * LABEL_SAMPLING_RATE)
        seg = labels[l_start:l_end]
        
        if len(seg) == 0: continue
            
        unique_l, counts_l = np.unique(seg, return_counts=True)
        majority = int(unique_l[np.argmax(counts_l)])

        # Filtro label e stabilità della finestra (80% stessa label)
        if majority not in VALID_LABELS or np.max(counts_l) / len(seg) < 0.8:
            continue

        window_data = {}
        valid = True
        for sig_name, sr in WRIST_SAMPLING_RATES.items():
            s_start = int(i * window_shift_sec * sr)
            s_end = int((i * window_shift_sec + window_size_sec) * sr)
            
            if s_end <= len(wrist[sig_name]):
                sig_segment = wrist[sig_name][s_start:s_end]
                
                if sig_name != 'ACC':
                    sig_segment = sig_segment.flatten()
                
                if apply_filters:
                    if sig_name == 'BVP':
                        sig_segment = butter_bandpass_filter(sig_segment, 0.5, 8.0, sr)
                    elif sig_name == 'EDA':
                        sig_segment = butter_lowpass_filter(sig_segment, 1.0, sr)
                
                window_data[sig_name] = sig_segment
            else:
                valid = False
                break

        # OUTLIER REMOVAL (Controllo qualità sensori)
        if valid:
            # Se la temperatura è troppo bassa (<25°C), il sensore non era a contatto
            if np.mean(window_data['TEMP']) < 25.0:
                continue
            
            windows.append(window_data)
            window_labels.append(majority)

    return windows, np.array(window_labels)

# Aggiungo anche una funzione per la Normalizzazione (da usare dopo l'estrazione feature)
def normalize_features_by_subject(df, feature_cols):
    """Applica Standard Scaling per ogni soggetto per gestire la variabilità individuale."""
    from sklearn.preprocessing import StandardScaler
    df_norm = df.copy()
    for sid in df_norm['subject_id'].unique():
        mask = df_norm['subject_id'] == sid
        scaler = StandardScaler()
        df_norm.loc[mask, feature_cols] = scaler.fit_transform(df_norm.loc[mask, feature_cols])
    return df_norm


def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99
    b, a = scipy_signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = scipy_signal.filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0:
        high = 0.99
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    y = scipy_signal.filtfilt(b, a, data)
    return y

# === Feature extraction functions ===

def stat_features(sig, prefix):
    """Time-domain statistical features for a 1D signal."""
    return {
        f'{prefix}_mean': np.mean(sig),
        f'{prefix}_std': np.std(sig),
        f'{prefix}_min': np.min(sig),
        f'{prefix}_max': np.max(sig),
        f'{prefix}_median': np.median(sig),
        f'{prefix}_range': np.ptp(sig),
        f'{prefix}_skew': skew(sig),
        f'{prefix}_kurtosis': kurtosis(sig),
        f'{prefix}_rms': np.sqrt(np.mean(sig ** 2)),
        f'{prefix}_mad': np.mean(np.abs(sig - np.mean(sig))),
        f'{prefix}_sad': np.sum(np.abs(np.diff(sig))),
    }

def freq_features(sig, sr, prefix):
    """Frequency-domain features via Welch PSD."""
    freqs, psd = scipy_signal.welch(sig, fs=sr, nperseg=min(len(sig), sr))
    psd_norm = psd / (np.sum(psd) + 1e-12)
    return {
        f'{prefix}_total_power': np.sum(psd),
        f'{prefix}_dom_freq': freqs[np.argmax(psd)] if len(psd) > 0 else 0,
        f'{prefix}_spectral_entropy': -np.sum(psd_norm * np.log2(psd_norm + 1e-12)),
    }

def eda_extra_features(sig, sr, prefix='EDA'):
    """EDA-specific features (peaks, derivatives)."""
    peaks, props = scipy_signal.find_peaks(sig, height=0, distance=sr)
    deriv = np.diff(sig)
    return {
        f'{prefix}_n_peaks': len(peaks),
        f'{prefix}_peak_mean_h': np.mean(props['peak_heights']) if len(peaks) > 0 else 0,
        f'{prefix}_deriv_mean': np.mean(deriv),
        f'{prefix}_deriv_std': np.std(deriv),
    }

def extract_window_features(window_data, WRIST_SAMPLING_RATES):
    """Extract all features from one window."""
    feats = {}

    # ACC: 3 axes + magnitude
    acc = window_data['ACC']
    for axis, name in enumerate(['ACC_x', 'ACC_y', 'ACC_z']):
        a = acc[:, axis].flatten()
        feats.update(stat_features(a, name))
        feats.update(freq_features(a, WRIST_SAMPLING_RATES['ACC'], name))
    acc_mag = np.linalg.norm(acc, axis=1)
    feats.update(stat_features(acc_mag, 'ACC_mag'))
    feats.update(freq_features(acc_mag, WRIST_SAMPLING_RATES['ACC'], 'ACC_mag'))

    # BVP
    bvp = window_data['BVP'].flatten()
    try:
        bvp = butter_bandpass_filter(bvp, 0.5, 8.0, WRIST_SAMPLING_RATES['BVP'])
    except Exception:
        pass
    feats.update(stat_features(bvp, 'BVP'))
    feats.update(freq_features(bvp, WRIST_SAMPLING_RATES['BVP'], 'BVP'))

    # EDA (with extras)
    eda = window_data['EDA'].flatten()
    try:
        eda = butter_lowpass_filter(eda, 1.0, WRIST_SAMPLING_RATES['EDA'])
    except Exception:
        pass
    feats.update(stat_features(eda, 'EDA'))
    feats.update(freq_features(eda, WRIST_SAMPLING_RATES['EDA'], 'EDA'))
    feats.update(eda_extra_features(eda, WRIST_SAMPLING_RATES['EDA']))

    # TEMP
    temp = window_data['TEMP'].flatten()
    feats.update(stat_features(temp, 'TEMP'))

    return feats



