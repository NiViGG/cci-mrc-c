import numpy as np
import streamlit as st
import torch
from sklearn.decomposition import PCA

from core.engine import CCIEngine
from core.models import MRCCore
from data.sample_eeg import generate_synthetic_eeg


st.set_page_config(page_title="CCI EEG Experimental Demo", layout="wide")
st.title("CCI EEG Experimental Extension")
st.caption(
    "Exploratory diagnostic interface. This tool does not provide causal proof or consciousness claims."
)

mode = st.selectbox("Signal mode", ["conscious-like", "anesthesia-like", "deep-noise"])
noise_scale = st.slider("Noise scale", min_value=0.01, max_value=0.8, value=0.15, step=0.01)
seq_len = st.slider("Sequence length", min_value=128, max_value=1024, value=512, step=32)
channels = st.slider("Channels", min_value=4, max_value=32, value=16, step=1)
seed = st.number_input("Seed", min_value=0, max_value=100000, value=42, step=1)

eeg = generate_synthetic_eeg(
    mode=mode,
    seq_len=int(seq_len),
    channels=int(channels),
    noise_scale=float(noise_scale),
    seed=int(seed),
)
eeg_t = torch.tensor(eeg, dtype=torch.float32)

model = MRCCore(input_dim=int(channels), hidden_dim=64, z_dim=16)
hidden, _, _ = model.forward_sequence(eeg_t)
engine = CCIEngine()
result = engine.process_signal(hidden, eeg_t)

st.metric("CCI (bits)", f"{result['cci_bits']:.4f}")
st.write(f"Used jitter: `{result['used_jitter']}`")

left, right = st.columns(2)
with left:
    st.subheader("EEG Signal (channel 0)")
    st.line_chart(eeg[:, 0])

with right:
    st.subheader("Latent 2D projection")
    h_np = hidden.detach().cpu().numpy()
    pca = PCA(n_components=2)
    h2 = pca.fit_transform(h_np)
    st.scatter_chart({"x": h2[:, 0], "y": h2[:, 1]})

st.info(
    "Scope note: this EEG extension provides an exploratory dependence signal only. "
    "Interpret with benchmark context and CLAIMS.md boundaries."
)
