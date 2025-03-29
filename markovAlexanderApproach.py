import streamlit as st
import math
import random
import pandas as pd
from Bio import SeqIO
from collections import Counter
from io import StringIO, BytesIO
import zipfile
import matplotlib.pyplot as plt
import tempfile
import os
import statistics

import seaborn as sns
from scipy.stats import shapiro, norm
import numpy as np

from fpdf import FPDF

#############################################################################
# Fancy Plot Functions
#############################################################################
def create_score_scatter_plot(df):
    """Generate a scatter plot of the chosen score vs index and return it as a BytesIO PNG."""
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    x_vals = range(len(df))
    y_vals = df["score"].values
    ax.scatter(x_vals, y_vals, alpha=0.7, color="#1ABC9C", edgecolors="black")
    ax.set_title("Scatter Plot of the Chosen Score", fontsize=14, color="#2C3E50")
    ax.set_xlabel("Sequence Index")
    ax.set_ylabel("Score")
    ax.grid(True)

    png_buffer = BytesIO()
    fig.savefig(png_buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    png_buffer.seek(0)
    return png_buffer

def create_zscore_vs_distance_plot(df):
    """Generate a scatter plot of zscore_normal vs. Distancia and return it as a BytesIO PNG."""
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        df["zscore_normal"],
        df["Distancia"],
        s=100,  
        c=df["Distancia"],  
        cmap="viridis",      
        alpha=0.8,
        edgecolors="black"
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.8)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Euclidean Distance", rotation=270, labelpad=15)

    ax.set_xlabel("Z-score (zscore_normal)")
    ax.set_ylabel("Euclidean Distance")
    ax.set_title("Z-score vs. Euclidean Distance")

    plt.tight_layout()
    png_buffer = BytesIO()
    fig.savefig(png_buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    png_buffer.seek(0)
    return png_buffer

#############################################################################
# Markov Model (Trinucleotides)
#############################################################################
def compute_markov_model_trinuc(genome_seq):
    """Build a 2nd-order Markov model (trinucleotides)."""
    genome_str = str(genome_seq).upper()
    length = len(genome_str)
    two_mer_counts = Counter(genome_str[i:i+2] for i in range(length - 1))
    three_mer_counts = Counter(genome_str[i:i+3] for i in range(length - 2))

    total_2mers = sum(two_mer_counts.values())
    initial_prob_2mer = {}
    for pair in two_mer_counts:
        if total_2mers > 0:
            initial_prob_2mer[pair] = two_mer_counts[pair] / total_2mers
        else:
            initial_prob_2mer[pair] = 1e-10

    cond_prob_3mer = {}
    for triplet in three_mer_counts:
        first_two = triplet[:2]
        denom = sum(count for t, count in three_mer_counts.items() if t.startswith(first_two))
        if denom > 0:
            cond_prob_3mer[triplet] = three_mer_counts[triplet] / denom
        else:
            cond_prob_3mer[triplet] = 1e-10

    return initial_prob_2mer, cond_prob_3mer

def calc_log_prob_trinuc(seq, initial_prob_2mer, cond_prob_3mer):
    """Compute log P(seq) under the 2nd-order Markov model (trinucleotides)."""
    seq = seq.upper()
    if len(seq) < 2:
        return float('-inf')
    log_p = math.log(initial_prob_2mer.get(seq[:2], 1e-10))
    for i in range(2, len(seq)):
        triplet = seq[i-2:i+1]
        p_cond = cond_prob_3mer.get(triplet, 1e-10)
        log_p += math.log(p_cond)
    return log_p

def generate_random_sequence_markov(init_2, cond_2, length):
    """Generate a single random sequence of length 'length' from the 2nd-order Markov model."""
    if length < 2:
        return "A" * length

    pairs, probs = zip(*init_2.items())
    initial_2mer = random.choices(pairs, weights=probs, k=1)[0]
    seq = initial_2mer

    for i in range(2, length):
        last2 = seq[-2:]
        next_candidates = {}
        for nt in ["A", "C", "G", "T"]:
            triplet = last2 + nt
            if triplet in cond_2:
                next_candidates[nt] = cond_2[triplet]
        if not next_candidates:
            seq += "A"
        else:
            bases, w = zip(*next_candidates.items())
            chosen_base = random.choices(bases, weights=w, k=1)[0]
            seq += chosen_base

    return seq

#############################################################################
# Utility: GC%, Normality
#############################################################################
def calc_gc_content(seq):
    """Return GC percentage (0-100) in a sequence."""
    seq = seq.upper()
    if len(seq) == 0:
        return 0.0
    gc_count = seq.count('G') + seq.count('C')
    return (gc_count / len(seq)) * 100

def check_normality(values, alpha=0.05):
    """Return (is_normal, p_value) from the Shapiro-Wilk test."""
    stat, p_value = shapiro(values)
    return (p_value >= alpha, p_value)

#############################################################################
# PDF Generation
#############################################################################
class ClassificationPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "DNA Markov Model Classification Report", ln=1, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

def add_classification_table(pdf, classification_results):
    """Add a table with ID, logP(seq), logP/nt, and the chosen score (Z-score or percentile)."""
    pdf.set_font("Arial", "B", 12)
    col_w = 40
    headers = ["Seq ID", "logP(seq)", "logP/nt", "Score"]
    for h in headers:
        pdf.cell(col_w, 8, h, border=1, align="C")
    pdf.ln(8)

    pdf.set_font("Arial", "", 12)
    for entry in classification_results:
        pdf.cell(col_w, 8, entry["seq_id"], border=1)
        pdf.cell(col_w, 8, entry["log_p_seq"], border=1)
        pdf.cell(col_w, 8, entry["log_p_per_nt"], border=1)
        pdf.cell(col_w, 8, entry["score"], border=1)
        pdf.ln(8)

def generate_pdf_report(classification_results):
    """Creates and returns a PDF with only the classification table."""
    pdf = ClassificationPDF()
    pdf.add_page()

    add_classification_table(pdf, classification_results)

    pdf_out = pdf.output(dest='S').encode('latin-1')
    pdf_buffer = BytesIO(pdf_out)
    pdf_buffer.seek(0)
    return pdf_buffer

#############################################################################
# Excel Generation
#############################################################################
def generate_top100_excel(classification_results):
    """Returns an Excel buffer with the 'top 100' sequences, sorted by 'score' descending."""
    df = pd.DataFrame(classification_results)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df_sorted = df.sort_values(by="score", ascending=False)
    df_top100 = df_sorted.head(100)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_top100.to_excel(writer, index=False, sheet_name="Top100_Score")
    output.seek(0)
    return output

def generate_merged_excel(merged_df):
    """Returns an Excel buffer with the merged Z-score and distance data."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        merged_df.to_excel(writer, index=False, sheet_name="Merged_Zscore_Distance")
    output.seek(0)
    return output

#############################################################################
# Decide N_sim Based on Sequence Length
#############################################################################
def get_n_sim_for_length(L):
    ##maximum number of simulations setted to 1000
    if L < 1000:
        return 100
    elif L < 5000:
        return 200
    elif L < 20000:
        return 500
    else:
        return 1000

#############################################################################
# Rank (Percentile) Score
#############################################################################
def percentile_score(value, distribution):
    sorted_vals = sorted(distribution)
    n = len(sorted_vals)
    count_less = sum(1 for x in sorted_vals if x < value)
    percentile = (count_less + 1) / (n + 1)
    return percentile

#############################################################################
# STREAMLIT APP
#############################################################################
# Page config
st.set_page_config(page_title="DNA Markov Model Classifier", layout="wide")

# --- CSS Styling for a More Appealing Design (No Big White Rectangle) ---
st.markdown("""
<style>
body {
    background-color: #FFF8E6; /* Light orange background */
    margin: 0;
    padding: 0;
    font-family: "Arial", sans-serif;
}
h1, h2, h3, h4, h5 {
    color: #2C3E50;
}
.light-orange-text {
    color: #FFA726; /* Light orange color for short description */
    font-size: 1.1rem;
    margin-bottom: 1rem;
}
.section-header {
    color: #D35400;
    font-size: 1.3rem;
    font-weight: 600;
    margin-top: 1.2rem;
    margin-bottom: 0.5rem;
}
hr {
    border: 1px solid #D35400;
    margin: 1rem 0;
}
.block-container {
    padding: 1rem 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("DNA Markov Model Classifier")

# Short description in light orange
st.markdown('<p class="light-orange-text">How likely each DNA sequence is generated by a reference Markov model?</p>',
            unsafe_allow_html=True)
st.markdown("---")

# Session State
if "classification_sequences" not in st.session_state:
    st.session_state.classification_sequences = []

if "ref_seq_str" not in st.session_state:
    st.session_state.ref_seq_str = None

#############################################################################
# 1) Reference Input
#############################################################################
st.markdown('<h2 class="section-header">1. Reference Genome</h2>', unsafe_allow_html=True)
ref_option = st.radio("Reference genome source:", ["Default snippet", "Upload FASTA (can be multi-FASTA)"])

if ref_option == "Default snippet":
    default_ref = """>example_single_ref
ACGTACGTACGTACGTACGTACGTACGT
"""
    records = list(SeqIO.parse(StringIO(default_ref), "fasta"))
    if len(records) == 0:
        st.error("No sequences in the default snippet!")
        st.stop()
    combined = "".join(str(r.seq) for r in records)
    st.session_state.ref_seq_str = combined
else:
    uploaded_ref = st.file_uploader("Upload the reference FASTA", type=["fa", "fasta", "fna"])
    if uploaded_ref is not None:
        recs = list(SeqIO.parse(StringIO(uploaded_ref.getvalue().decode("utf-8")), "fasta"))
        if not recs:
            st.error("No sequences found in the uploaded reference.")
        else:
            ref_seq_str = "".join(str(r.seq) for r in recs)
            st.session_state.ref_seq_str = ref_seq_str

#############################################################################
# 2) Sequences to Classify
#############################################################################
st.markdown('<h2 class="section-header">2. Sequences to Classify</h2>', unsafe_allow_html=True)
seq_input_method = st.radio(
    "How do you want to provide the sequences?",
    ("Upload a multi-FASTA", "Paste sequences", "Upload ZIP with many FASTA")
)

if seq_input_method == "Upload a multi-FASTA":
    seq_file = st.file_uploader("Upload your (multi-)FASTA", type=["fa", "fasta", "fna"])
    if seq_file and st.button("Load multi-FASTA Sequences"):
        st.session_state.classification_sequences.clear()
        seq_records = list(SeqIO.parse(StringIO(seq_file.getvalue().decode("utf-8")), "fasta"))
        for rec in seq_records:
            st.session_state.classification_sequences.append((rec.id, str(rec.seq)))
        st.success(f"Loaded {len(seq_records)} sequences from the multi-FASTA.")

elif seq_input_method == "Paste sequences":
    seq_text = st.text_area("Paste sequences here, one per line", "", height=150)
    if st.button("Load Pasted Sequences"):
        st.session_state.classification_sequences.clear()
        lines = [l.strip() for l in seq_text.split("\n") if l.strip()]
        count = 0
        for i, s in enumerate(lines, start=1):
            st.session_state.classification_sequences.append((f"user_seq_{i}", s))
            count += 1
        st.success(f"Loaded {count} pasted sequences.")

else:
    zip_file = st.file_uploader("Upload a ZIP containing multiple FASTA files", type=["zip"])
    if zip_file and st.button("Extract & Load Sequences from ZIP"):
        st.session_state.classification_sequences.clear()
        with zipfile.ZipFile(zip_file, "r") as myzip:
            for zip_info in myzip.infolist():
                if zip_info.filename.endswith((".fa", ".fasta", ".fna")):
                    with myzip.open(zip_info) as f:
                        content = f.read().decode("utf-8")
                        for rec in SeqIO.parse(StringIO(content), "fasta"):
                            st.session_state.classification_sequences.append((rec.id, str(rec.seq)))

    seq_count = len(st.session_state.classification_sequences)
    if seq_count > 0:
        st.success(f"Loaded {seq_count} sequences from the ZIP.")
    else:
        if seq_input_method == "Upload ZIP with many FASTA":
            st.warning("No valid FASTA files found in the ZIP, or no sequences extracted.")

st.write(f"**Currently loaded classification sequences:** {len(st.session_state.classification_sequences)}")

#############################################################################
# 3) Scoring Method
#############################################################################
st.markdown('<h2 class="section-header">3. Select Your Scoring Method</h2>', unsafe_allow_html=True)
score_method = st.radio("Choose the scoring metric:", ("Z‐score", "Rank (percentile)"))

#############################################################################
# 4) Main Analysis
#############################################################################
st.markdown("---")
if st.button("Analyze - ParamNsim + Chosen Score Method"):
    if not st.session_state.ref_seq_str:
        st.error("No reference genome available.")
        st.stop()
    if not st.session_state.classification_sequences:
        st.warning("No sequences to classify.")
        st.stop()

    # Compute Markov model on reference
    init_2_ref, cond_2_ref = compute_markov_model_trinuc(st.session_state.ref_seq_str)

    # We'll keep a cache of simulation stats for different lengths
    sim_stats_cache = {}  # key: length, value: {mu, sigma, sim_values}

    def get_simulation_stats_for_length(L, init_2_ref, cond_2_ref):
        if L in sim_stats_cache:
            return sim_stats_cache[L]

        N_sim = get_n_sim_for_length(L)
        sim_logp_per_nt = []
        for _ in range(N_sim):
            sim_seq = generate_random_sequence_markov(init_2_ref, cond_2_ref, L)
            sim_log_p = calc_log_prob_trinuc(sim_seq, init_2_ref, cond_2_ref)
            sim_logp_per_nt.append(sim_log_p / L if L > 0 else float('-inf'))

        mu_ = statistics.mean(sim_logp_per_nt)
        sigma_ = statistics.pstdev(sim_logp_per_nt)
        sim_stats_cache[L] = {
            "mu": mu_,
            "sigma": sigma_,
            "sim_values": sim_logp_per_nt
        }
        return sim_stats_cache[L]

    classification_results = []

    for (seq_id, seq_str) in st.session_state.classification_sequences:
        L = len(seq_str)
        gc_val = calc_gc_content(seq_str)

        if L < 2:
            log_p = float('-inf')
            log_p_per_nt = float('-inf')
            final_score = 0.0
        else:
            log_p = calc_log_prob_trinuc(seq_str, init_2_ref, cond_2_ref)
            log_p_per_nt = log_p / L
            stats_for_len = get_simulation_stats_for_length(L, init_2_ref, cond_2_ref)
            mu = stats_for_len["mu"]
            sigma = stats_for_len["sigma"]
            sim_values = stats_for_len["sim_values"]

            if score_method == "Z‐score":
                if sigma == 0:
                    final_score = 0.0
                else:
                    final_score = (log_p_per_nt - mu) / sigma
            else:  # Rank (percentile)
                final_score = percentile_score(log_p_per_nt, sim_values)

        classification_results.append({
            "seq_id": seq_id,
            "length": L,
            "gc_percent": f"{gc_val:.2f}",
            "log_p_seq": f"{log_p:.2e}",
            "log_p_per_nt": f"{log_p_per_nt:.4f}",
            "score": f"{final_score:.4f}",
        })

    st.session_state["classification_results"] = classification_results
    st.success("Analysis completed! Check below for additional options.")

    # Display a small preview
    st.write("**Classification Results (first 10 rows):**")
    st.dataframe(pd.DataFrame(classification_results).head(10))

    pdf_buffer = generate_pdf_report(classification_results)
    st.download_button(
        label="Download PDF (Classification Table)",
        data=pdf_buffer,
        file_name="report_score.pdf",
        mime="application/pdf"
    )

#############################################################################
# 5) Download Excel Top 100
#############################################################################
if "classification_results" in st.session_state:
    st.markdown("---")
    st.markdown('<h2 class="section-header">4. Download Top 100 in Excel</h2>', unsafe_allow_html=True)

    if st.button("Generate Top 100 Excel"):
        excel_buffer = generate_top100_excel(st.session_state["classification_results"])
        st.download_button(
            label="Download TOP 100 Excel",
            data=excel_buffer,
            file_name="top100_score.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

#############################################################################
# 6) Optional: Download a Score Plot
#############################################################################
if "classification_results" in st.session_state:
    st.markdown("---")
    st.markdown('<h2 class="section-header">5. Optional: Download the Chosen Score Plot</h2>', unsafe_allow_html=True)

    plot_score_option = st.radio("Do you want to generate & download the score plot?", ["No", "Yes"], index=0)

    if plot_score_option == "Yes":
        classification_df = pd.DataFrame(st.session_state["classification_results"]).copy()
        classification_df["score"] = pd.to_numeric(classification_df["score"], errors="coerce")

        # Create plot in a buffer
        plot_buffer = create_score_scatter_plot(classification_df)

        # Offer download
        st.download_button(
            label="Download the Score Plot (PNG)",
            data=plot_buffer.getvalue(),
            file_name="score_plot.png",
            mime="image/png"
        )

#############################################################################
# 7) Optional: Compare Z-score with Distance + Download Merged Plot
#############################################################################
if "classification_results" in st.session_state:
    st.markdown("---")
    st.markdown('<h2 class="section-header">6. Compare Z-score with an External Euclidean Distance</h2>', unsafe_allow_html=True)

    st.write("""
    **How it works**:
    1. Only applies if you used **Z-score** as your scoring method.
    2. We extract `(seq_id, zscore)` from your classification results.
    3. If you have a table with a distance/metric for the **same IDs**,
       upload it and we'll merge on `seq_id`.
    4. You can **download a plot** of Z-score vs. Distance and also an **Excel** of merged data.
    """)

    classification_df = pd.DataFrame(st.session_state["classification_results"]).copy()
    classification_df["score"] = pd.to_numeric(classification_df["score"], errors="coerce")

    if score_method == "Z‐score":
        classification_df.rename(columns={"score": "zscore_normal"}, inplace=True)
        zscore_df = classification_df[["seq_id", "zscore_normal"]]

        dist_file = st.file_uploader(
            "Upload a table (CSV, TSV, or Excel) with columns: seq_id and your distance/metric",
            type=["csv", "tsv", "xlsx"]
        )
        if dist_file is not None:
            file_name = dist_file.name.lower()
            if file_name.endswith(".csv"):
                distance_df = pd.read_csv(dist_file)
            elif file_name.endswith(".tsv"):
                distance_df = pd.read_csv(dist_file, sep="\t")
            else:
                distance_df = pd.read_excel(dist_file)

            st.write("Preview of your distance/metric data:")
            st.dataframe(distance_df.head())

            if "seq_id" not in distance_df.columns:
                st.error("No 'seq_id' column found in your distance table. Please check your file.")
            else:
                distance_columns = [col for col in distance_df.columns if col != "seq_id"]
                chosen_dist_col = st.selectbox("Which column is the distance/metric?", distance_columns)

                if chosen_dist_col:
                    merged_df = pd.merge(
                        zscore_df,
                        distance_df[["seq_id", chosen_dist_col]],
                        on="seq_id",
                        how="inner"
                    )
                    merged_df.rename(columns={chosen_dist_col: "Distancia"}, inplace=True)

                    st.write("Merged data preview (common seq_id only):")
                    st.dataframe(merged_df.head())

                    # Radio to download the plot
                    distance_plot_option = st.radio(
                        "Generate & download Z-score vs. Distance plot?",
                        ["No", "Yes"], index=0
                    )
                    if distance_plot_option == "Yes":
                        dist_plot_buffer = create_zscore_vs_distance_plot(merged_df)
                        st.download_button(
                            label="Download Z-score vs. Distance Plot (PNG)",
                            data=dist_plot_buffer.getvalue(),
                            file_name="zscore_vs_distance.png",
                            mime="image/png"
                        )

                    # Download merged Excel
                    if not merged_df.empty:
                        if st.button("Download Merged Data Excel"):
                            merged_excel_buffer = generate_merged_excel(merged_df)
                            st.download_button(
                                label="Download Merged Z-score & Distance Excel",
                                data=merged_excel_buffer,
                                file_name="merged_zscore_distance.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
    else:
        st.info("Z-score was not chosen as the scoring method, so no Z-score to compare.")
