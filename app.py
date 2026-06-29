
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
from datasets import load_dataset

# ── Load Datasets ─────────────────────────────────────────────────────────────
print("Loading datasets...")
poly_fever = load_dataset("fever", "v1.0")
wiki_hades  = load_dataset("wiki_hades")
fact_chd    = load_dataset("utahnlp/FactCHD")
halueval    = load_dataset("pminervini/HaluEval", "general")

split_map = {"HaluEval": "data"}

def to_df(name, ds_obj):
    if hasattr(ds_obj, "keys"):
        preferred = [split_map.get(name)] + ["train", "test", "validation"]
        for split in preferred:
            if split and split in ds_obj:
                return ds_obj[split].to_pandas()
    return ds_obj.to_pandas()

raw = {
    "PolyFever": to_df("PolyFever", poly_fever),
    "WikiHades":  to_df("WikiHades", wiki_hades),
    "FactCHD":    to_df("FactCHD",   fact_chd),
    "HaluEval":   to_df("HaluEval",  halueval),
}

# ── Config ────────────────────────────────────────────────────────────────────
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
dataset_names = ["PolyFever", "WikiHades", "FactCHD", "HaluEval"]
label_cols = {"PolyFever": "Label", "WikiHades": "hallucination", "FactCHD": "label", "HaluEval": "hallucination"}
text_cols  = {"PolyFever": "en", "WikiHades": "replaced", "FactCHD": "query", "HaluEval": "user_query"}

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    return Image.open(buf)

def show_chart(dataset_name, chart_type):
    df = raw[dataset_name]
    color = colors[dataset_names.index(dataset_name)]
    fig, ax = plt.subplots(figsize=(7, 4))

    if chart_type == "Label Distribution":
        col = label_cols[dataset_name]
        counts = df[col].value_counts()
        ax.bar(counts.index.astype(str), counts.values, color=color)
        ax.set_title(f"{dataset_name} - Label Distribution")
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        for i, val in enumerate(counts.values):
            ax.text(i, val + 50, f"{val:,}", ha="center", fontsize=9)

    elif chart_type == "Text Length":
        col = text_cols[dataset_name]
        lengths = df[col].astype(str).apply(lambda x: len(x.split()))
        ax.hist(lengths, bins=30, color=color, edgecolor="white")
        mean_len = lengths.mean()
        ax.axvline(mean_len, color="black", linestyle="--", linewidth=1.5)
        ax.text(mean_len + 1, ax.get_ylim()[1] * 0.85, f"avg: {mean_len:.0f}w", fontsize=9)
        ax.set_title(f"{dataset_name} - Text Length Distribution")
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Number of Rows")

    elif chart_type == "Missing Data":
        missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing = missing[missing > 0]
        if missing.empty:
            ax.text(0.5, 0.5, "No missing data!", ha="center", va="center",
                    fontsize=14, transform=ax.transAxes)
            ax.set_title(f"{dataset_name} - Missing Data")
        else:
            ax.barh(missing.index, missing.values, color=color)
            ax.set_title(f"{dataset_name} - Missing Data (%)")
            ax.set_xlabel("% Missing")

    plt.tight_layout()
    img = fig_to_image(fig)
    plt.close()
    return img

# ── Gradio App ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Hallucination Dataset Dashboard") as app:
    gr.Markdown("# Hallucination Dataset Comparison Dashboard")
    gr.Markdown("Explore PolyFever, WikiHades, FactCHD and HaluEval visually.")
    with gr.Row():
        dataset_dropdown = gr.Dropdown(choices=dataset_names, value="PolyFever", label="Select Dataset")
        chart_dropdown   = gr.Dropdown(choices=["Label Distribution", "Text Length", "Missing Data"], value="Label Distribution", label="Select Chart")
    output_image = gr.Image(label="Chart", type="pil")
    generate_btn = gr.Button("Generate Chart", variant="primary")
    generate_btn.click(fn=show_chart, inputs=[dataset_dropdown, chart_dropdown], outputs=output_image)

app.launch()
