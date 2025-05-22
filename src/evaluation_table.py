import os
import pandas as pd
import evaluate
import click

def evaluate_metrics(tsv_path, metrics=["bleu", "chrf"], comet_model="Unbabel/wmt22-comet-da"):
    df = pd.read_csv(tsv_path, sep='\t')
    refs = df["target"].tolist()
    preds = df["translation"].tolist()
    sources = df["source"].tolist()
    results = {}

    bleu_references = [[ref] for ref in refs]

    if "bleu" in metrics:
        bleu = evaluate.load("bleu")
        bleu_score = bleu.compute(predictions=preds, references=bleu_references)
        results["bleu"] = bleu_score["bleu"]

    if "chrf" in metrics:
        chrf = evaluate.load("chrf")
        chrf_score = chrf.compute(predictions=preds, references=refs)
        results["chrf"] = chrf_score["score"]

    if "chrf++" in metrics:
        chrf = evaluate.load("chrf")
        chrf_score = chrf.compute(predictions=preds, references=refs, word_order=2)
        results["chrf++"] = chrf_score["score"]

    if "comet" in metrics:
        comet = evaluate.load("comet", model=comet_model)
        comet_score = comet.compute(predictions=preds, references=refs, sources=sources)
        results["comet"] = comet_score["mean_score"]

    return results

@click.command()
@click.option('--results_path', required=True, help='Directory where results are stored.')
@click.option('--dataset_name', required=True, help='Name of the dataset (as used in the filename).')
@click.option('--metrics', default="bleu,chrf", help='Comma-separated list of metrics to compute (bleu,chrf,chrf++,comet).')
@click.option('--comet_model', default="Unbabel/wmt22-comet-da", help='COMET model to use.')
def main(results_path, dataset_name, metrics, comet_model):
    dataset_dir = os.path.join(results_path, dataset_name)
    metrics_list = [m.strip() for m in metrics.split(",")]
    summary = []

    for fname in os.listdir(dataset_dir):
        if fname.endswith(".tsv"):
            model_name = fname.replace("_", "/").replace(".tsv", "")
            tsv_path = os.path.join(dataset_dir, fname)
            print(f"Evaluating {fname} ...")
            scores = evaluate_metrics(tsv_path, metrics=metrics_list, comet_model=comet_model)
            row = {dataset_name: model_name}
            row.update(scores)
            summary.append(row)

    df = pd.DataFrame(summary)
    # Reorder columns: model first, then metrics
    columns = [dataset_name] + [m for m in metrics_list if m in df.columns]
    df = df[columns]
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()