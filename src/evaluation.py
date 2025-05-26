# pip install evaluate sacrebleu unbabel-comet git+https://github.com/google-research/bleurt.git
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

    # Format references for BLEU: list of list of references
    bleu_references = [[ref] for ref in refs]

    # BLEU
    if "bleu" in metrics:
        bleu = evaluate.load("bleu")
        bleu_score = bleu.compute(predictions=preds, references=bleu_references)
        results["bleu"] = bleu_score["bleu"]

    # chrF
    if "chrf" in metrics:
        chrf = evaluate.load("chrf")
        chrf_score = chrf.compute(predictions=preds, references=refs)
        results["chrf"] = chrf_score["score"]

    # chrF++
    if "chrf++" in metrics:
        chrf = evaluate.load("chrf")
        chrf_score = chrf.compute(predictions=preds, references=refs, word_order=2)
        results["chrf++"] = chrf_score["score"]

    # COMET
    if "comet" in metrics:
        comet = evaluate.load("comet", model=comet_model)
        comet_score = comet.compute(predictions=preds, references=refs, sources=sources)
        results["comet"] = comet_score["mean_score"]

    return results

@click.command()
@click.option('--results_path', required=True, help='Directory where results are stored.')
@click.option('--model_name', required=True, help='Name of the model (as used in the filename).')
@click.option('--dataset_name', required=True, help='Name of the dataset (as used in the filename).')
@click.option('--num_beam', default=1, show_default=True, help='Number of beams for beam search. If 0, use greedy search.')
@click.option('--metrics', default="bleu,chrf", help='Comma-separated list of metrics to compute (bleu,chrf,chrf++,comet).')
@click.option('--comet_model', default="Unbabel/wmt22-comet-da", help='COMET model to use.')
def main(results_path, model_name, dataset_name, num_beam, metrics, comet_model):
    if num_beam > 1:
        tsv_file = os.path.join(results_path, dataset_name, f"{model_name.replace('/','_')}_beam_{num_beam}.tsv")
    else:
        tsv_file = os.path.join(results_path, dataset_name, f"{model_name.replace('/','_')}.tsv")
    metrics_list = [m.strip() for m in metrics.split(",")]
    scores = evaluate_metrics(tsv_file, metrics=metrics_list, comet_model=comet_model)
    print(scores)

if __name__ == "__main__":
    main()
    # results_path = "../results"
    # model_name = "facebook/nllb-200-3.3B"
    # dataset_name = "flores"
    # tsv_file = os.path.join(results_path, dataset_name, f"{model_name.replace('/','_')}.tsv")
    # scores = main(tsv_file, metrics=["bleu", "chrf", "chrf++", "comet"])
    # print(scores)
