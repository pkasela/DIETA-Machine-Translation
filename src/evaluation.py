# pip install evaluate sacrebleu unbabel-comet
import os
import pandas as pd
import evaluate

def main(tsv_path, metrics=["bleu", "chrf"], comet_model="Unbabel/wmt22-comet-da"):
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

    # chrF++
    if "chrf" in metrics:
        chrf = evaluate.load("chrf")
        chrf_score = chrf.compute(predictions=preds, references=refs)
        results["chrf"] = chrf_score["score"]

    # COMET
    if "comet" in metrics:
        comet = evaluate.load("comet", model=comet_model)
        comet_score = comet.compute(predictions=preds, references=refs, sources=sources)
        results["comet"] = comet_score["mean_score"]

    return results

if __name__ == "__main__":
    results_dir = "../results"
    model_name = "Helsinki-NLP/opus-mt-tc-big-en-it"
    dataset_name = "tatoeba"
    tsv_file = os.path.join(results_dir, dataset_name, f"{model_name.replace('/','_')}.tsv")
    scores = main(tsv_file, metrics=["bleu", "chrf", "comet"])
    print(scores)