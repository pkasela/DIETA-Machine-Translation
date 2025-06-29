import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import evaluate
import click
import torch
from tqdm import tqdm

from comet import download_model, load_from_checkpoint
from metrix23.models import MT5ForRegression
from transformers import AutoTokenizer

def load_metric_objects(metrics, comet_model, bleurt_model="BLEURT-20"):
    """
    Load the specified evaluation metric objects.
    Args:
        metrics (list): List of metric names to load.
        comet_model (str): COMET model to use.
        bleurt_model (str): BLEURT model to use.
    Returns:
        dict: Dictionary of loaded metric objects.
    """
    metric_objs = {}
    for metric in metrics:
        if metric == "bleu":
            metric_objs["bleu"] = evaluate.load("bleu")
        elif metric == "chrf":
            metric_objs["chrf"] = evaluate.load("chrf")
        elif metric == "chrf++":
            metric_objs["chrf++"] = evaluate.load("chrf")
        elif metric == "comet":
            metric_objs["comet"] = evaluate.load("comet", model=comet_model)
        elif metric == "metricx":
            metrix_model_name = "google/metricx-24-hybrid-xl-v2p6-bfloat16"
            metrix_tokenizer_name = "google/mt5-xl"
            reference_free = True
            max_input_length = 1536
            metrix_tokenizer = AutoTokenizer.from_pretrained(metrix_tokenizer_name)
            metrix_model = MT5ForRegression.from_pretrained(metrix_model_name)
            metrix_model = metrix_model.cuda()
            metrix_model = metrix_model.eval()
            metric_objs["metricx"] = (metrix_model, metrix_tokenizer, reference_free, max_input_length)
        elif metric == "cometkiwi":
            reference_free = True
            cometkiwi_model_name = "Unbabel/wmt23-cometkiwi-da-xl"
            # cometkiwi_model_name = "Unbabel/XCOMET-XL"
            model_path = download_model(cometkiwi_model_name)
            cometkiwi_model = load_from_checkpoint(model_path)
            cometkiwi_model = cometkiwi_model.cuda()
            metric_objs["cometkiwi"] = cometkiwi_model, reference_free #evaluate.load("Unbabel/wmt23-cometkiwi-da-xl")
        elif metric == "sacrebleu":
            metric_objs["sacrebleu"] = evaluate.load("sacrebleu")
        elif metric == "bleurt":
            metric_objs["bleurt"] = evaluate.load("bleurt", module_type="metric", checkpoint=bleurt_model)
    return metric_objs

def evaluate_metrics(tsv_path, metrics, metric_objs):
    df = pd.read_csv(tsv_path, sep='\t', lineterminator='\n')
    df["translation"] = df["translation"].fillna(df["source"])
    refs = df["target"].tolist()
    preds = df["translation"].tolist()
    sources = df["source"].tolist()
    results = {}

    bleu_references = [[ref] for ref in refs]

    if "bleu" in metrics:
        bleu = metric_objs["bleu"]
        bleu_score = bleu.compute(predictions=preds, references=bleu_references)
        results["bleu"] = bleu_score["bleu"]

    if "chrf" in metrics:
        chrf = metric_objs["chrf"]
        chrf_score = chrf.compute(predictions=preds, references=refs)
        results["chrf"] = chrf_score["score"]

    if "chrf++" in metrics:
        chrfpp = metric_objs["chrf++"]
        chrfpp_score = chrfpp.compute(predictions=preds, references=refs, word_order=2)
        results["chrf++"] = chrfpp_score["score"]

    if "comet" in metrics:
        comet = metric_objs["comet"]
        comet_score = comet.compute(predictions=preds, references=refs, sources=sources)
        results["comet"] = comet_score["mean_score"]

    if "metricx" in metrics:
        metricx_model, metricx_tokenizer, reference_free, max_input_length = metric_objs["metricx"]
        metricx_values = []
        for i in tqdm(range(len(sources))):
            if reference_free:
                # input_text = f"candidate: {preds[i]} source: {sources[i]}"
                input_text = f"source: {sources[i]} candidate: {preds[i]}"
            else:
                # input_text = f"candidate: {preds[i]} reference: {refs[i]}"
                input_text = f"source: {sources[i]} candidate: {preds[i]} references: {refs[i]}"
            enc = metricx_tokenizer(
                input_text,
                max_length=max_input_length,
                truncation=True,
                padding=False,
                return_tensors="pt"
            )
            # Remove EOS token (last token)
            input_ids = enc["input_ids"][0][:-1].cuda()
            attention_mask = enc["attention_mask"][0][:-1].cuda()
            #with torch.autocast(device_type='cuda'):
            metricx_score = metricx_model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
            # import ipdb; ipdb.set_trace()
            metricx_values.append(metricx_score["predictions"].item())
        results["metricx"] = torch.tensor(metricx_values).mean().item()

    if "cometkiwi" in metrics:
        cometkiwi, reference_free = metric_objs["cometkiwi"]
        # comet_data = [{"src": src, "mt": mt} for src, mt in zip(sources, preds)]
        if reference_free:
            comet_data = [{"src": src, "mt": mt} for src, mt in zip(sources, preds)]
        else:
            comet_data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(sources, preds, refs)]
        # import ipdb; ipdb.set_trace()
        cometkiwi_score = cometkiwi.predict(comet_data, batch_size=32, gpus=1)

        # import ipdb; ipdb.set_trace()
        results["cometkiwi"] = cometkiwi_score["system_score"]

    if "sacrebleu" in metrics:
        sacrebleu = metric_objs["sacrebleu"]
        sacrebleu_score = sacrebleu.compute(predictions=preds, references=bleu_references)
        results["sacrebleu"] = sacrebleu_score["score"]

    if "bleurt" in metrics:
        bleurt = metric_objs["bleurt"]
        bleurt_score = bleurt.compute(predictions=preds, references=refs)
        # BLEURT returns a list of scores, take the mean
        results["bleurt"] = sum(bleurt_score["scores"]) / len(bleurt_score["scores"])

    return results

@click.command()
@click.option('--results_path', required=True, help='Directory where results are stored.')
@click.option('--dataset_name', required=True, help='Name of the dataset (as used in the filename).')
@click.option('--metrics', default="bleu,chrf", help='Comma-separated list of metrics to compute (bleu,chrf,chrf++,comet,metricx,cometkiwi,sacrebleu,bleurt).')
@click.option('--comet_model', default="Unbabel/wmt22-comet-da", help='COMET model to use.')
@click.option('--bleurt_model', default="BLEURT-20", help='BLEURT model to use.')
@click.option('--sort_by', default=None, help='Metric to sort the table by (ascending order).')
def main(results_path, dataset_name, metrics, sort_by, comet_model, bleurt_model):
    dataset_dir = os.path.join(results_path, dataset_name)
    metrics_list = [m.strip() for m in metrics.split(",")]
    summary = []

    # Load metric objects once
    metric_objs = load_metric_objects(metrics_list, comet_model, bleurt_model)

    for fname in os.listdir(dataset_dir):
        if fname.endswith(".tsv"):
            model_name = fname.replace("_", "/").replace(".tsv", "")
            tsv_path = os.path.join(dataset_dir, fname)
            print(f"Evaluating {fname} ...")
            try:
                scores = evaluate_metrics(tsv_path, metrics=metrics_list, metric_objs=metric_objs)
                row = {dataset_name: model_name}
                row.update(scores)
                summary.append(row)
            except Exception as e:
                print(f"Error evaluating {fname}: {repr(e)}")

    df = pd.DataFrame(summary)
    # Reorder columns: model first, then metrics
    columns = [dataset_name] + [m for m in metrics_list if m in df.columns]
    df = df[columns]

    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=True, na_position='last')

    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
