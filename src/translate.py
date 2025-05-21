# pip install sentencepiece sacremoses

import os

import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.flores import FloresDataset
from dataset.tatoeba import TatoebaDataset
from dataset.wmt24 import Wmt24Dataset


def get_model_and_tokenizer(model_name):
    """
    Load the pre-trained model and tokenizer from Hugging Face.

    Args:
        model_name (str): Name of the pre-trained model.

    Returns:
        model: Pre-trained MarianMTModel.
        tokenizer: Pre-trained MarianTokenizer.
    """
    if model_name.startswith("Helsinki-NLP/opus-mt"):
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return model, tokenizer
    if model_name.startswith("facebook/mbart"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return model, tokenizer
    if model_name.startswith("facebook/nllb"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang='eng_Latn', tgt_lang='ita_Latn')
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        return model, tokenizer

def main(model_name, dataset_name, dataset_path, results_path, batch_size=128, device="cuda"):
    """
    Main function to load the model and dataset, and perform translation.

    Args:
        model_name (str): Name of the pre-trained model.
        dataset_name (str): Name of the dataset to use ("flores", "tatoeba", "wmt24").
        dataset_path (str): Path to the dataset.
        results_path (str): Path to save the results.
        device (str): Device to use for computation ("cuda" or "cpu").
    """
    # Make the results path 
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(os.path.join(results_path, dataset_name), exist_ok=True)
    # Load the pre-trained model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_name)
    # Move the model to the specified device
    model = model.to(device)
    
    # Load the dataset
    if dataset_name == "flores":
        # Load the Flores dataset
        dataset = FloresDataset(dataset_path, "dev", "eng_Latn", "ita_Latn")
    elif dataset_name == "tatoeba":
        # Load the Tatoeba dataset
        dataset = TatoebaDataset(dataset_path, "test", "eng", "ita")
    elif dataset_name == "wmt24":
        # Load the WMT24 dataset
        dataset = Wmt24Dataset(dataset_path, "it_IT")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    final = {'source': [], 'target': [], 'translation': []}
    # Iterate through the dataset
    for batch in tqdm(dataloader):
        # Translate the source text
        src_text = batch["source_lang"]
        if model_name.startswith("facebook/nllb"):
            # For NLLB models, we need to set the target languages
            translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True).to(device), forced_bos_token_id=tokenizer.lang_code_to_id['ita_Latn'])
        else:
            translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True).to(device))
        # Decode the translated text
        translations = tokenizer.batch_decode(translated, skip_special_tokens=True)
        for i, translation in enumerate(translated):
            final['source'].append(src_text[i])
            final['target'].append(batch["target_lang"][i])
            final['translation'].append(translations[i])
    # Save the results as TSV
    df = pd.DataFrame(final)
    
    tsv_path = os.path.join(results_path, dataset_name, f"{model_name.replace('/','_')}.tsv")
    df.to_csv(tsv_path, sep='\t', index=False)

if __name__ == "__main__":
    # base_path = "../../datasets/wmt24pp"
    # base_path = "../../datasets/tatoeba"
    # base_path = "../../datasets/flores200_dataset"
    # model_name = facebook/nllb-200-distilled-600M

    main("Helsinki-NLP/opus-mt-tc-big-en-it", "tatoeba", "../datasets/tatoeba", '../results', 16, 'cuda:0')
    main("facebook/nllb-200-distilled-1.3B", "tatoeba", "../datasets/tatoeba", '../results', 16, 'cuda:0')
    main("facebook/nllb-200-3.3B", "tatoeba", "../datasets/tatoeba", '../results', 16, 'cuda:0')
    main("facebook/nllb-200-distilled-600M", "tatoeba", "../datasets/tatoeba", '../results', 16, 'cuda:0')