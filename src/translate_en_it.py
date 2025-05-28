# pip install sentencepiece sacremoses

import os
import pandas as pd
import click
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.flores import FloresDataset
from dataset.tatoeba import TatoebaDataset
from dataset.wmt24 import Wmt24Dataset
from dataset.ntrex import NtrexDataset


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
    if model_name.startswith("facebook/nllb"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang='eng_Latn', tgt_lang='ita_Latn')
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if model_name.startswith("jbochi/madlad400") or model_name.startswith("google/madlad400"):
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    if model_name.startswith("facebook/mbart"):
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        tokenizer.src_lang = "en_XX"
        model = MBartForConditionalGeneration.from_pretrained(model_name)
    if model_name.startswith("ModelSpace/Gemma"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
    return model, tokenizer
    

def make_prompt(src_text, model_name):
    """
    Create a prompt for the translation task.

    Args:
        src_text (str): Source text to be translated.
        model_name (str): Name of the pre-trained model.

    Returns:
        str: Prompt for the translation task.
    """
    if model_name.startswith("jbochi/madlad400") or model_name.startswith("google/madlad400"):
        # For T5 models, we need to add the task prefix
        # For example, "translate English to Italian: <2it> Hello, how are you?"
        return f"<2it> {src_text}"
    elif model_name.startswith("ModelSpace/Gemma"):
        # Prompt for ModelSpace/Gemma
        return f"Translate this from English to Italian:\nEnglish: {src_text}\nItalian:"
    else:
        return src_text

@click.command()
@click.option('--model_name', required=True, help='Name of the pre-trained model.')
@click.option('--dataset_name', required=True, type=click.Choice(['flores', 'tatoeba', 'wmt24', 'ntrex']), help='Dataset to use.')
@click.option('--dataset_path', required=True, help='Path to the dataset.')
@click.option('--results_path', required=True, help='Path to save the results.')
@click.option('--batch_size', default=128, show_default=True, help='Batch size for DataLoader.')
@click.option('--num_beam', default=1, show_default=True, help='Number of beams for beam search. If 0, use greedy search.')
@click.option('--device', default='cuda', show_default=True, help='Device to use for computation.')
def main(model_name, dataset_name, dataset_path, results_path, batch_size=128, num_beam=0, device="cuda"):
    """
    Main function to load the model and dataset, and perform translation.

    Args:
        model_name (str): Name of the pre-trained model.
        dataset_name (str): Name of the dataset to use ("flores", "tatoeba", "wmt24", "ntrex").
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
    model = model.eval()  # Set the model to evaluation mode
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
    elif dataset_name == "ntrex":
        # Load the NTREX-128 dataset
        dataset = NtrexDataset(dataset_path, "eng-US", "ita")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    final = {'source': [], 'target': [], 'translation': []}
    # Iterate through the dataset
    for batch in tqdm(dataloader):
        # Translate the source text
        src_text = batch["source_lang"]
        # Create the prompt
        prompted_src_text = [make_prompt(text, model_name) for text in src_text]
        # Generate the translation
        if model_name.startswith("facebook/nllb"):
            # For NLLB models, we need to set the target languages
            tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=True).to(device)
            if num_beam > 1:
                translated = model.generate(**tokenized_src_text, forced_bos_token_id=tokenizer.convert_tokens_to_ids("ita_Latn"), num_beams=num_beam)
            else:
                translated = model.generate(**tokenized_src_text, forced_bos_token_id=tokenizer.convert_tokens_to_ids("ita_Latn"))
        elif model_name.startswith("facebook/mbart"):
            # For MBART models, we need to set the target languages
            tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=True).to(device)
            if num_beam > 1:
                translated = model.generate(**tokenized_src_text, forced_bos_token_id=tokenizer.convert_tokens_to_ids("it_IT"), num_beams=num_beam)
            else:
                translated = model.generate(**tokenized_src_text, forced_bos_token_id=tokenizer.convert_tokens_to_ids("it_IT"))
            # translated = model.generate(**tokenized_src_text, forced_bos_token_id=tokenizer.lang_code_to_id['it_IT'])
        else:
            tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=True).to(device)
            if num_beam > 1:
                translated = model.generate(**tokenized_src_text, num_beams=num_beam, max_new_tokens=1024)
            else:
                translated = model.generate(**tokenized_src_text, max_new_tokens=1024)
        # Decode the translated text
        translations = tokenizer.batch_decode(translated, skip_special_tokens=True)
        for i, translation in enumerate(translated):
            final['source'].append(src_text[i])
            final['target'].append(batch["target_lang"][i])
            if translations[i].startswith(prompted_src_text[i]) and not model_name.startswith("facebook/mbart"):
                # Remove the prompt from the translation
                translations[i] = translations[i][len(prompted_src_text[i]):]
            final['translation'].append(translations[i])
    # Save the results as TSV
    df = pd.DataFrame(final)
    
    if num_beam > 1:
        tsv_path = os.path.join(results_path, dataset_name, f"{model_name.replace('/','_')}_beam_{num_beam}.tsv")
    else:
        tsv_path = os.path.join(results_path, dataset_name, f"{model_name.replace('/','_')}.tsv")
    df.to_csv(tsv_path, sep='\t', index=False)

if __name__ == "__main__":
    # base_path = "../../datasets/wmt24pp"
    # base_path = "../../datasets/tatoeba"
    # base_path = "../../datasets/flores200_dataset"
    # model_name = facebook/nllb-200-distilled-600M
    main()
    # main("Helsinki-NLP/opus-mt-tc-big-en-it", "tatoeba", "../datasets/tatoeba", '../results', 16, 'cuda:0')
    # main("facebook/nllb-200-distilled-1.3B", "tatoeba", "../datasets/tatoeba", '../results', 16, 'cuda:0')
    # main("facebook/nllb-200-3.3B", "tatoeba", "../datasets/tatoeba", '../results', 16, 'cuda:0')
    # main("facebook/nllb-200-distilled-600M", "tatoeba", "../datasets/tatoeba", '../results', 16, 'cuda:0')