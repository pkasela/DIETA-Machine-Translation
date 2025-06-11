# pip install sentencepiece sacremoses 
# pip install -U accelerate bitsandbytes

import os
import pandas as pd
import click
import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    BitsAndBytesConfig
)
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.flores import FloresDataset
from dataset.tatoeba import TatoebaDataset
from dataset.wmt24 import Wmt24Dataset
from dataset.ntrex import NtrexDataset
from dataset.wikinews import WikinewsDataset

def get_model_and_tokenizer(model_name, device="cuda"):
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
        model = model.eval()
        model = model.to(device)
    if model_name.startswith("facebook/nllb"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang='eng_Latn', tgt_lang='ita_Latn')
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = model.eval()
        model = model.to(device)
    if model_name.startswith("jbochi/madlad400") or model_name.startswith("google/madlad400"):
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model = model.eval()
        model = model.to(device)
    if model_name.startswith("facebook/mbart"):
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        tokenizer.src_lang = "en_XX"
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        model = model.eval()
        model = model.to(device)
    if (
        model_name.startswith("ModelSpace/Gemma") or model_name.startswith("mii-llm/maestrale") or 
        model_name.startswith("sapienzanlp/Minerva") or model_name.startswith("DeepMount00/Llama") or
        model_name.startswith("swap-uniba/LLaMAntino") or model_name.startswith("sapienzanlp/modello-italia") or
        model_name.startswith("Unbabel/TowerInstruct") or model_name.startswith("galatolo/cerbero") or
        model_name.startswith("LeonardPuettmann/PhiMaestra") or model_name.startswith("utter-project/EuroLLM")
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.eval()
        model = model.to(device)
    if model_name.startswith("sag-uniroma2/extremITA-Camoscio"):
        tokenizer = AutoTokenizer.from_pretrained("yahma/llama-7b-hf")
        tokenizer.padding_side = "left"
        # tokenizer.pad_token = "[PAD]"
        tokenizer.pad_token = tokenizer.bos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.bfloat16()
        model = model.eval()
        model = model.to(device)     
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
    elif model_name.startswith("mii-llm/maestrale"):
        prompt = f'''<|im_start|>system
Sei un assistente utile.<|im_end|>
<|im_start|>user
Traduci questa frase da Inglese a Italiano. Dai solo la traduzione non scrivere altro. Inglese: {src_text}
Italiano:<|im_end|>
<|im_start|>assistant'''
        return prompt
    elif model_name.startswith("sapienzanlp/Minerva"):
        # Prompt for SapienzaNLP/Minerva
        return f'''<s><|start_header_id|> user<|end_header_id|>

Traduci questa frase da Inglese a Italiano. Dai solo la traduzione non scrivere altro.
Inglese: {src_text}
Italiano: <|eot_id|>
'''
    elif model_name.startswith("DeepMount00/Llama"):
        # Prompt for DeepMount00/Llama
        return f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Traduci questa frase da Inglese a Italiano. Dai solo la traduzione non scrivere altro.
Inglese: {src_text}
Italiano: <|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
    elif model_name.startswith("swap-uniba/LLaMAntino"):
        # Prompt for swap-uniba/LLaMAntino
        return f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Sei un an assistente AI per la lingua Italiana di nome LLaMAntino-3 ANITA (Advanced Natural-based interaction for the ITAlian language). Rispondi nella lingua usata per la domanda in modo chiaro, semplice ed esaustivo.<|eot_id|><|start_header_id|>user<|end_header_id|>

Traduci questa frase da Inglese a Italiano. Dai solo la traduzione non scrivere altro.
Inglese: {src_text}
Italiano: <|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
    elif model_name.startswith("sapienzanlp/modello-italia"):
        # Prompt for SapienzaNLP/modello-italia
        return f'''<|system|>
Tu sei Modello Italia, un modello di linguaggio naturale addestrato da iGenius.</s>
<|user|>
Traduci questa frase da Inglese a Italiano. Dai solo la traduzione non scrivere altro.
Inglese: {src_text}
Italiano: </s>
<|assistant|>

'''
    elif model_name.startswith("sag-uniroma2/extremITA-Camoscio"):
        return f"Traduci la seguente frase da Inglese a Italiano:\nInglese: {src_text}\nItaliano:"
    elif model_name.startswith("Unbabel/TowerInstruct"):
        return f'''<|im_start|>user
Translate the following text from English into Italian.
English: {src_text}
Italian: <|im_end|>
<|im_start|>assistant
'''
    elif model_name.startswith("galatolo/cerbero"):
        return f"[|Umano|] Traduci questa frase da Inglese a Italiano. Dai solo la traduzione non scrivere altro.\nInglese: {src_text}\nItaliano: \n[|Assistente|]"
    elif model_name.startswith("LeonardPuettmann/PhiMaestra"):
        return f'''<|system|>
translate English to Italian: <|end|>
<|user|>
{src_text}
<|assistant|> 
'''
    elif model_name.startswith("utter-project/EuroLLM"):
        return f'''<|im_start|>system
You are EuroLLM --- an AI assistant specialized in European languages that provides safe, educational and helpful answers.<|im_end|>
<|im_start|>user
Translate the following English source text to Italian:
English: {src_text}
Italian: <|im_end|>
<|im_start|>assistant
'''
    else:
        return src_text

@click.command()
@click.option('--model_name', required=True, help='Name of the pre-trained model.')
@click.option('--dataset_name', required=True, type=click.Choice(['flores', 'tatoeba', 'wmt24', 'ntrex', 'wikinews']), help='Dataset to use.')
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
    model, tokenizer = get_model_and_tokenizer(model_name, device)
    # Move the model to the specified device
    # model = model.eval()  # Set the model to evaluation mode
    # model = model.to(device)
    
    # Load the dataset
    if dataset_name == "flores":
        # Load the Flores dataset
        dataset = FloresDataset(dataset_path, "devtest", "eng_Latn", "ita_Latn")
    elif dataset_name == "tatoeba":
        # Load the Tatoeba dataset
        dataset = TatoebaDataset(dataset_path, "test", "eng", "ita")
    elif dataset_name == "wmt24":
        # Load the WMT24 dataset
        dataset = Wmt24Dataset(dataset_path, "it_IT")
    elif dataset_name == "ntrex":
        # Load the NTREX-128 dataset
        dataset = NtrexDataset(dataset_path, "eng-US", "ita")
    elif dataset_name == "wikinews":
        dataset = WikinewsDataset(dataset_path, "en", "it")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    final = {'source': [], 'target': [], 'translation': []}
    # Iterate through the dataset
    for batch in tqdm(dataloader):
        with torch.no_grad():
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
            elif model_name.startswith("ModelSpace/Gemma"):
                tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=True).to(device)
                if num_beam > 1:
                    translated = model.generate(**tokenized_src_text, num_beams=num_beam, max_new_tokens=512)
                else:
                    translated = model.generate(**tokenized_src_text, max_new_tokens=512)
                translated = translated[:, len(tokenized_src_text['input_ids'][0]):] # Remove the prompt
            elif model_name.startswith("mii-llm/maestrale"):
                tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=True).to(device)
                generation_config = GenerationConfig(
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    top_k=50,
                    top_p=0.95,
                    max_new_tokens=512,
                    num_beams=num_beam,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
                )
                translated = model.generate(**tokenized_src_text, generation_config=generation_config)
                translated = translated[:, len(tokenized_src_text['input_ids'][0]):]  # Remove the prompt
            elif model_name.startswith("sapienzanlp/Minerva"):
                tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=False).to(device)
                translated = model.generate(**tokenized_src_text, do_sample=True, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
                translated = translated[:, len(tokenized_src_text['input_ids'][0]):] # Remove the prompt
            elif model_name.startswith("DeepMount00/Llama"):
                tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=False).to(device)
                translated = model.generate(**tokenized_src_text, do_sample=True, max_new_tokens=200, temperature=0.001, pad_token_id=tokenizer.eos_token_id)
                translated = translated[:, len(tokenized_src_text['input_ids'][0]):] # Remove the prompt
            elif model_name.startswith("swap-uniba/LLaMAntino"):
                tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=False).to(device)
                translated = model.generate(**tokenized_src_text, do_sample=True, max_new_tokens=200, temperature=0.6, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
                translated = translated[:, len(tokenized_src_text['input_ids'][0]):] # Remove the prompt
            elif model_name.startswith("sapienzanlp/modello-italia"):
                tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=False).to(device)
                translated = model.generate(**tokenized_src_text, do_sample=False, max_new_tokens=200, eos_token_id=tokenizer.convert_tokens_to_ids('|'), pad_token_id=tokenizer.eos_token_id)
                translated = translated[:, len(tokenized_src_text['input_ids'][0]):translated.shape[1]-2] # Remove the prompt
            elif model_name.startswith("sag-uniroma2/extremITA-Camoscio"):
                tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=False).to(device)
                generation_config = GenerationConfig(
                    # temperature=0.2,
                    # top_p=0.75,
                    # top_k=40,
                    num_beams=4,
                    # do_sample=True
                )
                translated = model.generate(**tokenized_src_text, max_new_tokens=200, generation_config=generation_config)
                translated = translated[:, len(tokenized_src_text['input_ids'][0]):] # Remove the prompt
            elif model_name.startswith("Unbabel/TowerInstruct"):
                tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=False).to(device)
                translated = model.generate(**tokenized_src_text, do_sample=False, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
                translated = translated[:, len(tokenized_src_text['input_ids'][0]):] # Remove the prompt
            elif model_name.startswith("galatolo/cerbero"):
                tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=False).to(device)
                translated = model.generate(**tokenized_src_text, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.convert_tokens_to_ids('|'))
                translated = translated[:, len(tokenized_src_text['input_ids'][0]):translated.shape[1]-2] # Remove the prompt
            elif model_name.startswith("LeonardPuettmann/PhiMaestra"):
                tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=False).to(device)
                translated = model.generate(**tokenized_src_text, max_new_tokens=200, do_sample=False)
                translated = translated[:, len(tokenized_src_text['input_ids'][0]):] # Remove the prompt
            elif model_name.startswith("utter-project/EuroLLM"):
                tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=False).to(device)
                translated = model.generate(**tokenized_src_text, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
                translated = translated[:, len(tokenized_src_text['input_ids'][0]):] # Remove the prompt
            else:
                tokenized_src_text = tokenizer(prompted_src_text, return_tensors="pt", padding=True).to(device)
                if num_beam > 1:
                    translated = model.generate(**tokenized_src_text, num_beams=num_beam, max_new_tokens=512)
                else:
                    translated = model.generate(**tokenized_src_text, max_new_tokens=512)
        # Decode the translated text
        translations = tokenizer.batch_decode(translated, skip_special_tokens=True)
        translations = [t.replace('\n','') for t in translations]
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
