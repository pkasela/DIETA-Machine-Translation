import os
import pandas as pd

from src.dataset.flores import FloresDataset
from src.dataset.tatoeba import TatoebaDataset
from src.dataset.wmt24 import Wmt24Dataset
from src.dataset.ntrex import NtrexDataset
from src.dataset.wikinews import WikinewsDataset


clicit_path = 'results/clicit_predictions'

 # Load the Flores dataset
dataset_path = 'datasets/flores200_dataset'
dataset = FloresDataset(dataset_path, "devtest", "eng_Latn", "ita_Latn")
clicit_res = os.path.join(clicit_path, 'flores200_it_pred.txt')

with open(clicit_res, 'r') as f:
    data = [line.strip('\n') for line in f]

final = {'source': [], 'target': [], 'translation': []}
for i, line in enumerate(data):
    final['source'].append(dataset[i]["source_lang"])
    final['target'].append(dataset[i]["target_lang"])
    final['translation'].append(line)

dataset = FloresDataset(dataset_path, "devtest", "eng_Latn", "ita_Latn")
clicit_res = os.path.join(clicit_path, 'flores200_en_pred.txt')
with open(clicit_res, 'r') as f:
    data = [line.strip('\n') for line in f]

final = {'source': [], 'target': [], 'translation': []}
for i, line in enumerate(data):
    final['source'].append(dataset[i]["target_lang"])
    final['target'].append(dataset[i]["source_lang"])
    final['translation'].append(line)

df = pd.DataFrame(final)
tsv_path = os.path.join("results/it_en/flores/clicit.tsv")
df.to_csv(tsv_path, sep='\t', index=False)

 # Load the Tatoeba dataset
dataset_path = "datasets/tatoeba"
dataset = TatoebaDataset(dataset_path, "test", "eng", "ita")
clicit_res = os.path.join(clicit_path, 'tatoeba-test-v2023-09-26_it_pred.txt')
with open(clicit_res, 'r') as f:
    data = [line.strip('\n') for line in f]

final = {'source': [], 'target': [], 'translation': []}
for i, line in enumerate(data):
    final['source'].append(dataset[i]["source_lang"])
    final['target'].append(dataset[i]["target_lang"])
    final['translation'].append(line)

df = pd.DataFrame(final)
tsv_path = os.path.join("results/en_it/tatoeba/clicit.tsv")
df.to_csv(tsv_path, sep='\t', index=False)

dataset = TatoebaDataset(dataset_path, "test", "eng", "ita")
clicit_res = os.path.join(clicit_path, 'tatoeba-test-v2023-09-26_en_pred.txt')
with open(clicit_res, 'r') as f:
    data = [line.strip('\n') for line in f]

final = {'source': [], 'target': [], 'translation': []}
for i, line in enumerate(data):
    final['source'].append(dataset[i]["target_lang"])
    final['target'].append(dataset[i]["source_lang"])
    final['translation'].append(line)

df = pd.DataFrame(final)
tsv_path = os.path.join("results/it_en/tatoeba/clicit.tsv")
df.to_csv(tsv_path, sep='\t', index=False)


 # Load the WMT24 dataset
dataset_path = "datasets/wmt24pp"
dataset = Wmt24Dataset(dataset_path, "it_IT")
clicit_res = os.path.join(clicit_path, 'wm24pp_it_pred.txt')
with open(clicit_res, 'r') as f:
    data = [line.strip('\n') for line in f]

final = {'source': [], 'target': [], 'translation': []}
for i, line in enumerate(data):
    final['source'].append(dataset[i]["source_lang"])
    final['target'].append(dataset[i]["target_lang"])
    final['translation'].append(line)

df = pd.DataFrame(final)
tsv_path = os.path.join("results/en_it/wmt24/clicit.tsv")
df.to_csv(tsv_path, sep='\t', index=False)

dataset = Wmt24Dataset(dataset_path, "it_IT")
clicit_res = os.path.join(clicit_path, 'wm24pp_en_pred.txt')
with open(clicit_res, 'r') as f:
    data = [line.strip('\n') for line in f]

final = {'source': [], 'target': [], 'translation': []}
for i, line in enumerate(data):
    final['source'].append(dataset[i]["target_lang"])
    final['target'].append(dataset[i]["source_lang"])
    final['translation'].append(line)

df = pd.DataFrame(final)
tsv_path = os.path.join("results/it_en/wmt24/clicit.tsv")
df.to_csv(tsv_path, sep='\t', index=False)

 # Load the NTREX-128 dataset
dataset_path = "datasets/NTREX-128"
dataset = NtrexDataset(dataset_path, "eng-US", "ita")
clicit_res = os.path.join(clicit_path, 'newstest2019-ref.eng-US._it_pred.txt')
with open(clicit_res, 'r') as f:
    data = [line.strip('\n') for line in f]

final = {'source': [], 'target': [], 'translation': []}
for i, line in enumerate(data):
    final['source'].append(dataset[i]["source_lang"])
    final['target'].append(dataset[i]["target_lang"])
    final['translation'].append(line)

df = pd.DataFrame(final)
tsv_path = os.path.join("results/en_it/ntrex/clicit.tsv")
df.to_csv(tsv_path, sep='\t', index=False)

dataset = NtrexDataset(dataset_path, "eng-US", "ita")
clicit_res = os.path.join(clicit_path, 'newstest2019-ref.ita_en_pred.txt')
with open(clicit_res, 'r') as f:
    data = [line.strip('\n') for line in f]

final = {'source': [], 'target': [], 'translation': []}
for i, line in enumerate(data):
    final['source'].append(dataset[i]["target_lang"])
    final['target'].append(dataset[i]["source_lang"])
    final['translation'].append(line)

df = pd.DataFrame(final)
tsv_path = os.path.join("results/it_en/ntrex/clicit.tsv")
df.to_csv(tsv_path, sep='\t', index=False)



dataset_path = "datasets/wikinews"
dataset = WikinewsDataset(dataset_path, "en", "it")
clicit_res = os.path.join(clicit_path, 'clicit_annotation2025.en2it._pred.txt')
with open(clicit_res, 'r') as f:
    data = [line.strip('\n') for line in f]

final = {'source': [], 'target': [], 'translation': []}
for i, line in enumerate(data):
    final['source'].append(dataset[i]["source_lang"])
    final['target'].append(dataset[i]["target_lang"])
    final['translation'].append(line)

df = pd.DataFrame(final)
tsv_path = os.path.join("results/en_it/wikinews/clicit.tsv")
df.to_csv(tsv_path, sep='\t', index=False)


dataset = WikinewsDataset(dataset_path, "en", "it")
clicit_res = os.path.join(clicit_path, 'clicit_annotation2025.it2en._pred.txt')
with open(clicit_res, 'r') as f:
    data = [line.strip('\n') for line in f]

final = {'source': [], 'target': [], 'translation': []}
for i, line in enumerate(data):
    final['source'].append(dataset[i]["target_lang"])
    final['target'].append(dataset[i]["source_lang"])
    final['translation'].append(line)

df = pd.DataFrame(final)
tsv_path = os.path.join("results/it_en/wikinews/clicit.tsv")
df.to_csv(tsv_path, sep='\t', index=False)
