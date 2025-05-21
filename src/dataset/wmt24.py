import os
import json
import subprocess

from torch.utils.data import Dataset
from tqdm import tqdm

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                            stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def read_jsonl(path, verbose=True):
    with open(path) as f:
        pbar = tqdm(f, total=file_len(path), desc=f'Loading {path}') if verbose else f
        data = [json.loads(line) for line in pbar]
    return data

class Wmt24Dataset(Dataset):
    def __init__(self, base_path, target_lang):
        """
        base_path: path to 'wmt24_dataset'
        target_lang: language codes, e.g., 'it_IT', 'fr_FR'
        """
        self.base_path = base_path
        self.target_lang = target_lang
        self.file_path = os.path.join(
            self.base_path,
            f"en-{self.target_lang}.jsonl"
        )
        self.data = self._load_data()

    def _load_data(self):
        # Load the JSONL file
        data = read_jsonl(self.file_path, verbose=False)
        paired = []
        # Filter out entries with missing source or target text
        for entry in data:
            if entry['is_bad_source'] == False:
                paired.append({
                    "source_lang": entry.pop('source'),
                    "target_lang": entry.pop('target'),
                    "metadata": entry
                })
        return paired

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
    
# Example usage
if __name__ == "__main__":
    import pprint
    base_path = "../../datasets/wmt24pp"
    dataset = Wmt24Dataset(base_path, "it_IT")
    for i in range(5):
        pprint.pp(dataset[i], width=150)
