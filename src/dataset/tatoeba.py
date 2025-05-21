import gzip
import os

from torch.utils.data import Dataset

class TatoebaDataset(Dataset):
    def __init__(self, base_path, split, source_lang, target_lang, v_date="v2021-08-07"):
        """
        base_path: path to 'tatoeba_dataset'
        split: 'dev' or 'test'
        source_lang, target_lang: language codes, e.g., 'eng', 'ita'
        """
        self.base_path = base_path
        self.split = split
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.language_pair = f"{source_lang}-{target_lang}"
        self.v_date = v_date
        self.file_path = os.path.join(
            self.base_path,
            'data',
            'release',
            self.split,
            self.v_date,
            f"tatoeba-{self.split}-{self.v_date}.{self.language_pair}.txt.gz"
        )

        self.data = self._load_data()

    def _load_data(self):
        paired = []
        with gzip.open(self.file_path,'rt') as f:
            for line in f:                
                src_lang, trg_lang, lines1, lines2 = line.split('\t')
                lines1 = lines1.strip()
                lines2 = lines2.strip()
                paired.append({
                    "source_lang": lines1,
                    "target_lang": lines2,
                    "metadata": {
                        "source_lang": src_lang,
                        "target_lang": trg_lang
                    }
                })
        return paired
        
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
# Example usage
if __name__ == "__main__":
    import pprint
    base_path = "../../datasets/tatoeba"
    dataset = TatoebaDataset(base_path, "test", "eng", "ita")
    for i in range(5):
        pprint.pp(dataset[i], width=150)
