import os

from torch.utils.data import Dataset

class WikinewsDataset(Dataset):
    def __init__(self, base_path, source_lang, target_lang):
        """
        base_path: path to Wikinews dataset
        split: 'dev' or 'test'
        source_lang, target_lang: language codes, e.g., 'eng', 'ita'
        """
        self.base_path = base_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.file_path = os.path.join(
            self.base_path,
	    f"clicit_annotation2025.{self.source_lang}{self.target_lang}.tsv"
        )

        self.data = self._load_data()

    def _load_data(self):
        paired = []
        with open(self.file_path,'rt') as f:
            for line in f:                
                lines1, lines2 = line.split('\t')
                lines1 = lines1.strip()
                lines2 = lines2.strip()
                paired.append({
                    "source_lang": lines1,
                    "target_lang": lines2,
                    "metadata": {
                        "source_lang": self.source_lang,
                        "target_lang": self.target_lang
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
    base_path = "../../datasets/wikinews"
    dataset = WikinewsDataset(base_path, "en", "it")
    for i in range(5):
        pprint.pp(dataset[i], width=150)
