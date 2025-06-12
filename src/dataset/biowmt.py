import os
from torch.utils.data import Dataset

class BiowmtDataset(Dataset):
    def __init__(self, base_path, source_lang, target_lang, direction):
        """
        base_path: path to 'Biowmt'
        lang1, lang2: language codes, e.g., 'en', 'it'
        direction: en2it or it2en
        """
        self.base_path = base_path
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.direction = direction
        self.data = self._load_data()


    def _load_data(self):
        file1 = os.path.join(self.base_path, f"{self.direction}_{self.source_lang}.txt")
        file2 = os.path.join(self.base_path, f"{self.direction}_{self.target_lang}.txt")

        with open(file1, encoding="utf-8") as f1, open(file2, encoding="utf-8") as f2:
            lines1 = [line.strip() for line in f1]
            lines2 = [line.strip() for line in f2]

        # Pair sentences and attach metadata if needed
        paired = []
        for doc_id, (sent1, sent2) in enumerate(zip(lines1, lines2)):
            meta = {
               "source_lang": self.source_lang,
               "target_lang": self.target_lang
            }
            paired.append({
                "source_lang": sent1,
                "target_lang": sent2,
                "metadata": meta
            })
        return paired

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
# Example usage
if __name__ == "__main__":
    import pprint
    base_path = "../../datasets/biowmt"
    dataset = BiowmtDataset(base_path, "en", "it", "en2it")
    for i in range(5):
        pprint.pp(dataset[i], width=150)
