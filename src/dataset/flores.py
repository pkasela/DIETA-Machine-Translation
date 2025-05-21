import os
import csv
from torch.utils.data import Dataset

class FloresDataset(Dataset):
    def __init__(self, base_path, split, source_lang, target_lang):
        """
        base_path: path to 'flores200_dataset'
        split: 'dev' or 'devtest'
        lang1, lang2: language codes, e.g., 'eng_Latn', 'ita_Latn'
        """
        self.base_path = base_path
        self.split = split
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.metadata = self._load_metadata()
        self.data = self._load_data()

    def _load_metadata(self):
        meta_file = os.path.join(
            self.base_path,
            f"metadata_{self.split}.tsv"
        )
        metadata = {}
        with open(meta_file, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row_n, row in enumerate(reader):
                metadata[row_n] = row
        return metadata

    def _load_data(self):
        split_dir = os.path.join(self.base_path, self.split)
        file1 = os.path.join(split_dir, f"{self.source_lang}.dev" if self.split == "dev" else f"{self.source_lang}.devtest")
        file2 = os.path.join(split_dir, f"{self.target_lang}.dev" if self.split == "dev" else f"{self.target_lang}.devtest")

        with open(file1, encoding="utf-8") as f1, open(file2, encoding="utf-8") as f2:
            lines1 = [line.strip() for line in f1]
            lines2 = [line.strip() for line in f2]

        # Pair sentences and attach metadata if needed
        paired = []
        for doc_id, (sent1, sent2) in enumerate(zip(lines1, lines2)):
            meta = self.metadata.get(doc_id, {})
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
    base_path = "../../datasets/flores200_dataset"
    dataset = FloresDataset(base_path, "dev", "eng_Latn", "ita_Latn")
    for i in range(5):
        pprint.pp(dataset[i], width=150)
