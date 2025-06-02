import string
import re
from pathlib import Path
import warnings
import ftfy
import emoji
import datefinder
from dateutil.parser import UnknownTimezoneWarning
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UnknownTimezoneWarning)
tqdm.pandas()


def preprocess_text(text):
    # First, use ftfy to fix any encoding issues
    if hasattr(text, '__len__'):
        text = ftfy.fix_text(text)

        # Remove dates
        text = remove_dates(text)
        
        # Custom rules for punctuation fixing
        rules = [
            # Remove http and https links
            (r'https?://\S+', ''),
            # Remove mentions
            (r'@\S+', ''),
            # Remove tags
            (r'#\S+', ''),
            # Remove IPs
            (r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ''),
            # Remove NEWLINE_TOKEN
            ('NEWLINE_TOKEN', ''),
            # Remove consecutive repetitive punctuation, but keep a maximum of two for emphasis (e.g., !!)
            (r'([,\.?!])\1{2,}', r'\1\1'),
            # Add space after comma, period, question mark, or exclamation mark if not followed by space
            (r'([,\.?!])(?=[^\s])', r'\1 '),
            # Remove space before comma, period, question mark, or exclamation mark
            (r'\s+([,\.?!])', r'\1'),
            # Fix multiple spaces
            (r'\s{2,}', ' '),
            # Ensure numbers have space before and after, except when punctuation or hyphen follows
            (r'(\d)(?=[^\s\d,\.?!-])', r'\1 '),
            (r'(?<=[^\s\d-])(\d)', r' \1')
        ]
    
        # Replace emoji with :shortcode:
        text = emoji.demojize(text, delimiters=(" ::", ":: "))
        # text = emoji.replace_emoji(text)
    
        # Apply each rule
        for pattern, replacement in rules:
            text = re.sub(pattern, replacement, text)

        stripped = string.punctuation + string.whitespace    
        text = text.strip(stripped)
    else:
        text = ''
    return text

def remove_dates(text):
    matches = datefinder.find_dates(text, source=True)
    replaced = []
    for match in matches:
        replaced.append(match[1])
    for replacement in replaced:
        text = text.replace(replacement, '')
    return text


def load_data(data_dir, task='Semantika', dedup_dir="../dedup"):
    if task =='Semantika':
        df = pd.read_csv(Path(data_dir)/'Semantika_2.txt', sep='__', header=None, usecols=[2], engine='python')
        df = df[2].str.split(" ", n=1, expand=True)
        di = {"neutral": 0, "offensive": 1}
        labels = df[0].map(di)
        texts = df.progress_apply(lambda x : preprocess_text(x[1]), axis=1).tolist()
        return texts, labels
    elif task == 'EmotionLT':
        df = pd.read_excel(Path(data_dir)/'Dabartiniai Duomenys.xlsx', sheet_name='Komentarai', 
                   usecols=['Comment', 'Emocinis manipuliavimas stiprumas'])
        df.columns = ['Comment', 'Target']
        df = df[~pd.isnull(df['Target'])]
        labels_map = {'No': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        labels = df["Target"].map(labels_map)
        texts = df['Comment'].progress_apply(preprocess_text).tolist()
        return texts, labels
    elif task == "ManipulationLT":
        file = Path(data_dir)/"manipuliaciniu_tekstynas_V1.xlsx"
        df = pd.concat([
            pd.read_excel(file, sheet_name="Manipuliaciniai", header=0, usecols=["Komentaras"]).assign(label=1),
            pd.read_excel(file, sheet_name="Neutralūs", header=0, usecols=["Komentaras"]).assign(label=0)
        ]).rename(columns={"Komentaras": "text"}).reset_index(drop=True)
        texts = df['text'].progress_apply(preprocess_text).tolist()
        return texts, df['label']
    elif task == 'Russian':
        df = pd.read_csv(Path(data_dir)/'Russian Kaggle.txt', sep='\t', header=None)
        labels = df[0].apply(lambda x: re.findall('__label__([A-Z]+) ', x)[-1])
        texts = df[0].apply(lambda x: re.sub('(__label__[A-Z]+) ', '', x).strip())
        labels_map = {'NORMAL': 0, 'INSULT': 1, 'OBSCENITY': 1, 'THREAT': 1}
        labels = labels.map(labels_map)
        return texts, labels
    elif task == 'Metahate':
        df = pd.read_csv(Path(data_dir)/"MetaHate.tsv", sep='\t')
        print("Initial dataset size:", df.shape[0])
        index_removed = pd.read_csv(Path(dedup_dir)/"removed_all.csv", header=None)[0].tolist()
        df = df.loc[df.index.difference(index_removed)].reset_index(drop=True)
        print("Filtered dataset size:", df.shape[0])
        texts = df.progress_apply(lambda x : preprocess_text(x[1]), axis=1).tolist()
        labels = df['label']
        return texts, labels


if __name__ == '__main__':
    load_data()