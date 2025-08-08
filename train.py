import pandas as pd
import re
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

def download_nltk_resources():
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab"
    }
    for resource_path, resource_name in resources.items():
        try:
            nltk.data.find(resource_path)
            print(f"resourse '{resource_name}' is already downloaded.")
        except LookupError:
            print(f"resourse '{resource_name}' was not found. Downloading...")
            nltk.download(resource_name, quiet=True)
            print(f"Downloading '{resource_name}' complete.")

def main():
    download_nltk_resources()

    FILE_PATH = 'data/chat.csv'
    TEXT_COLUMN_NAME = 'key'

    processed_data = preprocess_chat_data(FILE_PATH, TEXT_COLUMN_NAME)

    if processed_data:
        w2v_model = train_w2v_model(processed_data, vector_size=50, window=5, min_count=20)

        MODEL_PATH = 'word2vec_dota_chat.model'
        w2v_model.save(MODEL_PATH)
        print(f'\nModel was saved in {MODEL_PATH} file')

        demonstrate_model(w2v_model)

def preprocess_chat_data(file_path, text_column):
    print(f'loading and processing "{file_path}" file...')
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f'Error: "{file_path}" was not found.')
        return []
    
    df.dropna(subset=[text_column], inplace=True)
    
    processed_sentences = df[text_column].apply(
        lambda msg:
        word_tokenize(re.sub(r'[^a-zA-Zа-яА-Я\с]', '', str(msg).lower()))
    )
    final_sentences = processed_sentences.tolist()
    print(f'Text is processed. {len(final_sentences)} messages were found')
    return final_sentences

def train_w2v_model(sentences, vector_size=100, window=5, min_count=5, workers=4):
    print('\nStarting Word2Vec model training...')
    print(f'Parameters: vector_size={vector_size}, window={window}, min_count={min_count}')
    
    model = Word2Vec(
        sentences=sentences, vector_size=vector_size, window=window, 
        min_count=min_count, workers=workers
    )
    print('Model was trained successfully!')
    return model

def demonstrate_model(model):
    words_to_test = ['mid', 'gank', 'gg', 'support', 'carry', 'мид', 'саппорт']
    for word in words_to_test:
        try:
            similar_words = model.wv.most_similar(word, topn=5)
            print(f'\nwords the most similar to: "{word}":')
            for w, score in similar_words:
                print(f'- "{w}" (similarity: {score:.2f})')
        except KeyError:
            print(f'\nCouldn`t find "{word}" in model`s dictionary.')

    try:
        result = model.wv.most_similar(positive=['tinker','carry'], negative=['support'], topn=1)
        print('\nVector Arithmetic: tinker - support + carry ≈ ?')
        print(f' - result: "{result[0][0]}"(similarity: {result[0][1]:.2f})')
    except (KeyError, ValueError):
        print('\nCannot complete vector arithmetic.')

if __name__ == '__main__':
    main()