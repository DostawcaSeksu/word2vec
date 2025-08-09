# word2vec

## Project Overview

This module serves as the foundational step in our "NLP Evolution" project. The primary goal is to train a `Word2Vec` model on a domain-specific corpus: the in-game chat logs from thousands of Dota 2 matches.

Unlike generic text corpora like Wikipedia, Dota 2 chat has its own unique vocabulary, slang, sentiment, and context. The objective is to create word embeddings (vectors) that capture these unique semantic relationships, which are often tied to the game's situations, emotions, and specific terminology.

This model doesn't just learn dictionary definitions; it learns the contextual meaning of words as they are used by players in the heat of a game. The resulting word vectors will serve as the input layer for more advanced neural network models in subsequent modules.

## How It Works

The `train.py` script performs the following steps:

1.  **Data Loading & Preprocessing**:
    *   Loads the `chat.csv` dataset using `pandas`.
    *   Removes any rows with empty chat messages.
    *   Applies a text cleaning pipeline to each message:
        *   Converts text to lowercase.
        *   Removes all characters except for English and Cyrillic letters and whitespace using regular expressions.
        *   Tokenizes the cleaned message into a list of words using `nltk`.
    *   The result is a list of sentences, where each sentence is a list of word tokens, ready for training.

2.  **Word2Vec Model Training**:
    *   Initializes and trains a `gensim.Word2Vec` model on the processed chat data.
    *   Key hyperparameters used:
        *   `vector_size=50`: Each word is represented by a 50-dimensional vector.
        *   `window=5`: The model considers a context window of 5 words around a target word.
        *   `min_count=20`: Words that appear fewer than 20 times in the entire corpus are ignored. This helps filter out noise, typos, and rare slang.

3.  **Model Demonstration**:
    *   After training, the script showcases the model's capabilities by performing several tests:
        *   **Finding Similar Words**: For key Dota 2 terms (e.g., `mid`, `gank`, `gg`), it finds the top 5 most similar words from the vocabulary based on cosine similarity between their vectors.
        *   **Vector Arithmetic**: It attempts to solve semantic analogies, such as `tinker - support + carry ≈ ?` *(P.S. The result is `"fuckingtrash"(similarity: 0.44)` xdd)*, to test if the model has captured deeper relationships between words.

## How to Run

1.  **Prerequisites**: Ensure you have Python and the required libraries installed.
    ```bash
    pip install -r req.txt
    ```

2.  **Data**: Place your `chat.csv` file inside the `data/` directory. **Important**: Open the script and verify that the `TEXT_COLUMN_NAME` variable matches the actual name of the column containing chat messages in your CSV file.

3.  **Execution**: Run the script from the command line while inside the directory.
    ```bash
    python train.py
    ```
    The script will first download necessary `nltk` resources if they are missing, then process the data, train the model, save it as `word2vec_dota_chat.model`, and finally print the demonstration results.

## Analysis of Results

The trained model produced fascinating results, highlighting the unique nature of the Dota 2 chat environment.

### Key Findings:

*   **Context over Semantics**: The model excelled at learning situational and emotional context rather than strict dictionary definitions. For example, the word `mid` was found to be similar to words related to conflict, blame, and toxicity (`cykablyat`, `fucktard`, `noteam`), accurately reflecting the high-pressure environment of the mid-lane.

*   **Toxicity Patterns**: Words like `gg` were strongly associated with calls to `report` players, indicating that "gg" is often used sarcastically at the end of a lost game.

*   **Domain-Specific Vocabulary**: The model successfully captured relationships between game-specific terms (e.g., `invo` as a common abbreviation for a mid hero).

*   **Limitations and Noise**: The results for broader terms like `support` and `carry` were less coherent. This is likely due to the "noisy" nature of the data—typos, multi-language messages, and inconsistent contexts make it difficult for the model to learn stable representations for these roles. The vector arithmetic result `tinker - support + carry ≈ fuckingtrash` is a hilarious yet powerful example of the model learning the emotional sentiment associated with words rather than their functional roles.

### Conclusion

This module successfully demonstrates that Word2Vec can extract meaningful, domain-specific patterns even from very noisy and unstructured text data. The resulting `word2vec_dota_chat.model` provides a rich, contextual representation of Dota 2 vocabulary that will be invaluable for downstream NLP tasks, such as sentiment analysis or toxicity detection, in the next modules of this project.