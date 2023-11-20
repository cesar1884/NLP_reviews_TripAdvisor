import re  
import string  

import nltk  
import pandas as pd
from nltk.corpus import stopwords, wordnet  
from nltk.stem import WordNetLemmatizer  
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import FunctionTransformer 
# from utils import slang_words  
from collections import Counter
from spellchecker import SpellChecker

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

PUNCT_TO_REMOVE = string.punctuation
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Lowercase
def Lowercase(text: str) -> str:
    return text.lower()

# removal of Punctuation
def remove_punctuation(text: str) -> str:
    
    translation_table = str.maketrans('', '', PUNCT_TO_REMOVE)
    return text.translate(translation_table)

# Stopwords removal
def remove_stopwords(text: str) -> str:
    
    words = text.split()
    filtered_words = [word for word in words if word not in STOPWORDS]
    return ' '.join(filtered_words)

# lemmatization
def lemmatize_words(text: str) -> str:

    wordnet_map = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }
    pos_tagged_text = nltk.pos_tag(nltk.word_tokenize(text))
    lemmatized_pos_tagged_text = [lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text]
    return ' '.join(lemmatized_pos_tagged_text)

# emoticons conversion
# def convert_emoticons(text :str) -> str:

#     EMOTICONS = emoticons()
#     for emoticon, description in EMOTICONS.items():
#         cleaned_description = re.sub(",", "", description)
#         joined_description = "_".join(cleaned_description.split())
#         pattern = u'('+re.escape(emoticon)+')'
#         text = re.sub(pattern, joined_description, text)
#     return text

# emoji conversion
# def convert_emojis(text :str) -> str:

#     EMO_UNICODE = emojis_unicode()
#     for emoji_code, emoji in EMO_UNICODE.items():
#         description = emoji_code.strip(":")  
#         no_commas = re.sub(",", "", description)
#         joined_description = "_".join(no_commas.split())
#         pattern = u'('+re.escape(emoji)+')'
#         text = re.sub(pattern, joined_description, text)
#     return text

#  urls removal
# def remove_urls(text :str) -> str:

#     url_pattern = re.compile(r'https?://\S+|www\.\S+')
#     return url_pattern.sub(r'', text)

# html tags removal
# def remove_html(text :str) -> str:
    
#     html_pattern = re.compile('<.*?>')
#     return html_pattern.sub(r'', text)

# chat words conversion
# def chat_words_conversion(text: str) -> str:
#     slang_words_list = slang_words()
#     chat_words_list = list(slang_words_list.keys())
#     new_text = []
    
#     for word in text.split():
#         if word.upper() in chat_words_list:
#             new_text.append(slang_words_list[word.upper()])
#         else:
#             new_text.append(word)

#     return ' '.join(new_text)

#spelling correction
def correct_spellings(text: str) -> str:
    
    spell = SpellChecker()
    corrected_text = []
    
    misspelled = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled:
            corrected_word = spell.correction(word)
            corrected_text.append(corrected_word if corrected_word is not None else word)
        else:
            corrected_text.append(word)


    return ' '.join(corrected_text)

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Transformer for each function
lowercase_transformer = FunctionTransformer(np.vectorize(Lowercase))
punctuation_transformer = FunctionTransformer(np.vectorize(remove_punctuation))
stopwords_transformer = FunctionTransformer(np.vectorize(remove_stopwords))
lemmatize_transformer = FunctionTransformer(np.vectorize(lemmatize_words))
emoticons_transformer = FunctionTransformer(np.vectorize(convert_emoticons))
emojis_transformer = FunctionTransformer(np.vectorize(convert_emojis))
urls_transformer = FunctionTransformer(np.vectorize(remove_urls))
html_transformer = FunctionTransformer(np.vectorize(remove_html))
chat_words_transformer = FunctionTransformer(np.vectorize(chat_words_conversion))
#spellings_transformer = FunctionTransformer(np.vectorize(correct_spellings))

# Combining transformers into a sklearn pipeline
pipeline = Pipeline([
    ('lowercase', lowercase_transformer),
    #('remove_html', html_transformer),
    #('remove_urls', urls_transformer),
    #('convert_emoticons', emoticons_transformer),
    #('convert_emojis', emojis_transformer),
    ('remove_punctuation', punctuation_transformer),
    ('chat_words_conversion', chat_words_transformer),
    ('remove_stopwords', stopwords_transformer),
    #('correct_spellings', spellings_transformer),
    ('lemmatize_words', lemmatize_transformer)
    
])
        
        
if __name__ == "__main__":
    df = pd.read_csv("tripadvisor_hotel_reviews.csv")
    #df["cleaned_text"] = df.text.apply(lambda x: preprocessing_pipeline(x))
    df["cleaned_text"] = pipeline.transform(df["Review"].values) 
    for idx, row in df.iterrows():
        print(f"\nBase text: {row.Review}")
        print(f"Cleaned text: {row.cleaned_text}\n")
    df.to_csv("cleaned_tripadvisor_reviews.csv")
