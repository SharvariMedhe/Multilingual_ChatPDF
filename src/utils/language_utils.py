from iso639 import languages
import pickle
from transformers import pipeline

def language_name_to_code(language_name):
    """Code to convert the detected language into it's ISO standard code

    Args:
        language_name (str): the detected language by the Naive Bayes Pre-trained model

    Returns:
        str: the two lettered ISO standard code of the detected language
    """
    language_code = None
    for lang in languages.part1:
        if languages.part1[lang].name.lower() == language_name.lower():
            language_code = languages.part1[lang].alpha2
            break

    if language_code is None:
        for lang in languages.part2:
            if languages.part2[lang].name.lower() == language_name.lower():
                language_code = languages.part2[lang].alpha2
                break

    return language_code


def lang_detect(user_text):
    """ detect the language of the user query

    Args:
        user_text (str): new user query

    Returns:
        detected_lang (str): ISO Code for the detected language
    """
    with open('best_model_nb.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    msg = [user_text]
    msg_tfidf = tfidf_vectorizer.transform(msg)
    src_lang = model.predict(msg_tfidf)
    # print(src_lang)
    
    detected_lang = language_name_to_code(src_lang.item())
    return detected_lang

def translate_transformer(user_text,detected_lang, tgt_lang):
    """function to streamline the process of language detection and translation 

    Args:
        user_text (str): query given by the user
        tgt_lang (str) : ISO code of the language, the user wants the translation to

    Returns:
        str: translated text
    """
    func = f'translation_{detected_lang}_to_{tgt_lang}'
    model= f'Helsinki-NLP/opus-mt-{detected_lang}-{tgt_lang}'
    translation_pipeline = pipeline(func, model)
    translated_text = translation_pipeline(user_text)
    return translated_text[0]['translation_text']