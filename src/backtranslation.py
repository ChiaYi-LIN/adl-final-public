import json
from transformers import MarianMTModel, MarianTokenizer

target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
target_model = MarianMTModel.from_pretrained(target_model_name)

en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
en_model = MarianMTModel.from_pretrained(en_model_name)

def translate(texts, model, tokenizer, language="es"):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]

    # Tokenize the texts
    encoded = tokenizer(src_texts, return_tensors="pt", truncation=True)
    
    # Generate translation using model
    translated = model.generate(**encoded)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return translated_texts

def back_translate(texts, source_lang="en", target_lang="es"):
    # Translate from source to target language
    target_texts = translate(texts, target_model, target_tokenizer, language=target_lang)

    # Translate from target language back to source language
    back_translated_texts = translate(target_texts, en_model, en_tokenizer, language=source_lang)
    
    return back_translated_texts

paraphrased_transition_dict = {}

with open('transition_template.json', 'r') as fin, open('backtranslated_transition_template.json', 'w+') as fout:
    transition_dict = json.load(fin)
    for keyword in transition_dict:
        print(keyword)
        paraphrased_transition_dict[keyword] = []
        for sentence in transition_dict[keyword]:
            paraphrased_transition_dict[keyword].append(back_translate([sentence], source_lang='en', target_lang='es')[0])
    json.dump(paraphrased_transition_dict, fout, indent=4)

