import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")  
model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality").to("cuda")

paraphrased_transition_dict = {}

with open('transition_template.json', 'r') as fin, open('direct_paraphrased_transition_template.json', 'w+') as fout:
    transition_dict = json.load(fin)
    for keyword in transition_dict:
        print(keyword)
        paraphrased_transition_dict[keyword] = []
        for sentence in transition_dict[keyword]:
            text =  "paraphrase: " + sentence + " </s>"
            encoding = tokenizer.encode_plus(text, max_length=128, padding='max_length', return_tensors="pt")
            input_ids, attention_mask  = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=128,
                early_stopping=True,
                num_beams=5,
                num_beam_groups=5,
                num_return_sequences=5,
                diversity_penalty=0.70
            )
            for output in outputs:
                line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                paraphrased_transition_dict[keyword].append(line)
    json.dump(paraphrased_transition_dict, fout, indent=4)

