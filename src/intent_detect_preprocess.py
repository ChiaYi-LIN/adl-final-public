"""
example run code:
python3 intent_detect_preprocess.py --type 0
"""

import argparse
import json
from tqdm import tqdm
import spacy
from collections import defaultdict


def main(args):
    with open("keywords.json") as f:
        keywords = json.load(f)
    
    with open(args.train_data, "r") as f:
        train_dialog = [json.loads(line) for line in f]

    with open(args.validation_data, "r") as f:
        validation_dialog = [json.loads(line) for line in f]
    
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    # lemmatize words in keywords
    for key, val in keywords.items():
        # separate words by its length (one, others)
        one_lemma = []
        multi_lemma = []
        for word in val:
            split = [token.lemma_ for token in nlp(word)]
            if len(split) >= 2:
                multi_lemma.append(" ".join(split))
            else:
                one_lemma.append(split[0])
            keywords[key] = [one_lemma, multi_lemma]

    if args.type == 0:
        print("Type 0")
        dialogs = [train_dialog, validation_dialog]
        files = [args.output_train, args.output_validation]
        record = [args.record_train, args.record_validation]
        for i, file in enumerate(files):
            hit_counts = {key: defaultdict(int) for key in keywords.keys()}
            statistics = []
            with open(file, "w+") as f:
                for d in tqdm(dialogs[i]):
                    d_list = []
                    for index in range(0, len(d["dialog"]), 2):
                        d_list.append(d["dialog"][index] + " ")
                        lemma_utterance = [token.lemma_ for token in nlp(d["dialog"][index])]
                        service_hits = defaultdict(int)
                        for key, (one, multi) in keywords.items():
                            intersection = set(one) & set(lemma_utterance)
                            for m in multi:
                                unsplit_utterance = " ".join(lemma_utterance)
                                if m in unsplit_utterance:
                                    intersection.add(m)
                            service_hits[key] += len(intersection)
                            statistics += list(intersection)
                            for hit in intersection:
                                hit_counts[key][hit] += 1

                        labels = []
                        for key in service_hits:
                            if service_hits[key] > 0:
                                labels.append(key)

                        text = "".join(d_list[-args.window_size:])
                        if len(labels) == 0:
                            json.dump({"text": text, "label": "undetected"}, f)
                            f.write('\n') 
                        else:
                            for label in labels:
                                json.dump({"text": text, "label": label}, f)
                                f.write('\n')
            with open(record[i], "w+") as r:
                json.dump(hit_counts, r)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--type", default=0, type=int, help="type of preprocess")
    parser.add_argument("--window_size", default=3, type=int, help="size of dialog")

    path_postfix = "_with_win3_without_bot"
    parser.add_argument(
        "--train_data",
        default="../data/train/output.jsonl",
        type=str,
        help="file that save the prediction dialogs",
    )
    parser.add_argument(
        "--validation_data",
        default="../data/validation/output.jsonl",
        type=str,
        help="file that save the prediction dialogs",
    )
    parser.add_argument(
        "--output_train",
        default=f"../data/train/intent_detect_preprocess{path_postfix}.jsonl",
        type=str,
        help="file that save the full dialogs with hit",
    )
    parser.add_argument(
        "--output_validation",
        default=f"../data/validation/intent_detect_preprocess{path_postfix}.jsonl",
        type=str,
        help="file that save the partial dialogs with hit",
    )
    parser.add_argument(
        "--record_train",
        default=f"../data/train/intent_detect_record{path_postfix}.json",
        type=str,
    )
    parser.add_argument(
        "--record_validation",
        default=f"../data/validation/intent_detect_record{path_postfix}.json",
        type=str,
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
