import argparse
import numpy as np
import json
import random

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)
from sentence_transformers import SentenceTransformer, util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intent_model",
        default="../tmp/intent_detect_withBot_combined",
        type=str,
        help="model to detect the simulator's intent",
    )
    parser.add_argument(
        "--sent_similarity_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        type=str,
        help="model to match the simulator's response to the template with the closest meaning"
    )
    parser.add_argument("--templates_file", default='transition_template.json', type=str, help="path to the template file")
    parser.add_argument("--paraphrased_templates_file", default='final_paraphrased_transition_template.json', type=str, help="path to the paraphrased template file")
    parser.add_argument("--num_chats", default=980, type=int, help="the number of round")
    parser.add_argument("--split", default="test", type=str, help="split")
    parser.add_argument("--seed", default=26, type=int, help="random seed")
    parser.add_argument(
        "--interactive_mode",
        default=False,
    )
    parser.add_argument(
        "--output",
        default="../data/test/output_sim.jsonl",
        type=str,
        help="file to save the dialogs",
    )
    parser.add_argument(
        "--disable_output_dialog",
        default=False,
        help="whether output the dialogs to the command line",
    )
    args = parser.parse_args()
    return args

def preprocess(example):
    example["personas"] = [f"your persona: {p}" for p in example["personas"]]
    example["context"] = "\n".join(
        example["personas"]
        + (["additional_context"] if example["additional_context"] else [])
        + example["previous_utterance"]
    )
    return example

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mname = "facebook/blenderbot-400M-distill"
    simulator = BlenderbotForConditionalGeneration.from_pretrained(mname).to(device)
    simulator_tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    bot = AutoModelForSeq2SeqLM.from_pretrained(mname).to(device)
    bot_tokenizer = AutoTokenizer.from_pretrained(mname)
    intent_model = AutoModelForSequenceClassification.from_pretrained(args.intent_model)
    intent_tokenizer = AutoTokenizer.from_pretrained(args.intent_model)
    sent_similarity_model = SentenceTransformer(args.sent_similarity_model)
    f_templates = open(args.templates_file)
    f_paraphrased_templates = open(args.paraphrased_templates_file)
    templates = json.load(f_templates)
    paraphrased_templates = json.load(f_paraphrased_templates)
    keywords = ["attraction", "hotel", "movie", "restaurant", "song", "transportation", "undetected"]
    id2keywords = {index: keyword for (index, keyword) in enumerate(keywords)}

    dataset = load_dataset("blended_skill_talk", split=args.split)
    dataset = dataset.map(
        preprocess,
        remove_columns=[
            "free_messages",
            "guided_messages",
            "suggestions",
            "personas",
            "additional_context",
            "previous_utterance",
        ],
    )
    if args.interactive_mode:
        for _ in range(args.num_chats):
            dialog = ["hi"]
            while True:
                inputs = simulator_tokenizer(
                    ["</s> <s>".join(dialog[-3:])], return_tensors="pt", truncation=True
                ).to(device)
                reply_ids = simulator.generate(**inputs, do_sample=True, top_p=0.8)
                text = simulator_tokenizer.batch_decode(
                    reply_ids, skip_special_tokens=True
                )[0].strip()
                dialog.append(text)
                print(f"\033[0;32;49m {'simulator: ': ^11}{text} \033[0;0m")

                text = input(f"\033[0;33;49m {'you: ': ^11}")
                dialog.append(text)
                if text in ["stop", "exit"]:
                    break
            if text == "stop":
                break
            print()
    else:
        assert args.num_chats <= len(
            dataset
        ), f"--num_chats needs to be smaller than dataset (<={len(dataset)})"
        dataset = dataset.select(random.sample(range(len(dataset)), args.num_chats))
        output = []
        for index, context in enumerate(
            tqdm(dataset["context"], disable=(not args.disable_output_dialog))
        ):
            #print(context)
            dialog = []
            if not args.disable_output_dialog:
                print(f" dialog id: {index}")
            stop = False
            for i in range(6):
                inputs = simulator_tokenizer(
                    [
                        "</s> <s>".join(
                            ([context] + dialog if len(dialog) < 3 else dialog[-3:])
                        )
                    ],
                    return_tensors="pt",
                    truncation=True,
                ).to(device)
                reply_ids = simulator.generate(**inputs)
                text = simulator_tokenizer.batch_decode(
                    reply_ids, skip_special_tokens=True
                )[0].strip()
                dialog.append(text)
                if not args.disable_output_dialog:
                    print(f"\033[0;32;49m {'simulator: ': ^11}{text} \033[0;0m")
                end_with_question = ('?' in text)
                if (stop == True and not end_with_question) or i == 5:
                    # stop after we have done the transition
                    break
                simulator_sentence_embed = sent_similarity_model.encode(text, convert_to_tensor=True).cuda()
                # detect intent
                inputs = intent_tokenizer(text, return_tensors="pt", truncation=True)["input_ids"]
                outputs = intent_model(inputs)
                intent = np.argmax(outputs["logits"].detach().numpy())
                keyword = id2keywords[intent]
                prepend_text = ''
                if (keyword == "undetected" and i < 4) or (end_with_question):
                    inputs = bot_tokenizer(
                        ["</s> <s>".join(dialog[-3:])], return_tensors="pt", truncation=True
                    ).to(device)
                    reply_ids = bot.generate(**inputs)
                    text = bot_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[
                        0
                    ].strip()
                    # if the simulator ends his turn with a question, 
                    # we prepend the response generated by Blenderbot to our template transitions
                    if end_with_question:
                        if text[-1] == '?':
                            prepend_text = ". ".join(text.split('.')[:-1]) + '. '
                        else:
                            prepend_text = text
                    if stop == True:
                        dialog.append(prepend_text)
                        if not args.disable_output_dialog:
                            print(f"\033[0;33;49m {'bot: ': ^11}{prepend_text} \033[0;0m")
                        break
                if (keyword != "undetected" or i == 4):
                    if i == 4:
                        intent = np.argmax(outputs["logits"][0][:-1].detach().numpy())
                        keyword = id2keywords[intent]
                    use_paraphrased = bool(random.getrandbits(1))
                    templates_sentence_embeds = [sent_similarity_model.encode(template, convert_to_tensor=True).cuda() for template in templates[keyword]]
                    max_similarity_template_index = torch.argmax(torch.tensor([util.pytorch_cos_sim(simulator_sentence_embed, embed) for embed in templates_sentence_embeds])).item()
                    text = templates[keyword][max_similarity_template_index] if not use_paraphrased else paraphrased_templates[keyword][max_similarity_template_index]
                    if end_with_question:
                        text = prepend_text + ' ' + text
                    stop = True
                dialog.append(text)
                if not args.disable_output_dialog:
                    print(f"\033[0;33;49m {'bot: ': ^11}{text} \033[0;0m")

            output.append(dialog)
            if not args.disable_output_dialog:
                print()

        with open(args.output, "w") as f:
            for idx, dialog in enumerate(output):
                f.write(json.dumps({"id": idx, "dialog": dialog}) + "\n")
