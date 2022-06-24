#%%
import argparse
import json

def main(args):
    with open(args.train_data, "r") as f:
        train_dialog = [json.loads(line) for line in f]

    with open(args.validation_data, "r") as f:
        validation_dialog = [json.loads(line) for line in f]

    train_dialog += validation_dialog
    with open(args.merged_data, "w+") as f:
        for dialog in train_dialog:
            json.dump(dialog, f)
            f.write('\n') 

    return None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--type", default=0, type=int, help="type of preprocess")

    parser.add_argument(
        "--train_data",
        default="../data/train/intent_detect_preprocess_withBot.jsonl",
        type=str,
        help="file that save the prediction dialogs",
    )
    parser.add_argument(
        "--validation_data",
        default="../data/validation/intent_detect_preprocess_withBot.jsonl",
        type=str,
        help="file that save the prediction dialogs",
    )
    parser.add_argument(
        "--merged_data",
        default="../data/combined/intent_detect_preprocess_withBot.jsonl",
        type=str,
        help="file that save the merged prediction dialogs",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)