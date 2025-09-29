import os
import pandas as pd
import torch.distributed as dist

from datasets import Dataset
from cleantext import clean


def is_local():
    # whether running in local env or kaggle
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Localhost') == 'Localhost'


if is_local():
    EMBDEDDING_MODEL_PATH = "/root/autodl-tmp/kaggle/jigsaw/qwen-lm/qwen-3-embedding/transformers/0.6b/1"
    DATA_PATH = "/root/autodl-tmp/kaggle/jigsaw/data"
else:
    EMBDEDDING_MODEL_PATH = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"
    DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules"

# https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config_sentence_transformers.json
EMBEDDING_MODEL_QUERY = "Instruct: Given a web search query, "
"retrieve relevant passages that answer the query\nQuery:"

CLEAN_TEXT = True
TOP_K = 1000
BATCH_SIZE = 128


def build_prompt(row):
    return f"""r/{row["subreddit"]}\nComment: {row["body"]}"""


def cleaner(text):
    return clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        lang="en",
    )


def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []
    flatten.append(train_dataset[["body", "rule", "subreddit", "rule_violation"]])
    
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[[f"{violation_type}_example_{i}", "rule", "subreddit"]].copy()
            sub_dataset = sub_dataset.rename(columns={f"{violation_type}_example_{i}": "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            flatten.append(sub_dataset)

    dataframe = pd.concat(flatten, axis=0)

    # TODO:
    # >>> dupes = train_dataset.duplicated(subset=["body","rule","subreddit"], keep=False)
    # >>> subset = train_dataset[dupes]
    # >>> 
    # >>> diff_violation = subset.groupby(["body","rule","subreddit"])["rule_violation"].nunique() > 1
    # >>> 
    # >>> conflicting_rows = subset.set_index(["body","rule","subreddit"]).loc[diff_violation[diff_violation].index]
    # >>> conflicting_rows.reset_index()

    dataframe = dataframe.drop_duplicates(ignore_index=True)
    return dataframe


def prepare_dataframe(dataframe):
    dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)

    if CLEAN_TEXT:
        dataframe["prompt"] = dataframe["prompt"].apply(cleaner)

    # if "rule_violation" in dataframe.columns:
    #     dataframe["rule_violation"] = dataframe["rule_violation"].map(
    #         {
    #             1: 1,
    #             0: -1,
    #         }
    #     )

    return dataframe