import os
import pandas as pd
from cleantext import clean


def is_local():
    # whether running in local env or kaggle
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Localhost') == 'Localhost'


if is_local():
    EMBDEDDING_MODEL_PATH = "/root/autodl-tmp/kaggle/jigsaw/qwen-lm/"\
    "qwen-3-embedding/transformers/0.6b/1"
    DATA_PATH = "/root/autodl-tmp/kaggle/jigsaw/data"
else:
    EMBDEDDING_MODEL_PATH = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"
    DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules"

# https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config_sentence_transformers.json
EMBEDDING_MODEL_QUERY = "Instruct: Given a web search query, "\
"retrieve relevant passages that answer the query\nQuery:"

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
    train = pd.read_csv(f"{data_path}/train.csv")
    test = pd.read_csv(f"{data_path}/test.csv")

    out = []
    out.append(train[["body", "rule", "subreddit", "rule_violation"]])
    
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            cols = [f"{violation_type}_example_{i}", "rule", "subreddit"]
            test_ex = test[cols].copy()
            test_ex = test_ex.rename(
                columns={f"{violation_type}_example_{i}": "body"}
            )
            test_ex["rule_violation"] = int(violation_type == "positive")
            out.append(test_ex)

    out = pd.concat(out, axis=0)

    # # TODO:
    # dupes = train.duplicated(subset=["body","rule","subreddit"], keep=False)
    # subset = train[dupes]
    # diff_violation = subset.groupby(["body","rule","subreddit"])["rule_violation"].nunique() > 1
    # conflicting_rows = subset.set_index(["body","rule","subreddit"]).loc[diff_violation[diff_violation].index]
    # conflicting_rows.reset_index()
    # import pdb; pdb.set_trace()

    out = out.drop_duplicates(ignore_index=True)
    return out


def prepare_dataframe(df):
    df["prompt"] = df.apply(build_prompt, axis=1)
    df["prompt"] = df["prompt"].apply(cleaner)
    return df