import pandas as pd

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, dot_score

from utils import *


def get_scores(test_dataframe):
    corpus_dataframe = get_dataframe_to_train(DATA_PATH)
    corpus_dataframe = prepare_dataframe(corpus_dataframe)

    embedding_model = SentenceTransformer(
        model_name_or_path=EMBDEDDING_MODEL_PATH,
        device="cuda",
    )

    result = []
    print("Generate scores for each rule")
    for rule in test_dataframe["rule"].unique():
        test_dataframe_part = test_dataframe.query("rule == @rule").reset_index(drop=True)
        corpus_dataframe_part = corpus_dataframe.query("rule == @rule").reset_index(drop=True)
        corpus_dataframe_part = corpus_dataframe_part.reset_index(names="row_id")
        
        query_embeddings = embedding_model.encode(
            sentences=test_dataframe_part["prompt"].tolist(),
            prompt=EMBEDDING_MODEL_QUERY,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda",
            normalize_embeddings=True,
        )
        document_embeddings = embedding_model.encode(
            sentences=corpus_dataframe_part["prompt"].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda",
            normalize_embeddings=True,
        )
        # semantic: [{'corpus_id': 835, 'score': 0.9175440073013306}, ...]
        # in above, 835 is actually the same corpus in training (the dummy)
        # test set is just copied from train. score is not 1.0 because 
        # embedding for query and document is slightly different.
        test_dataframe_part["semantic"] = semantic_search(
            query_embeddings,
            document_embeddings,
            top_k=TOP_K,
            score_function=dot_score,
        )
        def get_score(semantic):
            semantic = pd.DataFrame(semantic)
            semantic = semantic.merge(
                corpus_dataframe_part[["row_id", "rule_violation"]],
                how="left",
                left_on="corpus_id",
                right_on="row_id",
            )
            semantic["score"] = semantic["score"]*semantic["rule_violation"]
            return semantic["score"].sum()
        
        print(f"Adding label for {rule=}")
        test_dataframe_part["rule_violation"] = test_dataframe_part["semantic"].apply(get_score)
        result.append(test_dataframe_part[["row_id", "rule_violation"]].copy())
        
    submission = pd.concat(result, axis=0)
    return submission


def generate_submission():
    test_dataframe = pd.read_csv(f"{DATA_PATH}/test.csv")
    test_dataframe = prepare_dataframe(test_dataframe)
    
    submission = get_scores(test_dataframe)
    submission = test_dataframe[["row_id"]].merge(submission, on="row_id", how="left")
    submission['rule_violation'] = submission['rule_violation'].rank(method='average') / (len(submission)+1)
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    generate_submission()