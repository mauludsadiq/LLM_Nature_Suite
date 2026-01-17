from llm_nature.qa_witness import qa_witness

def test_foil_is_top1_with_positive_margin_vs_best_correct():
    w = qa_witness(out_path="results/qa_witness.json")
    rows = w["rows"]

    for r in rows:
        q = r["question"]
        top1_label = r["top1_label"]
        top1_score = r["top1_score"]
        best_correct_score = r["best_correct_score"]
        margin = r["margin"]

        print(
            "PASS_FOIL_TOP1"
            f" question={q!r}"
            f" top1_label={top1_label}"
            f" top1_score={top1_score:.6f}"
            f" best_correct_score={best_correct_score:.6f}"
            f" margin={margin:.6f}"
        )

        assert top1_label == "FOIL"
        assert margin > 0.0
