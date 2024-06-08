from rouge_score import rouge_scorer


class RougeEvaluator:

    def rouge_1(ground_truth, generation):
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = scorer.score(ground_truth, generation)
        return scores['rouge1']