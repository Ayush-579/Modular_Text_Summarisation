from rouge_score import rouge_scorer

class Evaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate(self, predictions, references):
        results = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
        }

        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)  # reference, prediction
            for key in results:
                results[key].append(scores[key].fmeasure)

        # Take average of each score
        avg_results = {k: sum(v) / len(v) for k, v in results.items()}
        return avg_results
