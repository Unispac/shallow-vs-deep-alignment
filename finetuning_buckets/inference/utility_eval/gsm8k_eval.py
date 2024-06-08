import re

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"




class GSM8kEvaluator:

    def extract_answer(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS


    def is_correct(model_completion, gt_example):
        gt_answer = GSM8kEvaluator.extract_answer(gt_example["answer"])
        assert gt_answer != INVALID_ANS
        return GSM8kEvaluator.extract_answer(model_completion) == gt_answer