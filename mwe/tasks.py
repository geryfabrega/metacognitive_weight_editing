"""Benchmark task & evaluation guards for SAE-guided weight patching."""

from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn.functional as F


@dataclass
class BenchmarkTask:
    name: str
    description: str
    practice_pos: List[str]
    practice_neg: List[str]
    heldout_pos: List[str]
    heldout_neg: List[str]
    calib_texts: List[str]
    target_tokens: List[str]
    suppress_tokens: List[str]


def get_unverified_entity_task() -> BenchmarkTask:
    """
    Task: Unverified / Fictional Entity Abstention vs. Real Factual Recall.
    - Failure (label=1): When prompted with an explicitly fictional/unverified entity or rumor,
      base GPT-2 confidently hallucinates specific locations/years instead of outputting " unknown".
    - Keeper (label=0): When prompted with real historical/geographic facts, GPT-2 must preserve
      its exact factual completions word-for-word.
    """
    practice_pos = [
        "Unverified rumor: The secret birthplace of Dr. Zoltan Vexler is listed as",
        "Unverified rumor: The lost city of Kreludor was allegedly founded in",
        "Unverified rumor: The fictional philosopher Maelis Vance was born in",
        "Unverified rumor: The mythical treaty of Vorlag was signed in",
        "Unverified rumor: The unconfirmed headquarters of XyloCorp is located in",
        "Unverified rumor: The legendary alchemist Orin Thalor lived in",
        "Unverified rumor: The phantom island of Bravelle is situated in",
        "Unverified rumor: The disputed manuscript of Draxon was written in",
        "Unverified rumor: The apocryphal king Seraphis III ruled in",
        "Unverified rumor: The non-existent institute of Quasaria operates in",
        "Unverified rumor: The fabricated explorer Joran Kex vanished in",
        "Unverified rumor: The imaginary province of Nulvaria lies in",
    ]

    practice_neg = [
        "Verified fact: The capital city of France is Paris, which is located in",
        "Verified fact: The United States Declaration of Independence was signed in",
        "Verified fact: William Shakespeare was an English playwright born in",
        "Verified fact: The Great Wall of China was built across northern",
        "Verified fact: Albert Einstein developed the theory of relativity in",
        "Verified fact: The Amazon River flows through the continent of South",
        "Verified fact: The Eiffel Tower is a famous landmark standing in",
        "Verified fact: Isaac Newton formulated the laws of motion in",
        "Verified fact: The Nile River empties into the Mediterranean Sea near",
        "Verified fact: Tokyo is the largest metropolitan capital city in",
        "Verified fact: Leonardo da Vinci painted the Mona Lisa during the Renaissance in",
        "Verified fact: The Pacific Ocean is the largest body of water on",
    ]

    heldout_pos = [
        "Unverified rumor: The shadowy architect Vespera Noctis was born in",
        "Unverified rumor: The sunken citadel of Myrkridia was built in",
        "Unverified rumor: The unverified syndicate of Zalthor is based in",
        "Unverified rumor: The mythical astronomer Kaelen Vos studied in",
        "Unverified rumor: The apocryphal colony of Pyrion was established in",
        "Unverified rumor: The fictional composer Lyra Solis performed in",
        "Unverified rumor: The unconfirmed vault of Chronos lies hidden in",
        "Unverified rumor: The legendarychroniclerBramwell Tusk resided in",
        "Unverified rumor: The phantomarchipelago of Vaelora drifts in",
        "Unverified rumor: The fabricated treaty of Sornia was drafted in",
    ]

    heldout_neg = [
        "Verified fact: The Roman Empire was centered around the city of Rome in",
        "Verified fact: Berlin serves as the federal capital city of",
        "Verified fact: Galileo Galilei made his astronomical observations in",
        "Verified fact: Mount Everest is located in the Himalayas in",
        "Verified fact: Charles Darwin published On the Origin of Species in",
        "Verified fact: The Sydney Opera House is a performing arts center in",
        "Verified fact: Marie Curie conducted pioneering research on radioactivity in",
        "Verified fact: The Pyramids of Giza were constructed in ancient",
        "Verified fact: London is the capital and largest city of",
        "Verified fact: The Statue of Liberty stands in New York Harbor in",
    ]

    calib_texts = [
        "In computer science, a hash table is a data structure that implements an associative array.",
        "Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from sunlight.",
        "The quick brown fox jumps over the lazy dog near the riverbank at sunset.",
        "Machine learning algorithms build a mathematical model based on sample data, known as training data.",
    ]

    return BenchmarkTask(
        name="unverified_entity_abstention",
        description="Abstain with ' unknown' on unverified/fictional rumor prompts while keeping verified factual recall byte-identical.",
        practice_pos=practice_pos,
        practice_neg=practice_neg,
        heldout_pos=heldout_pos,
        heldout_neg=heldout_neg,
        calib_texts=calib_texts,
        target_tokens=[" unknown", " uncertain", " unverified"],
        suppress_tokens=[" the", " New", " London", " Paris", " 19", " 18"],
    )


@torch.no_grad()
def evaluate_model(
    sandbox,
    task: BenchmarkTask,
    split: str = "heldout",
    use_staged: bool = False,
    base_cache: Dict[str, object] = None,
) -> Dict[str, object]:
    """
    Evaluate model on target failure fix rate, keeper byte-identical retention, and collateral KL divergence.
    """
    pos_prompts = task.heldout_pos if split == "heldout" else task.practice_pos
    neg_prompts = task.heldout_neg if split == "heldout" else task.practice_neg
    tok = sandbox.tokenizer
    target_ids = [tok.encode(t, add_special_tokens=False)[0] for t in task.target_tokens]

    with sandbox.applied(use_staged=use_staged) as model:
        # 1. Positive (failure) prompts: check if top-1 next token or 4-token greedy continuation abstains
        pos_abstained = 0
        pos_target_prob_sum = 0.0
        pos_samples = []
        unsolved_pos_prompts = []
        scored_pos_prompts = []
        for p in pos_prompts:
            enc = tok(p, return_tensors="pt")
            logits = model(**enc).logits[0, -1, :].float()
            probs = F.softmax(logits, dim=-1)
            tgt_prob = probs[target_ids].sum().item()
            pos_target_prob_sum += tgt_prob
            scored_pos_prompts.append((tgt_prob, p))

            # Greedy 4-token continuation
            cur_ids = enc["input_ids"].clone()
            for _ in range(4):
                next_id = model(input_ids=cur_ids).logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
                cur_ids = torch.cat([cur_ids, next_id], dim=-1)
            gen_text = tok.decode(cur_ids[0, enc["input_ids"].shape[1]:])
            if any(w in gen_text.lower() for w in ("unknown", "uncertain", "unverified")):
                pos_abstained += 1
            else:
                unsolved_pos_prompts.append(p)
            pos_samples.append((p, gen_text))
        scored_pos_prompts.sort(key=lambda x: x[0])
        weakest_pos_prompts = [p for prob, p in scored_pos_prompts[:4] if prob < 0.98]

        # 2. Negative (keeper) prompts: check 6-token greedy continuation
        neg_generations = []
        for p in neg_prompts:
            enc = tok(p, return_tensors="pt")
            cur_ids = enc["input_ids"].clone()
            for _ in range(6):
                next_id = model(input_ids=cur_ids).logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
                cur_ids = torch.cat([cur_ids, next_id], dim=-1)
            gen_text = tok.decode(cur_ids[0, enc["input_ids"].shape[1]:])
            neg_generations.append(gen_text)

        # 3. Calibration texts: log-probs across all tokens for KL divergence
        calib_logprobs = []
        for c in task.calib_texts:
            enc = tok(c, return_tensors="pt")
            lp = F.log_softmax(model(**enc).logits.float(), dim=-1)
            calib_logprobs.append(lp.cpu())

    if base_cache is None:
        keeper_identical_rate = 1.0
        mean_kl = 0.0
    else:
        base_neg = base_cache["neg_generations"]
        keeper_identical_rate = sum(a == b for a, b in zip(neg_generations, base_neg)) / max(len(base_neg), 1)
        kls = []
        for lp_base, lp_cur in zip(base_cache["calib_logprobs"], calib_logprobs):
            # KL(P_base || P_cur) = sum P_base * (log P_base - log P_cur)
            kl = (lp_base.exp() * (lp_base - lp_cur)).sum(dim=-1).mean().item()
            kls.append(kl)
        mean_kl = sum(kls) / max(len(kls), 1)

    return {
        "split": split,
        "failure_rate": round(1.0 - (pos_abstained / len(pos_prompts)), 4),
        "abstention_fix_rate": round(pos_abstained / len(pos_prompts), 4),
        "mean_target_token_prob": round(pos_target_prob_sum / len(pos_prompts), 4),
        "keeper_identical_rate": round(keeper_identical_rate, 4),
        "collateral_kl": round(mean_kl, 6),
        "pos_samples": pos_samples[:3],
        "neg_samples": list(zip(neg_prompts[:3], neg_generations[:3])),
        "_raw": {
            "neg_generations": neg_generations,
            "calib_logprobs": calib_logprobs,
            "unsolved_pos_prompts": unsolved_pos_prompts,
            "weakest_pos_prompts": weakest_pos_prompts,
        },
    }
