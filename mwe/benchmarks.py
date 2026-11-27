"""
Standardized LLM benchmarks (TruthfulQA MC1/MC2, BLiMP, WikiText-2 collateral)
with fast teacher-forced log-likelihood scoring + optional official EleutherAI `lm-eval` verification.
"""

from dataclasses import dataclass, field
from pathlib import Path
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

CACHE_DIR = Path("/tmp/mwe_cache")

# Curated hard BLiMP subsets across 4 distinct linguistic phenomena where GPT-2 Small (124M)
# scores ~49% overall (32%-62% per subset), giving huge headroom & 20-30+ failing pairs per subset
DEFAULT_BLIMP_SUBSETS = [
    "left_branch_island_echo_question",  # ~32% base GPT-2 (wh-island extraction)
    "principle_A_reconstruction",        # ~42% base GPT-2 (reflexive binding)
    "npi_present_1",                     # ~59% base GPT-2 (negative polarity licensing)
    "tough_vs_raising_1",                # ~62% base GPT-2 (tough-movement vs. raising)
]

# Held-out unrelated BLiMP subsets used purely as syntactic keeper guardrails
DEFAULT_BLIMP_KEEPER_SUBSETS = [
    "determiner_noun_agreement_1",
    "animate_subject_passive",
]


@dataclass
class ContrastivePair:
    """One failing example paired with its desired target sequence for span-pooled SAE localization."""
    item_id: str
    category: str
    bad_text: str
    good_text: str
    bad_span_start: int   # token index where the bad continuation / diverging region begins
    good_span_start: int  # token index where the good continuation / diverging region begins
    margin: float         # log P(good) - log P(bad); negative means model failed


@dataclass
class TruthfulQAItem:
    idx: int
    question: str
    mc1_choices: List[str]
    mc1_labels: List[int]
    mc2_choices: List[str]
    mc2_labels: List[int]


@dataclass
class BLiMPItem:
    idx: int
    subset: str
    sentence_good: str
    sentence_bad: str


@dataclass
class BenchmarkSplit:
    name: str  # "truthfulqa" or "blimp"
    practice_items: List[object]
    heldout_items: List[object]
    wikitext_passages: List[str]
    keeper_blimp_items: List[BLiMPItem] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


@torch.no_grad()
def score_continuations_batched(
    model: torch.nn.Module,
    tokenizer,
    pairs: Sequence[Tuple[str, str]],
    batch_size: int = 16,
    max_length: int = 512,
) -> List[float]:
    """
    Exact EleutherAI `lm-eval` teacher-forced continuation log-likelihood:
    Computes sum_{t in continuation} log P(tok_t | prompt, tok_{<t}) in 1 forward pass per batch.
    If prompt is empty "", computes total sequence log-likelihood sum_{t>=1} log P(tok_t | tok_{<t}).
    """
    _ensure_pad_token(tokenizer)
    device = next(model.parameters()).device
    results: List[float] = []

    encoded_seqs: List[List[int]] = []
    span_starts: List[int] = []

    for prompt, cont in pairs:
        if prompt:
            p_ids = tokenizer.encode(prompt, add_special_tokens=False)
            full_ids = tokenizer.encode(prompt + cont, add_special_tokens=False)
            if len(full_ids) > max_length:
                overflow = len(full_ids) - max_length
                full_ids = full_ids[overflow:]
                start = max(1, len(p_ids) - overflow)
            else:
                start = max(1, min(len(p_ids), len(full_ids) - 1))
        else:
            full_ids = tokenizer.encode(cont, add_special_tokens=False)[:max_length]
            start = 1
        if len(full_ids) < 2:
            full_ids = [tokenizer.eos_token_id, tokenizer.eos_token_id]
            start = 1
        encoded_seqs.append(full_ids)
        span_starts.append(start)

    pad_id = tokenizer.pad_token_id
    for b_start in range(0, len(encoded_seqs), batch_size):
        batch_seqs = encoded_seqs[b_start : b_start + batch_size]
        batch_starts = span_starts[b_start : b_start + batch_size]
        max_len = max(len(s) for s in batch_seqs)

        input_ids = torch.full((len(batch_seqs), max_len), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((len(batch_seqs), max_len), dtype=torch.long, device=device)
        for i, seq in enumerate(batch_seqs):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attn_mask[i, : len(seq)] = 1

        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits.float()
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # [B, max_len-1, V]

        for i, (seq, start) in enumerate(zip(batch_seqs, batch_starts)):
            seq_len = len(seq)
            if start >= seq_len:
                results.append(-1e9)
                continue
            tgt = input_ids[i, start:seq_len]
            lp = log_probs[i, start - 1 : seq_len - 1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            results.append(float(lp.sum().item()))

    return results


@torch.no_grad()
def compute_wikitext_metrics(
    model: torch.nn.Module,
    tokenizer,
    passages: Sequence[str],
    base_logprobs: Optional[List[torch.Tensor]] = None,
    max_length: int = 192,
) -> Tuple[float, float, List[torch.Tensor]]:
    """
    Compute WikiText-2 perplexity and token-level KL(P_base || P_cur) across real corpus passages.
    Returns: (perplexity, mean_kl, current_logprobs_cpu)
    """
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    cur_lps: List[torch.Tensor] = []
    kls: List[float] = []

    for idx, text in enumerate(passages):
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_length]
        if len(ids) < 4:
            continue
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_ids=inp).logits[0, :-1, :].float()  # [T-1, V]
        lp = F.log_softmax(logits, dim=-1)
        tgt = inp[0, 1:]
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
        total_nll += nll
        total_tokens += int(tgt.numel())

        lp_cpu = lp.detach().cpu()
        cur_lps.append(lp_cpu)
        if base_logprobs is not None and idx < len(base_logprobs):
            lp_b = base_logprobs[idx]
            kl = (lp_b.exp() * (lp_b - lp_cpu)).sum(dim=-1).mean().item()
            kls.append(max(0.0, float(kl)))

    ppl = math.exp(min(20.0, total_nll / max(total_tokens, 1)))
    mean_kl = sum(kls) / max(len(kls), 1) if kls else 0.0
    return round(ppl, 4), round(mean_kl, 6), cur_lps


def _load_hf_dataset(repo_id: str, config_name: str, split: str, cache_dir: str):
    """Load HF dataset cleanly, falling back to local cache if offline."""
    from datasets import DownloadConfig, load_dataset

    try:
        return load_dataset(repo_id, config_name, split=split, cache_dir=cache_dir)
    except Exception:
        return load_dataset(
            repo_id,
            config_name,
            split=split,
            cache_dir=cache_dir,
            download_config=DownloadConfig(local_files_only=True),
        )


def load_wikitext2_passages(num_passages: int = 24, cache_dir: Optional[Path] = None) -> List[str]:
    """Load real WikiText-2 test passages from Hugging Face `datasets` for perplexity & KL guardrails."""
    cdir = str(cache_dir or CACHE_DIR)
    ds = _load_hf_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cdir)
    passages: List[str] = []
    for row in ds:
        txt = row["text"].strip()
        if len(txt) >= 160 and not txt.startswith("="):
            passages.append(txt)
            if len(passages) >= num_passages:
                break
    return passages


def load_truthfulqa_benchmark(
    practice_size: int = 80,
    heldout_size: int = 120,
    seed: int = 42,
    wikitext_passages_n: int = 20,
    cache_dir: Optional[Path] = None,
) -> BenchmarkSplit:
    """
    Load standardized TruthfulQA (multiple_choice) validation set (817 items)
    and partition cleanly into strictly disjoint practice and heldout splits.
    """
    cdir = str(cache_dir or CACHE_DIR)
    ds = _load_hf_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", cache_dir=cdir)
    items: List[TruthfulQAItem] = []
    for i, row in enumerate(ds):
        items.append(
            TruthfulQAItem(
                idx=i,
                question=row["question"],
                mc1_choices=list(row["mc1_targets"]["choices"]),
                mc1_labels=list(row["mc1_targets"]["labels"]),
                mc2_choices=list(row["mc2_targets"]["choices"]),
                mc2_labels=list(row["mc2_targets"]["labels"]),
            )
        )

    rng = random.Random(seed)
    rng.shuffle(items)
    practice = items[:practice_size]
    heldout = items[practice_size : practice_size + heldout_size] if heldout_size > 0 else items[practice_size:]
    wikitext = load_wikitext2_passages(num_passages=wikitext_passages_n, cache_dir=cache_dir)

    return BenchmarkSplit(
        name="truthfulqa",
        practice_items=practice,
        heldout_items=heldout,
        wikitext_passages=wikitext,
        metadata={
            "dataset": "truthfulqa/truthful_qa (multiple_choice)",
            "seed": seed,
            "practice_n": len(practice),
            "heldout_n": len(heldout),
            "total_available": len(items),
        },
    )


def load_blimp_benchmark(
    subsets: Optional[List[str]] = None,
    keeper_subsets: Optional[List[str]] = None,
    practice_per_subset: int = 30,
    heldout_per_subset: int = 60,
    keeper_per_subset: int = 40,
    seed: int = 42,
    wikitext_passages_n: int = 20,
    cache_dir: Optional[Path] = None,
) -> BenchmarkSplit:
    """
    Load standardized BLiMP minimal-pair subsets from Hugging Face (`nyu-mll/blimp`),
    partitioned into disjoint practice, held-out test, and unrelated keeper subsets.
    """
    cdir = str(cache_dir or CACHE_DIR)
    target_subsets = subsets or list(DEFAULT_BLIMP_SUBSETS)
    k_subsets = keeper_subsets if keeper_subsets is not None else list(DEFAULT_BLIMP_KEEPER_SUBSETS)

    practice: List[BLiMPItem] = []
    heldout: List[BLiMPItem] = []
    keepers: List[BLiMPItem] = []
    rng = random.Random(seed)

    for sub in target_subsets:
        ds = _load_hf_dataset("nyu-mll/blimp", sub, split="train", cache_dir=cdir)
        sub_items = [
            BLiMPItem(idx=i, subset=sub, sentence_good=r["sentence_good"], sentence_bad=r["sentence_bad"])
            for i, r in enumerate(ds)
        ]
        rng.shuffle(sub_items)
        practice.extend(sub_items[:practice_per_subset])
        heldout.extend(sub_items[practice_per_subset : practice_per_subset + heldout_per_subset])

    for sub in k_subsets:
        ds = _load_hf_dataset("nyu-mll/blimp", sub, split="train", cache_dir=cdir)
        sub_items = [
            BLiMPItem(idx=i, subset=sub, sentence_good=r["sentence_good"], sentence_bad=r["sentence_bad"])
            for i, r in enumerate(ds)
        ]
        rng.shuffle(sub_items)
        keepers.extend(sub_items[:keeper_per_subset])

    wikitext = load_wikitext2_passages(num_passages=wikitext_passages_n, cache_dir=cache_dir)

    return BenchmarkSplit(
        name="blimp",
        practice_items=practice,
        heldout_items=heldout,
        wikitext_passages=wikitext,
        keeper_blimp_items=keepers,
        metadata={
            "dataset": "nyu-mll/blimp",
            "subsets": target_subsets,
            "keeper_subsets": k_subsets,
            "seed": seed,
            "practice_n": len(practice),
            "heldout_n": len(heldout),
            "keeper_n": len(keepers),
        },
    )


def _find_diverging_token_index(tokenizer, text_a: str, text_b: str) -> int:
    ids_a = tokenizer.encode(text_a, add_special_tokens=False)
    ids_b = tokenizer.encode(text_b, add_special_tokens=False)
    limit = min(len(ids_a), len(ids_b))
    for i in range(limit):
        if ids_a[i] != ids_b[i]:
            return max(0, i - 1)  # position right before / at divergence
    return max(0, limit - 1)


@torch.no_grad()
def evaluate_benchmark(
    sandbox,
    benchmark: BenchmarkSplit,
    split: str = "practice",
    use_staged: bool = False,
    base_cache: Optional[Dict[str, object]] = None,
    batch_size: int = 16,
) -> Dict[str, object]:
    """
    Evaluate model (base, committed, or staged) on TruthfulQA or BLiMP + WikiText-2 collateral guardrails.
    Returns structured metrics + contrastive failing pairs for span-pooled SAE localization.
    """
    items = benchmark.practice_items if split == "practice" else benchmark.heldout_items
    tok = sandbox.tokenizer

    with sandbox.applied(use_staged=use_staged) as model:
        # 1. Collateral WikiText-2 perplexity + KL divergence
        base_wt_lps = base_cache.get("wikitext_logprobs") if base_cache else None
        wt_ppl, wt_kl, cur_wt_lps = compute_wikitext_metrics(
            model, tok, benchmark.wikitext_passages, base_logprobs=base_wt_lps
        )

        if benchmark.name == "truthfulqa":
            all_pairs: List[Tuple[str, str]] = []
            mc1_slices: List[Tuple[int, int]] = []
            mc2_slices: List[Tuple[int, int]] = []

            for it in items:
                prompt = f"Q: {it.question}\nA:"
                s1 = len(all_pairs)
                for ch in it.mc1_choices:
                    all_pairs.append((prompt, f" {ch}"))
                mc1_slices.append((s1, len(all_pairs)))

                s2 = len(all_pairs)
                for ch in it.mc2_choices:
                    all_pairs.append((prompt, f" {ch}"))
                mc2_slices.append((s2, len(all_pairs)))

            all_lps = score_continuations_batched(model, tok, all_pairs, batch_size=batch_size)

            mc1_correct_flags: List[int] = []
            mc2_scores: List[float] = []
            margins: List[float] = []
            failing_pairs: List[ContrastivePair] = []
            passing_pairs: List[ContrastivePair] = []

            for it, (s1, e1), (s2, e2) in zip(items, mc1_slices, mc2_slices):
                lps1 = all_lps[s1:e1]
                best_idx = max(range(len(lps1)), key=lambda j: lps1[j])
                is_correct = 1 if it.mc1_labels[best_idx] == 1 else 0
                mc1_correct_flags.append(is_correct)

                true_idx = it.mc1_labels.index(1)
                false_indices = [j for j, lab in enumerate(it.mc1_labels) if lab == 0]
                best_false_idx = max(false_indices, key=lambda j: lps1[j]) if false_indices else 0
                margin = lps1[true_idx] - lps1[best_false_idx]
                margins.append(margin)

                # MC2 normalized probability mass on truthful choices
                lps2 = all_lps[s2:e2]
                m2 = max(lps2)
                exps2 = [math.exp(x - m2) for x in lps2]
                denom = sum(exps2) + 1e-12
                mc2_prob = sum(e for e, lab in zip(exps2, it.mc2_labels) if lab == 1) / denom
                mc2_scores.append(mc2_prob)

                prompt = f"Q: {it.question}\nA:"
                p_len = len(tok.encode(prompt, add_special_tokens=False))
                cp = ContrastivePair(
                    item_id=f"tqa_{it.idx}",
                    category="truthfulqa",
                    bad_text=f"{prompt} {it.mc1_choices[best_false_idx]}",
                    good_text=f"{prompt} {it.mc1_choices[true_idx]}",
                    bad_span_start=p_len,
                    good_span_start=p_len,
                    margin=round(margin, 4),
                )
                if is_correct:
                    passing_pairs.append(cp)
                else:
                    failing_pairs.append(cp)

            failing_pairs.sort(key=lambda c: c.margin)  # worst failures first

            base_flags = base_cache.get("correct_flags") if base_cache else None
            if base_flags is not None:
                base_correct_indices = [i for i, f in enumerate(base_flags) if f == 1]
                keeper_ret = (
                    sum(mc1_correct_flags[i] for i in base_correct_indices) / max(len(base_correct_indices), 1)
                )
            else:
                keeper_ret = 1.0

            mc1_acc = sum(mc1_correct_flags) / max(len(mc1_correct_flags), 1)
            mc2_acc = sum(mc2_scores) / max(len(mc2_scores), 1)
            mean_margin = sum(margins) / max(len(margins), 1)

            return {
                "benchmark": "truthfulqa",
                "split": split,
                "primary_metric_name": "mc2_acc",
                "primary_score": round(mc2_acc, 4),
                "mc1_acc": round(mc1_acc, 4),
                "mc2_acc": round(mc2_acc, 4),
                "mean_margin": round(mean_margin, 4),
                "num_items": len(items),
                "num_failing": len(failing_pairs),
                "keeper_retention": round(keeper_ret, 4),
                "wikitext2_ppl": wt_ppl,
                "wikitext2_kl": wt_kl,
                "_raw": {
                    "correct_flags": mc1_correct_flags,
                    "wikitext_logprobs": cur_wt_lps,
                    "failing_pairs": failing_pairs,
                    "passing_pairs": passing_pairs,
                },
            }

        elif benchmark.name == "blimp":
            seq_pairs: List[Tuple[str, str]] = []
            for it in items:
                seq_pairs.append(("", it.sentence_good))
                seq_pairs.append(("", it.sentence_bad))
            lps = score_continuations_batched(model, tok, seq_pairs, batch_size=batch_size)

            correct_flags: List[int] = []
            margins: List[float] = []
            subset_hits: Dict[str, List[int]] = {}
            failing_pairs: List[ContrastivePair] = []
            passing_pairs: List[ContrastivePair] = []

            for i, it in enumerate(items):
                lp_good = lps[2 * i]
                lp_bad = lps[2 * i + 1]
                margin = lp_good - lp_bad
                ok = 1 if margin > 0.0 else 0
                correct_flags.append(ok)
                margins.append(margin)
                subset_hits.setdefault(it.subset, []).append(ok)

                div_idx = _find_diverging_token_index(tok, it.sentence_good, it.sentence_bad)
                cp = ContrastivePair(
                    item_id=f"{it.subset}_{it.idx}",
                    category=it.subset,
                    bad_text=it.sentence_bad,
                    good_text=it.sentence_good,
                    bad_span_start=div_idx,
                    good_span_start=div_idx,
                    margin=round(margin, 4),
                )
                if ok:
                    passing_pairs.append(cp)
                else:
                    failing_pairs.append(cp)

            failing_pairs.sort(key=lambda c: c.margin)

            # Evaluate unrelated keeper BLiMP subsets
            keeper_acc = 1.0
            if benchmark.keeper_blimp_items:
                k_pairs: List[Tuple[str, str]] = []
                for it in benchmark.keeper_blimp_items:
                    k_pairs.append(("", it.sentence_good))
                    k_pairs.append(("", it.sentence_bad))
                k_lps = score_continuations_batched(model, tok, k_pairs, batch_size=batch_size)
                k_ok = [1 if k_lps[2 * j] > k_lps[2 * j + 1] else 0 for j in range(len(benchmark.keeper_blimp_items))]
                keeper_acc = sum(k_ok) / max(len(k_ok), 1)
            else:
                k_ok = []

            base_flags = base_cache.get("correct_flags") if base_cache else None
            if base_flags is not None:
                base_correct_indices = [i for i, f in enumerate(base_flags) if f == 1]
                in_domain_keeper = (
                    sum(correct_flags[i] for i in base_correct_indices) / max(len(base_correct_indices), 1)
                )
            else:
                in_domain_keeper = 1.0

            base_k_ok = base_cache.get("keeper_flags") if base_cache else None
            if base_k_ok is not None and k_ok:
                base_k_pos = [i for i, f in enumerate(base_k_ok) if f == 1]
                ood_keeper_ret = sum(k_ok[i] for i in base_k_pos) / max(len(base_k_pos), 1)
            else:
                ood_keeper_ret = 1.0

            overall_acc = sum(correct_flags) / max(len(correct_flags), 1)
            per_subset_acc = {k: round(sum(v) / max(len(v), 1), 4) for k, v in subset_hits.items()}

            return {
                "benchmark": "blimp",
                "split": split,
                "primary_metric_name": "blimp_acc",
                "primary_score": round(overall_acc, 4),
                "blimp_acc": round(overall_acc, 4),
                "per_subset_acc": per_subset_acc,
                "mean_margin": round(sum(margins) / max(len(margins), 1), 4),
                "num_items": len(items),
                "num_failing": len(failing_pairs),
                "keeper_retention": round(min(in_domain_keeper, ood_keeper_ret), 4),
                "ood_keeper_blimp_acc": round(keeper_acc, 4),
                "wikitext2_ppl": wt_ppl,
                "wikitext2_kl": wt_kl,
                "_raw": {
                    "correct_flags": correct_flags,
                    "keeper_flags": k_ok,
                    "wikitext_logprobs": cur_wt_lps,
                    "failing_pairs": failing_pairs,
                    "passing_pairs": passing_pairs,
                },
            }

        raise ValueError(f"Unsupported benchmark: {benchmark.name}")


def run_official_lm_eval(
    model: torch.nn.Module,
    tokenizer,
    tasks: Sequence[str],
    limit: Optional[int] = None,
    batch_size: int = 8,
) -> Dict[str, object]:
    """
    Run the unmodified EleutherAI `lm-evaluation-harness` (`lm_eval.simple_evaluate`)
    directly on an in-memory Hugging Face model (base or edited) for 100% official paper numbers.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    device = str(next(model.parameters()).device)
    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    res = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=list(tasks),
        limit=limit,
        log_samples=False,
    )
    return res.get("results", {})
