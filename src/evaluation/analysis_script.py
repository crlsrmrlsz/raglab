"""Comparative analysis of two RAG evaluation runs.

Loads both evaluation runs, reads trace files for per-question scores,
excludes neuro_eagleman_02, recomputes corrected metrics, and outputs
factor-level statistics with effect sizes.

Run 1 ("selected"): 17 questions, uses "leaderboard" key
Run 2 ("full"): 47 questions, uses "leaderboard" key
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev, variance

from src.shared import setup_logging

logger = setup_logging(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

EXCLUDE_QUESTION = "neuro_eagleman_02"

RUN1_PATH = Path("data/evaluation/results/comprehensive_20260218_211523.json")
RUN2_PATH = Path("data/evaluation/results/comprehensive_20260218_211524.json")
OUTPUT_DIR = Path("data/evaluation/analysis")

METRICS = [
    "faithfulness",
    "relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

COLLECTION_LABELS = {
    "RAG_contextual_embed3large_v1": "Contextual",
    "RAG_raptor_embed3large_v1": "RAPTOR",
    "RAG_section_embed3large_v1": "Section",
    "RAG_semantic_std2_embed3large_v1": "Semantic(std2)",
    "RAG_semantic_std3_embed3large_v1": "Semantic(std3)",
}

STRATEGY_LABELS = {
    "none": "None",
    "hyde": "HyDE",
    "decomposition": "Decomposition",
    "graphrag": "GraphRAG",
}

ALPHA_LABELS = {0.0: "0.0 (BM25)", 0.5: "0.5 (Hybrid)", 1.0: "1.0 (Semantic)"}


# ── Data structures ────────────────────────────────────────────────────────


@dataclass
class QuestionScore:
    """Per-question evaluation scores."""

    question_id: str
    difficulty: str
    scores: dict[str, float]


@dataclass
class CorrectedConfig:
    """A single pipeline configuration with corrected scores."""

    collection: str
    alpha: float
    reranking: bool
    strategy: str
    question_scores: list[QuestionScore] = field(default_factory=list)
    corrected_overall: dict[str, float] = field(default_factory=dict)
    corrected_single: dict[str, float] = field(default_factory=dict)
    corrected_cross: dict[str, float] = field(default_factory=dict)
    n_single: int = 0
    n_cross: int = 0
    n_total: int = 0

    @property
    def config_label(self) -> str:
        """Human-readable configuration label."""
        chunking = COLLECTION_LABELS.get(self.collection, self.collection)
        strategy = STRATEGY_LABELS.get(self.strategy, self.strategy)
        rerank = "rerank" if self.reranking else "no-rerank"
        return f"{chunking} | alpha={self.alpha} | {strategy} | {rerank}"

    @property
    def config_key(self) -> str:
        """Unique key for matching configs across runs."""
        return f"{self.collection}|{self.alpha}|{self.reranking}|{self.strategy}"


@dataclass
class FactorStats:
    """Statistics for one level of one factor."""

    level: str
    n_configs: int
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    # metrics[metric_name] = {"mean": ..., "std": ..., "values": [...]}


@dataclass
class RunAnalysis:
    """Complete analysis of one evaluation run."""

    run_name: str
    configs: list[CorrectedConfig]
    factor_stats: dict[str, list[FactorStats]]  # factor_name -> [FactorStats]
    effect_sizes: dict[str, dict[str, float]]  # factor_name -> {metric: effect_size}


# ── Loading ────────────────────────────────────────────────────────────────


def load_run(path: Path, key: str) -> list[dict]:
    """Load evaluation run results from JSON.

    Args:
        path: Path to the results JSON file.
        key: Top-level key containing the configuration list
            ("leaderboard" for Run 1, "results" for Run 2).

    Returns:
        List of raw configuration dictionaries.
    """
    logger.info("Loading %s from %s", key, path)
    with open(path) as f:
        data = json.load(f)
    configs = data[key]
    logger.info("Found %d configurations", len(configs))
    return configs


def load_trace(trace_path: str) -> list[dict]:
    """Load per-question scores from a trace file.

    Args:
        trace_path: Absolute path to the trace JSON file.

    Returns:
        List of question dictionaries from the trace.
    """
    with open(trace_path) as f:
        trace = json.load(f)
    return trace["questions"]


# ── Correction ─────────────────────────────────────────────────────────────


def build_corrected_config(raw_config: dict) -> CorrectedConfig:
    """Build a CorrectedConfig by reading trace data and excluding the bad question.

    Args:
        raw_config: Raw configuration dict from the results JSON.

    Returns:
        CorrectedConfig with recomputed metrics excluding neuro_eagleman_02.
    """
    trace_questions = load_trace(raw_config["trace_path"])

    config = CorrectedConfig(
        collection=raw_config["collection"],
        alpha=raw_config["alpha"],
        reranking=raw_config["reranking"],
        strategy=raw_config["strategy"],
    )

    single_scores = defaultdict(list)
    cross_scores = defaultdict(list)

    for q in trace_questions:
        if q["question_id"] == EXCLUDE_QUESTION:
            continue

        qs = QuestionScore(
            question_id=q["question_id"],
            difficulty=q["difficulty"],
            scores=q["scores"],
        )
        config.question_scores.append(qs)

        bucket = single_scores if q["difficulty"] == "single_concept" else cross_scores
        for metric in METRICS:
            if metric in q["scores"] and q["scores"][metric] is not None:
                bucket[metric].append(q["scores"][metric])

    # Compute corrected means
    for metric in METRICS:
        s_vals = single_scores[metric]
        c_vals = cross_scores[metric]
        all_vals = s_vals + c_vals

        config.corrected_single[metric] = mean(s_vals) if s_vals else 0.0
        config.corrected_cross[metric] = mean(c_vals) if c_vals else 0.0
        config.corrected_overall[metric] = mean(all_vals) if all_vals else 0.0

    config.n_single = len(single_scores.get("answer_correctness", []))
    config.n_cross = len(cross_scores.get("answer_correctness", []))
    config.n_total = config.n_single + config.n_cross

    return config


# ── Factor analysis ────────────────────────────────────────────────────────


def _extract_factor(config: CorrectedConfig, factor: str) -> str:
    """Extract the factor level label from a config.

    Args:
        config: A corrected configuration.
        factor: Factor name (chunking, alpha, strategy, reranking).

    Returns:
        Human-readable label for this config's level of the given factor.
    """
    if factor == "chunking":
        return COLLECTION_LABELS.get(config.collection, config.collection)
    elif factor == "alpha":
        return ALPHA_LABELS.get(config.alpha, str(config.alpha))
    elif factor == "strategy":
        return STRATEGY_LABELS.get(config.strategy, config.strategy)
    elif factor == "reranking":
        return "On" if config.reranking else "Off"
    else:
        raise ValueError(f"Unknown factor: {factor}")


def compute_factor_stats(
    configs: list[CorrectedConfig],
    factor: str,
    score_type: str = "overall",
) -> list[FactorStats]:
    """Compute mean and std for each level of a factor.

    Args:
        configs: List of corrected configurations.
        factor: Factor name (chunking, alpha, strategy, reranking).
        score_type: Which scores to use ("overall", "single", "cross").

    Returns:
        List of FactorStats, one per factor level.
    """
    groups: dict[str, list[CorrectedConfig]] = defaultdict(list)
    for c in configs:
        level = _extract_factor(c, factor)
        groups[level].append(c)

    results = []
    for level, cfgs in sorted(groups.items()):
        fs = FactorStats(level=level, n_configs=len(cfgs))

        for metric in METRICS:
            if score_type == "overall":
                values = [c.corrected_overall[metric] for c in cfgs]
            elif score_type == "single":
                values = [c.corrected_single[metric] for c in cfgs]
            else:
                values = [c.corrected_cross[metric] for c in cfgs]

            fs.metrics[metric] = {
                "mean": mean(values) if values else 0.0,
                "std": stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
        results.append(fs)

    return results


def compute_effect_size(
    configs: list[CorrectedConfig],
    factor: str,
    metric: str,
    score_type: str = "overall",
) -> float:
    """Compute eta-squared effect size for a factor on a metric.

    Eta-squared = variance of group means / total variance of individual scores.

    Args:
        configs: List of corrected configurations.
        factor: Factor name.
        metric: Metric name.
        score_type: Which scores to use.

    Returns:
        Effect size (0.0 to 1.0). Returns 0.0 if total variance is zero.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for c in configs:
        level = _extract_factor(c, factor)
        if score_type == "overall":
            groups[level].append(c.corrected_overall[metric])
        elif score_type == "single":
            groups[level].append(c.corrected_single[metric])
        else:
            groups[level].append(c.corrected_cross[metric])

    all_values = []
    for vals in groups.values():
        all_values.extend(vals)

    if len(all_values) < 2:
        return 0.0

    total_var = variance(all_values)
    if total_var == 0:
        return 0.0

    group_means = [mean(vals) for vals in groups.values()]
    group_var = variance(group_means) if len(group_means) > 1 else 0.0

    return group_var / total_var


# ── Interaction analysis ───────────────────────────────────────────────────


def compute_interaction_table(
    configs: list[CorrectedConfig],
    factor_a: str,
    factor_b: str,
    metric: str,
    score_type: str = "overall",
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute mean metric for each combination of two factors.

    Args:
        configs: List of corrected configurations.
        factor_a: First factor name (rows).
        factor_b: Second factor name (columns).
        metric: Metric name.
        score_type: Which scores to use.

    Returns:
        Nested dict: {level_a: {level_b: {"mean": ..., "n": ...}}}
    """
    cells: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for c in configs:
        a = _extract_factor(c, factor_a)
        b = _extract_factor(c, factor_b)
        if score_type == "overall":
            val = c.corrected_overall[metric]
        elif score_type == "single":
            val = c.corrected_single[metric]
        else:
            val = c.corrected_cross[metric]
        cells[a][b].append(val)

    result = {}
    for a_level, b_dict in sorted(cells.items()):
        result[a_level] = {}
        for b_level, vals in sorted(b_dict.items()):
            result[a_level][b_level] = {
                "mean": mean(vals) if vals else 0.0,
                "n": len(vals),
            }

    return result


# ── Analysis pipeline ──────────────────────────────────────────────────────


def analyze_run(run_name: str, raw_configs: list[dict]) -> RunAnalysis:
    """Run complete analysis pipeline for one evaluation run.

    Args:
        run_name: Name for this run ("selected" or "full").
        raw_configs: List of raw config dicts from the JSON file.

    Returns:
        RunAnalysis with corrected configs, factor stats, and effect sizes.
    """
    logger.info("Analyzing run: %s (%d configs)", run_name, len(raw_configs))

    # Build corrected configs
    configs = []
    for rc in raw_configs:
        cc = build_corrected_config(rc)
        configs.append(cc)
        logger.info(
            "  %s: n=%d (single=%d, cross=%d) answer_correctness=%.4f",
            cc.config_label,
            cc.n_total,
            cc.n_single,
            cc.n_cross,
            cc.corrected_overall.get("answer_correctness", 0.0),
        )

    # Factor-level statistics
    factors = ["chunking", "alpha", "strategy", "reranking"]
    factor_stats = {}
    for factor in factors:
        factor_stats[factor] = {
            "overall": compute_factor_stats(configs, factor, "overall"),
            "single": compute_factor_stats(configs, factor, "single"),
            "cross": compute_factor_stats(configs, factor, "cross"),
        }

    # Effect sizes
    effect_sizes = {}
    for factor in factors:
        effect_sizes[factor] = {}
        for metric in ["answer_correctness", "context_recall"]:
            for score_type in ["overall", "single", "cross"]:
                key = f"{metric}_{score_type}"
                effect_sizes[factor][key] = compute_effect_size(
                    configs, factor, metric, score_type
                )

    return RunAnalysis(
        run_name=run_name,
        configs=configs,
        factor_stats=factor_stats,
        effect_sizes=effect_sizes,
    )


# ── Output ─────────────────────────────────────────────────────────────────


def _rank_configs(
    configs: list[CorrectedConfig],
    metric: str,
    score_type: str = "overall",
) -> list[tuple[CorrectedConfig, float]]:
    """Sort configs by a metric, descending.

    Args:
        configs: Configs to rank.
        metric: Metric name.
        score_type: Which scores to use.

    Returns:
        List of (config, score) tuples sorted descending by score.
    """
    if score_type == "overall":
        scored = [(c, c.corrected_overall[metric]) for c in configs]
    elif score_type == "single":
        scored = [(c, c.corrected_single[metric]) for c in configs]
    else:
        scored = [(c, c.corrected_cross[metric]) for c in configs]

    return sorted(scored, key=lambda x: x[1], reverse=True)


def write_corrected_json(analysis: RunAnalysis, output_path: Path) -> None:
    """Write corrected scores to JSON for verification.

    Args:
        analysis: Completed run analysis.
        output_path: Path to write the JSON file.
    """
    data = []
    for c in analysis.configs:
        entry = {
            "config_label": c.config_label,
            "config_key": c.config_key,
            "collection": c.collection,
            "alpha": c.alpha,
            "reranking": c.reranking,
            "strategy": c.strategy,
            "n_total": c.n_total,
            "n_single": c.n_single,
            "n_cross": c.n_cross,
            "corrected_overall": c.corrected_overall,
            "corrected_single": c.corrected_single,
            "corrected_cross": c.corrected_cross,
            "per_question_scores": [
                {
                    "question_id": qs.question_id,
                    "difficulty": qs.difficulty,
                    "scores": qs.scores,
                }
                for qs in c.question_scores
            ],
        }
        data.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote corrected JSON to %s", output_path)


def write_raw_stats_json(
    analysis: RunAnalysis,
    output_path: Path,
) -> None:
    """Write factor statistics and effect sizes to JSON.

    Args:
        analysis: Completed run analysis.
        output_path: Path to write the JSON file.
    """
    data = {
        "run_name": analysis.run_name,
        "n_configs": len(analysis.configs),
        "rankings": {},
        "factor_stats": {},
        "effect_sizes": analysis.effect_sizes,
        "interaction_tables": {},
    }

    # Rankings
    for metric in ["answer_correctness", "context_recall"]:
        for score_type in ["overall", "single", "cross"]:
            key = f"{metric}_{score_type}"
            ranked = _rank_configs(analysis.configs, metric, score_type)
            data["rankings"][key] = [
                {"rank": i + 1, "config": c.config_label, "score": round(s, 4)}
                for i, (c, s) in enumerate(ranked)
            ]

    # Factor stats
    for factor, score_types in analysis.factor_stats.items():
        data["factor_stats"][factor] = {}
        for score_type, stats_list in score_types.items():
            data["factor_stats"][factor][score_type] = [
                {
                    "level": fs.level,
                    "n_configs": fs.n_configs,
                    "metrics": {
                        m: {"mean": round(v["mean"], 4), "std": round(v["std"], 4)}
                        for m, v in fs.metrics.items()
                    },
                }
                for fs in stats_list
            ]

    # Interaction tables
    interactions = [
        ("chunking", "strategy"),
        ("chunking", "reranking"),
        ("strategy", "alpha"),
        ("chunking", "alpha"),
    ]
    for fa, fb in interactions:
        for metric in ["answer_correctness", "context_recall"]:
            key = f"{fa}_x_{fb}_{metric}"
            table = compute_interaction_table(
                analysis.configs, fa, fb, metric, "overall"
            )
            data["interaction_tables"][key] = {
                a_level: {
                    b_level: {"mean": round(v["mean"], 4), "n": v["n"]}
                    for b_level, v in b_dict.items()
                }
                for a_level, b_dict in table.items()
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote raw stats JSON to %s", output_path)


def write_summary_markdown(analysis: RunAnalysis, output_path: Path) -> None:
    """Write a markdown summary of the analysis.

    Args:
        analysis: Completed run analysis.
        output_path: Path to write the markdown file.
    """
    lines = []
    lines.append(f"# Raw Analysis Data: {analysis.run_name}")
    lines.append("")
    lines.append(
        f"Corrected scores excluding `{EXCLUDE_QUESTION}`. "
        f"{len(analysis.configs)} configurations analyzed."
    )
    lines.append("")

    # Top 10 + Bottom 3 for answer_correctness
    for metric in ["answer_correctness", "context_recall"]:
        for score_type in ["overall", "single", "cross"]:
            ranked = _rank_configs(analysis.configs, metric, score_type)
            lines.append(f"## {metric} — {score_type}")
            lines.append("")
            lines.append("### Top 10")
            lines.append("")
            lines.append("| Rank | Configuration | Score |")
            lines.append("|------|--------------|-------|")
            for i, (c, s) in enumerate(ranked[:10]):
                lines.append(f"| {i + 1} | {c.config_label} | {s:.4f} |")
            lines.append("")
            lines.append("### Bottom 3")
            lines.append("")
            lines.append("| Rank | Configuration | Score |")
            lines.append("|------|--------------|-------|")
            for i, (c, s) in enumerate(ranked[-3:]):
                rank = len(ranked) - 2 + i
                lines.append(f"| {rank} | {c.config_label} | {s:.4f} |")
            lines.append("")

    # Factor stats
    lines.append("## Factor Analysis")
    lines.append("")
    for factor in ["chunking", "alpha", "strategy", "reranking"]:
        lines.append(f"### {factor.title()}")
        lines.append("")
        for score_type in ["overall", "single", "cross"]:
            stats = analysis.factor_stats[factor][score_type]
            lines.append(f"**{score_type}**")
            lines.append("")
            lines.append(
                "| Level | n | answer_correctness | context_recall | faithfulness | relevancy | context_precision |"
            )
            lines.append(
                "|-------|---|-------------------|----------------|--------------|-----------|-------------------|"
            )
            for fs in stats:
                ac = fs.metrics["answer_correctness"]
                cr = fs.metrics["context_recall"]
                fa = fs.metrics["faithfulness"]
                rel = fs.metrics["relevancy"]
                cp = fs.metrics["context_precision"]
                lines.append(
                    f"| {fs.level} | {fs.n_configs} | "
                    f"{ac['mean']:.4f} +/- {ac['std']:.4f} | "
                    f"{cr['mean']:.4f} +/- {cr['std']:.4f} | "
                    f"{fa['mean']:.4f} +/- {fa['std']:.4f} | "
                    f"{rel['mean']:.4f} +/- {rel['std']:.4f} | "
                    f"{cp['mean']:.4f} +/- {cp['std']:.4f} |"
                )
            lines.append("")

    # Effect sizes
    lines.append("## Effect Sizes (eta-squared)")
    lines.append("")
    lines.append("| Factor | AC overall | AC single | AC cross | CR overall | CR single | CR cross |")
    lines.append("|--------|-----------|-----------|----------|-----------|-----------|----------|")
    for factor in ["chunking", "alpha", "strategy", "reranking"]:
        es = analysis.effect_sizes[factor]
        lines.append(
            f"| {factor.title()} | "
            f"{es['answer_correctness_overall']:.4f} | "
            f"{es['answer_correctness_single']:.4f} | "
            f"{es['answer_correctness_cross']:.4f} | "
            f"{es['context_recall_overall']:.4f} | "
            f"{es['context_recall_single']:.4f} | "
            f"{es['context_recall_cross']:.4f} |"
        )
    lines.append("")

    # Interaction tables
    lines.append("## Interaction Tables (answer_correctness, overall)")
    lines.append("")
    interactions = [
        ("chunking", "strategy"),
        ("chunking", "reranking"),
        ("strategy", "alpha"),
        ("chunking", "alpha"),
    ]
    for fa, fb in interactions:
        lines.append(f"### {fa.title()} x {fb.title()}")
        lines.append("")
        table = compute_interaction_table(
            analysis.configs, fa, fb, "answer_correctness", "overall"
        )
        # Collect all b-levels
        b_levels = sorted(
            {bl for b_dict in table.values() for bl in b_dict}
        )
        header = f"| {fa.title()} | " + " | ".join(b_levels) + " |"
        sep = "|" + "---|" * (len(b_levels) + 1)
        lines.append(header)
        lines.append(sep)
        for a_level in sorted(table):
            row = f"| {a_level} |"
            for bl in b_levels:
                cell = table[a_level].get(bl)
                if cell:
                    row += f" {cell['mean']:.4f} (n={cell['n']}) |"
                else:
                    row += " — |"
            lines.append(row)
        lines.append("")

    # Gap analysis
    lines.append("## Single-Concept vs Cross-Domain Gap")
    lines.append("")
    lines.append("| Configuration | Single AC | Cross AC | Gap | Gap % |")
    lines.append("|--------------|-----------|----------|-----|-------|")
    gap_data = []
    for c in analysis.configs:
        s = c.corrected_single["answer_correctness"]
        x = c.corrected_cross["answer_correctness"]
        gap = s - x
        gap_pct = (gap / s * 100) if s > 0 else 0
        gap_data.append((c, s, x, gap, gap_pct))
    # Sort by smallest absolute gap
    gap_data.sort(key=lambda t: abs(t[3]))
    for c, s, x, gap, gap_pct in gap_data[:10]:
        lines.append(
            f"| {c.config_label} | {s:.4f} | {x:.4f} | {gap:+.4f} | {gap_pct:+.1f}% |"
        )
    lines.append("")
    lines.append("*(Sorted by smallest absolute gap, top 10)*")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Wrote summary markdown to %s", output_path)


# ── Spot-check verification ───────────────────────────────────────────────


def spot_check(analysis: RunAnalysis, n_checks: int = 3) -> None:
    """Verify corrected scores by recomputing from per-question data.

    Args:
        analysis: Completed run analysis.
        n_checks: Number of configs to spot-check.
    """
    logger.info("Spot-checking %d configs...", n_checks)
    for config in analysis.configs[:n_checks]:
        # Recompute answer_correctness from per-question scores
        ac_vals = [
            qs.scores["answer_correctness"]
            for qs in config.question_scores
            if "answer_correctness" in qs.scores
            and qs.scores["answer_correctness"] is not None
        ]
        recomputed = mean(ac_vals) if ac_vals else 0.0
        stored = config.corrected_overall["answer_correctness"]

        # Verify no excluded question
        excluded_present = any(
            qs.question_id == EXCLUDE_QUESTION for qs in config.question_scores
        )

        match = abs(recomputed - stored) < 1e-10
        logger.info(
            "  %s: recomputed=%.6f stored=%.6f match=%s excluded_absent=%s n=%d",
            config.config_label,
            recomputed,
            stored,
            match,
            not excluded_present,
            len(ac_vals),
        )
        if not match:
            logger.warning("    MISMATCH in spot check!")
        if excluded_present:
            logger.warning("    EXCLUDED QUESTION FOUND!")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the complete analysis pipeline for both evaluation runs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load runs
    run1_raw = load_run(RUN1_PATH, "leaderboard")
    run2_raw = load_run(RUN2_PATH, "leaderboard")

    # Analyze
    run1 = analyze_run("run1_selected", run1_raw)
    run2 = analyze_run("run2_full", run2_raw)

    # Spot-check
    spot_check(run1)
    spot_check(run2)

    # Write corrected JSON
    write_corrected_json(run1, OUTPUT_DIR / "run1_corrected.json")
    write_corrected_json(run2, OUTPUT_DIR / "run2_corrected.json")

    # Write raw stats JSON
    write_raw_stats_json(run1, OUTPUT_DIR / "run1_stats.json")
    write_raw_stats_json(run2, OUTPUT_DIR / "run2_stats.json")

    # Write markdown summaries
    write_summary_markdown(run1, OUTPUT_DIR / "run1_raw_data.md")
    write_summary_markdown(run2, OUTPUT_DIR / "run2_raw_data.md")

    logger.info("Analysis complete. Output in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
