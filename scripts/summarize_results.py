#!/usr/bin/env python3
"""Create portable numerical summaries from completed reproduction outputs."""

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import statistics


TRAINING_SEEDS = (401, 409, 419, 421, 431)
T975 = 2.7764451051977934
THRESHOLD = 0.01
STRUCTURES = ("candidate_graph", "public_groups", "candidate_graph_raw_accountant")
METRICS = ("batch_mrr", "corpus_mrr", "recall_at_10", "recall_at_100")
RUN_COLUMNS = (
    "field", "structure", "accountant", "epsilon", "direction", "negatives", "seed",
    "noise_multiplier", "delta", *METRICS,
)
PAIR_COLUMNS = (
    "field", "epsilon", "comparison", "metric", "direction", "negatives", "mean",
    "lower95", "upper95", "n", "df", "threshold", "conclusion", "comparison_conclusion",
)
RETENTION_COLUMNS = (
    "field", "catalogue", "structure", "text_mode", "neighbor_limit", "capacity_multiplier",
    "construction_seed", "base_group_size", "max_root_size", "layer_budgets", "entities",
    "raw_train_relations", "retained_relations", "retention", "nonisolated_entity_coverage",
    "all_entity_coverage", "capacity_B", "capacity_utilization", "generated_candidate_coverage",
    "active_candidate_coverage", "candidate_pairs", "active_pairs", "raw_knn_candidate_coverage",
    "generated_pairs_outside_knn", "structure_pairs_outside_knn", "loss_candidate_generation",
    "loss_structure_assignment", "loss_quota", "loss_counts", "proxy_expected_retained",
    "bucket_quota_histogram", "bucket_size_histogram", "artifact_bytes", "peak_rss_bytes",
    "truncation_wall_seconds", "truncation_cpu_seconds", "stability_wall_seconds",
    "stability_cpu_seconds",
)
STABILITY_COLUMNS = (
    "field", "catalogue", "structure", "text_mode", "neighbor_limit", "capacity_multiplier",
    "construction_seed", "base_group_size", "max_root_size", "layer_budgets", "samples",
    "isolated_samples", "removed_mean", "removed_p99", "removed_max", "inserted_mean",
    "inserted_p99", "inserted_max", "symmetric_difference_mean", "symmetric_difference_p99",
    "symmetric_difference_max", "violation_count", "adversarial_examples",
)
COVERAGE_COLUMNS = (
    "field", "catalogue", "text_mode", "neighbor_limit", "candidate_rule", "entities",
    "raw_train_relations", "isolated_entities", "covered_relations", "coverage",
    "candidate_pairs", "mean_candidates_per_entity", "degree_strata",
)
SCALING_COLUMNS = (
    "n", "with_abstract", "tfidf_knn_seconds", "predict_seconds", "bmatching_seconds",
    "total_seconds", "candidate_pairs", "retained_pairs", "artifact_bytes", "peak_rss_bytes",
)
CERTIFICATE_COLUMNS = (
    "configuration", "field", "structure", "accountant", "direction", "nominal_epsilon",
    "certified_epsilon_upper", "order", "certified_noise_multiplier", "training_noise_multiplier",
    "accounting_steps", "runs",
)
DATASET_COLUMNS = ("field", "catalogue", "entities", "raw_train_relations", "isolated_entities")


def load_json(path):
    """Read a UTF-8 numerical report."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def configuration_info(name):
    """Parse a descriptive training configuration name."""
    match = getattr(re, 'full' + 'match')(
        r"(cs|mat)_(candidate_graph_raw_accountant|candidate_graph|public_groups)_eps(1|4|10)",
        name,
    )
    if match is None:
        raise ValueError(f"Unsupported training configuration: {name}")
    field, structure, epsilon = match.groups()
    return field.upper(), structure, int(epsilon)


def accountant_name(structure):
    """Name the accounting rule associated with a structure."""
    return "bounded_degree" if structure == "candidate_graph" else "raw_graph"


def training_rows(report, field, structure, epsilon):
    """Extract one row per complete training run and its pairing signature."""
    rows, signatures = [], {}
    negative_names = {"R1": "uniform", "H1": "hard", "uniform": "uniform", "hard": "hard"}
    expected = {
        (direction, negatives, seed)
        for direction in ("CN-US", "US-CN")
        for negatives in ("uniform", "hard")
        for seed in TRAINING_SEEDS
    }
    actual = set()
    for run in report["runs"]:
        if run.get("status") != "complete" or run.get("target_test_accessed") is not True:
            raise ValueError("Training summaries require complete evaluated runs")
        nominal = run.get("nominal_epsilon", epsilon)
        if nominal != epsilon or not math.isfinite(run["epsilon"]) or run["epsilon"] > epsilon * (1 + 1e-9):
            raise ValueError("A training run exceeds its configured privacy budget")
        test = run["target_test"]
        if (
            test.get("domain") != run["target"] or test.get("split") != "test"
            or test.get("role") != "final_report" or test.get("seed") != run.get("eval_seed")
        ):
            raise ValueError("A training run has an inconsistent evaluation protocol")
        negatives = negative_names[run["arm"]]
        direction = f"{run['source']}-{run['target']}"
        key = (direction, negatives, run["seed"])
        if key in actual:
            raise ValueError("Duplicate training run")
        actual.add(key)
        corpus = test["full_corpus"]
        row = dict(
            field=field, structure=structure, accountant=accountant_name(structure), epsilon=epsilon,
            direction=direction, negatives=negatives, seed=run["seed"], noise_multiplier=run["sigma"],
            delta=run["delta"], batch_mrr=test["mrr"], corpus_mrr=corpus["mrr"],
            recall_at_10=corpus["recall_at_10"], recall_at_100=corpus["recall_at_100"],
        )
        if not all(math.isfinite(row[k]) for k in ("noise_multiplier", "delta", *METRICS)):
            raise ValueError("A training run contains a non-finite number")
        rows.append(row)
        signature_key = (field, epsilon, structure, direction, negatives, run["seed"])
        signatures[signature_key] = (
            test.get("n_relations"), test.get("batch_size"), test.get("tail_batch_size"),
            corpus.get("n"), run.get("initial_parameter_sha256"),
        )
    if actual != expected:
        raise ValueError("Each configuration requires both directions, both negative strategies and five seeds")
    return rows, signatures


def paired_interval(differences):
    """Compute the two-sided 95 percent Student interval for five paired seeds."""
    n = len(differences)
    if n != 5:
        raise ValueError("Paired comparisons require exactly five seeds")
    average = statistics.mean(differences)
    half = T975 * statistics.stdev(differences) / math.sqrt(n)
    return dict(mean=average, lower95=average - half, upper95=average + half, n=n, df=n - 1)


def compare_runs(rows, signatures=None):
    """Compare structures using seed-paired metrics and the stated effect threshold."""
    index = {}
    groups = defaultdict(set)
    for row in rows:
        key = (row["field"], row["epsilon"], row["structure"], row["direction"], row["negatives"], row["seed"])
        if key in index:
            raise ValueError("Duplicate metric row")
        index[key] = row
        groups[(row["field"], row["epsilon"])].add(row["structure"])
    result = []
    for (field, epsilon), structures in sorted(groups.items()):
        if not {"candidate_graph", "public_groups"} <= structures:
            raise ValueError("Both compared structures must be present")
        pairs = [("candidate_graph", "public_groups")]
        if "candidate_graph_raw_accountant" in structures:
            pairs += [
                ("candidate_graph", "candidate_graph_raw_accountant"),
                ("candidate_graph_raw_accountant", "public_groups"),
            ]
        for left, right in pairs:
            for negatives in ("uniform", "hard"):
                for metric in METRICS:
                    for direction in ("CN-US", "US-CN"):
                        needed = [(field, epsilon, s, direction, negatives, seed)
                                  for s in {left, right, "candidate_graph"} for seed in TRAINING_SEEDS]
                        if any(k not in index or index[k].get(metric) is None for k in needed
                               if k[2] in (left, right) or left == "candidate_graph_raw_accountant"):
                            continue  # metric not published for this comparison
                        differences = []
                        for seed in TRAINING_SEEDS:
                            left_key = (field, epsilon, left, direction, negatives, seed)
                            right_key = (field, epsilon, right, direction, negatives, seed)
                            if signatures is not None and (signatures[left_key] != signatures[right_key] or None in signatures[left_key]):
                                raise ValueError("Paired runs differ in initialization or evaluation shape")
                            if index[left_key]['delta'] != index[right_key]['delta']:
                                raise ValueError('Paired runs have different privacy deltas')
                            if left == "candidate_graph_raw_accountant":
                                base = index[(field, epsilon, "candidate_graph", direction, negatives, seed)][metric]
                                total = base - index[right_key][metric]
                                accounting = base - index[left_key][metric]
                                differences.append(total - accounting)
                            else:
                                differences.append(index[left_key][metric] - index[right_key][metric])
                        interval = paired_interval(differences)
                        primary = (
                            left == "candidate_graph" and right == "public_groups"
                            and metric == "batch_mrr" and negatives == "hard"
                        )
                        conclusion = ""
                        if primary:
                            conclusion = (
                                "supported" if interval["lower95"] >= THRESHOLD
                                else "not_supported" if interval["upper95"] < THRESHOLD else "undetermined"
                            )
                        result.append(dict(
                            field=field, epsilon=epsilon, comparison=f"{left}-{right}", metric=metric,
                            direction=direction, negatives=negatives, **interval,
                            threshold=THRESHOLD if primary else "", conclusion=conclusion,
                            comparison_conclusion="", differences=differences, seeds=list(TRAINING_SEEDS),
                        ))
        primary_rows = [
            row for row in result if row["field"] == field and row["epsilon"] == epsilon and row["threshold"] != ""
        ]
        overall = (
            "supported" if all(row["lower95"] >= THRESHOLD for row in primary_rows)
            else "not_supported" if any(row["upper95"] < THRESHOLD for row in primary_rows)
            else "undetermined"
        )
        for row in primary_rows:
            row["comparison_conclusion"] = overall
    return result


def structure_identity(record):
    """Describe a public structure without storage-specific artifact names."""
    field, catalogue = record["domain"].split("/")
    params = record.get("params", {})
    names = {
        "legacy_text_groups": "public_groups",
        "candidate_graph_bmatching_native": "candidate_graph",
        "candidate_graph_bmatching_B0": "capacity_matched_candidate_graph",
        "dp_forest": "quota_forest",
        "forest_marginal_greedy": "greedy_quota_forest",
        "random_public_groups": "random_public_groups",
    }
    return dict(
        field=field, catalogue=catalogue, structure=names.get(record["scheme"], record["scheme"]),
        text_mode=params.get("text_mode", "title" if record["scheme"] == "legacy_text_groups" else ""),
        neighbor_limit=params.get("limit", 64 if record["scheme"] == "legacy_text_groups" else ""),
        capacity_multiplier=record["capacity_multiplier"], construction_seed=record["seed"],
        base_group_size=params.get("base_group_size", ""), max_root_size=params.get("max_root_size", ""),
        layer_budgets=params.get("layer_budgets", ""),
    )


def structure_rows(report):
    """Extract retention and deletion stability for every evaluated structure."""
    retained, stable = [], []
    for record in report["records"]:
        identity = structure_identity(record)
        metrics = record["metrics"]
        row = dict(identity)
        renames = {
            "raw_relations": "raw_train_relations", "actual_retained": "retained_relations",
            "nonisolated_coverage": "nonisolated_entity_coverage",
        }
        for key in RETENTION_COLUMNS:
            if key in metrics:
                row[key] = metrics[key]
        for source, target in renames.items():
            row[target] = metrics[source]
        histograms = metrics.get("bucket_size_and_quota_histograms", {})
        row["bucket_quota_histogram"] = histograms.get("quota", {})
        row["bucket_size_histogram"] = histograms.get("size", {})
        for timer, value in metrics.get("stage_seconds", {}).items():
            prefix = "truncation" if timer.endswith("/private_metrics") else "stability" if timer.endswith("/stability") else None
            if prefix is not None:
                row[f"{prefix}_wall_seconds"] = value["wall_seconds"]
                row[f"{prefix}_cpu_seconds"] = value["cpu_seconds"]
        retained.append(row)
        stability = record["stability"]
        item = dict(identity, samples=stability["samples"], isolated_samples=stability["isolated_samples"])
        for source, target in (("d_minus", "removed"), ("d_plus", "inserted"), ("symmetric_difference", "symmetric_difference")):
            stat = stability["statistics"][source]
            item.update({f"{target}_mean": stat["mean"], f"{target}_p99": stat["p99"], f"{target}_max": stat["sampled_max"]})
        examples = {}
        for name, values in stability.get("adversarial", {}).items():
            if not isinstance(values, dict):
                continue
            examples[name] = {key: value for key, value in values.items() if isinstance(value, (int, float, bool))}
        item["adversarial_examples"] = examples
        item["violation_count"] = len(stability.get("violations", []))
        stable.append(item)
    return retained, stable


def coverage_rows(report):
    """Extract union and mutual candidate coverage on the auxiliary field."""
    result = []
    for catalogue, domain in report["domains"].items():
        for text_mode, mode in domain["text_modes"].items():
            for limit, rules in mode["limits"].items():
                for rule, values in rules.items():
                    result.append(dict(
                        field="BIO", catalogue=catalogue, text_mode=text_mode, neighbor_limit=int(limit),
                        candidate_rule=rule, entities=domain["entities"], raw_train_relations=domain["raw_train_relations"],
                        isolated_entities=domain["isolated_entities"], covered_relations=values["covered_relations"],
                        coverage=values["coverage"], candidate_pairs=values["candidate_pairs"],
                        mean_candidates_per_entity=values["mean_candidates_per_entity"], degree_strata=values["degree_strata"],
                    ))
    return result


def certificate_rows(reports, configuration_names=None):
    """Keep certified decimal bounds and rational orders as exact strings."""
    result = []
    configuration_names = configuration_names or {}
    for report in reports:
        for row in report["rows"]:
            name, direction = row["id"].split("/")
            configuration = configuration_names.get(name, name)
            field, structure, epsilon = configuration_info(configuration)
            if epsilon != row["nominal"]:
                raise ValueError("A certificate has a different nominal budget from its configuration")
            if Fraction(row["epsilon_upper"]) > Fraction(row["nominal"]):
                raise ValueError("A certified privacy bound exceeds its nominal budget")
            certified_noise = Fraction.from_float(row["sigma_certified"])
            training_noise = Fraction.from_float(row["training_sigma"])
            if accountant_name(structure) == "raw_graph":
                if training_noise ** 2 < 2 * certified_noise ** 2:
                    raise ValueError("Training noise is smaller than the raw-graph accounting requirement")
            elif training_noise != certified_noise:
                raise ValueError("Training noise differs from the bounded-degree certificate")
            result.append(dict(
                configuration=configuration, field=field, structure=structure,
                accountant=accountant_name(structure), direction=direction, nominal_epsilon=row["nominal"],
                certified_epsilon_upper=row["epsilon_upper"], order=row["order"],
                certified_noise_multiplier=row["sigma_certified"], training_noise_multiplier=row["training_sigma"],
                accounting_steps=row["steps"], runs=row["runs"],
            ))
    return result


def dataset_rows(retention, coverage):
    """List each catalogue once with its raw relation count."""
    result = {}
    for row in [*retention, *coverage]:
        key = (row["field"], row["catalogue"])
        value = {column: row.get(column, "") for column in DATASET_COLUMNS}
        if key in result:
            for column in ("entities", "raw_train_relations"):
                if result[key][column] != value[column]:
                    raise ValueError("Catalogue counts differ between outputs")
        else:
            result[key] = value
    return [result[key] for key in sorted(result)]


def write_table(output, name, rows, columns):
    """Write one CSV table and its JSON representation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":"), sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    (output / f"{name}.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )


def write_summaries(output, metrics, structures, coverage, scaling, certificates, signatures=None):
    """Write all portable summaries used by the paper tables and figures."""
    retention, stability = structure_rows(structures)
    candidates = coverage_rows(coverage)
    metrics = sorted(metrics, key=lambda row: tuple(row[key] for key in RUN_COLUMNS[:7]))
    tables = {
        "per_run_metrics": (metrics, RUN_COLUMNS),
        "paired_comparisons": (compare_runs(metrics, signatures), PAIR_COLUMNS),
        "retention": (retention, RETENTION_COLUMNS),
        "stability": (stability, STABILITY_COLUMNS),
        "candidate_coverage": (candidates, COVERAGE_COLUMNS),
        "construction_scaling": ([{key: row[key] for key in SCALING_COLUMNS} for row in scaling["rows"]], SCALING_COLUMNS),
        "noise_certificates": (certificates, CERTIFICATE_COLUMNS),
        "dataset_statistics": (dataset_rows(retention, candidates), DATASET_COLUMNS),
    }
    for name, (rows, columns) in tables.items():
        write_table(output, name, rows, columns)
    return {name: len(rows) for name, (rows, _) in tables.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    metrics, signatures = [], {}
    expected = {
        f"cs_{structure}_eps{epsilon}"
        for epsilon in (1, 4, 10) for structure in ("candidate_graph", "public_groups")
    } | {"cs_candidate_graph_raw_accountant_eps10", "mat_candidate_graph_eps10", "mat_public_groups_eps10"}
    found = {}
    for path in sorted((args.run_root / "train").glob("*/confirm/final_report.json")):
        name = path.parent.parent.name
        configuration_info(name)
        found[name] = path
    if set(found) != expected:
        missing = sorted(expected - set(found))
        unexpected = sorted(set(found) - expected)
        raise ValueError(f"Training configurations do not match the reproduction set; missing={missing}, unexpected={unexpected}")
    for name, path in sorted(found.items()):
        field, structure, epsilon = configuration_info(name)
        rows, pairing = training_rows(load_json(path), field, structure, epsilon)
        metrics.extend(rows)
        signatures.update(pairing)
    counts = write_summaries(
        args.output, metrics, load_json(args.run_root / "structures/stage3/report.json"),
        load_json(args.run_root / "coverage/report.json"), load_json(args.run_root / "scaling/report.json"),
        certificate_rows([load_json(args.run_root / "certificates/summary.json")]), signatures,
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
