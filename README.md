# QuotaGuard

Reproduction code for **Stable Degree Bounding with Public Candidate Graphs for Entity-Level Private Learning on Raw Relational Data**. QuotaGuard constructs a public candidate graph from entity texts, applies a fixed degree bound to confidential relations, and trains a private cross-catalogue retriever. The package includes the complete data-to-figure pipeline, paired-seed comparisons, interval privacy certificates, and portable numerical results for every experimental table and all four figures.

## Layout

```text
configs/data/       OpenAlex extraction for CS, MAT and BIO
configs/train/      Nine training configurations
scripts/           Data, structures, training, comparison and plotting commands
src/quotaguard/     Public structures, private training and accounting
tests/             Synthetic structure, data and paired-analysis tests
results/           CSV and JSON numerical summaries
figures/           Four vector PDF figures
data/              Downloaded data and model cache (created locally)
runs/              Computed structures, checkpoints and reports (created locally)
```

Run all commands below from this repository root. The full training pipeline uses Linux, an NVIDIA GPU, CUDA and POSIX process locks. The numerical tables and figures can be inspected on a CPU machine. The C++ tests also support Windows with a compatible compiler.

## Environment

Use Python **3.11** or later. The recorded training environment used Python 3.11.17 and PyTorch 2.4.1 with CUDA 11.8; the test suite also passes on Python 3.13. `requirements.txt` preserves the recorded scientific-package versions: NumPy 2.2.6, SciPy 1.17.1, scikit-learn 1.9.1, PyTorch 2.4.1, Transformers 4.51.3, PEFT 0.21.0 and Opacus 1.5.4. The remaining pins come from the recorded environment package list. The recorded version of `python-flint` is unavailable, so that dependency is deliberately unpinned; it provides Arb interval arithmetic for certificates.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
```

`python scripts/check.py` runs compilation, the complete synthetic test suite and all four figure commands. On Linux the same check is available as `bash scripts/check.sh`.

The local CPU checks used NumPy 2.1.3, SciPy 1.18.0, scikit-learn 1.9.0, PyTorch 2.10.0, Matplotlib 3.11.0, PyYAML 6.0.3 and pytest 9.1.1. These checks cover synthetic data and summaries; they do not establish GPU training equivalence across package versions. Tests that require Opacus are explicitly skipped when it is absent.

### C++ kernel

Install `g++` or `clang++` with C++17 support and put it on `PATH`. The first structure operation automatically compiles `src/quotaguard/data/structures_kernel.cpp` with `-O3 -std=c++17 -shared` and `-fPIC` on Linux. Windows uses the corresponding DLL flags. Products are cached under `tmp/structure_kernel/`, keyed by the source checksum. To compile and load it explicitly:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from quotaguard.data.structures import kernel; kernel()"
```

## Data acquisition

[OpenAlex](https://openalex.org/) metadata is released under **CC0**. The extraction uses English articles and conference papers from 2015–2019 in Computer Science (CS), Materials Science (MAT), and Biochemistry (BIO). The first author's institution countries determine the CN and US catalogues. Within-catalogue citations become undirected relations, with duplicates and self loops removed. Each catalogue reserves 10,000 relations for testing, then 10% of the remaining relations for validation. BIO-CN calibrates public pair probabilities; BIO-US selects structures; CS and MAT provide the final retrieval comparisons.

An optional API key is read **only** from `OPENALEX_API_KEY`. Set it in your shell before retrieval. If it is unset, requests use anonymous access. The key is sent as an HTTP bearer header and excluded from URLs, cache keys and manifests. Pages are cached, checked and reused; quota exhaustion stops retrieval according to `Retry-After`.

```bash
python scripts/build_openalex.py --config configs/data/cs.yaml
python scripts/build_openalex.py --config configs/data/mat.yaml
python scripts/build_openalex.py --config configs/data/bio.yaml
```

Each output under `data/openalex_{cs,mat,bio}/` includes entity titles, abstracts, raw relations, disjoint splits and a checksum manifest. `--cache-only` repeats extraction using cached pages without network access. `--smoke` uses a small subset for checking acquisition and is not a paper-data replacement. OpenAlex changes over time: retrieval dates and checksums identify each local extract. The bundled summaries preserve the numerical results of the paper's extracts.

Manifest pins in configurations are optional. A supplied pin is verified; without one, the locally rebuilt manifest is used and its file checksums are still verified. Training logs explicitly identify an omitted dataset-manifest pin. Never modify the data, source code or configuration during a multi-stage structure run: each stage seals its artifacts and verifies earlier stages.

## Reproduce the tables and figures

### 1. Candidate coverage — Figure 1 (`coverage`)

```bash
python scripts/candidate_coverage.py --config configs/candidate_coverage.yaml
```

This computes union and mutual coverage for both text representations and neighbour counts 16, 64 and 256. It checks entity disjointness before accessing auxiliary training relations.

### 2. Public structures — Tables 1–3 (data, retention and stability)

```bash
python scripts/build_structures.py --stage 1 --output runs/structures
python scripts/build_structures.py --stage 2 --output runs/structures
python scripts/build_structures.py --stage 3 --output runs/structures
```

Stage 1 calibrates on BIO-CN, stage 2 selects on BIO-US, and stage 3 measures the selected family and its controls on CS and MAT. The grid, all five construction seeds, capacity settings and 1,000 deletion checks are specified in `configs/structures.yaml`. The public-groups control is specified in `configs/public_groups.yaml` and constructed by the same pipeline. Stage 3 writes retention and stability measurements to `runs/structures/stage3/report.json` and serialized public structures beside it.

### 3. Build the two training datasets per field

```bash
python scripts/build_training_data.py --config configs/training_data_cs.yaml
python scripts/build_training_data.py --config configs/training_data_mat.yaml
```

The fixed structure is applied to raw training relations. Validation and test files are copied unchanged and verified. The candidate-graph raw-accountant configuration shares the candidate-graph dataset. Training uses the public proxy-expected retained count for the sampling rate, public entity counts for delta, and the specified fixed structure for truncation.

### 4. Retrieval — Table 4 (utility), Figures 2 and 3 (`decomposition`, `privacy`)

Each command trains both negative strategies, both directions and seeds 401, 409, 419, 421 and 431. Choose the GPU IDs and concurrent processes per GPU to fit available memory. The examples use one process per GPU:

```bash
python scripts/train.py --config configs/train/cs_candidate_graph_eps10.yaml --mode confirm --gpus 0,1,2,3 --per-gpu 1
python scripts/train.py --config configs/train/cs_public_groups_eps10.yaml --mode confirm --gpus 0,1,2,3 --per-gpu 1
python scripts/train.py --config configs/train/cs_candidate_graph_raw_accountant_eps10.yaml --mode confirm --gpus 0,1,2,3 --per-gpu 1
python scripts/train.py --config configs/train/mat_candidate_graph_eps10.yaml --mode confirm --gpus 0,1,2,3 --per-gpu 1
python scripts/train.py --config configs/train/mat_public_groups_eps10.yaml --mode confirm --gpus 0,1,2,3 --per-gpu 1
python scripts/train.py --config configs/train/cs_candidate_graph_eps4.yaml --mode confirm --gpus 0,1,2,3 --per-gpu 1
python scripts/train.py --config configs/train/cs_public_groups_eps4.yaml --mode confirm --gpus 0,1,2,3 --per-gpu 1
python scripts/train.py --config configs/train/cs_candidate_graph_eps1.yaml --mode confirm --gpus 0,1,2,3 --per-gpu 1
python scripts/train.py --config configs/train/cs_public_groups_eps1.yaml --mode confirm --gpus 0,1,2,3 --per-gpu 1
python scripts/compare_structures.py --output runs/comparison_cs
python scripts/compare_budgets.py --output runs/comparison_fields_budgets
```

The two comparison scripts compare the candidate graph with public groups: hard-negative batch MRR, paired by seed, two-sided 95% Student intervals with four degrees of freedom, and an effect threshold of **0.01** in both directions. Corpus metrics and the decomposition through the raw-graph accountant are also reported. `R1` denotes uniform negatives and `H1` denotes hard negatives in the training reports.

The pretrained model is `bert-base-uncased`, downloaded through the model library to `data/model_cache`. For offline training, populate that cache beforehand or point `model.name` to the same local model and tokenizer. Keep the model weights, configuration and tokenizer identical across all compared runs.

### 5. Interval privacy certificates

```bash
python scripts/certify_noise.py --runs runs/train/cs_candidate_graph_eps10/confirm runs/train/cs_public_groups_eps10/confirm runs/train/cs_candidate_graph_raw_accountant_eps10/confirm runs/train/mat_candidate_graph_eps10/confirm runs/train/mat_public_groups_eps10/confirm runs/train/cs_candidate_graph_eps4/confirm runs/train/cs_public_groups_eps4/confirm runs/train/cs_candidate_graph_eps1/confirm runs/train/cs_public_groups_eps1/confirm --output runs/certificates --workers 8
```

This checks all 18 configuration/direction combinations with 192- and 256-bit interval arithmetic. Candidate graphs use the bounded-degree accountant at the training noise over T steps. Public groups and the raw-accountant control use the raw-graph accounting rule, including the exact noise-squared comparison and 2T accounting steps.

### 6. Construction cost — Figure 4 (`scaling`)

```bash
python scripts/construction_scaling.py --config configs/construction_scaling.yaml --output runs/scaling
```

The cost experiment reads public entity texts only. It uses nested samples of the six-catalogue union, four CPU threads, and a separate process for each size.

### 7. Summaries and vector figures

```bash
python scripts/summarize_results.py --run-root runs --output results
python scripts/make_figures.py
```

The summary command requires all nine complete training configurations, structure and coverage reports, scaling results and certificates. Save a copy of the bundled `results/` before replacing it with newly computed summaries. Analysis output directories must be empty when first used.

To regenerate the four paper figures directly from the included numerical summaries, run just `python scripts/make_figures.py`. It reads only package-local `results/` and produces `figures/coverage.pdf`, `decomposition.pdf`, `scaling.pdf` and `privacy.pdf`. All plots retain editable vector labels. Decomposition bars show mean paired gains, whiskers show two-sided 95% t intervals, and dots show the five training seeds.

## Numerical results

Each CSV has a corresponding JSON array. Metrics are fractions; multiply by 100 for the utility table. Retention and stability rows cover the structures reported in the paper at the deployment construction seed (101) with native capacity and `L=64`.

| File | Contents |
|---|---|
| `dataset_statistics.csv` | Six catalogue sizes and raw training relation counts; CS/MAT give Table 1 |
| `retention.csv` | Retained relations, coverage, quotas and cost of the reported structures; Table 2 |
| `stability.csv` | Deletion summaries, sample counts and maximum changes; Table 3 |
| `per_run_metrics.csv` | Per-run field, structure, accountant, budget, direction, negatives, seed, noise, delta and the retrieval metrics behind the reported comparisons; Table 4 and privacy curves |
| `paired_comparisons.csv` | Seed-paired intervals of the reported comparisons, including the structure/accounting decomposition |
| `candidate_coverage.csv` | 24 auxiliary coverage measurements; Figure 1 |
| `construction_scaling.csv` | Five public-entity sizes, neighbour-search/scoring/selection time and memory; Figure 4 |
| `noise_certificates.csv` | 18 certified privacy upper bounds and exact rational orders |

Table 2 uses public groups and the candidate graphs built from titles and from titles and abstracts; Table 3 uses public groups and the title-and-abstract candidate graph. For Table 4 use epsilon 10 and hard negatives, averaging the five seeds within each field/direction/structure; the candidate-minus-group intervals are in `paired_comparisons.csv`. Uniform-negative results and the budgets 1 and 4 use the same columns.

## Hardware and time

The training reference used an NVIDIA A800 GPU with multiple concurrent runs. Allow approximately **1–2 hours per training run**; the full set contains 180 runs. Actual throughput depends on GPU sharing, model cache and evaluation. The largest public construction has about 1.02 million entities and 50 million candidate pairs; exact nearest-neighbour search takes about 3.5 hours on four CPU threads with peak memory around 10.7 GB. Pair scoring and greedy selection take seconds. Exact sizes and separate timings are in `construction_scaling.csv`. Structure-grid evaluation also uses CPU and memory; its configured process limit is 16, with two CPU threads per process.

## License

The code is released under the MIT License; see `LICENSE`. OpenAlex metadata is CC0.

## Citation

```bibtex
@article{quotaguard,
  title = {Stable Degree Bounding with Public Candidate Graphs for Entity-Level Private Learning on Raw Relational Data},
  author = {AUTHORS},
  journal = {Proceedings of the VLDB Endowment},
  year = {YEAR},
  volume = {VOLUME},
  number = {NUMBER},
  pages = {PAGES},
  doi = {DOI}
}
```
