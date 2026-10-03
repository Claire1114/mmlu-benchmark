# MMLU Benchmark Pipeline

Enterprise-grade automated evaluation pipeline for Large Language Models (LLMs), covering local Hugging Face CausalLM checkpoints and cloud Gemini APIs.

## 1. Project Overview & Testing Objectives

This repository implements a systematic MMLU (`cais/mmlu`) benchmark workflow for Pegatron R&D. The pipeline is designed to:

- Load the configured multi-subject MMLU suite (`dataset.categories`, four domains × two subjects) with mode-driven sampling (`dataset.active_mode` → `modes.<mode>.sample_size_per_subject`, seeded by `project.seed`).
- Run inference against heterogeneous model backends (local Hugging Face CausalLM and Gemini).
- Score predictions independently of model I/O and persist raw JSONL logs plus summary tables.

**Testing objectives**

- Unit tests with mocks for data loading, model adapters, and metrics (no live API keys, no weight downloads).
- Smoke tests against a small `sample_size` (default: 5) before full-subject runs.
- Regression coverage for configuration parsing and credential lookup via environment variables.

## 2. Architecture & File Structure

Modules are strictly decoupled:

| Concern | Location | Responsibility |
| --- | --- | --- |
| Data loading | `src/dataset_loader.py` | MMLU fetch across configured subjects, mode-driven seeded sampling, category mapping |
| Model interfaces | `src/models/` | Hugging Face and Gemini drivers |
| Metric evaluation | `src/evaluator.py` | Accuracy and report aggregation |
| Runtime parameters | `configs/eval_config.yaml` | Dataset and model settings |

```text
mmlu-benchmark/
├── .cursorrules
├── .gitignore
├── pytest.ini
├── requirements.txt
├── README.md
├── configs/
│   └── eval_config.yaml
├── src/
│   ├── __init__.py
│   ├── dataset_loader.py
│   └── models/
│       └── __init__.py
├── tests/
│   └── __init__.py
├── results/
│   ├── raw/          # JSONL inference logs
│   └── summary/      # Exported metrics tables
├── docs/
│   └── screenshots/  # AI tool evidence
├── report/           # Technical PDF
└── logs/             # Execution and review logs
```

Credentials are never stored in Git. Gemini drivers must call `os.getenv("GEMINI_API_KEY")`.

## 3. Environment Setup & Dependency Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For cloud Gemini evaluation, export the key in the shell (do not write it into YAML or source):

```bash
export GEMINI_API_KEY="your-key-here"
```

Optional local overrides belong in `configs/eval_config.local.yaml` (gitignored).

## 4. How to Run Unit Tests & Smoke Tests

From the repository root:

```bash
pytest
```

With coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

Smoke tests should honor `dataset.modes[dataset.active_mode].sample_size_per_subject` from `configs/eval_config.yaml` (smoke_test baseline: 10 per subject) and mock model backends. Do not download `Qwen/Qwen2.5-0.5B-Instruct` or `HuggingFaceTB/SmolLM2-1.7B-Instruct` inside unit tests.

## 5. Execution Guide (`main.py`)

`main.py` is the planned CLI entry point (implemented in a later engineering step). Expected usage:

```bash
python main.py --config configs/eval_config.yaml
```

The runner should:

1. Parse YAML configuration (Pydantic-validated).
2. Load the MMLU subset via `src/dataset_loader.py`.
3. Dispatch each configured model in `src/models/`.
4. Score outputs in `src/evaluator.py`.
5. Write raw JSONL under `results/raw/` and summaries under `results/summary/`.

## 6. AI Tool Usage & Development Logs

Record Cursor/agent sessions, review notes, and screenshots under:

- `docs/screenshots/` — interaction evidence
- `logs/` — execution logs and AI code-review transcripts
- `report/` — 2–3 page PDF technical report

When adding a module, also add `tests/` coverage with mocks, following `.cursorrules`.
