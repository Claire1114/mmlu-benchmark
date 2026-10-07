# Pegatron MMLU Benchmark Pipeline

Enterprise-grade automated evaluation pipeline for Large Language Models (LLMs), supporting heterogeneous model architectures including local Hugging Face CausalLM models and cloud-based Gemini APIs.

---

## 1. Project Overview & Testing Objectives

This repository implements a systematic and reproducible MMLU (`cais/mmlu`) benchmark workflow tailored for Pegatron R&D.

The pipeline is designed to provide a standardized evaluation framework across different model architectures and inference backends.

### Key Objectives

- **Multi-Subject Domain Sampling**: Evaluates models across 4 core domains and 8 representative subjects:
  - `college_computer_science`
  - `high_school_mathematics`
  - `philosophy`
  - `world_religions`
  - `econometrics`
  - `high_school_psychology`
  - `clinical_knowledge`
  - `professional_law`
- **Flexible Evaluation Modes**: Configured with deterministic seeded sampling (`project.seed: 42`):
  - `smoke_test`: 10 samples per subject (80 questions) for end-to-end pipeline verification.
  - `demo`: 30 samples per subject (240 questions) for model performance comparison.
- **Heterogeneous Model Backends**: Provides a unified evaluation interface for:
  - **Local PyTorch / Hugging Face models**: `Qwen2.5-0.5B-Instruct`, `SmolLM2-1.7B-Instruct`
  - **Cloud API**: `gemini-3.1-flash-lite`
- **Standardized Output & Fault Isolation**:
  - Multi-tier answer extraction: Strict → Fallback → Sentinel
  - Immediate record flushing to reduce data-loss risk
  - Automated metric calculation and result persistence
  - Defensive handling of malformed or unexpected model outputs

---

## 2. Benchmark Results

Models were evaluated on the official MMLU test split using **5-shot** prompting across 240 questions (8 subjects × 30 samples).

| Model | Backend / Device | Overall Accuracy | Avg. Latency (s/q) | Status |
| :--- | :--- | :---: | :---: | :---: |
| **`gemini-3.1-flash-lite`** | Cloud API (Google) | **86.67%** (208/240) | 2.12s | Completed |
| **`smollm2-1.7b`** | Local HF (Auto / MPS) | **50.42%** (121/240) | 2.23s | Completed |
| **`qwen2.5-0.5b`** | Local HF (Auto / MPS) | **38.75%** (93/240) | 0.81s | Completed |

The observed performance gradient demonstrates that the pipeline can distinguish meaningful differences in model capability across architectures and parameter scales.

> **Evaluation Note**: Accuracy and latency are reported on the same 240-question evaluation set under the same 5-shot configuration.

---

## 3. Project Architecture

```text
mmlu-benchmark/
├── .cursorrules
├── .env
├── .gitignore
├── pytest.ini
├── requirements.txt
├── README.md
│
├── configs/
│   └── eval_config.yaml
│
├── src/
│   ├── dataset_loader.py
│   ├── evaluator.py
│   ├── runner.py
│   │
│   └── models/
│       ├── base.py
│       ├── huggingface.py
│       ├── gemini.py
│       └── openai_compatible.py
│
├── tests/
│   ├── test_dataset_loader.py
│   ├── test_evaluator.py
│   ├── test_models.py
│   ├── test_runner.py
│   └── test_main.py
│
├── results/
│   ├── demo/
│   └── smoke_test/
│
├── logs/
│
└── docs/
    └── step8_few_shot_extension.md

### Core Components

| Component | Description |
| :--- | :--- |
| `.cursorrules` | AI agent engineering rules and defensive development guidelines |
| `configs/eval_config.yaml` | Centralized evaluation settings and model registry |
| `src/dataset_loader.py` | MMLU dataset loading, validation, deterministic sampling, and slicing |
| `src/evaluator.py` | Answer parsing and hierarchical metric calculation |
| `src/runner.py` | Evaluation execution, request throttling, and resource management |
| `src/models/huggingface.py` | Local Hugging Face CausalLM adapter |
| `src/models/gemini.py` | Gemini API client |
| `src/models/openai_compatible.py` | Generic OpenAI-compatible API adapter |
| `tests/` | Offline unit and integration-level tests |
| `results/` | Benchmark predictions and metric summaries |
| `logs/` | Execution and AI-assisted development logs |
| `docs/` | Technical specifications and development documentation |

---

## 4. Environment Setup & Configuration

### Prerequisites
- Python 3.10+
- Tested with Python 3.13
- macOS with Apple Silicon / MPS support or Linux with CUDA support

### Installation

Clone the repository:
```bash
git clone [https://github.com/Claire1114/mmlu-benchmark.git](https://github.com/Claire1114/mmlu-benchmark.git)
cd mmlu-benchmark

Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate

Install dependencies:
```bash
pip install -r requirements.txt

### API Key Setup
Create a .env file in the project root:
```bash
GEMINI_API_KEY=your_gemini_api_key_here

Security: Never commit .env or API keys to version control.


## 5. Execution Guide

The evaluation pipeline is executed through `main.py`.

### 5.1 Smoke Test

Run a lightweight end-to-end verification using 10 samples per subject:

```bash
python main.py \
  --mode smoke_test \
  --models gemini-3.1-flash-lite \
  --shots 5


This mode is intended to verify:

⚬ Dataset loading
⚬ Prompt construction
⚬ Model inference
⚬ Answer extraction
⚬ Metric calculation
⚬ Result persistence
⚬ Error handling

5.2 Demo Benchmark

Run the standard 240-question evaluation across all configured models:

python main.py \
  --mode demo \
  --models qwen2.5-0.5b smollm2-1.7b gemini-3.1-flash-lite \
  --shots 5


Configuration:

⚬ 8 subjects × 30 questions = 240 questions

5.3 Key CLI Options

Option	Description
--mode	Evaluation mode: smoke_test or demo
--models	Target model(s) to evaluate
--shots	Number of few-shot examples, e.g. 0 for zero-shot or 5 for 5-shot
--delay	Delay between API requests to reduce rate-limit risk
--limit	Optional limit for debugging a smaller number of test samples

Example:

python main.py \
  --mode smoke_test \
  --models gemini-3.1-flash-lite \
  --shots 0 \
  --limit 5

6. Unit Testing & Code Quality

The codebase follows defensive engineering practices with fully mocked offline testing.

Run the complete test suite:

pytest tests/ -v


Test Results

⚬ 293 tests passed in approximately 5.4 seconds.

Test Coverage Highlights

Answer Parsing Robustness

Tests validate:

⚬ Strict answer extraction
⚬ Fallback recovery
⚬ Regex boundary protection
⚬ Non-string and malformed model outputs
⚬ Safe handling of ambiguous answer formats

Example extraction pattern:

(?i)\bthe correct answer is\s*\(?\b([A-D])\b\)?


The parser uses a hierarchical strategy:

Strict Extraction
       ↓
Fallback Extraction
       ↓
Sentinel / Unresolved


Fault Tolerance

Mocked tests simulate:

⚬ Network timeouts
⚬ HTTP 429 rate-limit errors
⚬ HTTP 5xx server errors
⚬ Retry behavior
⚬ Exponential backoff

Resource Management

Tests verify:

⚬ Deterministic CUDA/MPS memory cleanup
⚬ Local model batch handling
⚬ Resource cleanup after inference
⚬ Recovery from inference failures

Reproducibility

Dataset tests verify:

⚬ Deterministic seeded sampling
⚬ Stable question IDs
⚬ Subject-level sample selection
⚬ Dataset validation
⚬ Edge-case handling

7. AI Tool Utilization & Refactoring Logs

AI-assisted development tools were used throughout the implementation and review process, including:

⚬ Cursor IDE
⚬ Cline
⚬ Claude / Gemini coding assistants

AI tools were used as development assistants rather than as substitutes for automated verification.

Development Applications

Refactoring & Modularity

AI-assisted code review was used to improve separation of concerns between:

Dataset Loading
       ↓
Model Interface
       ↓
Inference Runner
       ↓
Answer Evaluation
       ↓
Metric Calculation
       ↓
Result Persistence


This helped maintain modular interfaces across local and cloud-based model backends.

Defensive Error Handling

AI-assisted review was used to identify and strengthen:

⚬ Input validation
⚬ Output validation
⚬ Regex boundary protection
⚬ Malformed response handling
⚬ API failure recovery
⚬ Prompt-injection-aware processing

Unit Test Generation

AI assistance was used to develop and review edge-case tests covering:

⚬ Dataset validation
⚬ Sampling reproducibility
⚬ Answer parsing
⚬ API retry behavior
⚬ Resource management
⚬ Error recovery

All generated or modified code was subsequently verified through the project’s automated test suite.

Development Audit Trail

Detailed development records, prompts, refactoring notes, and AI-assisted review logs are maintained under:

⚬ docs/
⚬ logs/

These records provide an auditable development history of AI-assisted implementation and refactoring.

8. Reproducibility

The benchmark is designed to support reproducible evaluation through:

⚬ Fixed random seed: 42
⚬ Explicit evaluation configuration
⚬ Deterministic dataset sampling
⚬ Standardized prompt construction
⚬ Consistent answer parsing
⚬ Persisted prediction records
⚬ Automated metric calculation
⚬ Version-controlled source code

For comparable results, use the same:

⚬ Dataset split
⚬ Sampling seed
⚬ Number of shots
⚬ Subject selection
⚬ Sample count
⚬ Model configuration
⚬ Evaluation pipeline version

9. Results & Artifacts

Benchmark artifacts are stored under:

results/
├── demo/
│   ├── predictions_*.jsonl
│   ├── summary_*.csv
│   └── summary_*.json
│
└── smoke_test/
    └── ...


Prediction files contain per-question evaluation records, while summary files contain aggregated benchmark metrics.

10. Engineering Principles

The project follows several engineering principles:

⚬ Reproducibility — deterministic sampling and explicit configuration
⚬ Modularity — interchangeable model backends
⚬ Defensive Validation — validate inputs and model outputs at system boundaries
⚬ Fault Isolation — isolate individual request failures from the complete benchmark
⚬ Observability — persist predictions, metrics, and execution logs
⚬ Testability — favor deterministic offline tests over live API-dependent tests
⚬ Resource Awareness — explicitly manage local model memory and API request rates
⚬ Auditability — maintain development and AI-assisted refactoring records

11. Project Status

Component	Status
MMLU dataset integration	✅ Complete
Deterministic sampling	✅ Complete
Local Hugging Face inference	✅ Complete
Gemini API integration	✅ Complete
Multi-tier answer parsing	✅ Complete
Metric calculation	✅ Complete
Fault-tolerant execution	✅ Complete
Offline unit testing	✅ Complete
5-shot benchmark	✅ Complete
Demo evaluation	✅ Complete
AI-assisted development audit	✅ Complete

12. License

This project is intended for research, evaluation, and technical demonstration purposes.