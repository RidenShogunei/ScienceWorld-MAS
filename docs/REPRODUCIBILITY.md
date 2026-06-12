# Reproducibility Guide

## Supported Setup

- Windows or Linux
- Python 3.10 or 3.11
- Java runtime available on `PATH`
- NVIDIA GPU recommended for training
- ScienceWorld 1.2.3

The verified development machine used Python 3.11.4, Java 25, PyTorch
2.4.1+cu121, Transformers 5.5.4, PEFT 0.19.1, and bitsandbytes 0.48.1.
These versions are recorded for reference, not all imposed as universal
requirements.

## Fresh Machine

```powershell
git clone <repository-url>
cd ScienceWorld-MAS
conda env create -f environment.yml
conda activate scienceworld-mas

python prepare_multisquare.py
python audit_multisquare.py --output artifacts/data_audit.json
python generate_sft_data.py
python doctor.py --smoke-environment --output artifacts/doctor.json
python -m pytest -q
```

`doctor.py` verifies Java, Python, package availability, exact raw-data SHA256
hashes, and an actual ScienceWorld reset/step cycle.

## Models

Model weights and checkpoints are intentionally not committed. Pass either a
Hugging Face model ID or a local directory:

```powershell
$MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
```

For offline machines, download the model in advance and set `$MODEL` to that
local directory. Never add a machine-specific absolute model path to source
files or committed experiment configs.

## SFT

```powershell
python sft_trainer.py `
  --base-model $MODEL `
  --agents both `
  --epochs 2 `
  --save-dir artifacts/checkpoints/sft `
  --use-4bit
```

Every run writes adapter-local `metrics.json`. Preserve the command, Git commit,
data manifest, model identity, seed, and output directory when reporting a
result. Experiment JSON records adapter weight hashes, so checkpoints with the
same LoRA configuration remain distinguishable.

## Executable Evaluation

```powershell
python evaluate_environment.py `
  --base-model $MODEL `
  --main-adapter artifacts/checkpoints/sft/main_agent/best `
  --sub-adapter artifacts/checkpoints/sft/sub_agent/best `
  --split dev `
  --episodes 10 `
  --output artifacts/eval/sft_dev10.json
```

The output contains the complete trajectory for every episode, including Main
subtasks, raw generations, Sub actions, action validity, reward, score, and
observations.

## Data Provenance

Source dataset:

- Repository: `sangeun-park/Multi-Square`
- Repository revision used during initial validation:
  `13c81adfcab1fae83d67ec90317efab09a70f14f`
- High-level SHA256:
  `59cf6b2e78445fae67032b17ff391314c7e52ac7db8d5078c4ac1d1322e9e441`
- Low-level SHA256:
  `da9c35f82d1dae5c363b13c53572d50af07b453fd78457a9dc45f381ec39d29b`

The raw and converted datasets are ignored by Git. They must be regenerated
with the checked-in scripts.
