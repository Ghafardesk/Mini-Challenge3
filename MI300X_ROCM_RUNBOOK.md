# MI300X / ROCm real-hardware runbook

This guide takes the repository from an AMD-hosted Jupyter environment to a
hardware validation run. The challenge contract is authoritative: the graded
artifact is a Docker image, not a notebook. Keep the image based on the required
ROCm base image, and test the image with network access disabled.

## 1. Confirm the notebook really has MI300X access

Use the notebook kernel/environment supplied by the AMD host. Do **not** try to
install or replace the host kernel driver from inside a hosted notebook or
container; the provider must expose the GPU and ROCm device nodes.

Run in a notebook cell:

```python
import sys
import torch

print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("ROCm/HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn((1024, 1024), device="cuda")
    print("GPU smoke test:", (x @ x).mean().item())
```

The host must identify an MI300X (`gfx942`), `torch.version.hip` must be
non-empty, and the matrix operation must complete on the GPU. In a terminal,
also run:

```bash
amd-smi list
rocminfo | grep -m 5 -E 'Name:.*gfx942|Marketing Name:.*MI300X'
ls -l /dev/kfd /dev/dri
```

If these fail, stop and ask the notebook provider to enable the MI300X/ROCm
runtime; installing Python packages will not fix a missing host driver/device.
AMD's current [ROCm GPU support list](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
identifies MI300X as `gfx942`.

## 2. Use one compatible ROCm, PyTorch, Python, and vLLM stack

The repository's challenge image is pinned by the contract to:

```text
rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
```

Do not install CUDA PyTorch, and do not run an unqualified `pip install -U
torch`. vLLM contains compiled GPU components: install a vLLM build documented
for the **exact** Python, ROCm, and PyTorch combination provided by the AMD
notebook/base image. Start with the [official vLLM GPU installation matrix](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
and the [ROCm installation guide](https://rocm.docs.amd.com/en/latest/install/rocm.html).
Do not assume that a wheel for a different ROCm or Python version will work.

The current `requirements.txt` is a broad development manifest (`vllm>=0.3.0`),
not a tested ROCm lockfile. The root `Dockerfile` installs it with `--no-deps`
to avoid replacing the base image's ROCm PyTorch. Consequently, a successful
`pip install` alone does **not** prove that vLLM or its runtime dependencies
are installed. Before a long model download, confirm these imports in the
notebook's selected kernel:

```python
import torch
import transformers
import sentence_transformers
import vllm

assert torch.cuda.is_available() and torch.version.hip
print("torch", torch.__version__, "HIP", torch.version.hip)
print("transformers", transformers.__version__)
print("sentence-transformers", sentence_transformers.__version__)
print("vLLM", vllm.__version__)
```

If an import is missing or incompatible, install the matching AMD/vLLM stack
first, then install the project's Python dependencies in that same kernel. Do
not proceed until the imports above succeed and PyTorch still reports HIP and
GPU availability. The daemon directly imports `transformers.AutoTokenizer`
and vLLM during vision-engine startup.

## 3. Review dependencies and prepare gated model access

Before changing the kernel environment, inspect what pip would install:

```bash
python3 -m pip install --dry-run --report /tmp/mc3-install-plan.json \
  -r requirements.txt
```

Do not apply this generic manifest blindly: `vllm>=0.3.0` is not a ROCm
version pin. Confirm that the plan keeps the existing ROCm PyTorch and the
selected compatible vLLM build unchanged. If it proposes replacing either,
cancel and install the remaining project packages through a dependency plan
that preserves the tested AMD stack. Once installed, rerun the import/GPU
checks in step 2. The root Dockerfile also uses `--no-deps`; therefore the
final image must separately be checked for all imports and dependencies rather
than assuming the notebook's installed packages carry over.

The model `meta-llama/Llama-3.2-11B-Vision-Instruct` is gated: accept its
Hugging Face terms and use an account that has been approved for access.

For a notebook-only local test, enter the token without putting it in a cell's
source or a committed file, then download the models from the repository root:

```python
from getpass import getpass
import os
import subprocess
import sys

os.environ["HF_TOKEN"] = getpass("Hugging Face access token: ")
subprocess.run([sys.executable, "download_model.py"], check=True)
os.environ.pop("HF_TOKEN", None)
```

This stores the model snapshots under `./models/`. Keep enough persistent
storage for the weights, and do not include token values in logs or notebook
outputs. The download must finish before any offline/container test.

## 4. Build the challenge image without embedding the token

The Docker build needs BuildKit and the gated-model token as a build secret.
From the repository root, set `HF_TOKEN` in the terminal environment without
echoing it, then build:

```bash
read -rsp "Hugging Face token: " HF_TOKEN
export HF_TOKEN
echo
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HF_TOKEN \
  -t mini-challenge3:mi300x .
unset HF_TOKEN
```

The `Dockerfile` mounts that secret only for `download_model.py`; it is not
saved in an image layer. The `.dockerignore` excludes local virtual
environments, model directories, and output files from the build context.
Check that Docker is available to the notebook user and can access the host's
ROCm devices. On a self-managed Linux Docker host, a typical MI300X container
launch must expose `/dev/kfd` and `/dev/dri`; use the AMD host's documented
container flags/runtime. A hosted notebook that does not grant Docker/device
access can still run the local Python checks, but cannot validate the final
challenge image there.

Before scoring, verify the image is based on the mandated `FROM` line, its
uncompressed size is at most **60 GiB**, and model weights are present in the
image. The evaluator has no network access; never rely on a model download at
container startup or query time.

## 5. Run on the notebook GPU before building the image

Recreate the challenge's empty-directory and unreadable-file cases, then index
the supplied sample corpus:

```bash
sh mc3-starter-kit/mc3-starter-kit/setup.sh
python3 app.py --index mc3-starter-kit/mc3-starter-kit/mc3-corpus
```

The `app.py --index` command starts the daemon, loads both models, and indexes
the corpus.
The daemon stays alive so subsequent app invocations reuse the loaded model and
index. Do not restart the notebook kernel/container between index and query
tests. Watch startup and peak VRAM in a second terminal:

```bash
watch -n1 'amd-smi metric --mem | grep USED_VRAM'
```

Then test both image questions and a refusal, using the exact sample prompts in
`mc3-starter-kit/mc3-starter-kit/sample-questions.json`:

```bash
python3 app.py --corpus mc3-starter-kit/mc3-starter-kit/mc3-corpus \
  --query-id query_07 \
  --query "Which backplane pin carries THERM_ALERT# on the TQ-40?"
cat output/query_07_output.json

python3 app.py --corpus mc3-starter-kit/mc3-starter-kit/mc3-corpus \
  --query-id query_08 \
  --query "What board revision is printed on the asset label?"
cat output/query_08_output.json

python3 app.py --corpus mc3-starter-kit/mc3-starter-kit/mc3-corpus \
  --query-id query_10 \
  --query "What is the unit price of the TQ-40 at 10,000 unit volume?"
cat output/query_10_output.json
```

Expected answers/citations are respectively `B14` / `specs/backplane_pinout.png`,
`REV-C2` / `support/asset_label.jpg`, and empty answer / empty citations. Check
the complete ten-question sample set, not just the general one-query selfcheck:
the challenge awards no credit for an answer with incorrect citations.

To score all ten sample cases from a notebook cell after the index step:

```python
import json
import re
import subprocess
import sys
from pathlib import Path

repo = Path.cwd()
corpus = repo / "mc3-starter-kit/mc3-starter-kit/mc3-corpus"
questions = json.loads(
    (repo / "mc3-starter-kit/mc3-starter-kit/sample-questions.json").read_text()
)["queries"]

def normalize(value):
    return re.sub(r"[-.·_\s]", "", str(value).upper().strip())

subprocess.run(
    [sys.executable, str(repo / "app.py"), "--index", str(corpus)],
    cwd=repo,
    check=True,
    timeout=600,
)
score = 0
for item in questions:
    query_id = f"query_{item['n']:02}"
    subprocess.run(
        [
            sys.executable, str(repo / "app.py"),
            "--corpus", str(corpus),
            "--query-id", query_id,
            "--query", item["query"],
        ],
        cwd=repo,
        check=True,
        timeout=30,
    )
    result = json.loads(
        (repo / "output" / f"{query_id}_output.json").read_text()
    )
    answers = {normalize(item["expected_answer"])}
    answers.update(normalize(alias) for alias in item.get("answer_aliases", []))
    answer_ok = normalize(result["answer"]) in answers
    citations_ok = set(result["citations"]) == set(item["expected_citations"])
    passed = answer_ok and citations_ok
    score += int(passed)
    print(
        f"{'PASS' if passed else 'FAIL'} {query_id}: "
        f"answer={result['answer']!r}, citations={result['citations']}"
    )

print(f"Sample score: {score}/{len(questions)}")
assert score == len(questions), "Fix every answer/citation mismatch before release."
```

## 6. Test the built image against the actual contract

Run the bundled setup and selfcheck from the starter-kit directory on a host
whose Docker runtime exposes the GPU:

```bash
cd mc3-starter-kit/mc3-starter-kit
./setup.sh
python3 selfcheck.py mini-challenge3:mi300x mc3-corpus
```

The selfcheck checks image existence, no-network startup, indexing, one query,
JSON shape, and prints the VRAM-monitoring command. It is **not** a complete
accuracy test and does not itself compare all ten expected answers/citation
sets. If Docker does not expose AMD devices by default, launch a container with
the provider's ROCm device flags and run the same `/app/app.py` index/query
commands inside it. Also perform the unreadable-file case without root's
`DAC_OVERRIDE` capability, as the contract requires; a root-only local test can
read a mode-000 file and give a false pass.

## 7. Final acceptance gates to measure on MI300X

Use the contract's exact limits. Record evidence from the final image, not just
from a notebook-only run:

| Gate | Required result |
|---|---|
| GPU | MI300X/gfx942; ROCm PyTorch reports HIP and GPU available |
| OCR | Both image questions correct, with exact corpus-relative citations |
| Refusal | The encrypted-PDF price query returns `""` and `[]` |
| Other questions | All ten sample answers and citation sets match; no answer is guessed from model memory |
| Network | Index/query pass with container network disabled |
| Startup | Container start + model load + full-corpus index complete within 10 minutes |
| Query | Each query process writes `/app/output/<query-id>_output.json` within 30 seconds |
| Total run | All ten questions complete within 10 minutes after startup |
| VRAM | GPU memory peak stays from 1 GiB through 48 GiB, inclusive of tolerance rules |
| Image size | Uncompressed image is no larger than 60 GiB |
| Corpus edge cases | Empty folder, unreadable file, encrypted PDF, and unsupported binary do not abort the index |

An MI300X may have more physical VRAM than the **48 GiB challenge ceiling**;
the challenge limit still applies. If the measured peak exceeds it, lower vLLM
concurrency/context allocation in the model initialization and repeat the full
index/OCR/query benchmark. If it stays below 1 GiB, investigate CPU fallback
instead of adding artificial GPU allocations. The daemon currently pins
`CUDA_VISIBLE_DEVICES=0` and initializes vLLM with `max_model_len=4096` and
`max_num_seqs=16`, so verify these actual settings on the target runtime.

## What Codespaces cannot certify

The Codespaces environment cannot establish ROCm kernel/driver compatibility,
real gfx942 dispatch, vision-model OCR quality on the MI300X path, GPU VRAM
peaks, GPU runtime container passthrough, or the challenge's measured startup
and per-question timing. Treat all of those as hardware release gates and keep
the command output/metrics from the AMD notebook as the test record.
