# ChainTranslate

English-to-French translation of Ethereum interface and educational text.

[Open notebook in Colab](https://colab.research.google.com/github/themisogynist/chaintranslate-web3-mt/blob/main/ChainTranslate.ipynb) · [Results](RESULTS.md) · [Analysis](docs/analysis.md) · [Translation samples](results/translations.txt)

**Assignment 3 · ICT 4403 · Foundations of Generative AI**  
Suraj Shuban · Registration **230953492** · MAHE, MIT Manipal

## Problem statement and objective

Web3 applications need translations that preserve technical meaning, quantities and security warnings. ChainTranslate compares a Transformer trained from scratch, a fine-tuned multilingual encoder-decoder, and two prompted decoder-only models on a public Ethereum parallel corpus. The objective is to measure translation quality and computational cost, and examine errors with ethical consequences. This is an experimental translation benchmark; the measured systems are not suitable for unattended publication.

**Experiment status:** basic Transformer and mT5 results are measured. Official LLaMA and Gemma runs are pending approved model access. Consequently, the four-model evaluation is incomplete; only measured systems are ranked.

## Dataset and provenance

Source: [ethereum.org website repository](https://github.com/ethereum/ethereum-org-website), revision `fb3514bd596848d74500429c43b955c999af839b`. English and French JSON entries are aligned by the same localization file and key. The corpus contains interface labels and educational passages about Ethereum, scaling, privacy and applications, with domain-specific vocabulary and named entities.

The downloader examines 85 paired files and obtains 3,541 pairs after length/content filtering. It removes 204 duplicate-source or identical-language pairs, leaving 3,337 unique parallel pairs before splitting. It strips HTML, decodes entities, normalizes whitespace, excludes placeholders and URLs, and retains English entries of 4–18 whitespace-separated words and French entries of 3–30 words. These limits favor short passages and restrict generalization to longer documents.

| Split | Used pairs | Available after grouped split | Represented files |
|---|---:|---:|---:|
| Training | 512 | 2,240 | 56 |
| Validation | 64 | 138 | 7 |
| Test | 32 | 959 | 4 |

Exact English-source deduplication occurs before splitting. Localization files are assigned to disjoint splits with seed 42 before bounded sampling, reducing page-level leakage. Semantic paraphrases and pretrained-model exposure remain possible. Every retained row has a stable ID, localization key and source/reference URLs. See [dataset summary](data/summary.json) and [attribution](DATASET_NOTICE.md).

## Models implemented

| Model | Architecture and adaptation | Tokenization |
|---|---|---|
| Basic Transformer | Two encoder and two decoder layers, hidden size 128, four heads, feed-forward 256, dropout 0.1; trained from scratch | 2,000-token SentencePiece BPE trained only on training pairs |
| mT5-small | `google/mt5-small`; LoRA fine-tuning of attention q/v projections, rank 8, alpha 16, dropout 0.05 | Native multilingual tokenizer |
| LLaMA 3.2 1B Instruct | Official `meta-llama/Llama-3.2-1B-Instruct`; three-shot prompting, no fine-tuning | Native tokenizer and chat template |
| Gemma 3 270M IT | Official `google/gemma-3-270m-it`; three-shot prompting, no fine-tuning | Native tokenizer and chat template |

Pinned model revisions are recorded in `chaintranslate/experiment.py`. The basic model uses AdamW, learning rate 0.001, weight decay 0.001, batch 32 and 30 epochs. mT5 uses AdamW, learning rate 0.001, default weight decay 0.01, batch 4 and three epochs. Both clip gradient norm at 1 and select the checkpoint with lowest validation token loss. mT5 updates 344,064 LoRA parameters, rather than all 300,176,768 pretrained parameters.

## Comparable evaluation conditions

All models receive the same English test strings and French references, identical test ordering, a 96-token output limit and greedy single-beam decoding for the main comparison. Both prompted models receive the same three training examples and instruction through their native chat templates. No validation/test examples appear in prompts. Random seeds are 42. The temperature experiment uses the same first four test pairs at 0.5, 1.0 and 1.5, with full-vocabulary sampling and seed 42.

The adaptation strategies, parameter counts, pretraining and tokenizers necessarily differ. Equal token limits do not imply equal word limits. This is a comparison of concrete systems, not a controlled causal test of architecture alone. The current CPU runs overlapped in execution; timing is descriptive and cannot establish a reliable speed ranking. The notebook runs all models sequentially for a more comparable hardware session.

## Execution and dependencies

Python 3.12 was used for the measured runs. Core dependencies are PyTorch, Transformers, PEFT, Accelerate, SentencePiece, SacreBLEU, NumPy and Matplotlib; versions are in [requirements.txt](requirements.txt). GPU acceleration is recommended for the pretrained models; CPU execution is supported but slower. Model downloads need internet access and sufficient disk space.

```bash
git clone https://github.com/themisogynist/chaintranslate-web3-mt.git
cd chaintranslate-web3-mt
python -m pip install -r requirements.txt
# Optional: recreate the fixed corpus from its pinned public source.
python -m chaintranslate.data --out data
python -m chaintranslate.experiment --model basic
python -m chaintranslate.experiment --model mt5
# Obtain approved access on the official model pages, then authenticate locally.
hf auth login
python -m chaintranslate.experiment --model llama
python -m chaintranslate.experiment --model gemma
python -m chaintranslate.report
python -m unittest discover -s tests -v
```

[ChainTranslate.ipynb](ChainTranslate.ipynb) provides the same workflow and secure interactive authentication. Access to [LLaMA](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and [Gemma](https://huggingface.co/google/gemma-3-270m-it) must be granted by their providers. Tokens and downloaded weights must not be committed. 

Each experiment writes model-specific outputs and can be run independently. A repeated training command retrains that model and replaces its result files. `results/checkpoints/` is ignored by Git. 

The notebook checks model access before launching training, runs experiments in a separate `runs/` directory, and exports the new outputs without overwriting the published measurements. CPU installations can use the [PyTorch CPU package index](https://download.pytorch.org/whl/cpu) for the pinned PyTorch version.

### Colab dependency recovery

The notebook removes Colab's optional `torchvision` package before installation because this text-only benchmark does not use it. A preinstalled `torchvision` build may not match the pinned PyTorch version and can prevent Transformers from importing mT5 with `RuntimeError: operator torchvision::nms does not exist`. The setup now checks all three pretrained model-class imports in a fresh process before training.

For an existing affected Colab session, run `python -m pip uninstall -y torchvision` using the notebook's Python interpreter, then retry the failed experiment in its existing output directory. Experiment subprocesses start with fresh imports, so a runtime restart is not required for this recovery. The tokenizer legacy and dtype deprecation warnings are separate from this import failure.

The training cell streams subprocess output and retains `run_dir` when retried. Models with metrics, translations and temperature outputs are skipped. To intentionally start a new experiment, assign a new `run_dir` first. Reporting still validates all recorded predictions before accepting results.

## Results obtained and model comparison

| Model | BLEU ↑ | chrF ↑ | Training/adaptation seconds | Test inference seconds | Base parameters |
|---|---:|---:|---:|---:|---:|
| Basic Transformer | 0.44 | 13.16 | 134.99 | 2.49 | 1,177,040 |
| mT5-small + LoRA | 1.44 | 18.04 | 523.22 | 29.68 | 300,176,768 |
| LLaMA | Pending | Pending | Not applicable: prompting | Pending | Measured on load |
| Gemma | Pending | Pending | Not applicable: prompting | Pending | Measured on load |

Scores use corpus-level SacreBLEU 2.6.0 BLEU with 13a tokenization and chrF (character order 6, beta 2, no word n-grams), on a 0–100 scale. Complete signatures, hardware and per-item inference times are retained. Adaptation time includes training, validation and checkpoint writes but excludes downloads; inference includes tokenization and decoding and excludes one warm-up call. These are single runs on a small test set, without confidence intervals.

mT5 improves BLEU by 1.00 and chrF by 4.88 points over the basic model, but both exhibit major semantic omissions and repetition. mT5 frequently emits `<extra_id_0>`, reflecting an unresolved mismatch between its pretrained behavior and the short adaptation run. Raw decoded output is preserved and scored without manually cleaning these artifacts. The basic model's falling training loss and rising later validation loss indicate overfitting; validation selects an earlier checkpoint.

See [generated results](RESULTS.md), [raw translations](results/translations.txt), [machine-readable translations](results/translations.csv), [score plot](results/scores.png), [learning curves](results/learning_curves.png) and [qualitative analysis](docs/analysis.md). All 32 measured test outputs per completed model are retained, including failures.

## Sampling-temperature analysis

Temperature scales logits before sampling: lower values concentrate probability on preferred tokens; higher values flatten the distribution. This does not guarantee faithfulness. The same four source texts are sampled at all three values for each completed model. [Temperature metrics](results/temperature_metrics.json), [raw samples](results/temperature_samples.json) and the [analysis](docs/analysis.md) preserve the comparison. Four examples and one sampling seed are exploratory, not a robust ranking. Greedy results above remain the primary comparison; no favorable temperature is selected using test scores and then presented as an independent test result.

## Ethical analysis

The [output-based analysis](docs/analysis.md) records omissions, repetitions and quantity failures. The [review protocol](docs/ethical_review_protocol.md) covers bias, hallucination, cultural interpretation and sensitive content. Automatic flags are review candidates, not validated findings. Human bilingual review, terminology constraints, quantity checks and preservation of security warnings are necessary before deployment. No demographic fairness claim is supported by this small corpus.

## Conclusion

**mT5 is the better-performing of the two measured models** on both BLEU and chrF. Its absolute quality remains inadequate for reliable Web3 translation. A best model across all four systems and an encoder-decoder versus decoder-only conclusion remain undetermined until the official LLaMA and Gemma runs are complete. More training data, validation-led adaptation and larger, independently reviewed test sets are needed before any production claim.

## Repository organization

- `chaintranslate/`: downloader, model definitions, training/generation, metrics and reporting.
- `data/`: fixed parallel splits, provenance and corpus summary.
- `results/`: measured scores, histories, outputs and plots; checkpoints are excluded from version control.
- `docs/`: methodology, qualitative findings and ethical review.
- `tests/`: split isolation, autoregressive masking and evaluation alignment checks.
- `ChainTranslate.ipynb`: reproducible notebook workflow.
