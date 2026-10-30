# ChainTranslate

English-to-French translation of Ethereum interface and educational text.

[Open notebook in Colab](https://colab.research.google.com/github/themisogynist/chaintranslate-web3-mt/blob/main/ChainTranslate.ipynb) · [Results](RESULTS.md) · [Analysis](docs/analysis.md) · [Translation samples](results/translations.txt)

## Problem statement and objective

Web3 applications need translations that preserve technical meaning, quantities and security warnings. ChainTranslate compares a Transformer trained from scratch, a fine-tuned multilingual encoder-decoder, and two prompted decoder-only models on a public Ethereum parallel corpus. The objective is to measure translation quality and computational cost, and examine errors with ethical consequences. This is an experimental translation benchmark; the measured systems are not suitable for unattended publication.

**Experiment status:** all four required models have completed the same held-out test set. The published results contain 128 test translations and 48 temperature samples, verified against their raw outputs. LLaMA achieves the highest BLEU and chrF.

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

The adaptation strategies, parameter counts, pretraining and tokenizers necessarily differ. Equal token limits do not imply equal word limits. This is a comparison of concrete systems, not a controlled causal test of architecture alone. The published Colab model executions were sequential, with a dependency failure before mT5 loaded and a subsequent retry. All four runs report CPU execution with two PyTorch threads. Colab resource variability and different output lengths make timings descriptive rather than an isolated architecture-speed benchmark.

## Execution and dependencies

The final measurements were exported from a Python 3.13 Colab session; the original development environment used Python 3.12. Core dependencies are PyTorch, Transformers, PEFT, Accelerate, SentencePiece, SacreBLEU, NumPy and Matplotlib; versions are in [requirements.txt](requirements.txt). GPU acceleration is recommended for the pretrained models; CPU execution is supported but slower. Model downloads need internet access and sufficient disk space.

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
| basic | 0.36 | 13.10 | 206.27 | 3.61 | 1,177,040 |
| mt5 | 1.49 | 17.16 | 1102.46 | 64.35 | 300,176,768 |
| llama | 36.71 | 62.65 | N/A (prompting) | 610.69 | 1,235,814,400 |
| gemma | 0.89 | 25.55 | N/A (prompting) | 94.68 | 268,098,176 |

Scores use corpus-level SacreBLEU 2.6.0 BLEU with 13a tokenization and chrF (character order 6, beta 2, no word n-grams), on a 0–100 scale. All primary and temperature scores were recomputed from the raw translations. Adaptation time includes training, validation and checkpoint writes but excludes downloads; inference includes tokenization and decoding and excludes one warm-up call. Prompted models have no training-time measurement, rather than a measured zero. All four runs used CPU, despite the CUDA-capable PyTorch build name. These are single runs on 32 test pairs, without confidence intervals.

**LLaMA is the best-performing system:** BLEU 36.71 and chrF 62.65. It usually translates into French and preserves important quantities and warnings, although grammar and domain interpretation still fail on some examples. Its test inference time is the largest, at 610.69 seconds for 32 inputs, and its approximately 1.24 billion parameters exceed the other systems.

The basic Transformer overfits the small training corpus; minimum validation loss selects epoch 19 of 30. mT5 selects epoch 3 of three but emits `<extra_id_0>` in all 32 outputs and frequently omits meaning, showing insufficient task adaptation. Gemma exactly copies 28/32 English sources; the remaining four outputs also remain in English with punctuation, wording or omission changes. Its chrF of 25.55 must not be mistaken for adequate French translation: shared names, cognates and character fragments can overlap with references even when the target-language instruction is not followed. mT5 has higher BLEU than Gemma despite lower chrF.

The strongest decoder-only system outperforms these encoder-decoder baselines, but the other decoder-only system fails to translate. Model size, pretraining, tokenizer and adaptation differ, so the results do not establish inherent superiority of one architecture. Raw outputs, including failures, are retained without manual correction.

See [results tables](RESULTS.md), [raw translations](results/translations.txt), [CSV translations](results/translations.csv), [scores](results/scores.png), [learning curves](results/learning_curves.png), [run provenance](results/run_metadata.json) and [qualitative analysis](docs/analysis.md).

## Sampling-temperature analysis

The same first four test pairs are sampled at temperatures **0.5, 1.0 and 1.5**, with seed 42, top-k disabled, top-p 1 and the same output cap. Lower temperatures concentrate probability on likely tokens; higher temperatures broaden sampling, but do not necessarily improve diversity or meaning.

For LLaMA, temperature 0.5 obtains BLEU 24.67 / chrF 61.90; 1.0 obtains 21.03 / 53.36; 1.5 falls to 0.28 / 16.79 and increases repeated-bigram fraction to 0.313. Thus **0.5 is the best tested sampling temperature for LLaMA on this exploratory subset**. Greedy decoding remains the main 32-pair evaluation; these smaller-sample scores are not directly comparable to the full-test scores.

Gemma produces identical samples at 0.5 and 1.0 and largely copies English at every temperature. Higher temperature does not rescue target-language compliance. The weak basic/mT5 systems can gain character overlap while generating longer, less faithful output, illustrating why chrF alone is insufficient. [Temperature metrics](results/temperature_metrics.json), [all 48 samples](results/temperature_samples.json) and the [analysis](docs/analysis.md) report coherence, diversity and repetition. Four examples and one random seed do not establish an optimal temperature generally.

## Ethical analysis

The [output-based analysis](docs/analysis.md) records omissions, repetitions and quantity failures. The [review protocol](docs/ethical_review_protocol.md) covers bias, hallucination, cultural interpretation and sensitive content. Automatic flags are review candidates, not validated findings. Human bilingual review, terminology constraints, quantity checks and preservation of security warnings are necessary before deployment. No demographic fairness claim is supported by this small corpus.

## Conclusion

**LLaMA 3.2 1B Instruct is the best-performing model in this experiment**, leading both BLEU and chrF and providing the most faithful French translations among the four tested systems. Temperature 0.5 offers the strongest sampled results on the four-example exploration, while the main benchmark uses greedy decoding. LLaMA's quality advantage comes with greater parameter count and longer measured CPU inference time.

The basic Transformer and lightly adapted mT5 require substantially better training before practical use. Gemma 3 270M IT does not reliably follow the French-translation task under this prompting setup. Even the best system needs bilingual domain review before publishing financial or security material. Conclusions are limited to this small Ethereum corpus and these specific models and adaptation methods.

## Repository organization

- `chaintranslate/`: downloader, model definitions, training/generation, metrics and reporting.
- `data/`: fixed parallel splits, provenance and corpus summary.
- `results/`: measured scores, histories, outputs and plots; checkpoints are excluded from version control.
- `docs/`: methodology, qualitative findings and ethical review.
- `tests/`: split isolation, autoregressive masking and evaluation alignment checks.
- `ChainTranslate.ipynb`: reproducible notebook workflow.
