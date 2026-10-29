# Experimental results

Status: incomplete; missing llama, gemma

| Model | BLEU | chrF | Adaptation seconds | Inference seconds | Base parameters |
|---|---:|---:|---:|---:|---:|
| basic | 0.44 | 13.16 | 134.99 | 2.49 | 1,177,040 |
| mt5 | 1.44 | 18.04 | 523.22 | 29.68 | 300,176,768 |

All reported runs are checked against the same test IDs, source strings and references; BLEU and chrF are recomputed from raw predictions. Scores are corpus-level SacreBLEU BLEU (13a) and chrF (character order 6, beta 2, no word n-grams). Signatures are retained in metrics.csv.

Adaptation time includes training, validation and checkpoint writes, excluding downloads. Inference includes tokenization and decoding, excluding one warm-up call. Hardware is recorded per model. Runs on different hardware, or overlapping runs, do not establish an isolated speed comparison.

## Sampling temperature

| Model | Temperature | BLEU | chrF | Distinct 2 | Repeated bigrams |
|---|---:|---:|---:|---:|---:|
| basic | 0.5 | 0.85 | 14.38 | 0.966 | 0.034 |
| basic | 1.0 | 1.13 | 14.12 | 0.911 | 0.089 |
| basic | 1.5 | 0.63 | 17.86 | 0.888 | 0.112 |
| mt5 | 0.5 | 1.69 | 19.24 | 0.972 | 0.028 |
| mt5 | 1.0 | 1.49 | 20.56 | 0.925 | 0.075 |
| mt5 | 1.5 | 0.47 | 21.95 | 0.939 | 0.061 |

Temperature metrics use the same four examples at each temperature and seed 42. Distinct-2 and repeated-bigram fraction measure lexical diversity and repetition, not semantic correctness.

Highest chrF among measured models: **mt5**. BLEU, compute requirements and qualitative failures must also be considered.

A four-model winner and encoder-decoder versus decoder-only comparison remain undetermined.
