# Experimental results

Status: complete

| Model | BLEU | chrF | Adaptation seconds | Inference seconds | Base parameters |
|---|---:|---:|---:|---:|---:|
| basic | 0.36 | 13.10 | 206.27 | 3.61 | 1,177,040 |
| mt5 | 1.49 | 17.16 | 1102.46 | 64.35 | 300,176,768 |
| llama | 36.71 | 62.65 | N/A (prompting) | 610.69 | 1,235,814,400 |
| gemma | 0.89 | 25.55 | N/A (prompting) | 94.68 | 268,098,176 |

All reported runs are checked against the same test IDs, source strings and references; BLEU and chrF are recomputed from raw predictions. Scores are corpus-level SacreBLEU BLEU (13a) and chrF (character order 6, beta 2, no word n-grams). Signatures are retained in metrics.csv.

Adaptation time includes training, validation and checkpoint writes, excluding downloads. Inference includes tokenization and decoding, excluding one warm-up call. Hardware is recorded per model. Runs on different hardware, or overlapping runs, do not establish an isolated speed comparison.

## Sampling temperature

| Model | Temperature | BLEU | chrF | Distinct 2 | Repeated bigrams |
|---|---:|---:|---:|---:|---:|
| basic | 0.5 | 0.51 | 11.86 | 1.000 | 0.000 |
| basic | 1.0 | 1.15 | 14.87 | 0.902 | 0.098 |
| basic | 1.5 | 0.70 | 17.98 | 0.684 | 0.316 |
| mt5 | 0.5 | 2.83 | 20.37 | 0.963 | 0.037 |
| mt5 | 1.0 | 1.37 | 20.62 | 0.950 | 0.050 |
| mt5 | 1.5 | 0.44 | 22.81 | 0.973 | 0.027 |
| llama | 0.5 | 24.67 | 61.90 | 0.941 | 0.059 |
| llama | 1.0 | 21.03 | 53.36 | 0.948 | 0.052 |
| llama | 1.5 | 0.28 | 16.79 | 0.688 | 0.312 |
| gemma | 0.5 | 2.89 | 32.11 | 0.944 | 0.056 |
| gemma | 1.0 | 2.89 | 32.11 | 0.944 | 0.056 |
| gemma | 1.5 | 2.77 | 30.01 | 0.941 | 0.059 |

Temperature metrics use the same four examples at each temperature and seed 42. Distinct-2 and repeated-bigram fraction measure lexical diversity and repetition, not semantic correctness.

Highest chrF among measured models: **llama**. BLEU, compute requirements and qualitative failures must also be considered.
