# Translation quality and ethical analysis

## Measured comparison

Both completed systems translate the same 32 held-out English sources. mT5 obtains BLEU 1.44 and chrF 18.04, compared with BLEU 0.44 and chrF 13.16 for the basic Transformer. These low absolute scores agree with the qualitative evidence: translations frequently omit the central meaning, repeat phrases, or introduce unrelated text. mT5 has more pretrained knowledge but the limited LoRA adaptation does not produce a reliable translation system.

The basic model has 1,177,040 trainable parameters. The mT5 base has 300,176,768 parameters with 344,064 additional trainable adapter parameters. Adaptation times are 134.99 and 523.22 seconds; inference over the test set is 2.49 and 29.68 seconds. The CPU runs overlapped, so these timings are descriptive measurements with resource contention, not an isolated throughput benchmark. Greedy decoding and the same output cap are shared, but tokenizer, pretraining, parameter size and adaptation differ.

Only encoder-decoder outputs are currently measured. The absence of official LLaMA and Gemma outputs prevents a valid decoder-only comparison or a four-model winner. A model label in the code is not evidence of a completed experiment.

## Temperature results

The following corpus scores use the same first four test pairs, one sampling seed and a 96-token output cap. Distinct-2 is the number of unique lowercased whitespace bigrams divided by all such bigrams across these outputs. Repeated-bigram fraction counts occurrences beyond the first and is its complement. Neither measure proves coherence; fragments and unrelated words can score highly on diversity.

| Model | Temperature | BLEU | chrF | Distinct-2 | Repeated bigrams | Mean words |
|---|---:|---:|---:|---:|---:|---:|
| basic | 0.5 | 0.85 | 14.38 | 0.966 | 0.034 | 8.25 |
| basic | 1.0 | 1.13 | 14.12 | 0.911 | 0.089 | 12.25 |
| basic | 1.5 | 0.63 | 17.86 | 0.888 | 0.112 | 23.25 |
| mt5 | 0.5 | 1.69 | 19.24 | 0.972 | 0.028 | 10.00 |
| mt5 | 1.0 | 1.49 | 20.56 | 0.925 | 0.075 | 14.25 |
| mt5 | 1.5 | 0.47 | 21.95 | 0.939 | 0.061 | 25.75 |

Higher temperature lengthens outputs in this sample. chrF rises at temperature 1.5 for both models even while BLEU falls, illustrating that additional overlapping character fragments need not preserve meaning. Repetition does not follow a universal monotonic rule: mT5's repeated-bigram fraction peaks at 1.0, while the basic model's increases throughout. A larger sample and multiple random seeds would be needed for a stable conclusion.

For source ID `cdda03b5f11bc4be`, the English question asks what the regenerative-finance community builds on Ethereum. mT5 at temperature 0.5 produces `<extra_id_0> Ethereum ?`, omitting the substantive question. At 1.5 it produces `<extra_id_0> est Brno social 2007. What analysepievrage provisie ATPil??? Operations Aile certifiés CsASB?`, introducing unrelated names, a year and mixed-language fragments. Lower temperature gives a shorter, less varied result but still fails faithfulness. The basic model at 0.5 produces `L'Ethereum denance de v sur la communauté Ethereum :`; at 1.5 it produces `M00 amational avezisations des sontonomanc compte li2 Pourquoi clés`. Neither is a coherent translation.

These raw excerpts are generated outputs, not illustrative rewrites. The full sources, references and predictions are in `results/temperature_samples.json`. No temperature is established as deployment-ready.

## Output-level ethical findings

**Sensitive financial content — `29d5ee4767fc2718`.** The source states that a staker who cheats risks losing ETH. The basic model outputs `En savoir plus sur le réseau de la chaîne de blocs`; mT5 outputs `<extra_id_0> peut faire une société à l'ETH`. Both omit the loss warning. A superficially fluent interface could therefore conceal a material risk. Mitigation requires explicit warning/negation preservation checks and bilingual human approval of financial text.

**Security meaning — `d9bcc1a8661ec6f2`.** The source explains that private-key safety derives from one-way key derivation, not the nonce. The basic output is `Sécurité d'une de blocs sur les stablecoins.` and the mT5 output is `<extra_id_0> est une fois avec la nonce.` Both lose the causal explanation; mT5 also loses the critical exclusion of the nonce. A terminology glossary alone cannot correct this relational error. Security documentation needs semantic review and tests for negation and causality.

**Numerical omission — `6e7b69b1669a0621`.** The source compares Netflix at approximately 0.451 terawatt hours per year with Ethereum at approximately 0.0026. The basic model omits the quantities entirely. mT5 outputs `<extra_id_0> uses 0.0026 terawatt pour year.`, retaining one quantity but dropping the comparison and its entity assignment. This can misrepresent an environmental claim. Numeric matching should account for French decimal commas, preserve units and verify the entity attached to each number. Source claims themselves also require verification before publication; translation does not validate them.

**Cultural and inclusion context — `3c2a4295f7f09d69`.** The source discusses access in regions with limited banking infrastructure. The basic output `Sécurité d'Ethereum.` and mT5 output `<extra_id_0> est l'infrastructure Ethereum.` erase this context. This is evidence of contextual omission, not proof of a demographic stereotype or discriminatory intent. A French locale-specific terminology review and preservation of social context are appropriate safeguards.

**Hallucination and repetition.** In the Circom example `884a15bcaa21b5fd`, the basic model introduces token/operator/community fragments unrelated to the language/compiler definition. In `673ab94e261dcd05`, it repeats `est un portefeuille`, while mT5 repeats Ethereum clauses instead of the settlement-speed distinction. Higher-temperature samples introduce unsupported entities and numbers. Output checks should flag unsupported additions and loops for human review, without silently replacing the raw benchmark output.

**Bias coverage.** This small technical test set cannot establish fairness across gender, nationality, socioeconomic groups or dialects. No controlled demographic counterfactual experiment was performed. Such a suite, reviewed by fluent French speakers and domain experts, is necessary to investigate whether equivalent inputs receive systematically different treatment. Lack of observed demographic evidence must not be interpreted as lack of bias.

## Limitations and conclusion

The dataset is short, domain-specific and small; exact-source and file separation do not rule out semantic overlap or pretrained-model exposure to public Ethereum content. One reference cannot cover every valid French translation. Automated quality scores and the rule-based review candidates are insufficient for safety certification. The reported ethical review is a qualitative inspection of cited examples, not an independent professional bilingual audit or an exhaustive error-rate estimate.

mT5 is the better of the two measured systems on both primary scores. Neither preserves meaning consistently enough for real wallet, transaction or security documentation. More training pairs, validation-led fine-tuning, terminology constraints, a larger test set and human review are warranted. The final architecture comparison remains pending the official LLaMA and Gemma measurements.
