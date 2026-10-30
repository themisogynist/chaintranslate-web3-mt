# Translation quality and ethical analysis

## Experiment and verification

The final Colab export contains all four required systems: a trained basic Transformer, LoRA-fine-tuned mT5-small, prompted LLaMA 3.2 1B Instruct and prompted Gemma 3 270M IT. Every model generates 32 translations from identical held-out English inputs, and 12 additional samples from the same four inputs at three temperatures. All 128 primary outputs and 48 sampled outputs are retained, including unsuccessful translations.

The exported model, training, generation and metric code matches the repository implementation. The training, validation and test JSON files match the original fixed splits byte for byte. Exact English sources and localization files are disjoint across splits. Test IDs, ordering, sources and references match for all four models; summed per-example inference times reproduce the totals. All four BLEU/chrF pairs and twelve temperature-score pairs were independently recomputed with SacreBLEU 2.6.0. These checks verify the internal consistency of the exported measurements; they do not independently reproduce model training from checkpoints. Archive and split hashes are retained in `results/run_metadata.json`.

## Quantitative comparison

| Model | BLEU | chrF | Adaptation seconds | Test inference seconds | Parameters |
|---|---:|---:|---:|---:|---:|
| Basic Transformer | 0.36 | 13.10 | 206.27 | 3.61 | 1,177,040 |
| mT5-small | 1.49 | 17.16 | 1,102.46 | 64.35 | 300,176,768 base |
| LLaMA 3.2 1B Instruct | 36.71 | 62.65 | N/A | 610.69 | 1,235,814,400 |
| Gemma 3 270M IT | 0.89 | 25.55 | N/A | 94.68 | 268,098,176 |

LLaMA leads both primary quality measures, but has the highest parameter count and measured inference time. Its mean latency is approximately 19.08 seconds per test input, compared with 0.11 for the basic model, 2.01 for mT5 and 2.96 for Gemma. These are CPU wall-clock measurements including tokenization and decoding, with one warm-up excluded. The CUDA-capable PyTorch build name does not mean a GPU was used: all four metadata records explicitly identify CPU execution.

The basic model trains all its parameters for 30 epochs; epoch 19 has the lowest validation token loss (5.5094). Later validation deterioration despite falling training loss indicates overfitting. mT5 updates 344,064 adapter parameters for three epochs; its final epoch is selected by validation token loss (3.3667). Cross-entropy values are not directly comparable across different tokenizers and vocabularies. Training-loss aggregation also differs: the basic history is token-weighted, while mT5's training history averages batch losses by example count. Both validation losses are token-weighted within their own vocabularies.

LLaMA and Gemma receive three demonstrations from the training set and undergo no fine-tuning. Their adaptation time is not applicable, rather than measured as zero. All models share the evaluation inputs, references, output-token cap and decoding settings. They do not share pretraining, tokenizers, parameter count or adaptation budget. Colab resource variability, different generated lengths and the interrupted installation before the successful mT5 retry also limit timing comparisons. Failed startup and model downloads are outside the recorded training times.

## Encoder decoder versus decoder only

The best decoder-only system, LLaMA, substantially outperforms the small trained encoder-decoder and lightly adapted mT5. However, the other decoder-only system, Gemma, fails the target-language task. Therefore the evidence supports selecting this particular LLaMA system, not a universal claim that decoder-only architecture is superior.

LLaMA benefits from a much larger pretrained instruction-following model. The basic Transformer has only 512 training pairs and no pretraining, while mT5's multilingual pretraining is followed by a small adapter update. Tokenization changes the number of words represented by the common 96-token cap. A fair causal architecture study would need matched data, parameter budgets, pretraining and adaptation, which this resource-bounded comparison does not provide.

## Qualitative findings

**Basic Transformer:** output is often fragmented and unrelated to the input. For `29d5ee4767fc2718`, a warning that dishonest stakers risk losing ETH becomes `En savoir plus sur les actifs numériques de la couche d'une`. The output loses the central warning. Low BLEU and chrF agree with this lack of faithfulness.

**mT5:** all 32 primary outputs contain `<extra_id_0>`. These raw artifacts are retained in scoring rather than manually removed. The limited adaptation has not sufficiently adapted the model to translation. For `6e7b69b1669a0621`, a comparison of Netflix and Ethereum energy usage becomes `<extra_id_0> uses $0.451 terawatt pour year.` It mixes languages, adds a currency marker, omits Ethereum's quantity and loses the comparison.

**LLaMA:** usually produces recognizable French with substantially more preserved meaning. For the energy example it produces `Netflix utilise environ 0,451 TWh par an. Ethereum utilise environ 0,0026 TWh par an.`, preserving quantities, entity assignment and units. It also preserves the staking-loss warning. It is not flawless: the ReFi question `cdda03b5f11bc4be` becomes `Quelques-uns des communautés de financement rénovateur (ReFi) qui construisent sur Ethereum.`, changing a question into a fragment and mistranslating terminology. In `42b69c521edeb193`, scaling is rendered as `atteindre la vitesse de dépôt`, which distorts the technical meaning. Its high relative scores do not establish expert translation quality.

**Gemma:** exactly copies the English source for 28/32 inputs. The other four outputs also remain English, with minor punctuation/wording changes or omissions. For `6f58ec4ec68dc5cf`, it preserves the first stablecoin sentence but drops the statement that the network pays no interest. Its chrF exceeds mT5's because source copying can retain shared names, cognates and character fragments that also appear in French references. This is not evidence of better target-language compliance. BLEU instead places mT5 above Gemma. The four sampled outputs at temperatures 0.5 and 1.0 are identical, consistent with this setup's concentrated copying behavior, rather than evidence that temperature was omitted from the code.

## Temperature coherence diversity and repetition

All temperature experiments use the same first four test pairs and sampling seed 42. Top-k is disabled and top-p is one. Distinct-2 is the number of unique lowercased whitespace bigrams divided by all such bigrams across the four outputs. Repeated-bigram fraction counts occurrences beyond the first and is its complement. These lexical measurements do not prove coherence or faithfulness.

| Model | Temperature | BLEU | chrF | Distinct 2 | Repeated bigrams | Mean words |
|---|---:|---:|---:|---:|---:|---:|
| basic | 0.5 | 0.51 | 11.86 | 1.000 | 0.000 | 6.00 |
| basic | 1.0 | 1.15 | 14.87 | 0.902 | 0.098 | 11.25 |
| basic | 1.5 | 0.70 | 17.98 | 0.684 | 0.316 | 15.25 |
| mt5 | 0.5 | 2.83 | 20.37 | 0.963 | 0.037 | 7.75 |
| mt5 | 1.0 | 1.37 | 20.62 | 0.950 | 0.050 | 11.00 |
| mt5 | 1.5 | 0.44 | 22.81 | 0.973 | 0.027 | 29.00 |
| llama | 0.5 | 24.67 | 61.90 | 0.941 | 0.059 | 18.00 |
| llama | 1.0 | 21.03 | 53.36 | 0.948 | 0.052 | 15.50 |
| llama | 1.5 | 0.28 | 16.79 | 0.688 | 0.312 | 41.00 |
| gemma | 0.5 | 2.89 | 32.11 | 0.944 | 0.056 | 14.50 |
| gemma | 1.0 | 2.89 | 32.11 | 0.944 | 0.056 | 14.50 |
| gemma | 1.5 | 2.77 | 30.01 | 0.941 | 0.059 | 13.75 |

For LLaMA, temperature 0.5 yields the strongest sampled BLEU and chrF. Temperature 1.0 reduces both scores. At 1.5, mean output length rises to 41 words, repeated-bigram fraction reaches 0.313, and both quality scores collapse. For the ReFi input, its 1.5 sample is `Booket la reformULATION propre.`, which abandons the topic. This supports 0.5 as the best tested sampling value for this subset, while greedy decoding remains the main evaluation setting.

The basic model becomes longer and more repetitive as temperature increases. Its higher chrF at 1.5 does not correspond to coherent translation. mT5 similarly produces longer outputs at 1.5: chrF rises but BLEU falls below 0.5, showing that additional character overlap can coexist with poor semantic consistency. Gemma remains predominantly an English copier; its higher-temperature changes do not solve language compliance. Temperature cannot compensate for inadequate adaptation or instruction following.

These are exploratory results from four examples and one random seed. They do not establish a universal optimal temperature, and their corpus scores cannot be directly compared with the 32-pair primary scores. No sampled temperature was selected and re-presented as an independent full-test result.

## Responsible and ethical AI

**Sensitive financial information.** In `29d5ee4767fc2718`, the basic model and mT5 omit the staking-loss warning. LLaMA preserves it as `Si un staker essaie de tricher la réseau, il risque de perdre ses ETH.`, despite a grammatical error. Gemma copies the original warning in English, making it unsuitable for the requested French audience. In the stablecoin example, Gemma drops the no-interest statement. Financial warnings, negation and restrictions require explicit semantic checks and bilingual approval before publication.

**Hallucination and factual distortion.** mT5 adds a dollar sign to an energy quantity and omits the second entity in `6e7b69b1669a0621`. LLaMA preserves the numerical comparison but distorts the meaning of scaling in `42b69c521edeb193`. Mitigations include entity-number-unit alignment, a technical glossary and a review of claims and causal relations. A number-preservation rule alone would miss the scaling error. Translation also does not validate whether the source's environmental claim is accurate or current.

**Security content.** In `d9bcc1a8661ec6f2`, the source attributes private-key security to one-way key derivation rather than a nonce. The basic model substitutes unrelated infrastructure text; mT5 outputs `<extra_id_0> n'est à la nonce.`, losing the causal explanation. LLaMA preserves the one-way property and exclusion of the nonce, although phrasing is awkward. Gemma leaves the explanation in English. Security guidance needs human review of negation, causality and terminology; private keys and wallet credentials must never be supplied as translation examples.

**Cultural and domain interpretation.** LLaMA's rendering of regenerative finance as `financement rénovateur` in the ReFi example changes the concept. The basic and mT5 outputs for `3c2a4295f7f09d69` lose the context of people in regions with limited banking infrastructure, whereas LLaMA largely preserves it and Gemma copies it in English. These are observed semantic and social-context failures, not proof of discriminatory intent. A specified French locale, reviewed terminology and native-speaker domain review would reduce such misrepresentation.

**Bias evidence and limits.** This technical corpus is not a controlled demographic fairness test. No reliable conclusion about gender, regional or socioeconomic bias can be drawn from 32 examples. Observed omission of banking-access context is relevant to inclusion, but does not establish a systematic group disparity. A balanced counterfactual suite and bilingual reviewers would be needed to test that hypothesis. Absence of observed stereotypes is not evidence of absence of bias.

Automated candidates in `results/review_candidates.json` flag numerical and protected-term changes; they are not confirmed ethical violations. Decimal normalization handles simple comma/point differences, but the heuristic does not comprehensively parse every numerical convention or semantic relationship. Review must also consider outputs that receive no flag.

## Conclusion and limitations

LLaMA 3.2 1B Instruct is the strongest of the four tested systems on BLEU, chrF and the inspected translations. It preserves important quantities and warnings better than the other systems, at the cost of greater parameter count and longer measured CPU inference. Temperature 0.5 gives its strongest sampled results on the four-example exploration.

The basic model overfits; mT5 is insufficiently adapted; Gemma fails to translate into French under the tested setup. These failures are retained as part of the empirical comparison. More training data, validation-led adaptation and terminology-aware generation are possible improvements, but no improvement is claimed without further measurement.

The dataset is short, domain-specific and small, with only one French reference per input. Exact-source and file separation do not exclude semantic overlap or exposure to public Ethereum text during pretraining. The qualitative review cites actual outputs but is not an independent professional bilingual audit. Even the strongest system requires human review before financial or security deployment.
