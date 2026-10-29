# Responsible translation review

The review examines actual translations from the same held-out English/French test pairs. BLEU and chrF do not establish factual correctness or safety. Annotate individual outputs by model and test ID, retaining the original source, French reference and raw prediction.

## Review categories

| Concern | Evidence to examine | Mitigation |
|---|---|---|
| Bias | Changes in who is described as competent, trusted or eligible; unsupported gender assumptions | Paired counterfactual tests and bilingual review; avoid inferring absence of bias from a small technical corpus |
| Hallucination | Invented claims, amounts, guarantees or instructions not supported by the source | Preserve source context; flag unsupported additions; require review before publishing wallet or transaction instructions |
| Cultural misrepresentation | Literal renderings of idioms, regional terminology or connotations | Define locale and audience; use a reviewed French terminology guide and native-speaker review |
| Sensitive content | Loss of negation, warnings, consent or security restrictions | Check key phrases, numbers, addresses and access-control statements; escalate high-impact text to a human translator |
| Domain terminology | Changes to Ethereum, ETH, tokens or recovery-phrase concepts | Protect immutable identifiers and use a validated glossary; avoid translating addresses or secret values |

`results/review_candidates.json` flags potential numerical changes, missing protected terms and empty outputs. These flags are review candidates, not confirmed errors. French decimal conventions, abbreviations and legitimate paraphrases can produce false positives. The code cannot establish fairness or certify safety.

The corpus contains public educational/interface strings and no private wallet credentials. It is not a representative evaluation of all social groups or sensitive translation scenarios. Pretraining may include ethereum.org text, so benchmark contamination is possible for pretrained models even though the experiment's own train and test files are disjoint. Model size, pretraining and adaptation differences must be considered when comparing encoder-decoder and decoder-only systems.

Output-level findings for completed runs appear in `docs/analysis.md`. The four-model review remains incomplete until all required outputs are measured.
