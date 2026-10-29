"""Corpus metrics, independent of model training dependencies."""

from sacrebleu.metrics import BLEU, CHRF


def metrics(predictions, references):
    bleu = BLEU(tokenize="13a")
    chrf = CHRF(word_order=0)
    b = bleu.corpus_score(predictions, [references])
    c = chrf.corpus_score(predictions, [references])
    return dict(
        bleu=b.score,
        chrf=c.score,
        bleu_signature=str(bleu.get_signature()),
        chrf_signature=str(chrf.get_signature()),
    )
