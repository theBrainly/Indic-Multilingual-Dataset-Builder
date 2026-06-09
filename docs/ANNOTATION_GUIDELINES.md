# Annotation Guidelines — Indic Multilingual Quality Labeling

These guidelines standardize how annotators label documents so that
inter-annotator agreement stays high and the resulting labels are trustworthy.
They pair with the QA tooling in `annotate/` (kappa/alpha metrics, consensus
aggregation, annotator scorecards).

## 1. Task definitions

### 1.1 Document quality (3-way classification)
Label each document as one of:

- **keep** — fluent, on-topic, self-contained text in a single language.
- **borderline** — usable but noisy (minor boilerplate, mixed scripts, partial
  sentences). Route to a second reviewer.
- **reject** — spam, gibberish, machine-translated salad, unsafe content, or
  fewer than ~4 meaningful words.

### 1.2 Language correctness (boolean)
Does the document's content match its tagged language code? For scripts shared
across languages (Devanagari → hi/mr/ne; Bengali → bn/as), judge by vocabulary,
not script alone.

### 1.3 Preference judgement (pairwise)
Given a prompt and two responses, mark which response is **chosen**. Use
**margin** 0.25/0.5/0.75/1.0 to express how strong the preference is. Never mark
two identical responses as a preference.

## 2. Decision rules (to maximize agreement)

1. **Single-language rule.** If more than ~20% of tokens are in a different
   language than tagged, label `reject` for quality unless the mixing is natural
   code-switching common to the language pair.
2. **Length floor.** Documents under 4 content words are `reject` by default.
3. **Safety first.** Any unsafe content (hate, explicit, PII dump) is `reject`
   regardless of fluency.
4. **When unsure, choose `borderline`,** never guess `keep`/`reject`.

## 3. Quality assurance process

- Every item is labeled by **≥ 3 annotators** where budget allows.
- **Gold items** (~10%, pre-labeled by leads) are mixed in to measure
  per-annotator accuracy.
- Run `AnnotationQA.report()` after each batch:
  - Fleiss' kappa **≥ 0.6** required to accept a batch.
  - Annotators with gold accuracy **< 0.8** or consensus kappa **< 0.4** are
    flagged for retraining.
- Items with no majority or agreement **< 0.66** go to **adjudication** by a lead.

## 4. Reasoning-trace & code-correctness review (advanced tasks)

- **Reasoning traces:** verify each step follows from the previous; mark the
  first invalid step. A correct final answer with broken intermediate steps is
  still `reject`.
- **Code correctness:** run the provided tests; label `correct` only if all
  pass. Record failing test ids in annotation metadata.

## 5. Provenance

Every annotation records `annotator_id`, `confidence`, and timestamps so label
sets are reproducible and auditable, consistent with the dataset governance
model used across this project.
