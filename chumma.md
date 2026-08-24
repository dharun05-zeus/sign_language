# Top 1% Oncology RAG Systems: What They Actually Look Like
### Domain Specialization Layer — PubMed-Ingested Clinical/Research RAG

Generic RAG mastery (chunk → embed → retrieve → generate) gets you to L2 competence. Oncology is one of the highest-stakes domains you can apply RAG to — a wrong hazard ratio, a missed retraction, or a silently outdated treatment guideline has real clinical consequences. What separates a top 1% oncology RAG build from a generic one is almost entirely in the layers *around* the core pipeline: evidence quality, numerical fidelity, contradiction handling, and auditability. Below is what that looks like end to end.

---

## 1. Ingestion Layer — PubMed Is Not "Just Documents"

A generic RAG build treats PubMed abstracts like any other text corpus. A top-tier build treats each paper as a **structured evidence object**, not a blob of text.

**What gets extracted at ingestion, beyond raw text:**
- PMID, DOI, PMCID (for full-text linking via PMC OA subset)
- MeSH headings (Medical Subject Headings) — used later for concept-level filtering, not just keyword search
- Publication type tag (RCT, meta-analysis, systematic review, case report, preclinical/in vitro, editorial) — this becomes a first-class filter, not metadata you ignore
- Structured abstract sections (Background / Methods / Results / Conclusion) — oncology papers are heavily structured (IMRaD), and collapsing that structure at chunking time is one of the most common quality-killing mistakes
- Journal name + quartile/impact signal
- Clinical trial registry number (NCT ID) when present, cross-linked to ClinicalTrials.gov for trial phase, status, and enrollment data
- Author conflict-of-interest / funding source, when disclosed — surfaced later as a trust signal, not hidden
- Publication and last-revision date — oncology standard-of-care shifts fast (e.g., a first-line therapy can be superseded within 18 months); recency is a ranking signal, not an afterthought

**Non-negotiable pipeline components most builders skip:**
- **Retraction and correction monitoring.** Integration with Retraction Watch's database (or PubMed's own `PubMedCommentIn`/retraction metadata) so that retracted papers are flagged or removed from the retrievable index automatically, on a recurring schedule — not just at initial ingestion.
- **Preprint-to-publication linking.** A bioRxiv/medRxiv preprint that later gets peer-reviewed and published (sometimes with materially different results) needs to be reconciled — the system should prefer the published version and flag when a preprint's findings changed on peer review.
- **Continuous ingestion, not a static snapshot.** PubMed adds ~5,000+ records daily. A production oncology RAG system runs incremental daily ingestion with new-publication alerting for tracked topics/drugs, not a one-time index build.

---

## 2. Chunking — Section- and Claim-Aware, Not Fixed-Size

Fixed-size or naive sentence chunking is actively dangerous here. A hazard ratio, its confidence interval, and the patient population it applies to are often split across a sentence boundary a generic chunker won't respect.

**What top-tier chunking does differently:**
- **Section-aware boundaries**: never split across Methods/Results/Discussion — each becomes its own retrievable unit with the paper's title and PICO context attached as a prefix, so a chunk retrieved in isolation still carries what population/intervention it's describing.
- **Statistic-context binding**: numerical results (HR, OR, CI, p-values, response rates) are kept bound to their surrounding sentence and, where possible, their table caption — never allowed to float as an isolated number with no population/endpoint context.
- **Table and figure handling**: survival curves (Kaplan-Meier), forest plots, and outcome tables are extracted separately (often via layout-aware PDF parsing, not plain text extraction) and linked back to the discussing paragraph.
- **PICO tagging at chunk level**: each evidence chunk is tagged with Population, Intervention, Comparator, Outcome where extractable — this becomes the backbone of retrieval filtering in section 3.

---

## 3. Retrieval — Domain Embeddings + Evidence-Aware Filtering

**Embedding model choice is not generic.** Top builds use biomedical-domain embedding models rather than general-purpose ones — e.g., **PubMedBERT**, **BioBERT**, or **MedCPT** (NIH's own retriever, trained directly on real PubMed search click logs, purpose-built for this exact task). General-purpose embeddings measurably underperform on biomedical terminology, drug synonym clustering, and gene/protein nomenclature.

**Concept normalization is mandatory, not optional:**
- Drug brand names ↔ generic names (e.g., "Keytruda" ↔ "pembrolizumab") must resolve to the same concept
- Gene/protein synonym resolution (e.g., "HER2" / "ERBB2" / "neu")
- UMLS (Unified Medical Language System) or SNOMED CT concept mapping for disambiguating polysemous terms common in oncology (e.g., "ER" = estrogen receptor, not emergency room)

**Retrieval is evidence-hierarchy aware:**
- Study design becomes a rankable/filterable dimension: systematic reviews and meta-analyses > RCTs > cohort studies > case reports > preclinical/in vitro — following standard evidence hierarchies (GRADE, Oxford CEBM levels)
- Query decomposition into PICO components before retrieval, rather than a single dense query embedding — this is standard practice in clinical evidence retrieval and meaningfully improves precision over naive single-vector search
- Recency-weighted ranking specifically for treatment/guideline questions, separate from ranking for mechanism/biology questions where older foundational papers remain authoritative

**Hybrid retrieval + re-ranking**, same as general RAG mastery (L1–L2), but the re-ranker is often fine-tuned or prompted specifically to weigh study design and sample size, not just semantic relevance.

---

## 4. Generation — Numerical Fidelity and Contradiction Surfacing

This is where oncology RAG diverges most sharply from generic RAG, because the failure modes are higher-stakes.

- **Numbers are extracted, never generated.** Hazard ratios, confidence intervals, survival percentages, and p-values are pulled verbatim from the retrieved chunk and inserted, not paraphrased or "recalled" by the model — this is enforced architecturally (e.g., structured extraction + templated insertion) rather than trusted to prompting alone, because LLMs are known to subtly misstate statistics under paraphrase.
- **Contradiction detection, not silent averaging.** Oncology literature frequently contains genuinely conflicting findings across trials (different populations, different endpoints, evolving standard of care). A top-tier system detects when retrieved sources disagree and surfaces the disagreement explicitly ("Trial A found X in population Y; Trial B found conflicting results in population Z") rather than blending them into one falsely-confident answer.
- **Every claim is chunk-traceable**, down to the PMID and section, not just "grounded in the corpus generally" — this is what allows a clinician-reviewer to actually audit an answer in seconds.
- **Explicit evidence-strength labeling** on the answer itself (e.g., "based on a single small phase I trial" vs. "based on a Cochrane systematic review of 12 RCTs") — the system communicates confidence the way a clinician would, not as a flat, unqualified answer.
- **Hard boundaries on scope.** The system is explicitly constrained from producing individualized treatment recommendations or diagnostic conclusions — it answers "what does the evidence say," not "what should this patient do." This boundary is enforced in the prompt architecture and typically reviewed by clinical/legal stakeholders, not left to the model's judgment.

---

## 5. Evaluation — Domain Benchmarks + Clinician-in-the-Loop

Generic RAGAS-style faithfulness/precision/recall metrics still apply, but top builds add domain-specific evaluation layers:

| Evaluation Layer | What It Catches |
|---|---|
| **PubMedQA / BioASQ / MedQA benchmarks** | Baseline biomedical QA competence vs. published standards |
| **Numerical fidelity audits** | Sampled manual checks that every reported statistic in a generated answer exactly matches its cited source |
| **Contradiction-detection accuracy** | Whether the system correctly identifies and surfaces genuine disagreement across sources, tested on a curated set of known-conflicting trial pairs |
| **Retraction-sensitivity testing** | Whether a retracted or corrected paper is still influencing answers |
| **Clinician panel review** | Practicing oncologists or medical evidence specialists score a sample of answers for factual accuracy, appropriate hedging, and clinical relevance — this is the evaluation layer that generic RAG builds skip entirely and that separates a demo from a deployable system |
| **Demographic/population representation audit** | Checking whether the evidence surfaced reflects trial population diversity, since oncology trial cohorts have well-documented representation gaps that can bias retrieved "consensus" |

This is also, structurally, what a system like a **response validator against curator ground truth** is doing — comparing model/system answers against expert-curated reference answers with semantic similarity plus domain judgment, rather than exact string match. That pattern (YES / YES_PARTIAL / NO against a curated standard) is the right shape for this evaluation layer.

---

## 6. Governance, Safety, and Compliance — Where Top 1% Builds Spend Disproportionate Effort

This is the layer most engineers underweight and where actual expertise shows:

- **Scope boundary enforcement**: research-evidence-summarization tool vs. clinical-decision-support tool are legally and regulatorily different products. If the system's outputs could influence a treatment decision, it may fall under FDA Software as a Medical Device (SaMD) guidance for AI/ML-based tools — top builds get explicit legal/regulatory classification early, not after launch.
- **Audit trail by design**: every answer logs the exact chunks, PMIDs, and model version used to produce it, retained and queryable — required for any post-hoc review of a disputed or incorrect answer.
- **Bias and representation monitoring**: continuous tracking of whether the evidence base being surfaced reflects known gaps in oncology trial demographics (age, sex, race/ethnicity), since silently "following the evidence" can encode and launder real historical bias in who gets enrolled in trials.
- **Human-in-the-loop escalation paths**: low-confidence, high-stakes, or contradictory-evidence queries are routed to flag for expert review rather than answered with false confidence.
- **Data provenance and licensing**: PubMed abstracts are generally open, but full-text PMC content has per-article licensing (some CC-BY, some more restrictive) — a compliant system tracks and respects this at the chunk level, not just at ingestion.

---

## 7. Reference Architecture Shape

```
PubMed/PMC Entrez API ─┐
bioRxiv/medRxiv ────────┼─▶ Ingestion & Normalization
ClinicalTrials.gov ─────┤     - MeSH/PICO/study-type tagging
Retraction Watch DB ────┘     - retraction monitoring (recurring)
                                     │
                                     ▼
                        Section-aware, statistic-bound chunking
                                     │
                                     ▼
                  Biomedical embeddings (PubMedBERT/BioBERT/MedCPT)
                        + UMLS/MeSH concept normalization
                                     │
                                     ▼
        Hybrid retrieval (dense + sparse) + evidence-hierarchy-aware re-rank
                                     │
                                     ▼
         PICO-decomposed query → filtered/ranked chunk set
                                     │
                                     ▼
      Grounded generation: extracted (not generated) statistics,
      contradiction surfacing, evidence-strength labeling,
      mandatory PMID-level citation
                                     │
                                     ▼
   Evaluation: RAGAS + BioASQ/PubMedQA + clinician panel + numerical audit
                                     │
                                     ▼
        Governance: audit logging, bias monitoring, scope enforcement,
                    human escalation for low-confidence/high-stakes queries
```

---

## 8. What Actually Separates Top 1% From "Working Demo" Here

If you strip this down to the essential differentiators, it's these five things — everything else is execution detail:

1. **Numbers are extracted, never trusted to model recall.**
2. **Disagreement across sources is surfaced, never silently resolved.**
3. **Study design and evidence strength are first-class retrieval/ranking signals, not metadata.**
4. **Retractions and corrections are monitored continuously, not checked once at ingestion.**
5. **A clinician (or domain expert) evaluation loop exists and actually gates deployment — not just automated metrics.**

A system that gets the generic RAG pipeline right but skips these five is a good demo. A system that gets these five right, even with a mediocre generic pipeline, is closer to something a clinical or research team could actually trust.