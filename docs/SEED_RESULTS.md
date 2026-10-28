# Seed results, 28 October

Run four weeks before the deadline, deliberately, to find out what breaks while
there is still time to change course.

## What could and could not be run

HuggingFace is unreachable from this machine's sandbox, so **coreference (L3)
and the e5 encoder (L2) could not run**. GitHub is reachable, so spaCy and its
English model installed, which gives a real dependency parse.

That makes the layers below *the condition Chaturvedi et al. (2018) actually
ran*: characters from surface mentions, roles from the parse, **no coreference
anywhere**. They are labelled `l3n` and `l4n` for "no coreference". Running them
first is useful on its own — they fix the pre-coreference point on this
benchmark, so the eventual L3 minus L3n difference measures what coreference
contributes rather than what the whole pipeline contributes.

## Dev results (n = 200)

| layer | accuracy | 95% bootstrap CI |
|---|---|---|
| L0 random | 46.00 | [39.5, 52.5] |
| L0 Jaccard | 45.00 | [38.0, 52.0] |
| L1 TF-IDF | 51.50 | [44.0, 58.5] |
| L3n characters, *actors only* | 52.50 | [45.5, 59.5] |
| L3n characters, *all noun chunks* | **57.00** | [50.0, 63.5] |
| L4n roles | 53.50 | [47.0, 60.0] |

Paired comparisons (McNemar, same 200 items):

| comparison | delta | p |
|---|---|---|
| Jaccard → TF-IDF | +6.50 | 0.137 |
| TF-IDF → L3n (actors only) | +1.00 | 0.909 |
| TF-IDF → L3n (all noun chunks) | +5.50 | 0.242 |
| L3n → L4n roles | +1.00 | 0.850 |
| **Jaccard → L3n (all noun chunks)** | **+12.00** | **0.006** |

## The finding that matters

**Restricting the cast to things that behave like characters makes accuracy
worse: 57.00 drops to 52.50.**

The filter keeps a mention group if it is a proper noun or ever appears as the
subject of a verb. It is a defensible definition of "character" — it keeps
*the organization*, *citizens*, *mary murphy*, and discards *safety*,
*its mission*, *the book*. Applying it costs 4.5 points.

The straightforward reading is that the gain over Jaccard is **not coming from
character structure**. It is coming from noun-phrase content overlap: a richer
lexical feature, computed through an expensive pipeline. The only comparison
that clears significance, Jaccard → L3n at +12.00, p = 0.006, is also the
comparison where the baseline is weakest, and L3n's advantage over TF-IDF
(+5.50, p = 0.242) does not clear it.

This is evidence against the proposal's hypothesis in its current form. It is
not yet decisive, for three reasons worth stating plainly:

1. **Coreference is exactly what is missing**, and it is the component the
   hypothesis rests on. Chaturvedi et al. named missed coreference as a source
   of error in this same setting. The honest statement today is that the
   pre-coreference condition shows no character-specific signal.
2. **n = 200 on dev.** The power analysis puts the detectable effect at 5.9-10.7
   points. A real 3-point effect would be invisible here.
3. **Dev is the harder split** (Jaccard 45.00 on dev against 58.00 on test), so
   dev understates everything.

## What changes because of this

1. **Add the actors-only contrast to the report.** It is a clean, cheap ablation
   that asks "is it characters, or is it nouns?" and today the answer leans
   towards nouns. That question is more interesting than the original one.
2. **L2 is now the priority**, not L3. Without the embedding backbone there is
   no way to tell whether any of this adds to a dense representation, which is
   the actual claim. Everything above may be noise around a 66.75-level baseline.
3. **Budget compute for L4.** The first L4n implementation rebuilt each
   character's role profile inside the pair loop and did not finish a 200-item
   split. Fixed by computing profiles once per character; a dev run is now
   about 60 seconds.
4. **Expect the same filtering question for real L3.** When coreference lands,
   run it both ways.

## Engineering faults found and fixed

- **Result files silently overwrote each other.** Two runs of one layer under
  different parameters wrote to the same filename, and the second destroyed the
  first. Filenames now carry an 8-character hash of the scorer parameters.
- **L4 was quadratic in redundant work**, as above.
- **Background runs were being reaped** when the shell exited; they need
  `setsid` on this machine.
