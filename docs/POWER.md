# What this design can actually detect

Computed 28 October with `scripts/power_analysis.py`, before any structural
layer was run. Alpha 0.05 two-sided, 80% power.

## The finding

**The diagnostic, as the proposal words it, cannot be supported statistically.**

The proposal says the gain will "concentrate in the first group", which invites
comparing one stratum's accuracy against the other's. That comparison is
unpaired, across disjoint item sets, and the strata are small:

| contrast | sizes | gap needed to reach significance |
|---|---|---|
| test `outcomes` | 79 vs 133 | **17.3 pts** |
| test `course_of_actions` | 188 vs 32 | 23.1 pts |
| dev `outcomes` | 40 vs 71 | 22.8 pts |

The design doc expects **3–8 points**. The best available contrast needs more
than double that. No result in the plausible range would clear the bar, so a
null finding here would say nothing about the hypothesis — only about the
sample size.

## What does work

Comparing two **layers on the same items** is paired, and paired tests are far
more sensitive. McNemar, by item count and by how often the two layers disagree:

| comparison | n | 10% disc. | 20% disc. | 30% disc. |
|---|---|---|---|---|
| test, all items | 400 | **4.3** | **6.2** | 7.6 |
| dev, all items | 200 | 5.9 | 8.6 | 10.7 |
| test `outcomes` none_marked | 133 | **7.0** | 10.4 | 13.0 |
| test `outcomes` all_marked | 79 | 8.6 | 13.1 | 16.4 |

Only the full test set reaches into the 3–8 band, and only when the two layers
disagree on few items.

## What this changes

1. **The confirmatory claim is the overall paired comparison**: does L4 beat L2
   on all 400 test items? That is answerable at roughly 4–6 points.
2. **The per-stratum result is descriptive, not confirmatory.** Report it as a
   point estimate with a confidence interval and say plainly that the stratum
   sizes cannot support a significance claim. An interval that excludes zero in
   one stratum and includes it in the other is suggestive and should be
   described exactly that way.
3. **Run McNemar *within* each stratum, not between them.** Within the
   133-item stratum the bar drops from 17.3 to 7.0 at low discordance. Between
   strata there is no paired comparison to make, because the items differ.
4. **Do not develop the diagnostic on dev.** Dev's unanimous `outcomes` strata
   are 40 and 71, worse than test's 79 and 133.
5. **Say the sample-size limit in the report before showing the numbers.** It is
   a property of the benchmark, not a failure of the method, and stating it
   first is what separates an honest negative from a weak one.

## Already visible in the baselines

TF-IDF beats Jaccard by 6.5 points on dev and McNemar still returns p = 0.137
(b01 = 39, b10 = 26). A six-point gap on 200 items is not significant. That is
this analysis showing up in practice rather than in theory.
