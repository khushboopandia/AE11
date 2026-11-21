# Decisions

One row one decision. This week is the first one. By Week 5 this file will explain what decisions you made during this bootcamp.

## Retrieval, week 1

**Context**
> The problem and the data - This is an intake assistant for pediatric neuropsychological evaluation. A caregiver describes their concerns in their own words, and the system, given the child’s age, maps that statement to up to 15 concern domains (for example sleep, speech_language or skill_regression). It is non-diagnostic, uses only synthetic data, and has not been clinically validated.

> Corpus. I used AI to draft a plain-text reference corpus from public sources, mainly US government and professional bodies (CDC, NIH/NIMH/NIDCD, AAP, AASM, AACAP, ASHA), supplemented by a few lower-authority sites where age-by-age detail was scarce. It has 15 domains, each with an overview and four age bands (under 4, 4 to 5, 6 to 11, 12 to 17), which is 75 sections.

> Caregiver Input - The caregiver statements are synthetic, generated with Google’s Simula framework for synthetic data generation (“Reasoning-Driven Synthetic Data Generation and Evaluation”, Davidson, Seguin, Bacis, Ilharco and Harkous, published in Transactions on Machine Learning Research). I chose ten for the assignment: eight that map to one or more domains, and two (Dev and Priya) written as no-match cases to test whether the system declines. The expected domains are my own judgment and haven’t been validated by a clinician.

> This task has two parts. Retrieval finds evidence for the concern domains a caregiver’s statement could map to, and the model judges whether that evidence is good enough to confirm a domain. For example, if a statement could map to skill_regression, retrieval should return every chunk that could give clinical grounding for it, and the model decides whether the evidence supports the domain. So retrieval is built for recall, and the precision of the final answer is the model’s job, judged at the generation step.

> Because this assignment is about RAG, I evaluated retrieval only. There’s no pass or fail. For each of the ten cases I compared the retrieved chunks before and after the change and judged whether they were a better fit: whether chunks for each expected domain were present, whether the chunks were the child’s age band or “all ages”, and whether noisy or wrong-band chunks had been dropped. Each case is rated improved, same or regressed. Retrieval improved on 7 of 10, stayed the same on 3, and regressed on none.


**What I changed**

> I replaced fixed-size chunks with section-aware chunks and added an age-band filter, k=5 and a 0.4 score cutoff as one bundled retrieval change.

**Why**

>- Failures targeted: wrong-age band chunks and missing expected-domain chunks.
>- Before and after: retrieval improved on 7 of 10 cases, stayed the same on 3, and regressed on 0.
>- The change is one bundle, so it doesn’t show which part did the work.

**Results**
| # | Case | Expected primary (band) | Gap | Retrieval | Why |
|---|---|---|---|---|---|
| 1 | Corin, 9 | learning_academic (6–11) | Wrong age-band chunks (2 of 3 off target) | **Improved** | Round 1 had 1 of 3 chunks on target, and the other two were the wrong band or domain. Round 2 has the expected chunk at rank 1, and all 5 chunks are in band. |
| 2 | Haeun, 8 | sleep (6–11) | Wrong age-band chunk (under 4); expected 6–11 chunk missing | **Improved** | Round 1 returned the under-4 sleep chunk and missed the 6–11 section. Round 2 retrieves sleep 6–11 at rank 4, and every chunk is in band. |
| 3 | Antoni, 9 | sensory_processing (6–11) | Off-target noise (preamble, global_delay chunk). Still open: expected chunk's rank fell from 3 to 5 | **Same** | The expected chunk is retrieved in both rounds. The noise is gone, but the section chunk is shorter and ranks lower. |
| 4 | Freya, 12 | attention_executive (12–17) | Expected-domain chunk missing (learning_academic); exact 12–17 section mixed into other-band chunks | **Improved** | The 12–17 section is isolated at rank 2, and a learning_academic chunk is now retrieved (rank 5). |
| 5 | Saba, 3 | skill_regression, speech_language, sensory_processing, sleep (under 4) | Expected-domain chunks missing (3 of 4). Still open: sensory and sleep | **Improved** | Expected domains with a chunk went from 1 of 4 to 2 of 4 (speech_language, skill_regression). The corpus covers sensory and sleep, but those chunks didn't rank. |
| 6 | Samir, 2 | adaptive_daily_living (under 4) | Expected age-band section missing (only the all-ages overview is retrieved). Still open | **Same** | The adaptive overview is retrieved in both rounds. The under-4 adaptive section is never retrieved, so the band is untested in both. |
| 7 | Amara, 0.9 | emotional_mood, social_communication (under 4) | Expected-domain chunk missing (social_communication) | **Improved** | Round 1 had no social_communication chunk. Round 2 has the under-4 social_communication chunk at rank 2. |
| 8 | Soraya, 15 | learning_academic, emotional_mood, motor_coordination or visuospatial_nonverbal (12–17) | Expected-domain chunks missing (0 of 3). Still open: learning_academic and motor, plus a corpus gap on abstract learning | **Improved** | Expected domains with a chunk went from 0 of 3 to 2 of 3 (emotional_mood, visuospatial_nonverbal). |
| 9 | Dev, 12 | none (decline) | Decoy chunks returned (no cutoff effect) | **Same** | His scores (0.501 to 0.443) are above the cutoff, so decoys come back as before. |
| 10 | Priya, 7 | none (decline) | Decoy chunks returned | **Improved** | All her scores (0.375 to 0.331) fall below the 0.4 cutoff, so nothing is retrieved. |

**What would change my mind**

>- If the 16 unused test chats show a real-match case scoring below 0.4, I’d drop the cutoff. It was set after seeing these ten rows.
>- I would update k based on more test cases if added noise or retrieved too few relevant chunks.

