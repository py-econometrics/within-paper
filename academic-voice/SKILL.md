---
name: academic-voice
description: >-
  Enforce a human, scholarly writing voice and strip LLM mannerisms when drafting or
  editing academic prose — paper sections, abstracts, derivations, cover letters, and
  referee responses. Bans staccato antithesis, hype adjectives, signposting filler
  ("crucially", "importantly"), the "not just X but Y" construction, teaser transitions,
  summary bows, and bare-imperative derivations; prescribes the authorial "we",
  consistent terminology, and varied sentence rhythm. Apply whenever generating or
  revising sentences for a manuscript. Triggers: academic writing, paper, manuscript,
  draft, revise, section, abstract, journal, prose style, referee response.
---

# Academic voice

Write so that a careful scholar in the field would accept the prose as their own. The
rules below target word choice and sentence voice; they remove the machine *tells*
without flattening the writing into lifeless passive prose. Good academic writing still
has rhythm and intuition. The goal is to cut the habits that mark generated text, not
the life.

## The read-aloud test

Before keeping a sentence, read it aloud. Rewrite it if it sounds like a conference
keynote, a blog hook, or marketing copy: a punchy two-beat contrast, a triad of
adjectives, a cliff-hanger transition, or a tidy moral at the end of a paragraph.
Scholarly prose explains; it does not perform.

## 1. Rewrite these constructions on sight

**Staccato antithesis** — short, parallel clauses set against each other for punch. This
is the single most common tell.

- Avoid: "One piece is cheap and one is hard."
- Avoid: "The idea is simple. The execution is not."
- Prefer: "The two pieces differ sharply in cost: because $G_(W W)$ is diagonal,
  $G_(W W)^(-1)$ is immediate, whereas $S^(-1)$ requires solving the reduced worker–firm
  system."

State a contrast with a subordinating connective ("whereas", "while", "although", "by
contrast"), not with a drumbeat of clipped sentences.

**"Not just X, but Y" / "It is not about X; it is about Y."**

- Avoid: "This is not merely a preconditioner; it is a way of seeing the graph."
- Prefer: "The preconditioner also exposes the graph structure that diagonal scaling
  ignores."

**Rhetorical rule of three** — a triad of nouns, verbs, or adjectives used for cadence
rather than because there happen to be three things.

- Avoid: "The method is fast, principled, and elegant."
- Prefer: "The method converges in fewer iterations and needs no tuning."

Two precise claims beat three vague ones. Vary list length so nothing sounds chanted.

**Teaser / curiosity-gap transitions.**

- Avoid: "But there is a catch." / "This raises a natural question." / "Here is where it
  gets interesting."
- Prefer: state the catch or the question directly — "This construction fails when one
  factor is nested in another, because …"

**Summary bows and throat-clearing.** Openers such as "When it comes to …", "In the
context of …", "It is important to note that …", and closers such as "In essence,", "At
its core,", "Ultimately,", and "This not only … but also …". Delete them. If a sentence
only restates the paragraph, cut it.

**Em-dash drama.** One appositive dash in a sentence is fine. Two dashes in a sentence,
or a dash deployed for suspense, is a tell. Prefer commas, parentheses, or a second
sentence.

**Gerund-phrase subjects.** Opening a sentence with an "-ing" phrase as its grammatical
subject — "Applying it amounts to …", "Turning $P^(-1)$ into $M^(-1)$ means …",
"Inverting $G_(W W)$ is immediate" — gives every sentence the same abstract shape. Give
the sentence a concrete subject (usually "we", or the object itself) and a finite verb.

- Avoid: "Turning $P^(-1)$ into a usable operator $M^(-1)$ means computing each pair
  inverse cheaply."
- Prefer: "To make $P^(-1)$ usable, we must compute each pair inverse cheaply." — or "A
  usable $M^(-1)$ needs a cheap inverse for each pair."
- Avoid: "Applying it amounts to division by weighted level counts."
- Prefer: "We apply it as a single division by weighted level counts."

**"X means / amounts to / boils down to Y."** These reductive verbs narrate the real
action at one remove. Name the action: not "the update means subtracting the mean", but
"the update subtracts the mean".

**Abstract-noun summaries that label the argument.** A sentence whose subject is "the
difficulty", "the structural point", "the takeaway", "the upshot", "the intuition", or
"the key" — usually closing a paragraph. It comments on the argument instead of making
it. Cut it, or state the substantive claim directly.

- Avoid: "The difficulty of the fixed-effect solve is concentrated here."
- Avoid: "With three factors the closed form is messier, but the structural point is
  unchanged."
- Prefer: "$S^(-1)$ is therefore the only expensive part of the solve." / "With three
  factors the algebra is longer, but every block of $G^(-1)$ still depends on all three
  cross-tabulations."

**Editorializing the difficulty.** Adjectives that rate a computation rather than
describe it — "immediate", "trivial", "straightforward", "messier", "clean" — stand in
for the concrete fact. Lead with the fact and the rating is unnecessary.

- Avoid: "Inverting $G_(W W)$ is immediate because it is diagonal."
- Prefer: "$G_(W W)$ is diagonal, so inverting it is a division by worker counts."

**Finish the sentence.** Do not tack a specification onto a complete sentence as a
verbless fragment after a dash or colon — especially "here / namely + a list". It reads
as an unfinished half-sentence. Supply the missing verb, or make it its own sentence.

- Avoid: "It works through overlapping local subproblems — here the worker-firm,
  worker-year, and firm-year blocks."
- Prefer: "It works through overlapping local subproblems. In the AKM case these are the
  worker-firm, worker-year, and firm-year blocks."

## 2. Drop the LLM lexicon

Replace on contact:

| Avoid | Use instead |
|---|---|
| leverage, utilize | use |
| in order to | to |
| delve into, dive into | examine, study |
| shed light on, illuminate | show, clarify |
| underscore, highlight | emphasize — or just state the point |
| crucial, vital, key, pivotal | important, central — or quantify it |
| powerful, robust, seamless, elegant, rich | name the specific property |
| realm, landscape, tapestry, paradigm | the actual noun (field, setting, method) |
| plays a … role in | affects, determines, governs |
| a myriad of, a plethora of | many, or a specific count |
| navigate (challenges) | address, handle |
| immediate, trivial, straightforward (of a computation) | name the operation: "a single division", "one pass" |
| messier, cleaner, hairier | "longer", "has more terms" — or show the structure |
| amounts to, boils down to, comes down to | state the actual step |
| it is worth noting that | (cut) |

Cut empty intensifiers: "very", "extremely", "vastly", "truly", "simply", and "just" as
a minimizer. Reserve "significantly" for statistical significance, and then say so.

## 3. Use the authorial "we"

Narrate the paper's actions in the first-person plural, not as bare imperatives to the
reader. This is the standard register in economics, statistics, and applied mathematics.

- Avoid: "To see what $G^(-1)$ requires, write $G$ in block form."
- Prefer: "To see what $G^(-1)$ requires, we write $G$ in block form."
- Avoid: "Now consider the firm equation. Note that $G_(F F)$ is diagonal."
- Prefer: "We now turn to the firm equation, where $G_(F F)$ is again diagonal."

Keep genuine mathematical imperatives that address the object rather than the reader:
"Let $T = "diag"(I, -I)$", "Fix a factor pair $(q, r)$", "Suppose mobility is sparse."
The rule bans imperatives that narrate — "write", "consider", "note", "observe",
"recall" — not the definitional ones — "let", "fix", "suppose", "assume", "denote".

Do not use "I" (for a sole author, "we" remains conventional), avoid "the authors", and
do not address the reader as "you".

## 4. Repeat the exact term; do not elegant-vary

Generated prose swaps in synonyms to avoid repeating a word. Technical writing needs the
opposite: one name per concept, used every time. Alternating between "preconditioner",
"operator", "matrix", and "approximation" for the same object makes the reader ask
whether four different things are meant.

Choose the term — "the Schwarz preconditioner $M^(-1)$" — and reuse it verbatim.
Repetition of a defined term is correct, not lazy.

## 5. Sentence rhythm and register

- Vary sentence length. Generated prose drifts toward a uniform medium length; human
  academic prose mixes short claims with longer, subordinated sentences that carry the
  argument.
- No contractions: "doesn't" becomes "does not". Keep a formal register throughout.
- Prefer precise verbs: "yields", "gives", "reduces to", "implies", "follows from",
  "eliminates". Replace "is able to" with "can" and "make use of" with "use".
- Do not open consecutive sentences with "Thus,", "Hence,", "Therefore,", or
  "Moreover,". Use the connective once, where the logic actually turns.
- Avoid mechanical "First … Second … Third …" scaffolding in prose; reserve enumerated
  lists for genuinely enumerable items.
- Demonstratives: use "this / these" for the thing just named ("This structure", not
  "That structure"); reserve "that / those" for genuine distance or contrast. Never leave
  a demonstrative bare — follow "this" with its noun ("this preconditioner", not "this").

## 6. Claim with calibrated confidence

- State proven results plainly: "Proposition 2 shows that …", not "We believe this may
  suggest that …".
- For empirical patterns, hedge with precision rather than padding: "the benchmarks
  indicate", "is consistent with", "in our designs". Avoid stacked hedges such as "it
  could potentially be the case that".
- Do not hype. Let magnitudes and proofs carry the weight: "converges in 12 iterations
  rather than 400", not "converges dramatically faster".

## 7. Personification: deliberate, not reflexive

Light personification can aid intuition — "the diagonal preconditioner does not know how
a firm connects to the rest of the market" — but generated text reaches for it by
default. Use it once, knowingly, for a genuinely helpful image; do not let every
operator "know", "see", "want", or "try". When in doubt, state the mechanism: "the
diagonal carries no information about the firm's position in the mobility graph."

## Workflow

1. Draft for content first; do not self-edit while the argument is still forming.
2. Run a dedicated voice pass: search the draft for the vocabulary in §2, the
   constructions in §1, and the narrating imperatives in §3.
3. Read each changed paragraph aloud (the read-aloud test).
4. Check term consistency (§4): is each concept named the same way throughout?
