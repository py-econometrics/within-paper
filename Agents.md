# Goal

We are writing a Stata-journal like paper on our new algorithm for demeaning fixed effects. 
The audience is applied microeconomists. 
This implies that the math needs to be correct, but we do not need to be uber-precise; no need to die in formal rigor. 
The paper should provide the core ideas behind the method of alternating projections (MAP), why it can be slow, and 
explain how our solver adresses MAPs shortcoming. W
We then want to demonstrate this using empirical data / benchmarks.
We conclude by showing our software. 
The paper should be pleasant to read and informative. Any smart 4th-year undergraduate should be able to follow. 

# Language

Before editing prose, use the `avoid-ai-writing` skill. Prefer direct, specific sentences
and remove staged narration, slogans, vague pointers, and inflated claims.

Do not use "sweep" as shorthand for a collection of benchmark designs or parameter
values. State which feature varies instead. Reserve "sweep" for its precise algorithmic
meaning: one complete MAP pass over the fixed-effect dimensions.
