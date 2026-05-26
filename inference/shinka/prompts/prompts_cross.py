import random
from typing import List

from shinka.database import Program
from .prompts_base import perf_str


CROSS_SYS_FORMAT = """
You are given multiple code scripts implementing the same algorithm.
You are tasked with generating a new code snippet that combines these code scripts in a way that is more efficient. 
I.e. perform crossover between the code scripts.
Provide the complete new program code.
You MUST repond using a short summary name, description and the full code:

<NAME>
A shortened name summarizing the code you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
A description and argumentation process of the code you are proposing.
</DESCRIPTION>

<CODE>
```{language}
# The new rewritten program here.
```
</CODE>

* Keep the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the code. Do not change the code outside of these markers.
* Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.
* Make sure the file still runs after your changes.
* Use the <NAME>, <DESCRIPTION>, and <CODE> delimiters to structure your response. It will be parsed afterwards.
""".rstrip()


CROSS_ITER_MSG = """# Current program

Here is the current program we are trying to improve (you will need to propose a new program with the same inputs and outputs as the original program, but with improved internal implementation):

```{language}
{code_content}
```

Here are the performance metrics of the program:

{performance_metrics}{text_feedback_section}

# Task

Perform a cross-over between the code script above and the one below. Aim to combine the best parts of both code implementations that improves the score.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.

Key directions to explore:
1. The optimal arrangement may involve heterogeneous or variable-sized elements
2. Strong solutions often use hybrid global-local patterns
4. The optimization routine is critical - use models with carefully tuned parameters
5. Use scipy optimize, LP, or SLSQP to optimize variables given candidate structures
""".rstrip()


def get_cross_component(
    archive_inspirations: List[Program],
    top_k_inspirations: List[Program],
    language: str = "python",
) -> str:
    all_inspirations = archive_inspirations + top_k_inspirations

    # TODO(RobertTLange): Compute embedding distance between all inspirations and parent - max?! for more diversity

    # Sample a random inspiration
    inspiration = random.choice(all_inspirations)

    crossover_inspiration = "# Crossover Inspiration Programs\n"
    crossover_inspiration += f"```{language}\n{inspiration.code}\n```\n\n"
    crossover_inspiration += f"Performance metrics: {perf_str(inspiration.combined_score, inspiration.public_metrics)}\n\n"

    return crossover_inspiration


CROSS_COMBINE_ITER_MSG = """# Current program

Here is the current program we are trying to improve:

```{language}
{code_content}
```

Here are the performance metrics of the program:

{performance_metrics}{text_feedback_section}

# Task: trick combination

Below you will see SEVERAL inspiration programs. For EACH inspiration:
1. Identify the single most distinctive trick / mechanism it uses (a particular
   initialization, a refinement step, a numerical formulation, a heuristic, ...).
2. Decide whether that trick is compatible with the current program and is likely
   additive (i.e. attacks a different failure mode than what the current program
   already handles).

Then produce a NEW full program that is the current program PLUS the compatible
tricks from the inspirations stitched in. Be explicit in the <DESCRIPTION> about
which trick came from which inspiration and why you expect them to compose
without redundancy. Drop tricks that conflict.

IMPORTANT: This is a combination, not a free rewrite. The skeleton should follow
the current program; the inspirations only contribute identifiable plug-in tricks.

Key directions to explore:
1. The optimal arrangement may involve heterogeneous or variable-sized elements
2. Strong solutions often use hybrid global-local patterns
4. The optimization routine is critical - use models with carefully tuned parameters
5. Use scipy optimize, LP, or SLSQP to optimize variables given candidate structures
""".rstrip()


CROSS_TRANSLOCATE_ITER_MSG = """# Current program (the "near" parent — keep its skeleton)

```{language}
{code_content}
```

Performance metrics: {performance_metrics}{text_feedback_section}

# Task: trick translocation from a distant relative

Below you will see ONE inspiration program drawn from the archive (a "distant
relative" — likely structurally different from the current program). Your job:

1. Read it and pick the ONE trick that is most likely to help the current
   program — a specific initialization, refinement step, constraint formulation,
   numerical detail, or heuristic. Be concrete; name it.
2. Transplant ONLY that trick into the current program. Keep the rest of the
   current program intact. Do NOT also fold in other ideas from the donor and
   do NOT broadly rewrite the recipient.
3. Adapt naming / signatures so the transplant compiles, but do not refactor
   surrounding code beyond what the transplant strictly requires.

Argue in the <DESCRIPTION>: which trick, why this one, and why grafting it onto
the current skeleton is more promising than full crossover.

Key directions to explore:
1. The optimal arrangement may involve heterogeneous or variable-sized elements
2. Strong solutions often use hybrid global-local patterns
4. The optimization routine is critical - use models with carefully tuned parameters
5. Use scipy optimize, LP, or SLSQP to optimize variables given candidate structures
""".rstrip()


def get_cross_combine_component(
    archive_inspirations: List[Program],
    top_k_inspirations: List[Program],
    language: str = "python",
    max_inspirations: int = 3,
) -> str:
    """Show several inspirations side-by-side so the LLM can pick tricks from each."""
    all_inspirations = archive_inspirations + top_k_inspirations
    if not all_inspirations:
        return ""
    chosen = (
        random.sample(all_inspirations, max_inspirations)
        if len(all_inspirations) > max_inspirations
        else list(all_inspirations)
    )
    out = "# Inspiration Programs (extract one trick from each)\n\n"
    for i, insp in enumerate(chosen):
        out += f"## Inspiration {i + 1}\n"
        out += f"```{language}\n{insp.code}\n```\n\n"
        out += (
            f"Performance metrics: "
            f"{perf_str(insp.combined_score, insp.public_metrics)}\n\n"
        )
    return out


def get_cross_translocate_component(
    archive_inspirations: List[Program],
    top_k_inspirations: List[Program],
    language: str = "python",
) -> str:
    """Pick a single distant donor (prefer archive over top-k)."""
    pool = archive_inspirations or top_k_inspirations
    if not pool:
        return ""
    inspiration = random.choice(pool)
    out = "# Distant Donor Program (transplant exactly one trick)\n"
    out += f"```{language}\n{inspiration.code}\n```\n\n"
    out += (
        f"Performance metrics: "
        f"{perf_str(inspiration.combined_score, inspiration.public_metrics)}\n\n"
    )
    return out
