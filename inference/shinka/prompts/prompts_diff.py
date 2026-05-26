DIFF_SYS_FORMAT = """
You MUST repond using a edit name, description and the exact SEARCH/REPLACE diff format shown below to indicate changes:

<NAME>
A shortened name summarizing the edit you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
A description and argumentation process of the edit you are proposing.
</DESCRIPTION>

<DIFF>
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE

</DIFF>


Example of a valid diff format:
<DIFF>
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

</DIFF>

* You may only modify text that lies below a line containing "EVOLVE-BLOCK-START" and above the next "EVOLVE-BLOCK-END". Everything outside those markers is read-only.
* Do not repeat the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the SEARCH/REPLACE blocks.  
* Every block’s SEARCH section must be copied **verbatim** from the current file.  Including indentation.
* You can propose multiple independent edits. SEARCH/REPLACE blocks follow one after another. DO NOT ADD ANY OTHER TEXT BETWEEN THESE BLOCKS.
* Make sure the file still runs after your changes.
""".rstrip()


DIFF_ITER_MSG = """# Current program

Here is the current program we are trying to improve (you will need to propose a modification to it below):

```{language}
{code_content}
```

Here are the performance metrics of the program:

{performance_metrics}{text_feedback_section}

# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.

Key directions to explore:
1. The optimal arrangement may involve heterogeneous or variable-sized elements
2. Strong solutions often use hybrid global-local patterns
4. The optimization routine is critical - use models with carefully tuned parameters
5. Use scipy optimize, LP, or SLSQP to optimize variables given candidate structures
""".rstrip()


DIFF_ABLATE_SYS_FORMAT = """
You are given a program whose iterative refinement has plateaued. Your task is to abandon the current implementation and rewrite the program from a fundamentally new perspective. Provide the complete new program code.

You MUST respond using a short summary name, description and the full code:

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
* Make sure your rewritten program maintains the same inputs and outputs as the original program.
* Make sure the file still runs after your changes.
* Use the <NAME>, <DESCRIPTION>, and <CODE> delimiters to structure your response. It will be parsed afterwards.
""".rstrip()


DIFF_ABLATE_ITER_MSG = """# Current program

Here is the current program. The evolution loop has been stuck on iterations of approaches similar to this one — incremental tweaks have not been moving the score:

```{language}
{code_content}
```

Performance metrics of the current program:

{performance_metrics}{text_feedback_section}

{previous_attempts}

# Task

The current implementation has plateaued. Iterating on it further is unlikely to help. Instead:

1. Identify components of the current code that look unreasonable or that may be holding the search inside a local optimum (heuristics that don't pay off, design choices the search keeps committing to, dead branches, parameter sweeps that add little).
2. DELETE those components.
3. Rewrite the program from a fundamentally new perspective: pick an algorithm class, data structure, or strategy that the current program does NOT use, and commit fully to it.

Do not iterate on the current implementation. Do not stitch new code onto the old skeleton. Commit fully to a different approach.

A fundamental change replaces the solution representation (e.g., closed-form ↔ free coordinates ↔ discrete) or the search paradigm (e.g., gradient ↔ sampling ↔ enumeration). Swapping the optimizer, picking a sibling parametric family, or adding numerical guards are NOT fundamental changes — they leave the search trapped.

For example, the following are structurally orthogonal algorithm classes — two attempts in the same class are minor variants of each other no matter how the surface code differs:
- Closed-form analytical construction (orbit of a finite symmetry group, vertices of a known polytope, regular polygon, root-system points)
- Low-discrepancy / quasi-random sampling on a fixed domain (Halton, Sobol, Hammersley, sunflower spiral, Fibonacci lattice)
- Lattice / grid enumeration (G×G square grid, hexagonal lattice, crystallographic packing — search over subsets/labels)
- Continuous local optimization on free decision variables (gradient on a smoothed objective, SLSQP / Nelder-Mead / coordinate ascent on the raw objective)
- Population-based global search (CMA-ES, Differential Evolution, Genetic Algorithm — many parallel candidates with selection)
- Discrete combinatorial search over a finite candidate set (simulated annealing on subset selection, branch-and-bound, ILP, beam search over partial states)
- Constructive online insertion (farthest-first / k-center, max-min greedy adding one element at a time, beam search building a configuration step by step)
- Physics / relaxation methods (Lloyd / centroidal Voronoi tessellation, repulsive force fields, gradient flow with hard-margin barriers, simulated cooling on continuous coordinates)
- Algebraic / number-theoretic structure (lattice orbits of a Coxeter group, points related by a Möbius / projective map, modular-arithmetic constructions)

The list is illustrative, not exhaustive — feel free to commit to any class outside the previous attempts, including ones not above.

In the <DESCRIPTION>: name the OLD strategy in one sentence, the NEW strategy you committed to in one sentence, what you removed, and why a clean swap (not incremental tweaks) is the right move now — what local optimum the old strategy is stuck in and how the new one structurally avoids it.

Key directions to explore:
1. The optimal arrangement may involve heterogeneous or variable-sized elements
2. Strong solutions often use hybrid global-local patterns
4. The optimization routine is critical - use models with carefully tuned parameters
5. Use scipy optimize, LP, or SLSQP to optimize variables given candidate structures
""".rstrip()
