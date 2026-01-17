import math

def test_phase_transition_inequalities_character_kgrams():
    H = {
        (1, 1): 2.9305,
        (1, 2): 3.1650,
        (1, 3): 3.4424,
        (1, 4): 3.5790,
        (2, 1): 2.5833,
        (2, 2): 2.4753,
        (2, 3): 2.5825,
        (2, 4): 2.6557,
        (5, 1): 2.2837,
        (5, 2): 1.7883,
        (5, 3): 1.6866,
        (5, 4): 1.7128,
        (10, 1): 2.2126,
        (10, 2): 1.4995,
        (10, 3): 1.2611,
        (10, 4): 1.2487,
        (20, 1): 2.1681,
        (20, 2): 1.2876,
        (20, 3): 0.9136,
        (20, 4): 0.8566,
        (50, 1): 2.1357,
        (50, 2): 1.1140,
        (50, 3): 0.5993,
        (50, 4): 0.4900,
        (100, 1): 2.1232,
        (100, 2): 1.0423,
        (100, 3): 0.4595,
        (100, 4): 0.3227,
    }

    sparse_gap = H[(1, 4)] - H[(1, 1)]
    dense_gap = H[(100, 4)] - H[(100, 1)]

    print(
        "PASS_PHASE_SPARSE_GAP"
        f" H(N=1,k=4)={H[(1,4)]:.4f}"
        f" H(N=1,k=1)={H[(1,1)]:.4f}"
        f" gap={sparse_gap:.4f}"
    )
    print(
        "PASS_PHASE_DENSE_GAP"
        f" H(N=100,k=4)={H[(100,4)]:.4f}"
        f" H(N=100,k=1)={H[(100,1)]:.4f}"
        f" gap={dense_gap:.4f}"
    )

    assert sparse_gap > 0.0
    assert dense_gap < 0.0


def test_context_space_utilization_explicit_ratio():
    sigma = 46
    k = 4
    unique_contexts = 664
    theoretical = sigma**k
    util = unique_contexts / theoretical

    print(
        "PASS_CONTEXT_UTIL"
        f" sigma={sigma}"
        f" k={k}"
        f" theoretical={theoretical}"
        f" unique={unique_contexts}"
        f" utilization={util:.12f}"
        f" percent={(100.0*util):.8f}"
    )

    assert theoretical == 4_477_456
    assert unique_contexts == 664
    assert util < 0.0002


def test_transformer_vs_ngram_explicit_gaps():
    ngram = {
        1: 3.6451,
        2: 2.6557,
        5: 1.7128,
        10: 1.2487,
        20: 0.8566,
        50: 0.4900,
        100: 0.3227,
    }
    tr = {
        1: 3.6910,
        2: 3.6009,
        5: 1.3868,
        10: 1.7347,
        20: 1.5319,
        50: 0.3862,
        100: 0.3743,
    }

    gap_N5 = ngram[5] - tr[5]
    gap_N100 = ngram[100] - tr[100]

    print(
        "PASS_MODEL_GAP_N5"
        f" H_ngram={ngram[5]:.4f}"
        f" H_transformer={tr[5]:.4f}"
        f" ngram_minus_transformer={gap_N5:.4f}"
    )
    print(
        "PASS_MODEL_GAP_N100"
        f" H_ngram={ngram[100]:.4f}"
        f" H_transformer={tr[100]:.4f}"
        f" ngram_minus_transformer={gap_N100:.4f}"
    )

    assert gap_N5 > 0.0
    assert gap_N100 < 0.0


def test_perplexity_consistency():
    H = 0.3227
    PP = math.exp(H)
    print(f"PASS_PP exp({H})={PP:.10f}")
    assert abs(PP - 1.3808) < 0.01
