"""
Long run Gray Codes

"Gray Codes with Optimized Run Lengths", Goddyn et al. 1998.

Reference: https://stackoverflow.com/questions/30519621/generating-long-run-gray-codes
"""

import json
from pathlib import Path

import numpy as np

json_dump_path = Path("quanta_SL/encode/graycodes/long_run_graycode.json")

if json_dump_path.exists():
    with open(json_dump_path) as f:
        transition_dict = json.load(f)["transitions"]
    transition_dict = {int(k): v for k, v in transition_dict.items()}
else:
    transition_dict = {
        2: [0, 1, 0, 1],
        3: [0, 1, 0, 2, 0, 1, 0, 2],
        4: [0, 1, 2, 3, 2, 1, 0, 2, 0, 3, 0, 1, 3, 2, 3, 1],
        5: [
            0,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            0,
            1,
            4,
            3,
            2,
            1,
            4,
            3,
            0,
            1,
            2,
            3,
            4,
            1,
            2,
            3,
            0,
            1,
            4,
            3,
            2,
            1,
            4,
            3,
        ],
        6: [
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            2,
            4,
            1,
            3,
            2,
            0,
            5,
            4,
            2,
            3,
            1,
            4,
            0,
            2,
            5,
            3,
            4,
            2,
            1,
            0,
            4,
            3,
            5,
            2,
            4,
            0,
            1,
            2,
            3,
            4,
            5,
            0,
            2,
            4,
            1,
            3,
            2,
            0,
            5,
            4,
            2,
            3,
            1,
            4,
            0,
            2,
            5,
            3,
            4,
            2,
            1,
            0,
            4,
            3,
            5,
            2,
            4,
        ],
    }


def transition_to_code(transition_sequence):
    code_sequence = [0]

    n = np.int(np.log2(len(transition_sequence)))

    code = 0

    for pos in transition_sequence:
        code ^= 1 << int(pos)
        code_sequence.append(code)

    return code_sequence[:-1]


def print_code_from_transition(transition_sequence):
    n = np.int(np.log2(len(transition_sequence)))

    codes = transition_to_code(transition_sequence)

    format_string = "b: {:0" + str(n) + "b}"

    for c in codes:
        print(format_string.format(c))


def gap(transition_sequence):
    as_array = np.array(transition_sequence)
    gap = 1

    while gap < len(transition_sequence):
        if np.any(as_array == np.roll(as_array, gap)):
            return gap
        gap += 1

    return 0


def valid_circuit(transition_sequence):
    n = np.int(np.log2(len(transition_sequence)))

    if not len(transition_sequence) == 2 ** n:
        print("Length not valid")
        return False

    if not np.all(np.array(transition_sequence) < n):
        print("Transition codes not valid")
        return False

    sorted_codes = np.sort(transition_to_code(transition_sequence))

    if not np.all(sorted_codes == np.arange(0, 2 ** n)):
        print("Not all Unique")
        return False

    return True


def interleave(A, B):
    n = np.int(np.log2(len(A)))
    m = np.int(np.log2(len(B)))

    M = 2 ** m
    N = 2 ** n

    assert N >= M

    gap_A = gap(A)
    gap_B = gap(B)

    assert gap_A >= gap_B

    st_pairs = [(i, M - i) for i in range(M) if i % 2 == 1]

    sorted_pairs = sorted(st_pairs, key=lambda p: np.abs(p[1] / p[0] - gap_B / gap_A))

    best_pair = sorted_pairs[0]

    s, t = best_pair

    ratio = t / s

    P = "b"

    while len(P) < M:
        b_to_a_ratio = P.count("b") / (P.count("a") + 1)

        if b_to_a_ratio >= ratio:
            P += "a"
        else:
            P += "b"

    return P * N


def P_to_transition(P, A, B):
    Z = []

    pos_a = 0
    pos_b = 0

    n = np.int(np.log2(len(A)))

    delta = n

    for p in P:
        if p == "a":
            Z.append(A[pos_a % len(A)])
            pos_a += 1
        else:
            Z.append(B[pos_b % len(B)] + delta)
            pos_b += 1

    return Z


"""
Code for special case for 10-bits to fabric a gap of 8.

From: Binary gray codes with long bit runs
by: Luis Goddyn∗ & Pavol Gvozdjak

"""

T0 = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]


def to_4(i, sequence):
    permutations = []

    indices = [j for j, x in enumerate(sequence) if x == i]

    for pos in indices:
        permutation = sequence.copy()
        permutation[pos] = 4
        permutations.append(permutation)

    return permutations


def T_to_group(T):
    state = np.array([0, 0, 0, 0, 0])

    cycle = []

    for pos in T:
        cycle.append(state.copy())
        state[pos] += 1
        state[pos] %= 4

    return np.array(cycle)


def T_to_transition(T):
    ticker = [False, False, False, False, False]

    transitions = []

    for t in T:
        transistion = 2 * t + 1 * ticker[t]
        ticker[t] = not ticker[t]

        transitions.append(transistion)
    return transitions


if __name__ == "__main__":

    T1 = to_4(0, T0)[3] * 4
    T2 = to_4(1, T1)[0] * 4
    T3 = to_4(2, T2)[1] * 4

    transition_dict[10] = T_to_transition(T3)

    dump_json = False
    force_recompile = False
    code_dict = {
        k: transition_to_code(transition) for k, transition in transition_dict.items()
    }

    for bits in range(2, 14):
        if (bits in transition_dict) and not force_recompile:
            print(f"gray code for {bits} bits has gap: {gap(transition_dict[bits])}")

        else:
            dump_json = True
            print(f"finding code for {bits} bits...")

            all_partitions = [(i, bits - i) for i in range(bits) if i > 1]
            partitions = [(n, m) for (n, m) in all_partitions if n >= m and m > 1]
            current_gap = 0
            for n, m in partitions:
                P = interleave(transition_dict[n], transition_dict[m])
                Z = P_to_transition(P, transition_dict[n], transition_dict[m])
                candidate_gap = gap(Z)

                if candidate_gap > current_gap:
                    current_gap = candidate_gap
                    transition_dict[bits] = Z
            if valid_circuit(transition_dict[bits]):
                code_dict[bits] = transition_to_code(transition_dict[bits])
                print(
                    f"gray code for {bits} bits has gap: {gap(transition_dict[bits])}"
                )
            else:
                print("found in-valid gray code")

    if dump_json:
        print("Dumping transitions to JSON")
        with open(json_dump_path, "w") as f:
            dump_dict = {"transitions": transition_dict, "codes": code_dict}
            json.dump(dump_dict, f, indent=4, sort_keys=True)
