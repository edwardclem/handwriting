# beam search decoder implementation.
import collections
import math

import numpy as np
from tqdm import trange

NEG_INF = -float("inf")


def make_new_beam():
    fn = lambda: (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)


def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


# TODO: vectorize? this is pretty slow.


# based on https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0
# have a beam! https://youtu.be/llSvwpz6Vh0?t=340
# the various merging operations are ways of marginalizing over multiple
# alignments that could map to more than one sequence.
def beam_decode(log_probs, beam_size=100, blank=0):
    T, S = log_probs.shape
    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank
    # (in log space).
    beam = [(tuple(), (0.0, NEG_INF))]

    for t in trange(T, desc="Decoding timesteps"):  # Loop over time
        # A default dictionary to store the next step candidates.
        next_beam = make_new_beam()
        for s in range(S):  # loop over vocab
            p = log_probs[t, s]

            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, (p_b, p_nb) in beam:  # Loop over beam
                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == blank:
                    n_p_b, n_p_nb = next_beam[prefix]

                    # combine the probabilities, but only update p_blank for that
                    # prefix. If I understand correctly, this is essentially
                    # a way to distinguish between "true" repeat characters (separated by a blank)
                    # and repeats that will get merged. Each element in the beam
                    # tracks its likelihood of both cases.
                    n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)
                    continue

                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated.

                # any non-blank character will produce a new prefix.
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beam[n_prefix]

                prev_char = prefix[-1] if prefix else None

                # if s isn't a repeat, update the likelihood of ending
                # with a non-blank
                if s != prev_char:
                    n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:
                    # We don't include the previous probability of not ending
                    # in blank (p_nb) if s is repeated at the end. The CTC
                    # algorithm merges characters not separated by a blank.

                    # if s is repeated, then the previous string must have ended
                    # with a blank.
                    n_p_nb = logsumexp(n_p_nb, p_b + p)

                # *NB* this would be a good place to include an LM score.
                next_beam[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case - i.e. the case in which the previous
                # prefix DID end with a blank.
                if s == prev_char:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = logsumexp(n_p_nb, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)

        # Sort and trim the beam before moving on to the next time-step.
        # combine the scores for both ending with blank and not ending with blank when selecting.
        beam = sorted(next_beam.items(), key=lambda x: logsumexp(*x[1]), reverse=True)
        beam = beam[:beam_size]

    best = beam[0]
    return best[0], -logsumexp(*best[1])
