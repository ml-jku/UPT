def cumsum_of_sequence(sequence):
    r, s = [], 0
    for e in sequence:
        l = len(e)
        r.append(l + s)
        s += l
    return r
