import string
import re
import math
from dataset import *

alphabet = string.ascii_lowercase + "0123456789"

keywords = ["int", "include", "main", "float", "scanf",
            "printf", "if", "return", "for", "while", "else", "stdio"]

def ord(char):
    return alphabet.index(char) + 1


def tokenize(string):
    tokens = []
    for char in re.sub(r"\b" + "|".join(keywords) + r"\b" , "", string.lower()):
        if char in alphabet:
            tokens.append(char)
    return tokens


def rollingHash(k, string):
    base = ord("9")
    tokens = tokenize(string)
    initialHash = sum(ord(token) * base ** (k - idx - 1)
                      for idx, token in enumerate(tokens[:k]))
    hashes = [initialHash]
    for i in range(k, len(tokens)):
        token = tokens[i]
        prevToken = tokens[i-k]
        hashes.append((hashes[-1] - ord(prevToken) *
                       base ** (k - 1)) * base + ord(token))
    return list(hashes)


def chunks(l, n):
    n_chunks = math.ceil(len(l) / n)
    for i in range(n_chunks):
        yield l[i*n:(i+1)*n]


def fingerprint(string, w, k):
    return set(map(min, chunks(rollingHash(k, string), w)))


def fingerprints(assignment, w, k):
    prints = []
    i = 0
    for id in assignment.ids:
        with open(get_path(assignment, id)) as f:
            prints.append(fingerprint(f.read(), w, k))
        i += 1
    return prints


def matches(assignment, w=100, k=150, t=3):
    prints = fingerprints(assignment, w, k)
    matches = set()
    for i in range(len(prints)):
        for j in range(i+1, len(prints)):
            if len(set.intersection(prints[i], prints[j])) > t:
                matches.add(frozenset((assignment.ids[i], assignment.ids[j])))
    return matches
