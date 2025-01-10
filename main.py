import math

### Prime Generation Methods ###

def sieve_of_eratosthenes(n: int) -> list:

    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0], sieve[1] = False, False

    # Main Sieve logic - iterate until sqrt n
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False

    return [i for i in range(n + 1) if sieve[i]]


def sieve_of_sundaram(n: int) -> list:
    if n < 2:
        return []

    # n // 2 - used to compute the sieve range
    limit = (n - 1) // 2
    sieve = [True] * (limit + 1)

    for i in range(1, limit + 1):
        j = i
        while i + j + 2 * i * j <= limit:
            sieve[i + j + 2 * i * j] = False
            j += 1

    # handle case 2
    primes = []
    if n >= 2:
        primes.append(2)

    # convert back to actual numbers
    for i in range(1, limit + 1):
        if sieve[i]:
            p = 2 * i + 1
            if p <= n:
                primes.append(p)

    return primes


