import math
import random

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


def sieve_of_atkin(n: int) -> list:
    if n < 2:
        return []

    sieve = [False] * (n + 1)
    limit = int(math.isqrt(n)) + 1

    # candidate flipping
    for x in range(1, limit):
        for y in range(1, limit):
            number1 = 4 * x * x + y * y
            if number1 <= n and (number1 % 12 == 1 or number1 % 12 == 5):
                sieve[number1] = not sieve[number1]

            number3 = 3 * x * x + y * y
            if number3 <= n and number3 % 12 == 7:
                sieve[number3] = not sieve[number3]

            number3 = 3 * x * x - y * y
            if x > y and number3 <= n and number3 % 12 == 11:
                sieve[number3] = not sieve[number3]

    # handle known primes - 2 & 3
    if n >= 2:
        sieve[2] = True
    if n >= 3:
        sieve[3] = True

    # drop squares of primes
    for r in range(5, limit):
        if sieve[r]:
            for k in range(r * r, n + 1, r * r):
                sieve[k] = False

    return [i for i in range(n + 1) if sieve[i]]


# one iteration defined for the given d and n.
def miller_rabin_test(d: int, n: int) -> bool:

    # d * 2^r = n - 1
    a = 2 + random.randint(1, n - 4)
    x = pow(a, d, n)

    if x == 1 or x == n - 1:
        return True

    while d != n - 1:
        x = (x * x) % n
        d *= 2
        if x == 1:
            return False
        if x == n - 1:
            return True  # probably prime
    return False


def is_prime_miller_rabin(n: int, k: int = 5) -> bool:

    # handle known cases first
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    # find d such that n-1 = d * 2^r
    d = n - 1
    while d % 2 == 0:
        d //= 2

    # perform k number of tests
    for _ in range(k):
        if not miller_rabin_test(d, n):
            return False
    return True

# orchestrate Miller-Rabin test
def generate_primes_miller_rabin(n: int, k: int = 5) -> list:
    primes = []
    for num in range(2, n + 1):
        if is_prime_miller_rabin(num, k=k):
            primes.append(num)
    return primes





