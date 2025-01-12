import time
import math
import random
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

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


### Benchmarking

def measure_time_and_peak_memory(func, *args, **kwargs):

    ### time counter start
    start_time = time.perf_counter()

    # use wrapper to catch peak memory.
    result_container = {}
    def wrapper():
        result_container['output'] = func(*args, **kwargs)

    mem_usage_list = memory_usage(wrapper, interval=0.01, max_iterations=1_000_000)
    end_time = time.perf_counter()
    exec_time = end_time - start_time

    # baseline to be reduced from peak memory
    peak_mem = max(mem_usage_list) - min(mem_usage_list)
    result = result_container['output']

    return exec_time, peak_mem, result


def compare_algorithms(n_values=None):
    if n_values is None:
        n_values = [100_000, 500_000, 1_000_000]
    methods = {
        "Sieve of Eratosthenes": sieve_of_eratosthenes,
        "Sieve of Sundaram": sieve_of_sundaram,
        "Sieve of Atkin": sieve_of_atkin,
        "Miller-Rabin": generate_primes_miller_rabin
    }

    # print results
    for n in n_values:
        print(f"\n ======== Benchmarking for n = {n} ========")
        for method_name, method_func in methods.items():
            exec_time, peak_mem, _ = measure_time_and_peak_memory(method_func, n)
            print(f"{method_name:25} | Time: {exec_time:.4f} s | "
                  f"Peak Memory: {peak_mem:.2f} MB")


def plot_benchmarks(n_values=None):
    # draw bar plot
    if n_values is None:
        n_values = [100_000, 500_000, 1_000_000]
    methods = {
        "Eratosthenes": sieve_of_eratosthenes,
        "Sundaram": sieve_of_sundaram,
        "Atkin": sieve_of_atkin,
        "Miller-Rabin": generate_primes_miller_rabin
    }

    results = {method: [] for method in methods}

    for n in n_values:
        for method_name, method_func in methods.items():
            exec_time, _, _ = measure_time_and_peak_memory(method_func, n)
            results[method_name].append(exec_time)

    # plot arrangement
    bar_width = 0.2
    x_positions = range(len(n_values))

    plt.figure(figsize=(10, 6))

    for i, (method_name, times) in enumerate(results.items()):
        offsets = [x + (i * bar_width) for x in x_positions]
        plt.bar(offsets, times, width=bar_width, label=method_name)

    # x-axis
    mid_positions = [x + (bar_width * (len(methods) - 1) / 2) for x in x_positions]
    plt.xticks(mid_positions, [str(n) for n in n_values])

    plt.xlabel("n")
    plt.ylabel("Time (sec)")
    plt.title("Comparison of Prime Generation Algorithms")
    plt.legend()
    plt.tight_layout()
    plt.show()


# start
if __name__ == "__main__":
    N_VALUES = [100_000, 500_000, 1_000_000]
    compare_algorithms(N_VALUES)
    plot_benchmarks(N_VALUES)
