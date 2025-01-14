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


# simple trial division method - straightforward but slow
def trial_division(n: int) -> list:
    if n < 2:
        return []
    primes = []
    for num in range(2, n+1):
        # check divisibility
        root = int(math.isqrt(num))
        is_prime = True
        for p in range(2, root + 1):
            if num % p == 0:
                is_prime = False
                break
        if is_prime:
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
        n_values = [100_000, 500_000, 1_000_000, 5_000_000]

    methods = {
        "Sieve of Eratosthenes": sieve_of_eratosthenes,
        "Sieve of Sundaram": sieve_of_sundaram,
        "Sieve of Atkin": sieve_of_atkin,
        "Miller-Rabin": generate_primes_miller_rabin,
        "Trial Division": trial_division
    }

    # distinct tables to be used
    time_table = {method: {} for method in methods}
    memory_table = {method: {} for method in methods}

    # print results
    for n in n_values:
        print(f"\n ======== Benchmarking for n = {n} ========")
        for method_name, method_func in methods.items():
            exec_time, peak_mem, _ = measure_time_and_peak_memory(method_func, n)
            print(f"{method_name:25} | Time: {exec_time:.4f} s | "
                  f"Peak Memory: {peak_mem:.2f} MB")

            time_table[method_name][n] = exec_time
            memory_table[method_name][n] = peak_mem

    # display summary tables
    display_summary_tables(time_table, memory_table, n_values)


def display_summary_tables(time_table, memory_table, n_values):
    # time part
    fig_time, ax_time = plt.subplots(figsize=(7, 3))
    ax_time.axis("off")

    header_time = ["Method"] + ["n = " + str(n) for n in n_values]
    table_data_time = [header_time]

    for method in time_table.keys():
        row = [method]
        for n in n_values:
            row.append(f"{time_table[method][n]:.4f}")
        table_data_time.append(row)

    # create the table
    table_time = ax_time.table(
        cellText=table_data_time,
        loc='center',
        cellLoc='center'
    )
    ax_time.set_title("Execution Time Summary", pad=10)
    table_time.set_fontsize(10)
    table_time.scale(1, 1.5)
    fig_time.tight_layout()
    plt.show()

    # memory part
    fig_mem, ax_mem = plt.subplots(figsize=(7, 3))
    ax_mem.axis("off")

    header_mem = ["Method"] + ["n = " + str(n) for n in n_values]
    table_data_mem = [header_mem]

    for method in memory_table.keys():
        row = [method]
        for n in n_values:
            row.append(f"{memory_table[method][n]:.2f}")
        table_data_mem.append(row)

    table_mem = ax_mem.table(
        cellText=table_data_mem,
        loc='center',
        cellLoc='center'
    )
    ax_mem.set_title("Peak Memory Summary (MB)", pad=10)
    table_mem.set_fontsize(10)
    table_mem.scale(1, 1.5)
    fig_mem.tight_layout()
    plt.show()


def plot_benchmarks(n_values=None):
    # draw bar plot
    if n_values is None:
        n_values = [100_000, 500_000, 1_000_000, 5_000_000]

    methods = {
        "Eratosthenes": sieve_of_eratosthenes,
        "Sundaram": sieve_of_sundaram,
        "Atkin": sieve_of_atkin,
        "Miller-Rabin": generate_primes_miller_rabin,
        "TrialDiv": trial_division
    }

    # time and memory to be stored separately
    time_results = {m: [] for m in methods}
    mem_results = {m: [] for m in methods}

    for n in n_values:
        for method_name, method_func in methods.items():
            exec_time, peak_mem, _ = measure_time_and_peak_memory(method_func, n)
            time_results[method_name].append(exec_time)
            mem_results[method_name].append(peak_mem)

    # first figure for Time
    plt.figure(figsize=(8, 6))
    bar_width = 0.12
    x_positions = range(len(n_values))

    for i, (method_name, times) in enumerate(time_results.items()):
        offsets = [x + (i * bar_width) for x in x_positions]
        plt.bar(offsets, times, width=bar_width, label=method_name)

    # x-axis
    mid_positions = [x + (bar_width * (len(methods) - 1) / 2) for x in x_positions]
    plt.xticks(mid_positions, [str(n) for n in n_values])

    plt.xlabel("n")
    plt.ylabel("Time (sec)")
    plt.title("Execution Time Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # second figure for Memory Consumption
    plt.figure(figsize=(8, 6))
    for i, (method_name, mem_vals) in enumerate(mem_results.items()):
        offsets = [x + (i * bar_width) for x in x_positions]
        plt.bar(offsets, mem_vals, width=bar_width, label=method_name)

    plt.xticks(mid_positions, [str(n) for n in n_values])
    plt.xlabel("n")
    plt.ylabel("Peak Memory (MB)")
    plt.title("Memory Usage Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


# start
if __name__ == "__main__":
    N_VALUES = [100_000, 500_000, 1_000_000, 5_000_000]
    compare_algorithms(N_VALUES)
    plot_benchmarks(N_VALUES)
