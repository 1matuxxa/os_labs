import subprocess
import csv
import os
import matplotlib.pyplot as plt

PHYSICAL_CORES = 8
LOGICAL_CORES = 16

EXE_NAME = "matmul.exe" if os.name == "nt" else "./matmul"

# Сетка потоков по п. 2.3: шаг 1 до числа логических процессоров, далее до 2x, 4x и 8x
THREADS_LIST = list(range(1, LOGICAL_CORES + 1)) + [
    int(LOGICAL_CORES * 1.5),
    LOGICAL_CORES * 2,
    LOGICAL_CORES * 4,
    LOGICAL_CORES * 8,
]

SIZES_GRADE3 = [500, 1000, 1800]
SIZES_GRADE4 = [1000, 2000]
RUNS = 5


def run_trimmed_mean(cmd_args):
    """Выполняет 5 замеров, отбрасывает min и max, усредняет оставшиеся 3."""
    times = []
    for _ in range(RUNS):
        out = subprocess.check_output([EXE_NAME, "-q"] + [str(a) for a in cmd_args], text=True).strip()
        parts = out.split(",")
        times.append(float(parts[0]))
    raw = list(times)
    times_sorted = sorted(times)
    trimmed = times_sorted[1:-1]
    avg = sum(trimmed) / len(trimmed)
    return avg, raw


def benchmark_grade3():
    print("=== Замеры части на «3» (обычное умножение матриц) ===")
    data = {}
    with open("results_grade3.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Size", "Threads", "Run1", "Run2", "Run3", "Run4", "Run5", "AvgTime_s", "Speedup", "Efficiency"])

        for n in SIZES_GRADE3:
            data[n] = {"T": [], "time": [], "S": [], "E": []}
            t1 = None
            for t in THREADS_LIST:
                avg_t, raw = run_trimmed_mean(["mul", n, n, n, t, "itj"])
                if t == 1:
                    t1 = avg_t
                s = t1 / avg_t
                e = s / t
                data[n]["T"].append(t)
                data[n]["time"].append(avg_t)
                data[n]["S"].append(s)
                data[n]["E"].append(e)
                writer.writerow([n, t] + [f"{r:.4f}" for r in raw] + [f"{avg_t:.4f}", f"{s:.4f}", f"{e:.4f}"])
                print(f"  Матрица {n}x{n}, T={t}: ср. время = {avg_t:.3f} с, S = {s:.2f}, E = {e:.2f}")

    # График 1: Время от числа потоков
    plt.figure(figsize=(9, 5))
    for n in SIZES_GRADE3:
        plt.plot(data[n]["T"], data[n]["time"], marker="o", label=f"Матрица {n}×{n}")
    plt.title("Зависимость времени умножения от числа потоков")
    plt.xlabel("Число потоков, T")
    plt.ylabel("Время выполнения, с")
    plt.grid(True)
    plt.legend()
    plt.savefig("graph_grade3_time.png", dpi=150)

    # График 2: Ускорение S(T) от числа потоков
    plt.figure(figsize=(9, 5))
    max_ideal = LOGICAL_CORES * 2
    plt.plot([1, max_ideal], [1, max_ideal], "k--", label="Идеальное ускорение S = T")
    plt.axvline(PHYSICAL_CORES, color="r", linestyle=":", label=f"Физические ядра ({PHYSICAL_CORES})")
    plt.axvline(LOGICAL_CORES, color="b", linestyle=":", label=f"Логические процессоры ({LOGICAL_CORES})")
    for n in SIZES_GRADE3:
        plt.plot(data[n]["T"], data[n]["S"], marker="o", label=f"Матрица {n}×{n}")
    plt.title("Зависимость ускорения S(T) от числа потоков")
    plt.xlabel("Число потоков, T")
    plt.ylabel("Ускорение S(T)")
    plt.grid(True)
    plt.legend()
    plt.savefig("graph_grade3_speedup.png", dpi=150)


def benchmark_grade4():
    print("=== Замеры части на «4» (треугольная матрица static vs dynamic + IPC) ===")
    with open("results_grade4.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Size", "Mode", "Threads", "Run1", "Run2", "Run3", "Run4", "Run5", "AvgTime_s"])

        for n in SIZES_GRADE4:
            t_stat, t_dyn = [], []
            for t in THREADS_LIST:
                avg_s, raw_s = run_trimmed_mean(["tri", n, n, t, "static"])
                avg_d, raw_d = run_trimmed_mean(["tri", n, n, t, "dynamic", 8])
                t_stat.append(avg_s)
                t_dyn.append(avg_d)
                writer.writerow([n, "static", t] + [f"{r:.4f}" for r in raw_s] + [f"{avg_s:.4f}"])
                writer.writerow([n, "dynamic", t] + [f"{r:.4f}" for r in raw_d] + [f"{avg_d:.4f}"])
                print(f"  Треуг. {n}x{n}, T={t}: static = {avg_s:.3f} с | dynamic = {avg_d:.3f} с")

            # График сравнения static и dynamic для каждого размера
            plt.figure(figsize=(9, 5))
            plt.plot(THREADS_LIST, t_stat, marker="s", label="Статическое распределение (static)")
            plt.plot(THREADS_LIST, t_dyn, marker="o", label="Динамическое распределение (dynamic, порция=8)")
            plt.axvline(PHYSICAL_CORES, color="r", linestyle=":", label=f"Физические ядра ({PHYSICAL_CORES})")
            plt.axvline(LOGICAL_CORES, color="b", linestyle=":", label=f"Логические процессоры ({LOGICAL_CORES})")
            plt.title(f"Нижнетреугольная матрица {n}×{n}: Static vs Dynamic")
            plt.xlabel("Число потоков, T")
            plt.ylabel("Время выполнения, с")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"graph_grade4_tri_{n}.png", dpi=150)

    # Диаграмма сравнения времени IPC (генерация + передача по Pipe) и времени умножения
    ipc_times, mul_times = [], []
    for n in SIZES_GRADE3:
        out = subprocess.check_output([EXE_NAME, "-q", "ipc", str(n), str(n), str(n), str(PHYSICAL_CORES)], text=True).strip()
        ipc_t, mul_t, _ = out.split(",")
        ipc_times.append(float(ipc_t))
        mul_times.append(float(mul_t))

    x = range(len(SIZES_GRADE3))
    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.18 for i in x], ipc_times, width=0.35, label="Генерация + передача по Pipe (IPC)")
    plt.bar([i + 0.18 for i in x], mul_times, width=0.35, label=f"Многопоточное умножение ({PHYSICAL_CORES} потоков)")
    plt.xticks(list(x), [f"{n}×{n}" for n in SIZES_GRADE3])
    plt.title("Сравнение времени передачи данных (IPC) и времени умножения")
    plt.xlabel("Размер матриц")
    plt.ylabel("Время, с")
    plt.grid(True, axis="y")
    plt.legend()
    plt.savefig("graph_grade4_ipc.png", dpi=150)


if __name__ == "__main__":
    if not os.path.exists(EXE_NAME):
        print(f"Сначала скомпилируй программу: g++ -O2 -std=c++20 main.cpp -o {EXE_NAME} -lpthread")
    else:
        benchmark_grade3()
        benchmark_grade4()
        print("Все расчеты и графики на оценку «4» успешно созданы!")