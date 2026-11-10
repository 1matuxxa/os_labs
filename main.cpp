#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <string>
#include <cstdint>
#include <cstdio>

#ifdef _WIN32
#include <windows.h> // <-- Для SetConsoleOutputCP и SetConsoleCP
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

// Обычная прямоугольная матрица (хранится в одномерном массиве: i * cols + j)
struct Matrix {
    int rows;
    int cols;
    std::vector<int64_t> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c, int64_t init_val = 0)
        : rows(r), cols(c), data(static_cast<std::size_t>(r) * c, init_val) {}

    int64_t& at(int r, int c) {
        return data[static_cast<std::size_t>(r) * cols + c];
    }

    const int64_t& at(int r, int c) const {
        return data[static_cast<std::size_t>(r) * cols + c];
    }
};

// Нижнетреугольная матрица A (n x n): элементы выше диагонали НЕ хранятся
struct LowerTriangularMatrix {
    int n;
    std::vector<std::vector<int64_t>> rows_data;

    explicit LowerTriangularMatrix(int size) : n(size), rows_data(size) {
        for (int i = 0; i < n; ++i) {
            rows_data[i].resize(static_cast<std::size_t>(i + 1), 0);
        }
    }
};

Matrix generate_random_matrix(int rows, int cols, uint32_t seed = 42) {
    Matrix mat(rows, cols);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int64_t> dist(0, 9);
    for (std::size_t i = 0; i < mat.data.size(); ++i) {
        mat.data[i] = dist(gen);
    }
    return mat;
}

LowerTriangularMatrix generate_random_lower_triangular(int n, uint32_t seed = 42) {
    LowerTriangularMatrix mat(n);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int64_t> dist(0, 9);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            mat.rows_data[i][j] = dist(gen);
        }
    }
    return mat;
}

int64_t compute_checksum(const Matrix& mat) {
    int64_t sum = 0;
    for (int64_t val : mat.data) {
        sum += val;
    }
    return sum;
}

void print_if_small(const Matrix& mat, const std::string& name) {
    if (mat.rows <= 10 && mat.cols <= 10) {
        std::cout << "Матрица " << name << " (" << mat.rows << "x" << mat.cols << "):\n";
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                std::cout << mat.at(i, j) << "\t";
            }
            std::cout << "\n";
        }
    }
}

void run_with_progress(
    int total_work,
    std::atomic<int>& completed_work,
    std::atomic<bool>& is_done,
    bool quiet
) {
    if (quiet || total_work <= 0) {
        return;
    }
    int last_percent = -1;
    while (!is_done.load(std::memory_order_acquire)) {
        int done = completed_work.load(std::memory_order_relaxed);
        int percent = static_cast<int>((static_cast<int64_t>(done) * 100) / total_work);
        if (percent > 100) {
            percent = 100;
        }
        if (percent != last_percent) {
            std::cout << "\rПрогресс: " << percent << "%" << std::flush;
            last_percent = percent;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << "\rПрогресс: 100%\n";
}

Matrix multiply_matrices(
    const Matrix& A,
    const Matrix& B,
    int num_threads,
    std::atomic<int>& completed_rows,
    bool cache_friendly_itj = true
) {
    if (A.cols != B.rows || A.rows <= 0 || A.cols <= 0 || B.cols <= 0) {
        throw std::invalid_argument("Несовместимые или некорректные размеры матриц");
    }
    if (num_threads < 1) {
        throw std::invalid_argument("Число потоков должно быть >= 1");
    }

    int n = A.rows;
    int m = A.cols;
    int k = B.cols;
    Matrix C(n, k, 0);
    completed_rows.store(0, std::memory_order_relaxed);

    auto worker = [&](int start_row, int end_row) {
        if (cache_friendly_itj) {
            for (int i = start_row; i < end_row; ++i) {
                int64_t* c_row = &C.data[static_cast<std::size_t>(i) * k];
                const int64_t* a_row = &A.data[static_cast<std::size_t>(i) * m];
                for (int t = 0; t < m; ++t) {
                    int64_t a_it = a_row[t];
                    const int64_t* b_row = &B.data[static_cast<std::size_t>(t) * k];
                    for (int j = 0; j < k; ++j) {
                        c_row[j] += a_it * b_row[j];
                    }
                }
                completed_rows.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            for (int i = start_row; i < end_row; ++i) {
                for (int j = 0; j < k; ++j) {
                    int64_t sum = 0;
                    for (int t = 0; t < m; ++t) {
                        sum += A.at(i, t) * B.at(t, j);
                    }
                    C.at(i, j) = sum;
                }
                completed_rows.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(num_threads));

    int base_rows = n / num_threads;
    int remainder = n % num_threads;
    int current_row = 0;

    for (int t = 0; t < num_threads; ++t) {
        int rows_for_thread = base_rows + (t < remainder ? 1 : 0);
        int start_row = current_row;
        int end_row = start_row + rows_for_thread;
        current_row = end_row;

        threads.emplace_back(worker, start_row, end_row);
    }

    for (std::size_t t = 0; t < threads.size(); ++t) {
        threads[t].join();
    }

    return C;
}

#ifndef _WIN32
void write_all_fd(int fd, const void* buf, std::size_t bytes) {
    const char* ptr = static_cast<const char*>(buf);
    std::size_t left = bytes;
    while (left > 0) {
        ssize_t written = write(fd, ptr, left);
        if (written <= 0) {
            throw std::runtime_error("Ошибка записи в pipe");
        }
        ptr += written;
        left -= static_cast<std::size_t>(written);
    }
}

void read_all_fd(int fd, void* buf, std::size_t bytes) {
    char* ptr = static_cast<char*>(buf);
    std::size_t left = bytes;
    while (left > 0) {
        ssize_t r = read(fd, ptr, left);
        if (r <= 0) {
            throw std::runtime_error("Ошибка чтения из pipe или преждевременный EOF");
        }
        ptr += r;
        left -= static_cast<std::size_t>(r);
    }
}
#endif

double generate_matrices_in_child_process(
    const char* prog_path,
    int n, int m, int k,
    Matrix& A, Matrix& B,
    uint32_t seedA = 101, uint32_t seedB = 202
) {
#ifndef _WIN32
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        throw std::runtime_error("Не удалось создать канал pipe()");
    }

    auto start_ipc = std::chrono::steady_clock::now();
    pid_t pid = fork();
    if (pid < 0) {
        throw std::runtime_error("Не удалось создать дочерний процесс fork()");
    }

    if (pid == 0) {
        close(pipefd[0]);
        Matrix genA = generate_random_matrix(n, m, seedA);
        Matrix genB = generate_random_matrix(m, k, seedB);

        write_all_fd(pipefd[1], genA.data.data(), genA.data.size() * sizeof(int64_t));
        write_all_fd(pipefd[1], genB.data.data(), genB.data.size() * sizeof(int64_t));

        close(pipefd[1]);
        _exit(0);
    } else {
        close(pipefd[1]);
        A = Matrix(n, m);
        B = Matrix(m, k);

        read_all_fd(pipefd[0], A.data.data(), A.data.size() * sizeof(int64_t));
        read_all_fd(pipefd[0], B.data.data(), B.data.size() * sizeof(int64_t));

        close(pipefd[0]);
        int status = 0;
        waitpid(pid, &status, 0);
        auto end_ipc = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(end_ipc - start_ipc).count();
    }
#else
    auto start_ipc = std::chrono::steady_clock::now();
    std::string cmd = "\"" + std::string(prog_path) + "\" --child-gen " +
                      std::to_string(n) + " " + std::to_string(m) + " " +
                      std::to_string(k) + " " + std::to_string(seedA) + " " +
                      std::to_string(seedB);

    FILE* pipe_fp = _popen(cmd.c_str(), "rb");
    if (!pipe_fp) {
        throw std::runtime_error("Не удалось запустить дочерний процесс через _popen");
    }

    A = Matrix(n, m);
    B = Matrix(m, k);

    std::size_t readA = std::fread(A.data.data(), sizeof(int64_t), A.data.size(), pipe_fp);
    std::size_t readB = std::fread(B.data.data(), sizeof(int64_t), B.data.size(), pipe_fp);
    _pclose(pipe_fp);

    if (readA != A.data.size() || readB != B.data.size()) {
        throw std::runtime_error("Ошибка чтения бинарных матриц из канала дочернего процесса");
    }

    auto end_ipc = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end_ipc - start_ipc).count();
#endif
}

Matrix multiply_lower_triangular(
    const LowerTriangularMatrix& A,
    const Matrix& B,
    int num_threads,
    bool is_dynamic,
    int chunk_size,
    std::atomic<int>& completed_rows
) {
    if (A.n != B.rows || A.n <= 0 || B.cols <= 0) {
        throw std::invalid_argument("Несовместимые размеры матриц");
    }
    if (num_threads < 1 || chunk_size < 1) {
        throw std::invalid_argument("Некорректное число потоков или размер порции");
    }

    int n = A.n;
    int k = B.cols;
    Matrix C(n, k, 0);
    completed_rows.store(0, std::memory_order_relaxed);
    std::atomic<int> next_row{0};

    auto compute_single_row = [&](int i) {
        int64_t* c_row = &C.data[static_cast<std::size_t>(i) * k];
        const std::vector<int64_t>& a_row = A.rows_data[i];
        for (int t = 0; t <= i; ++t) {
            int64_t a_it = a_row[t];
            const int64_t* b_row = &B.data[static_cast<std::size_t>(t) * k];
            for (int j = 0; j < k; ++j) {
                c_row[j] += a_it * b_row[j];
            }
        }
        completed_rows.fetch_add(1, std::memory_order_relaxed);
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(num_threads));

    if (!is_dynamic) {
        int base_rows = n / num_threads;
        int remainder = n % num_threads;
        int current_row = 0;

        for (int t = 0; t < num_threads; ++t) {
            int rows_for_thread = base_rows + (t < remainder ? 1 : 0);
            int start_row = current_row;
            int end_row = start_row + rows_for_thread;
            current_row = end_row;

            threads.emplace_back([&, start_row, end_row]() {
                for (int i = start_row; i < end_row; ++i) {
                    compute_single_row(i);
                }
            });
        }
    } else {
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&]() {
                while (true) {
                    int start_row = next_row.fetch_add(chunk_size, std::memory_order_relaxed);
                    if (start_row >= n) {
                        break;
                    }
                    int end_row = std::min(n, start_row + chunk_size);
                    for (int i = start_row; i < end_row; ++i) {
                        compute_single_row(i);
                    }
                }
            });
        }
    }

    for (std::size_t t = 0; t < threads.size(); ++t) {
        threads[t].join();
    }

    return C;
}

void execute_grade3_mul(int n, int m, int k, int threads, bool itj_order, bool quiet) {
    Matrix A = generate_random_matrix(n, m, 101);
    Matrix B = generate_random_matrix(m, k, 202);

    std::atomic<int> completed_rows{0};
    std::atomic<bool> is_done{false};
    std::thread progress_thread(run_with_progress, n, std::ref(completed_rows), std::ref(is_done), quiet);

    auto start_time = std::chrono::steady_clock::now();
    Matrix C = multiply_matrices(A, B, threads, completed_rows, itj_order);
    auto end_time = std::chrono::steady_clock::now();

    is_done.store(true, std::memory_order_release);
    progress_thread.join();

    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    int64_t checksum = compute_checksum(C);

    if (!quiet) {
        print_if_small(A, "A");
        print_if_small(B, "B");
        print_if_small(C, "C");
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Порядок циклов: " << (itj_order ? "i->t->j (быстрый, кэш)" : "i->j->t (школьный)") << "\n";
        std::cout << "Потоков: " << threads << " | Время умножения: " << elapsed << " с\n";
        std::cout << "Контрольная сумма: " << checksum << "\n";
    } else {
        std::cout << std::fixed << std::setprecision(6) << elapsed << "," << checksum << "\n";
    }
}

void execute_grade4_ipc(const char* prog_path, int n, int m, int k, int threads, bool quiet) {
    Matrix A;
    Matrix B;

    double ipc_time = generate_matrices_in_child_process(prog_path, n, m, k, A, B, 101, 202);

    std::atomic<int> completed_rows{0};
    std::atomic<bool> is_done{false};
    std::thread progress_thread(run_with_progress, n, std::ref(completed_rows), std::ref(is_done), quiet);

    auto start_mul = std::chrono::steady_clock::now();
    Matrix C = multiply_matrices(A, B, threads, completed_rows, true);
    auto end_mul = std::chrono::steady_clock::now();

    is_done.store(true, std::memory_order_release);
    progress_thread.join();

    double mul_time = std::chrono::duration<double>(end_mul - start_mul).count();
    int64_t checksum = compute_checksum(C);

    if (!quiet) {
        print_if_small(A, "A (принята по IPC)");
        print_if_small(B, "B (принята по IPC)");
        print_if_small(C, "C");
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Время генерации в дочернем процессе + IPC (pipe): " << ipc_time << " с\n";
        std::cout << "Время многопоточного умножения: " << mul_time << " с\n";
        std::cout << "Контрольная сумма: " << checksum << "\n";
    } else {
        std::cout << std::fixed << std::setprecision(6) << ipc_time << "," << mul_time << "," << checksum << "\n";
    }
}

void execute_grade4_triangular(int n, int k, int threads, bool is_dynamic, int chunk_size, bool quiet) {
    LowerTriangularMatrix A = generate_random_lower_triangular(n, 101);
    Matrix B = generate_random_matrix(n, k, 202);

    std::atomic<int> completed_rows{0};
    std::atomic<bool> is_done{false};
    std::thread progress_thread(run_with_progress, n, std::ref(completed_rows), std::ref(is_done), quiet);

    auto start_time = std::chrono::steady_clock::now();
    Matrix C = multiply_lower_triangular(A, B, threads, is_dynamic, chunk_size, completed_rows);
    auto end_time = std::chrono::steady_clock::now();

    is_done.store(true, std::memory_order_release);
    progress_thread.join();

    double elapsed = std::chrono::duration<double>(end_time - start_time).count();
    int64_t checksum = compute_checksum(C);

    if (!quiet) {
        print_if_small(C, "C (треугольная A * B)");
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Распределение: " << (is_dynamic ? "dynamic (порция=" + std::to_string(chunk_size) + ")" : "static (поровну)") << "\n";
        std::cout << "Потоков: " << threads << " | Время умножения: " << elapsed << " с\n";
        std::cout << "Контрольная сумма: " << checksum << "\n";
    } else {
        std::cout << std::fixed << std::setprecision(6) << elapsed << "," << checksum << "\n";
    }
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    // Настраиваем вывод и ввод консоли Windows на работу с UTF-8
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    try {
        if (argc >= 7 && std::string(argv[1]) == "--child-gen") {
#ifdef _WIN32
            _setmode(_fileno(stdout), _O_BINARY);
#endif
            int n = std::stoi(argv[2]);
            int m = std::stoi(argv[3]);
            int k = std::stoi(argv[4]);
            uint32_t seedA = static_cast<uint32_t>(std::stoul(argv[5]));
            uint32_t seedB = static_cast<uint32_t>(std::stoul(argv[6]));

            Matrix genA = generate_random_matrix(n, m, seedA);
            Matrix genB = generate_random_matrix(m, k, seedB);
            std::fwrite(genA.data.data(), sizeof(int64_t), genA.data.size(), stdout);
            std::fwrite(genB.data.data(), sizeof(int64_t), genB.data.size(), stdout);
            std::fflush(stdout);
            return 0;
        }

        if (argc == 1) {
            std::cout << "=== Лабораторная работа №2 (Вариант на оценку «4») ===\n";
            std::cout << "Логических процессоров в системе: " << std::thread::hardware_concurrency() << "\n\n";
            std::cout << "Выберите задание:\n";
            std::cout << "  1 — Оценка «3»: Многопоточное умножение A(n x m) * B(m x k)\n";
            std::cout << "  2 — Оценка «4» (п. 3.1): Генерация матриц в отдельном процессе (IPC Pipe)\n";
            std::cout << "  3 — Оценка «4» (п. 3.2-3.3): Нижнетреугольная матрица (static vs dynamic)\n";
            std::cout << "Ваш выбор: ";

            int choice = 0;
            if (!(std::cin >> choice)) {
                return 0;
            }

            if (choice == 1) {
                int n = 0, m = 0, k = 0, t = 1, order_choice = 1;
                std::cout << "Введите n, m, k и число потоков T: ";
                std::cin >> n >> m >> k >> t;
                std::cout << "Порядок циклов (1 - быстрый i->t->j, 0 - школьный i->j->t): ";
                std::cin >> order_choice;
                execute_grade3_mul(n, m, k, t, order_choice != 0, false);
            } else if (choice == 2) {
                int n = 0, m = 0, k = 0, t = 1;
                std::cout << "Введите n, m, k и число потоков T: ";
                std::cin >> n >> m >> k >> t;
                execute_grade4_ipc(argv[0], n, m, k, t, false);
            } else if (choice == 3) {
                int n = 0, k = 0, t = 1, chunk = 8;
                std::string mode_str;
                std::cout << "Введите размер n, столбцы k и число потоков T: ";
                std::cin >> n >> k >> t;
                std::cout << "Способ распределения (static или dynamic): ";
                std::cin >> mode_str;
                bool is_dyn = (mode_str == "dynamic");
                if (is_dyn) {
                    std::cout << "Размер порции строк (например 1, 8, 64): ";
                    std::cin >> chunk;
                }
                execute_grade4_triangular(n, k, t, is_dyn, chunk, false);
            }
            return 0;
        }

        bool quiet = false;
        std::vector<std::string> args;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-q" || arg == "--quiet") {
                quiet = true;
            } else {
                args.push_back(arg);
            }
        }

        if (args.empty()) {
            return 0;
        }

        if (args[0] == "mul") {
            int n = std::stoi(args.at(1));
            int m = std::stoi(args.at(2));
            int k = std::stoi(args.at(3));
            int t = std::stoi(args.at(4));
            bool itj = (args.size() < 6 || args[5] != "ijk");
            execute_grade3_mul(n, m, k, t, itj, quiet);
        } else if (args[0] == "ipc") {
            int n = std::stoi(args.at(1));
            int m = std::stoi(args.at(2));
            int k = std::stoi(args.at(3));
            int t = std::stoi(args.at(4));
            execute_grade4_ipc(argv[0], n, m, k, t, quiet);
        } else if (args[0] == "tri") {
            int n = std::stoi(args.at(1));
            int k = std::stoi(args.at(2));
            int t = std::stoi(args.at(3));
            bool is_dyn = (args.at(4) == "dynamic");
            int chunk = (args.size() >= 6) ? std::stoi(args[5]) : 8;
            execute_grade4_triangular(n, k, t, is_dyn, chunk, quiet);
        } else if (args.size() >= 4 && (args[3] == "static" || args[3] == "dynamic")) {
            int n = std::stoi(args[0]);
            int k = std::stoi(args[1]);
            int t = std::stoi(args[2]);
            bool is_dyn = (args[3] == "dynamic");
            int chunk = (args.size() >= 5) ? std::stoi(args[4]) : 8;
            execute_grade4_triangular(n, k, t, is_dyn, chunk, quiet);
        } else if (args.size() >= 4) {
            int n = std::stoi(args[0]);
            int m = std::stoi(args[1]);
            int k = std::stoi(args[2]);
            int t = std::stoi(args[3]);
            execute_grade3_mul(n, m, k, t, true, quiet);
        } else {
            std::cerr << "Некорректные аргументы запуска.\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}