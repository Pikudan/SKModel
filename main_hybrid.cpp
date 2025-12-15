#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <limits>
#include <sstream>
#include <mpi.h>
#include <omp.h>

struct Point {
    double x, y;
    Point(double x = 0, double y = 0) : x(x), y(y) {}
};

const Point A(-3, 0), B(3, 0), C(2, 3), D(-2, 3);
const double TOL = 1e-8;

bool in_trapezoid(double x, double y) {
    double x_left, x_right;
    
    if (std::fabs(D.y - A.y) > TOL) {
        x_left = D.x + (A.x - D.x) * (y - D.y) / (A.y - D.y);
    } else {
        x_left = D.x;
    }
    
    if (std::fabs(C.y - B.y) > TOL) {
        x_right = B.x + (C.x - B.x) * (y - B.y) / (C.y - B.y);
    } else {
        x_right = B.x;
    }
    
    if (y < std::min(A.y, B.y) || y > std::max(D.y, C.y)) return false;
    if (std::fabs(y - A.y) < TOL) return x >= std::min(A.x, B.x) && x <= std::max(A.x, B.x);
    if (std::fabs(y - D.y) < TOL) return x >= std::min(D.x, C.x) && x <= std::max(D.x, C.x);
    
    return std::min(x_left, x_right) <= x && x <= std::max(x_left, x_right);
}

bool on_border(double x, double y, double tol = TOL) {
    if (std::fabs(y - A.y) < tol && std::min(A.x, B.x) - tol <= x && x <= std::max(A.x, B.x) + tol) return true;
    if (std::fabs(y - C.y) < tol && std::min(C.x, D.x) - tol <= x && x <= std::max(C.x, D.x) + tol) return true;
    
    if (std::fabs(B.y - C.y) > 1e-10) {
        double x_on_BC = B.x + (C.x - B.x) * (y - B.y) / (C.y - B.y);
        if (std::min(B.y, C.y) - tol <= y && y <= std::max(B.y, C.y) + tol && std::fabs(x - x_on_BC) < tol) return true;
    }
    
    if (std::fabs(D.y - A.y) > 1e-10) {
        double x_on_DA = D.x + (A.x - D.x) * (y - D.y) / (A.y - D.y);
        if (std::min(A.y, D.y) - tol <= y && y <= std::max(A.y, D.y) + tol && std::fabs(x - x_on_DA) < tol) return true;
    }
    
    return false;
}

double f(double x, double y) { return 1.0; }
double g(double x, double y) { return 0.0; }

class Grid2D {
public:
    std::vector<std::vector<double> > data;
    int rows, cols;
    
    Grid2D(int r = 0, int c = 0, double init_val = 0.0) : rows(r), cols(c) {
        if (r > 0 && c > 0) {
            data.resize(rows);
            for (int i = 0; i < rows; ++i) {
                data[i].resize(cols, init_val);
            }
        }
    }
};

class BoolGrid2D {
public:
    std::vector<std::vector<bool> > data;
    int rows, cols;
    
    BoolGrid2D(int r = 0, int c = 0, bool init_val = false) : rows(r), cols(c) {
        if (r > 0 && c > 0) {
            data.resize(rows);
            for (int i = 0; i < rows; ++i) {
                data[i].resize(cols, init_val);
            }
        }
    }
};

std::string int_to_string(int value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

// Структура для хранения информации о двумерной топологии процесса
struct ProcessTopology2D {
    int px, py;              // Размеры сетки процессов (px × py = size)
    int rank_x, rank_y;      // Координаты процесса в сетке
    int local_M, local_N;    // Локальные размеры домена (без ghost cells)
    int start_M, start_N;    // Начальные индексы в глобальной сетке
    int local_M_with_ghost, local_N_with_ghost;  // С ghost cells
};

// Структура для предвыделенных буферов обмена границами
struct ExchangeBuffers {
    std::vector<double> send_buf_left, recv_buf_left;
    std::vector<double> send_buf_right, recv_buf_right;
    int buffer_size;
    
    ExchangeBuffers(int size) : buffer_size(size) {
        send_buf_left.resize(size);
        recv_buf_left.resize(size);
        send_buf_right.resize(size);
        recv_buf_right.resize(size);
    }
};

// поиск оптимального двумерного разбиения
std::pair<int, int> find_optimal_2d_decomposition(int M, int N, int size) {
    int best_px = 1, best_py = size;
    double best_score = std::numeric_limits<double>::max();
    
    // Перебираем все возможные разбиения px × py = size
    for (int px = 1; px <= size; ++px) {
        if (size % px != 0) continue;  // px должно делить size
        
        int py = size / px;
        
        // Вычисляем размеры доменов
        int M_per_proc_base = (M + 1) / px;
        int M_remainder = (M + 1) % px;
        int N_per_proc_base = (N + 1) / py;
        int N_remainder = (N + 1) % py;
        
        int M_max = M_per_proc_base + (M_remainder > 0 ? 1 : 0);
        int M_min = M_per_proc_base;
        int N_max = N_per_proc_base + (N_remainder > 0 ? 1 : 0);
        int N_min = N_per_proc_base;
        

        if (M_max - M_min > 1 || N_max - N_min > 1) continue;
        
        bool valid_ratio = true;
        double max_ratio_diff = 0.0;
        
        for (int py_idx = 0; py_idx < py; ++py_idx) {
            for (int px_idx = 0; px_idx < px; ++px_idx) {
                int local_M_size = M_per_proc_base + (px_idx < M_remainder ? 1 : 0);
                int local_N_size = N_per_proc_base + (py_idx < N_remainder ? 1 : 0);
                
                if (local_M_size == 0 || local_N_size == 0) {
                    valid_ratio = false;
                    break;
                }
                
                double ratio = (double)local_M_size / local_N_size;
                if (ratio < 0.5 || ratio > 2.0) {
                    valid_ratio = false;
                    break;
                }
                
                // Вычисляем отклонение от идеального отношения 1:1
                double ratio_diff = std::fabs(ratio - 1.0);
                if (ratio_diff > max_ratio_diff) {
                    max_ratio_diff = ratio_diff;
                }
            }
            if (!valid_ratio) break;
        }
        
        if (!valid_ratio) continue;
        
        if (max_ratio_diff < best_score) {
            best_score = max_ratio_diff;
            best_px = px;
            best_py = py;
        }
    }
    
    return std::make_pair(best_px, best_py);
}

// Инициализация топологии процесса
ProcessTopology2D init_topology_2d(int M, int N, int rank, int size) {
    ProcessTopology2D topo;
    
    std::pair<int, int> decomp = find_optimal_2d_decomposition(M, N, size);
    topo.px = decomp.first;
    topo.py = decomp.second;
    
    // Вычисляем координаты процесса в сетке
    topo.rank_x = rank % topo.px;
    topo.rank_y = rank / topo.px;
    
    // Распределение по оси X
    int cols_per_proc = (M + 1) / topo.px;
    int cols_remainder = (M + 1) % topo.px;
    topo.local_M = cols_per_proc + (topo.rank_x < cols_remainder ? 1 : 0);
    topo.start_M = 0;
    for (int i = 0; i < topo.rank_x; ++i) {
        topo.start_M += cols_per_proc + (i < cols_remainder ? 1 : 0);
    }
    
    // Распределение по оси Y
    int rows_per_proc = (N + 1) / topo.py;
    int rows_remainder = (N + 1) % topo.py;
    topo.local_N = rows_per_proc + (topo.rank_y < rows_remainder ? 1 : 0);
    topo.start_N = 0;
    for (int i = 0; i < topo.rank_y; ++i) {
        topo.start_N += rows_per_proc + (i < rows_remainder ? 1 : 0);
    }
    
    // Добавляем ghost cells для обмена граничными данными
    topo.local_M_with_ghost = topo.local_M;
    if (topo.rank_x > 0) topo.local_M_with_ghost++;
    if (topo.rank_x < topo.px - 1) topo.local_M_with_ghost++;
    
    topo.local_N_with_ghost = topo.local_N;
    if (topo.rank_y > 0) topo.local_N_with_ghost++;
    if (topo.rank_y < topo.py - 1) topo.local_N_with_ghost++;
    
    return topo;
}

// Вычисление коэффициента a_ij
double compute_a_ij(double x_half, double y_j_minus_half, double y_j_plus_half, 
                   double h2, double eps) {
    // Проверяем положение отрезка относительно трапеции
    const int num_check_points = 5;
    bool all_inside = true;
    bool all_outside = true;
    
    for (int i = 0; i <= num_check_points; i++) {
        double y = y_j_minus_half + (y_j_plus_half - y_j_minus_half) * i / num_check_points;
        if (in_trapezoid(x_half, y)) {
            all_outside = false;
        } else {
            all_inside = false;
        }
    }
    
    if (all_inside) return 1.0;
    if (all_outside) return 1.0 / eps;
  
    // Отрезок пересекает границу - численное интегрирование
    const int num_segments = 100;
    double dy = (y_j_plus_half - y_j_minus_half) / num_segments;
    double sum = 0.0;
    
    for (int i = 0; i < num_segments; i++) {
        double y = y_j_minus_half + (i + 0.5) * dy;
        double k_val = in_trapezoid(x_half, y) ? 1.0 : 1.0/eps;
        sum += k_val;
    }
    
    return sum / num_segments;
}

// Вычисление коэффициента b_ij
double compute_b_ij(double y_half, double x_i_minus_half, double x_i_plus_half,
                   double h1, double eps) {
    // Проверяем положение отрезка относительно трапеции
    const int num_check_points = 5;
    bool all_inside = true;
    bool all_outside = true;
    
    for (int i = 0; i <= num_check_points; i++) {
        double x = x_i_minus_half + (x_i_plus_half - x_i_minus_half) * i / num_check_points;
        if (in_trapezoid(x, y_half)) {
            all_outside = false;
        } else {
            all_inside = false;
        }
    }
    
    if (all_inside) return 1.0;
    if (all_outside) return 1.0 / eps;
    
    // Отрезок пересекает границу - численное интегрирование
    const int num_segments = 100;
    double dx = (x_i_plus_half - x_i_minus_half) / num_segments;
    double sum = 0.0;
    
    for (int i = 0; i < num_segments; i++) {
        double x = x_i_minus_half + (i + 0.5) * dx;
        double k_val = in_trapezoid(x, y_half) ? 1.0 : 1.0/eps;
        sum += k_val;
    }
    
    return sum / num_segments;
}

// Вычисление площади пересечения ячейки с трапецией для F_ij
double compute_F_ij(double x_i, double y_j, double h1, double h2) {
    double x_start = x_i - h1/2;
    double x_end = x_i + h1/2;
    double y_start = y_j - h2/2;
    double y_end = y_j + h2/2;
    
    // Ограничиваем по y границами трапеции
    double y_low = std::max(0.0, y_start);
    double y_high = std::min(3.0, y_end);
    
    if (y_low >= y_high) return 0.0;
    
    // Численное интегрирование площади пересечения
    const int num_segments = 100;
    double dy = (y_high - y_low) / num_segments;
    double area = 0.0;
    
    for (int i = 0; i < num_segments; i++) {
        double y = y_low + (i + 0.5) * dy;
        
        // Границы трапеции на этой высоте y
        double left_bound = -3.0 + y/3.0;
        double right_bound = 3.0 - y/3.0;
        
        // Пересечение ячейки с горизонтальной линией на высоте y
        double cell_left = std::max(x_start, left_bound);
        double cell_right = std::min(x_end, right_bound);
        
        if (cell_left < cell_right) {
            area += (cell_right - cell_left) * dy;
        }
    }
    
    return area / (h1 * h2);
}

// Обмен граничными данными между процессами (двумерное разбиение)
// Использует предвыделенные буферы для оптимизации
void exchange_boundaries_2d(Grid2D& grid, const ProcessTopology2D& topo, 
                            ExchangeBuffers& buffers, MPI_Comm comm) {
    MPI_Status status;
    
    // Вычисляем ранги соседей
    int rank_up = -1, rank_down = -1, rank_left = -1, rank_right = -1;
    
    if (topo.rank_y > 0) {
        rank_up = (topo.rank_y - 1) * topo.px + topo.rank_x;
    }
    if (topo.rank_y < topo.py - 1) {
        rank_down = (topo.rank_y + 1) * topo.px + topo.rank_x;
    }
    if (topo.rank_x > 0) {
        rank_left = topo.rank_y * topo.px + (topo.rank_x - 1);
    }
    if (topo.rank_x < topo.px - 1) {
        rank_right = topo.rank_y * topo.px + (topo.rank_x + 1);
    }
    
    // Обмен с верхним соседом (по оси Y)
    if (rank_up >= 0) {
        int send_row = (topo.rank_y > 0) ? 1 : 0;
        int recv_row = 0;
        MPI_Sendrecv(grid.data[send_row].data(), grid.cols, MPI_DOUBLE, rank_up, 0,
                     grid.data[recv_row].data(), grid.cols, MPI_DOUBLE, rank_up, 1,
                     comm, &status);
    }
    
    // Обмен с нижним соседом (по оси Y)
    if (rank_down >= 0) {
        int send_row = grid.rows - ((topo.rank_y < topo.py - 1) ? 2 : 1);
        int recv_row = grid.rows - 1;
        MPI_Sendrecv(grid.data[send_row].data(), grid.cols, MPI_DOUBLE, rank_down, 1,
                     grid.data[recv_row].data(), grid.cols, MPI_DOUBLE, rank_down, 0,
                     comm, &status);
    }
    
    // Обмен с левым соседом (по оси X) - используем предвыделенные буферы
    if (rank_left >= 0) {
        int send_col = (topo.rank_x > 0) ? 1 : 0;
        int recv_col = 0;
        // Копируем данные в предвыделенный буфер
        #pragma omp parallel for
        for (int i = 0; i < grid.rows; ++i) {
            buffers.send_buf_left[i] = grid.data[i][send_col];
        }
        MPI_Sendrecv(buffers.send_buf_left.data(), grid.rows, MPI_DOUBLE, rank_left, 2,
                     buffers.recv_buf_left.data(), grid.rows, MPI_DOUBLE, rank_left, 3,
                     comm, &status);
        // Копируем данные из буфера обратно в сетку
        #pragma omp parallel for
        for (int i = 0; i < grid.rows; ++i) {
            grid.data[i][recv_col] = buffers.recv_buf_left[i];
        }
    }
    
    // Обмен с правым соседом (по оси X) - используем предвыделенные буферы
    if (rank_right >= 0) {
        int send_col = grid.cols - ((topo.rank_x < topo.px - 1) ? 2 : 1);
        int recv_col = grid.cols - 1;
        // Копируем данные в предвыделенный буфер
        #pragma omp parallel for
        for (int i = 0; i < grid.rows; ++i) {
            buffers.send_buf_right[i] = grid.data[i][send_col];
        }
        MPI_Sendrecv(buffers.send_buf_right.data(), grid.rows, MPI_DOUBLE, rank_right, 3,
                     buffers.recv_buf_right.data(), grid.rows, MPI_DOUBLE, rank_right, 2,
                     comm, &status);
        // Копируем данные из буфера обратно в сетку
        #pragma omp parallel for
        for (int i = 0; i < grid.rows; ++i) {
            grid.data[i][recv_col] = buffers.recv_buf_right[i];
        }
    }
}

void setup_fictitious_domain_2d(const Grid2D& X, const Grid2D& Y, const BoolGrid2D& mask,
                                Grid2D& k_coeffs, Grid2D& F_rhs, Grid2D& a_coeffs, 
                                Grid2D& b_coeffs, double hx, double hy,
                                const ProcessTopology2D& topo, int num_threads) {
    double h = std::max(hx, hy);
    double eps = h * h;
    
    int local_M = X.cols;
    int local_N = X.rows;
    
    omp_set_num_threads(num_threads);
    
    // k_coeffs
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < local_N; ++i) {
        for (int j = 0; j < local_M; ++j) {
            k_coeffs.data[i][j] = mask.data[i][j] ? 1.0 : 1.0/eps;
        }
    }
    
    // a_coeffs (на границах между ячейками по X)
    #pragma omp parallel for collapse(2)
    for (int j = 1; j < local_M; ++j) {
        for (int i = 0; i < local_N; ++i) {
            double x_half = X.data[i][j] - hx/2;
            double y_j_minus_half = Y.data[i][j] - hy/2;
            double y_j_plus_half = Y.data[i][j] + hy/2;
            
            a_coeffs.data[i][j] = compute_a_ij(x_half, y_j_minus_half, y_j_plus_half, hy, eps);
        }
    }
    
    // b_coeffs (на границах между ячейками по Y)
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < local_N; ++i) {
        for (int j = 0; j < local_M; ++j) {
            double y_half = Y.data[i][j] - hy/2;
            double x_i_minus_half = X.data[i][j] - hx/2;
            double x_i_plus_half = X.data[i][j] + hx/2;
            
            b_coeffs.data[i][j] = compute_b_ij(y_half, x_i_minus_half, x_i_plus_half, hx, eps);
        }
    }
    
    // F_rhs
    int start_i = (topo.rank_y > 0) ? 1 : 0;
    int end_i = (topo.rank_y < topo.py - 1) ? local_N - 1 : local_N;
    int start_j = (topo.rank_x > 0) ? 1 : 0;
    int end_j = (topo.rank_x < topo.px - 1) ? local_M - 1 : local_M;
    
    #pragma omp parallel for collapse(2)
    for (int i = start_i; i < end_i; ++i) {
        for (int j = start_j; j < end_j; ++j) {
            F_rhs.data[i][j] = compute_F_ij(X.data[i][j], Y.data[i][j], hx, hy);
        }
    }
}

Grid2D apply_A_2d(const Grid2D& w, const BoolGrid2D& mask, const BoolGrid2D& border_mask,
                  const Grid2D& a_coeffs, const Grid2D& b_coeffs, double hx, double hy,
                  const ProcessTopology2D& topo, int num_threads) {
    Grid2D Aw(w.rows, w.cols, 0.0);
    
    int start_i = (topo.rank_y > 0) ? 1 : 0;
    int end_i = (topo.rank_y < topo.py - 1) ? w.rows - 1 : w.rows;
    int start_j = (topo.rank_x > 0) ? 1 : 0;
    int end_j = (topo.rank_x < topo.px - 1) ? w.cols - 1 : w.cols;
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for collapse(2)
    for (int i = start_i; i < end_i; ++i) {
        for (int j = start_j; j < end_j; ++j) {
            if (mask.data[i][j] && !border_mask.data[i][j]) {
                double flux_x = 0.0, flux_y = 0.0;
                if (j > 0 && j < w.cols - 1 && i+1 < w.rows) {
                    flux_x = (a_coeffs.data[i+1][j] * (w.data[i+1][j] - w.data[i][j]) - 
                             a_coeffs.data[i][j] * (w.data[i][j] - w.data[i-1][j])) / (hx * hx);
                }
                if (i > 0 && i < w.rows - 1 && j+1 < w.cols) {
                    flux_y = (b_coeffs.data[i][j+1] * (w.data[i][j+1] - w.data[i][j]) - 
                             b_coeffs.data[i][j] * (w.data[i][j] - w.data[i][j-1])) / (hy * hy);
                }
                
                Aw.data[i][j] = -flux_x - flux_y;
            }
        }
    }
    
    return Aw;
}

Grid2D solve_D_2d(const Grid2D& prec, const BoolGrid2D& mask, const BoolGrid2D& border_mask,
                  const Grid2D& a_coeffs, const Grid2D& b_coeffs, double hx, double hy,
                  const ProcessTopology2D& topo, int num_threads) {
    Grid2D z(prec.rows, prec.cols, 0.0);
    
    int start_i = (topo.rank_y > 0) ? 1 : 0;
    int end_i = (topo.rank_y < topo.py - 1) ? prec.rows - 1 : prec.rows;
    int start_j = (topo.rank_x > 0) ? 1 : 0;
    int end_j = (topo.rank_x < topo.px - 1) ? prec.cols - 1 : prec.cols;
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for collapse(2)
    for (int i = start_i; i < end_i; ++i) {
        for (int j = start_j; j < end_j; ++j) {
            if (mask.data[i][j] && !border_mask.data[i][j]) {
                double D_diag = 0.0;
                if (i+1 < prec.rows) {
                    D_diag += (a_coeffs.data[i][j] + a_coeffs.data[i+1][j]) / (hx * hx);
                }
                if (j+1 < prec.cols) {
                    D_diag += (b_coeffs.data[i][j] + b_coeffs.data[i][j+1]) / (hy * hy);
                }
                
                if (D_diag > 1e-12) {
                    z.data[i][j] = prec.data[i][j] / D_diag;
                }
            }
        }
    }
    
    return z;
}

double dot_product_2d(const Grid2D& u, const Grid2D& v, const BoolGrid2D& mask, 
                      const ProcessTopology2D& topo, int num_threads) {
    double local_result = 0.0;
    int start_i = (topo.rank_y > 0) ? 1 : 0;
    int end_i = (topo.rank_y < topo.py - 1) ? u.rows - 1 : u.rows;
    int start_j = (topo.rank_x > 0) ? 1 : 0;
    int end_j = (topo.rank_x < topo.px - 1) ? u.cols - 1 : u.cols;
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for collapse(2) reduction(+:local_result)
    for (int i = start_i; i < end_i; ++i) {
        for (int j = start_j; j < end_j; ++j) {
            if (mask.data[i][j]) {
                local_result += u.data[i][j] * v.data[i][j];
            }
        }
    }
    
    double global_result = 0.0;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    return global_result;
}

double norm_2d(const Grid2D& u, const BoolGrid2D& mask, const ProcessTopology2D& topo, int num_threads) {
    return std::sqrt(dot_product_2d(u, u, mask, topo, num_threads));
}

std::pair<Grid2D, int> conjugate_gradient_2d(const Grid2D& U0, const Grid2D& F, const BoolGrid2D& mask,
                         const BoolGrid2D& border_mask, const Grid2D& a_coeffs,
                         const Grid2D& b_coeffs, double hx, double hy,
                         int max_iter, double tol, const ProcessTopology2D& topo, int num_threads) {
    
    omp_set_num_threads(num_threads);
    
    Grid2D w = U0;
    
    BoolGrid2D inner_mask(mask.rows, mask.cols);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            inner_mask.data[i][j] = mask.data[i][j] && !border_mask.data[i][j];
        }
    }
    
    int local_num_unknowns = 0;
    int start_i = (topo.rank_y > 0) ? 1 : 0;
    int end_i = (topo.rank_y < topo.py - 1) ? inner_mask.rows - 1 : inner_mask.rows;
    int start_j = (topo.rank_x > 0) ? 1 : 0;
    int end_j = (topo.rank_x < topo.px - 1) ? inner_mask.cols - 1 : inner_mask.cols;
    
    #pragma omp parallel for collapse(2) reduction(+:local_num_unknowns)
    for (int i = start_i; i < end_i; ++i) {
        for (int j = start_j; j < end_j; ++j) {
            if (inner_mask.data[i][j]) local_num_unknowns++;
        }
    }
    
    int global_num_unknowns = 0;
    MPI_Allreduce(&local_num_unknowns, &global_num_unknowns, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // Создаем предвыделенные буферы для обмена границами
    ExchangeBuffers exchange_buffers(w.rows);
    
    // Обмен границами перед вычислением Aw0
    exchange_boundaries_2d(w, topo, exchange_buffers, MPI_COMM_WORLD);
    
    Grid2D Aw0 = apply_A_2d(w, mask, border_mask, a_coeffs, b_coeffs, hx, hy, topo, num_threads);
    Grid2D r(F.rows, F.cols, 0.0);
    
    #pragma omp parallel for collapse(2)
    for (int i = start_i; i < end_i; ++i) {
        for (int j = start_j; j < end_j; ++j) {
            if (inner_mask.data[i][j]) {
                r.data[i][j] = F.data[i][j] - Aw0.data[i][j];
            }
        }
    }
    
    Grid2D z = solve_D_2d(r, mask, border_mask, a_coeffs, b_coeffs, hx, hy, topo, num_threads);
    Grid2D p = z;
    
    double rz_old = dot_product_2d(r, z, inner_mask, topo, num_threads);
    
    int iterations = 0;
    int rank = topo.rank_y * topo.px + topo.rank_x;
    
    for (int k = 0; k < max_iter && k < global_num_unknowns; ++k) {
        iterations++;
        
        // Вычисляем невязку
        double current_residual = norm_2d(r, inner_mask, topo, num_threads);
        
        if (current_residual < tol) {
            if (rank == 0) {
                std::cout << "Сходимость на итерации " << k 
                          << ", невязка = " << std::scientific << std::setprecision(10) << current_residual << std::endl;
            }
            break;
        }
        
        if (rank == 0 && k % 100 == 0) {
            std::cout << "Итерация " << k << ": невязка = " << std::scientific << std::setprecision(10) << current_residual 
                      << ", rz_old = " << std::scientific << std::setprecision(10) << rz_old << std::endl;
        }
        
        // Обмен границами для p перед вычислением Ap
        exchange_boundaries_2d(p, topo, exchange_buffers, MPI_COMM_WORLD);
        
        Grid2D Ap = apply_A_2d(p, mask, border_mask, a_coeffs, b_coeffs, hx, hy, topo, num_threads);
        double pAp = dot_product_2d(p, Ap, inner_mask, topo, num_threads);
        
        if (std::fabs(pAp) < 1e-15) {
            if (current_residual < tol) {
                if (rank == 0) {
                    std::cout << "Сходимость на итерации " << k 
                              << ", невязка = " << std::scientific << std::setprecision(10) << current_residual << std::endl;
                }
                break;
            }
            if (rank == 0) {
                std::cout << "Матрица вырождена, итерация " << k 
                          << ", pAp = " << std::scientific << std::setprecision(10) << pAp 
                          << ", невязка = " << std::scientific << std::setprecision(10) << current_residual << std::endl;
            }
            break;
        }
        
        double alpha = rz_old / pAp;
        
        #pragma omp parallel for collapse(2)
        for (int i = start_i; i < end_i; ++i) {
            for (int j = start_j; j < end_j; ++j) {
                if (inner_mask.data[i][j]) {
                    w.data[i][j] += alpha * p.data[i][j];
                    r.data[i][j] -= alpha * Ap.data[i][j];
                }
            }
        }
        
        exchange_boundaries_2d(w, topo, exchange_buffers, MPI_COMM_WORLD);
        z = solve_D_2d(r, mask, border_mask, a_coeffs, b_coeffs, hx, hy, topo, num_threads);
        double rz_new = dot_product_2d(r, z, inner_mask, topo, num_threads);
        
        if (std::fabs(rz_old) < 1e-15) {
            if (rank == 0) {
                std::cout << "Деление на ноль, итерация " << k 
                          << ", rz_old = " << std::scientific << std::setprecision(10) << rz_old << std::endl;
            }
            break;
        }
        
        double beta = rz_new / rz_old;
        
        #pragma omp parallel for collapse(2)
        for (int i = start_i; i < end_i; ++i) {
            for (int j = start_j; j < end_j; ++j) {
                if (inner_mask.data[i][j]) {
                    p.data[i][j] = z.data[i][j] + beta * p.data[i][j];
                }
            }
        }
        
        rz_old = rz_new;
    }
    
    return std::make_pair(w, iterations);
}

void gather_solution_2d(const Grid2D& local_U, Grid2D& global_U, 
                       const ProcessTopology2D& topo, int global_rows, int global_cols) {
    int start_i = (topo.rank_y > 0) ? 1 : 0;
    int end_i = (topo.rank_y < topo.py - 1) ? local_U.rows - 1 : local_U.rows;
    int start_j = (topo.rank_x > 0) ? 1 : 0;
    int end_j = (topo.rank_x < topo.px - 1) ? local_U.cols - 1 : local_U.cols;
    
    int local_data_size = (end_i - start_i) * (end_j - start_j);
    std::vector<double> local_data;
    local_data.reserve(local_data_size);
    
    for (int i = start_i; i < end_i; ++i) {
        for (int j = start_j; j < end_j; ++j) {
            local_data.push_back(local_U.data[i][j]);
        }
    }
    
    // Создаем структуру данных для сбора
    std::vector<int> recvcounts(topo.px * topo.py);
    std::vector<int> displs(topo.px * topo.py);
    
    // Вычисляем размеры данных для каждого процесса
    int cols_per_proc = global_cols / topo.px;
    int cols_remainder = global_cols % topo.px;
    int rows_per_proc = global_rows / topo.py;
    int rows_remainder = global_rows % topo.py;
    
    for (int py = 0; py < topo.py; ++py) {
        for (int px = 0; px < topo.px; ++px) {
            int proc_rank = py * topo.px + px;
            int proc_cols = cols_per_proc + (px < cols_remainder ? 1 : 0);
            int proc_rows = rows_per_proc + (py < rows_remainder ? 1 : 0);
            recvcounts[proc_rank] = proc_rows * proc_cols;
            
            if (proc_rank == 0) {
                displs[proc_rank] = 0;
            } else {
                displs[proc_rank] = displs[proc_rank - 1] + recvcounts[proc_rank - 1];
            }
        }
    }
    
    std::vector<double> global_data;
    int rank = topo.rank_y * topo.px + topo.rank_x;
    if (rank == 0) {
        global_data.resize(global_rows * global_cols);
    } else {
        global_data.resize(0);
    }
    
    MPI_Gatherv(local_data.data(), local_data.size(), MPI_DOUBLE,
                global_data.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        int idx = 0;
        for (int py = 0; py < topo.py; ++py) {
            int proc_rows = rows_per_proc + (py < rows_remainder ? 1 : 0);
            int row_start = 0;
            for (int i = 0; i < py; ++i) {
                row_start += rows_per_proc + (i < rows_remainder ? 1 : 0);
            }
            
            for (int px = 0; px < topo.px; ++px) {
                int proc_cols = cols_per_proc + (px < cols_remainder ? 1 : 0);
                int col_start = 0;
                for (int j = 0; j < px; ++j) {
                    col_start += cols_per_proc + (j < cols_remainder ? 1 : 0);
                }
                
                int proc_rank = py * topo.px + px;
                int data_idx = displs[proc_rank];
                
                for (int i = 0; i < proc_rows; ++i) {
                    for (int j = 0; j < proc_cols; ++j) {
                        if (row_start + i < global_rows && col_start + j < global_cols) {
                            global_U.data[row_start + i][col_start + j] = global_data[data_idx++];
                        }
                    }
                }
            }
        }
    }
}

void save_solution_csv(const Grid2D& X, const Grid2D& Y, const Grid2D& U, 
                      const BoolGrid2D& mask, const std::string& filename) {
    std::ofstream file(filename.c_str());
    file << "x,y,u,in_domain\n";
    
    for (int i = 0; i < X.rows; ++i) {
        for (int j = 0; j < X.cols; ++j) {
            file << std::fixed << std::setprecision(8) << X.data[i][j] << ","
                 << std::fixed << std::setprecision(8) << Y.data[i][j] << ","
                 << std::scientific << std::setprecision(10) << U.data[i][j] << ","
                 << (mask.data[i][j] ? "1" : "0") << "\n";
        }
    }
    file.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 4) {
        if (rank == 0) {
            std::cout << "Использование: mpirun -np <num_procs> " << argv[0] << " <M> <N> <num_threads>" << std::endl;
            std::cout << "Пример: mpirun -np 2 " << argv[0] << " 40 40 4" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int num_threads = std::atoi(argv[3]);
    
    if (M <= 0 || N <= 0 || num_threads <= 0) {
        if (rank == 0) {
            std::cout << "Ошибка: все параметры должны быть положительными числами" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Инициализация двумерной топологии
    ProcessTopology2D topo = init_topology_2d(M, N, rank, size);
    
    if (rank == 0) {
        std::cout << "Двумерное разбиение: " << topo.px << " x " << topo.py << " процессов" << std::endl;
        std::cout << "Число потоков OpenMP на процесс: " << num_threads << std::endl;
        std::cout << "Локальные размеры домена процесса 0: " << topo.local_M << " x " << topo.local_N << std::endl;
        
        std::cout << "\nИнформация о разбиении доменов:" << std::endl;
        for (int py = 0; py < topo.py; ++py) {
            for (int px = 0; px < topo.px; ++px) {
                int proc_rank = py * topo.px + px;
                int cols_per_proc = (M + 1) / topo.px;
                int cols_remainder = (M + 1) % topo.px;
                int rows_per_proc = (N + 1) / topo.py;
                int rows_remainder = (N + 1) % topo.py;
                
                int local_M_size = cols_per_proc + (px < cols_remainder ? 1 : 0);
                int local_N_size = rows_per_proc + (py < rows_remainder ? 1 : 0);
                double ratio = (double)local_M_size / local_N_size;
                
                std::cout << "Процесс " << proc_rank << " (" << px << "," << py << "): "
                          << local_M_size << " x " << local_N_size 
                          << " узлов, отношение = " << std::fixed << std::setprecision(3) << ratio << std::endl;
            }
        }
        std::cout << std::endl;
    }
    

    std::vector<double> x_vals(M + 1), y_vals(N + 1);
    for (int i = 0; i <= M; ++i) x_vals[i] = -3.0 + 6.0 * i / M;
    for (int i = 0; i <= N; ++i) y_vals[i] = 0.0 + 3.0 * i / N;
    
    Grid2D X(topo.local_N_with_ghost, topo.local_M_with_ghost);
    Grid2D Y(topo.local_N_with_ghost, topo.local_M_with_ghost);
    BoolGrid2D mask(topo.local_N_with_ghost, topo.local_M_with_ghost);
    BoolGrid2D border_mask(topo.local_N_with_ghost, topo.local_M_with_ghost);

    int local_i_offset = (topo.rank_y > 0) ? -1 : 0;
    int local_j_offset = (topo.rank_x > 0) ? -1 : 0;
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for
    for (int i = 0; i < topo.local_N_with_ghost; ++i) {
        int global_i = topo.start_N + local_i_offset + i;
        if (global_i >= 0 && global_i <= N) {
            for (int j = 0; j < topo.local_M_with_ghost; ++j) {
                int global_j = topo.start_M + local_j_offset + j;
                if (global_j >= 0 && global_j <= M) {
                    X.data[i][j] = x_vals[global_j];
                    Y.data[i][j] = y_vals[global_i];
                    mask.data[i][j] = in_trapezoid(X.data[i][j], Y.data[i][j]);
                    border_mask.data[i][j] = on_border(X.data[i][j], Y.data[i][j]);
                }
            }
        }
    }
    
    double hx = x_vals[1] - x_vals[0];
    double hy = y_vals[1] - y_vals[0];
    
    Grid2D k_coeffs(topo.local_N_with_ghost, topo.local_M_with_ghost);
    Grid2D F_rhs(topo.local_N_with_ghost, topo.local_M_with_ghost);
    Grid2D a_coeffs(topo.local_N_with_ghost, topo.local_M_with_ghost);
    Grid2D b_coeffs(topo.local_N_with_ghost, topo.local_M_with_ghost);
    
    setup_fictitious_domain_2d(X, Y, mask, k_coeffs, F_rhs, a_coeffs, b_coeffs, hx, hy, topo, num_threads);
    
    Grid2D U(topo.local_N_with_ghost, topo.local_M_with_ghost);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < topo.local_N_with_ghost; ++i) {
        for (int j = 0; j < topo.local_M_with_ghost; ++j) {
            if (border_mask.data[i][j]) {
                U.data[i][j] = g(X.data[i][j], Y.data[i][j]);
            }
        }
    }
    
    // Синхронизируем все процессы перед началом измерения времени
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    std::pair<Grid2D, int> result = conjugate_gradient_2d(U, F_rhs, mask, border_mask, 
                                       a_coeffs, b_coeffs, hx, hy, (N-1) * (M-1), 1e-8, topo, num_threads);
    
    // Синхронизируем перед окончанием измерения
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    
    double time = 0.0;
    MPI_Reduce(&local_time, &time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Время решения: " << time << " с" << std::endl;
        std::cout << "Количество итераций: " << result.second << std::endl;
    }
    
    // Собираем решение на процесс 0
    Grid2D global_U(N + 1, M + 1);
    Grid2D global_X(N + 1, M + 1);
    Grid2D global_Y(N + 1, M + 1);
    BoolGrid2D global_mask(N + 1, M + 1);
    
    gather_solution_2d(result.first, global_U, topo, N + 1, M + 1);
    
    // Восстанавливаем глобальные сетки X, Y и mask на процессе 0
    if (rank == 0) {
        for (int i = 0; i <= N; ++i) {
            for (int j = 0; j <= M; ++j) {
                global_X.data[i][j] = x_vals[j];
                global_Y.data[i][j] = y_vals[i];
                global_mask.data[i][j] = in_trapezoid(global_X.data[i][j], global_Y.data[i][j]);
            }
        }
        
        std::string filename_csv = "solution_" + int_to_string(M) + "_" + int_to_string(N) + "_" + int_to_string(size) + "_" + int_to_string(num_threads) + "_hybrid.csv";
        save_solution_csv(global_X, global_Y, global_U, global_mask, filename_csv);
        
        std::string filename_txt = "solution_" + int_to_string(M) + "_" + int_to_string(N) + "_" + int_to_string(size) + "_" + int_to_string(num_threads) + "_hybrid.txt";
        std::ofstream outfile(filename_txt.c_str());
        if (outfile.is_open()) {
            outfile << "Grid_Size: " << M << "x" << N << "\n";
            outfile << "Processes: " << size << " (" << topo.px << "x" << topo.py << ")\n";
            outfile << "Threads_per_process: " << num_threads << "\n";
            outfile << "Iterations: " << result.second << "\n";
            outfile << "Time(s): " << time << "\n";
            outfile << "Iterations_per_second: " << result.second / time << "\n";
            outfile.close();
        } else {
            std::cerr << "Error: Cannot open file " << filename_txt << " for writing!" << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}


