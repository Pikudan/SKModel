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
#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

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

// Структура для хранения времен выполнения
struct TimingInfo {
    double init_time;
    double setup_time;
    double cg_time;
    double gather_time;
    double total_time;
    double copy_to_gpu_time;
    double copy_from_gpu_time;
    double compute_time;
    double comm_time;
    // Детальные времена для отдельных операций
    double apply_A_time;
    double solve_D_time;
    double dot_product_time;
    double update_vectors_time;
    double update_p_time;
    
    TimingInfo() : init_time(0), setup_time(0), cg_time(0), gather_time(0), 
                   total_time(0), copy_to_gpu_time(0), copy_from_gpu_time(0),
                   compute_time(0), comm_time(0), apply_A_time(0), solve_D_time(0),
                   dot_product_time(0), update_vectors_time(0), update_p_time(0) {}
};

// Поиск оптимального двумерного разбиения
std::pair<int, int> find_optimal_2d_decomposition(int M, int N, int size) {
    int best_px = 1, best_py = size;
    double best_score = std::numeric_limits<double>::max();
    
    for (int px = 1; px <= size; ++px) {
        if (size % px != 0) continue;
        
        int py = size / px;
        
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
    
    topo.rank_x = rank % topo.px;
    topo.rank_y = rank / topo.px;
    
    int cols_per_proc = (M + 1) / topo.px;
    int cols_remainder = (M + 1) % topo.px;
    topo.local_M = cols_per_proc + (topo.rank_x < cols_remainder ? 1 : 0);
    topo.start_M = 0;
    for (int i = 0; i < topo.rank_x; ++i) {
        topo.start_M += cols_per_proc + (i < cols_remainder ? 1 : 0);
    }
    
    int rows_per_proc = (N + 1) / topo.py;
    int rows_remainder = (N + 1) % topo.py;
    topo.local_N = rows_per_proc + (topo.rank_y < rows_remainder ? 1 : 0);
    topo.start_N = 0;
    for (int i = 0; i < topo.rank_y; ++i) {
        topo.start_N += rows_per_proc + (i < rows_remainder ? 1 : 0);
    }
    
    topo.local_M_with_ghost = topo.local_M;
    if (topo.rank_x > 0) topo.local_M_with_ghost++;
    if (topo.rank_x < topo.px - 1) topo.local_M_with_ghost++;
    
    topo.local_N_with_ghost = topo.local_N;
    if (topo.rank_y > 0) topo.local_N_with_ghost++;
    if (topo.rank_y < topo.py - 1) topo.local_N_with_ghost++;
    
    return topo;
}

// CUDA device functions
__device__ bool in_trapezoid_device(double x, double y) {
    const double TOL = 1e-8;
    double x_left, x_right;
    
    if (fabs(3.0 - 0.0) > TOL) {
        x_left = -2.0 + (-3.0 - (-2.0)) * (y - 3.0) / (0.0 - 3.0);
    } else {
        x_left = -2.0;
    }
    
    if (fabs(3.0 - 0.0) > TOL) {
        x_right = 3.0 + (2.0 - 3.0) * (y - 0.0) / (3.0 - 0.0);
    } else {
        x_right = 3.0;
    }
    
    if (y < 0.0 || y > 3.0) return false;
    if (fabs(y - 0.0) < TOL) return x >= -3.0 && x <= 3.0;
    if (fabs(y - 3.0) < TOL) return x >= -2.0 && x <= 2.0;
    
    return fmin(x_left, x_right) <= x && x <= fmax(x_left, x_right);
}

// CUDA kernel: вычисление коэффициентов k_coeffs
__global__ void compute_k_coeffs_kernel(double* k_coeffs, char* mask, int rows, int cols, double eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    
    if (idx < total) {
        k_coeffs[idx] = (mask[idx] != 0) ? 1.0 : 1.0 / eps;
    }
}

// CUDA kernel: применение оператора A
__global__ void apply_A_kernel(double* Aw, const double* w, const char* mask, const char* border_mask,
                               const double* a_coeffs, const double* b_coeffs,
                               int rows, int cols, double hx, double hy,
                               int start_i, int end_i, int start_j, int end_j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (end_i - start_i) * (end_j - start_j);
    
    if (idx < total) {
        int local_i = idx / (end_j - start_j);
        int local_j = idx % (end_j - start_j);
        int i = start_i + local_i;
        int j = start_j + local_j;
        int pos = i * cols + j;
        
        if (mask[pos] != 0 && border_mask[pos] == 0) {
            double flux_x = 0.0, flux_y = 0.0;
            
            // Flux по X (используем a_coeffs[i][j] и a_coeffs[i][j+1])
            if (j > 0 && j < cols - 1) {
                int a_pos_j = i * cols + j;
                int a_pos_j1 = i * cols + (j + 1);
                flux_x = (a_coeffs[a_pos_j1] * (w[i * cols + (j + 1)] - w[pos]) - 
                         a_coeffs[a_pos_j] * (w[pos] - w[i * cols + (j - 1)])) / (hx * hx);
            }
            
            // Flux по Y (используем b_coeffs[i][j] и b_coeffs[i+1][j])
            if (i > 0 && i < rows - 1) {
                int b_pos_i = i * cols + j;
                int b_pos_i1 = (i + 1) * cols + j;
                flux_y = (b_coeffs[b_pos_i1] * (w[(i + 1) * cols + j] - w[pos]) - 
                         b_coeffs[b_pos_i] * (w[pos] - w[(i - 1) * cols + j])) / (hy * hy);
            }
            
            Aw[pos] = -flux_x - flux_y;
        } else {
            Aw[pos] = 0.0;
        }
    }
}

// CUDA kernel: решение системы с предобуславливателем D
__global__ void solve_D_kernel(double* z, const double* prec, const char* mask, const char* border_mask,
                               const double* a_coeffs, const double* b_coeffs,
                               int rows, int cols, double hx, double hy,
                               int start_i, int end_i, int start_j, int end_j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (end_i - start_i) * (end_j - start_j);
    
    if (idx < total) {
        int local_i = idx / (end_j - start_j);
        int local_j = idx % (end_j - start_j);
        int i = start_i + local_i;
        int j = start_j + local_j;
        int pos = i * cols + j;
        
        if (mask[pos] != 0 && border_mask[pos] == 0) {
            double D_diag = 0.0;
            
            // Диагональный элемент для X направления
            if (j > 0 && j < cols - 1) {
                int a_pos_j = i * cols + j;
                int a_pos_j1 = i * cols + (j + 1);
                D_diag += (a_coeffs[a_pos_j] + a_coeffs[a_pos_j1]) / (hx * hx);
            }
            
            // Диагональный элемент для Y направления
            if (i > 0 && i < rows - 1) {
                int b_pos_i = i * cols + j;
                int b_pos_i1 = (i + 1) * cols + j;
                D_diag += (b_coeffs[b_pos_i] + b_coeffs[b_pos_i1]) / (hy * hy);
            }
            
            if (D_diag > 1e-12) {
                z[pos] = prec[pos] / D_diag;
            } else {
                z[pos] = 0.0;
            }
        } else {
            z[pos] = 0.0;
        }
    }
}


__global__ void dot_product_kernel(const double* u, const double* v, const char* mask,
                                   int rows, int cols, double* partial_sum,
                                   int start_i, int end_i, int start_j, int end_j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (end_i - start_i) * (end_j - start_j);
    
    if (idx < total) {
        int local_i = idx / (end_j - start_j);
        int local_j = idx % (end_j - start_j);
        int i = start_i + local_i;
        int j = start_j + local_j;
        int pos = i * cols + j;
        
        // Каждый поток записывает результат в global memory (не в shared memory)
        if (mask[pos] != 0) {
            partial_sum[idx] = u[pos] * v[pos];
        } else {
            partial_sum[idx] = 0.0;
        }
    }
}

__global__ void update_vectors_kernel(double* w, double* r, const double* p, const double* Ap,
                                     const char* inner_mask, double alpha,
                                     int rows, int cols, int start_i, int end_i, int start_j, int end_j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (end_i - start_i) * (end_j - start_j);
    
    if (idx < total) {
        int local_i = idx / (end_j - start_j);
        int local_j = idx % (end_j - start_j);
        int i = start_i + local_i;
        int j = start_j + local_j;
        int pos = i * cols + j;
        
        if (inner_mask[pos] != 0) {
            w[pos] += alpha * p[pos];
            r[pos] -= alpha * Ap[pos];
        }
    }
}


__global__ void update_p_kernel(double* p, const double* z, double beta,
                                const char* inner_mask,
                                int rows, int cols, int start_i, int end_i, int start_j, int end_j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (end_i - start_i) * (end_j - start_j);
    
    if (idx < total) {
        int local_i = idx / (end_j - start_j);
        int local_j = idx % (end_j - start_j);
        int i = start_i + local_i;
        int j = start_j + local_j;
        int pos = i * cols + j;
        
        if (inner_mask[pos] != 0) {
            p[pos] = z[pos] + beta * p[pos];
        }
    }
}

__global__ void init_residual_kernel(double* r, const double* F, const double* Aw,
                                     const char* inner_mask,
                                     int rows, int cols, int start_i, int end_i, int start_j, int end_j) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (end_i - start_i) * (end_j - start_j);
    
    if (idx < total) {
        int local_i = idx / (end_j - start_j);
        int local_j = idx % (end_j - start_j);
        int i = start_i + local_i;
        int j = start_j + local_j;
        int pos = i * cols + j;
        
        if (inner_mask[pos] != 0) {
            r[pos] = F[pos] - Aw[pos];
        } else {
            r[pos] = 0.0;
        }
    }
}

void flatten_grid(const Grid2D& grid, double* flat_array) {
    for (int i = 0; i < grid.rows; ++i) {
        for (int j = 0; j < grid.cols; ++j) {
            flat_array[i * grid.cols + j] = grid.data[i][j];
        }
    }
}

void unflatten_grid(const double* flat_array, Grid2D& grid) {
    for (int i = 0; i < grid.rows; ++i) {
        for (int j = 0; j < grid.cols; ++j) {
            grid.data[i][j] = flat_array[i * grid.cols + j];
        }
    }
}


void flatten_bool_grid(const BoolGrid2D& grid, bool* flat_array) {
    for (int i = 0; i < grid.rows; ++i) {
        for (int j = 0; j < grid.cols; ++j) {
            flat_array[i * grid.cols + j] = grid.data[i][j];
        }
    }
}

// Обмен граничными данными (MPI коммуникации)
void exchange_boundaries_2d(Grid2D& grid, const ProcessTopology2D& topo, 
                            ExchangeBuffers& buffers, MPI_Comm comm, TimingInfo& timing) {
    double comm_start = MPI_Wtime();
    MPI_Status status;
    
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
    
    if (rank_up >= 0) {
        int send_row = (topo.rank_y > 0) ? 1 : 0;
        int recv_row = 0;
        MPI_Sendrecv(grid.data[send_row].data(), grid.cols, MPI_DOUBLE, rank_up, 0,
                     grid.data[recv_row].data(), grid.cols, MPI_DOUBLE, rank_up, 1,
                     comm, &status);
    }
    
    if (rank_down >= 0) {
        int send_row = grid.rows - ((topo.rank_y < topo.py - 1) ? 2 : 1);
        int recv_row = grid.rows - 1;
        MPI_Sendrecv(grid.data[send_row].data(), grid.cols, MPI_DOUBLE, rank_down, 1,
                     grid.data[recv_row].data(), grid.cols, MPI_DOUBLE, rank_down, 0,
                     comm, &status);
    }
    
    if (rank_left >= 0) {
        int send_col = (topo.rank_x > 0) ? 1 : 0;
        int recv_col = 0;
        for (int i = 0; i < grid.rows; ++i) {
            buffers.send_buf_left[i] = grid.data[i][send_col];
        }
        MPI_Sendrecv(buffers.send_buf_left.data(), grid.rows, MPI_DOUBLE, rank_left, 2,
                     buffers.recv_buf_left.data(), grid.rows, MPI_DOUBLE, rank_left, 3,
                     comm, &status);
        for (int i = 0; i < grid.rows; ++i) {
            grid.data[i][recv_col] = buffers.recv_buf_left[i];
        }
    }
    
    if (rank_right >= 0) {
        int send_col = grid.cols - ((topo.rank_x < topo.px - 1) ? 2 : 1);
        int recv_col = grid.cols - 1;
        for (int i = 0; i < grid.rows; ++i) {
            buffers.send_buf_right[i] = grid.data[i][send_col];
        }
        MPI_Sendrecv(buffers.send_buf_right.data(), grid.rows, MPI_DOUBLE, rank_right, 3,
                     buffers.recv_buf_right.data(), grid.rows, MPI_DOUBLE, rank_right, 2,
                     comm, &status);
        for (int i = 0; i < grid.rows; ++i) {
            grid.data[i][recv_col] = buffers.recv_buf_right[i];
        }
    }
    
    timing.comm_time += MPI_Wtime() - comm_start;
}

// Вычисление коэффициентов на CPU (для setup)
double compute_a_ij(double x_half, double y_j_minus_half, double y_j_plus_half, 
                   double h2, double eps) {
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

double compute_b_ij(double y_half, double x_i_minus_half, double x_i_plus_half,
                   double h1, double eps) {
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

double compute_F_ij(double x_i, double y_j, double h1, double h2) {
    double x_start = x_i - h1/2;
    double x_end = x_i + h1/2;
    double y_start = y_j - h2/2;
    double y_end = y_j + h2/2;
    
    double y_low = std::max(0.0, y_start);
    double y_high = std::min(3.0, y_end);
    
    if (y_low >= y_high) return 0.0;
    
    const int num_segments = 100;
    double dy = (y_high - y_low) / num_segments;
    double area = 0.0;
    
    for (int i = 0; i < num_segments; i++) {
        double y = y_low + (i + 0.5) * dy;
        double left_bound = -3.0 + y/3.0;
        double right_bound = 3.0 - y/3.0;
        double cell_left = std::max(x_start, left_bound);
        double cell_right = std::min(x_end, right_bound);
        
        if (cell_left < cell_right) {
            area += (cell_right - cell_left) * dy;
        }
    }
    
    return area / (h1 * h2);
}

void setup_fictitious_domain_2d(const Grid2D& X, const Grid2D& Y, const BoolGrid2D& mask,
                                Grid2D& k_coeffs, Grid2D& F_rhs, Grid2D& a_coeffs, 
                                Grid2D& b_coeffs, double hx, double hy,
                                const ProcessTopology2D& topo) {
    double h = std::max(hx, hy);
    double eps = h * h;
    
    int local_M = X.cols;
    int local_N = X.rows;
    
    for (int i = 0; i < local_N; ++i) {
        for (int j = 0; j < local_M; ++j) {
            k_coeffs.data[i][j] = mask.data[i][j] ? 1.0 : 1.0/eps;
        }
    }
    
    for (int j = 1; j < local_M; ++j) {
        for (int i = 0; i < local_N; ++i) {
            double x_half = X.data[i][j] - hx/2;
            double y_j_minus_half = Y.data[i][j] - hy/2;
            double y_j_plus_half = Y.data[i][j] + hy/2;
            a_coeffs.data[i][j] = compute_a_ij(x_half, y_j_minus_half, y_j_plus_half, hy, eps);
        }
    }
    
    for (int i = 1; i < local_N; ++i) {
        for (int j = 0; j < local_M; ++j) {
            double y_half = Y.data[i][j] - hy/2;
            double x_i_minus_half = X.data[i][j] - hx/2;
            double x_i_plus_half = X.data[i][j] + hx/2;
            b_coeffs.data[i][j] = compute_b_ij(y_half, x_i_minus_half, x_i_plus_half, hx, eps);
        }
    }
    
    int start_i = (topo.rank_y > 0) ? 1 : 0;
    int end_i = (topo.rank_y < topo.py - 1) ? local_N - 1 : local_N;
    int start_j = (topo.rank_x > 0) ? 1 : 0;
    int end_j = (topo.rank_x < topo.px - 1) ? local_M - 1 : local_M;
    
    for (int i = start_i; i < end_i; ++i) {
        for (int j = start_j; j < end_j; ++j) {
            F_rhs.data[i][j] = compute_F_ij(X.data[i][j], Y.data[i][j], hx, hy);
        }
    }
}

std::pair<Grid2D, int> conjugate_gradient_2d_cuda(
    const Grid2D& U0, const Grid2D& F, const BoolGrid2D& mask,
    const BoolGrid2D& border_mask, const Grid2D& a_coeffs,
    const Grid2D& b_coeffs, double hx, double hy,
    int max_iter, double tol, const ProcessTopology2D& topo, TimingInfo& timing) {
    
    int rows = U0.rows;
    int cols = U0.cols;
    int total_size = rows * cols;
    
    int start_i = (topo.rank_y > 0) ? 1 : 0;
    int end_i = (topo.rank_y < topo.py - 1) ? rows - 1 : rows;
    int start_j = (topo.rank_x > 0) ? 1 : 0;
    int end_j = (topo.rank_x < topo.px - 1) ? cols - 1 : cols;
    
    // Выделяем память на GPU
    double *d_w, *d_r, *d_p, *d_z, *d_Aw, *d_Ap, *d_F;
    double *d_a_coeffs, *d_b_coeffs;
    char *d_mask, *d_border_mask, *d_inner_mask;  // Используем char вместо bool для совместимости
    double *d_partial_sum;
    
    double copy_start = MPI_Wtime();
    CUDA_CHECK(cudaMalloc(&d_w, total_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, total_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p, total_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z, total_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Aw, total_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Ap, total_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_F, total_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_a_coeffs, total_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b_coeffs, total_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mask, total_size * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_border_mask, total_size * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_inner_mask, total_size * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_partial_sum, total_size * sizeof(double)));
    

    std::vector<double> flat_w(total_size), flat_F(total_size);
    std::vector<double> flat_a_coeffs(total_size), flat_b_coeffs(total_size);
    std::vector<char> flat_mask(total_size), flat_border_mask(total_size), flat_inner_mask(total_size);
    
    flatten_grid(U0, flat_w.data());
    flatten_grid(F, flat_F.data());
    flatten_grid(a_coeffs, flat_a_coeffs.data());
    flatten_grid(b_coeffs, flat_b_coeffs.data());
    
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            flat_mask[i * mask.cols + j] = mask.data[i][j] ? 1 : 0;
            flat_border_mask[i * mask.cols + j] = border_mask.data[i][j] ? 1 : 0;
        }
    }
    
    BoolGrid2D inner_mask(mask.rows, mask.cols);
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            inner_mask.data[i][j] = mask.data[i][j] && !border_mask.data[i][j];
            flat_inner_mask[i * mask.cols + j] = inner_mask.data[i][j] ? 1 : 0;
        }
    }
    
    // Копируем данные на GPU
    CUDA_CHECK(cudaMemcpy(d_w, flat_w.data(), total_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_F, flat_F.data(), total_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_coeffs, flat_a_coeffs.data(), total_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_coeffs, flat_b_coeffs.data(), total_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, flat_mask.data(), total_size * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_border_mask, flat_border_mask.data(), total_size * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inner_mask, flat_inner_mask.data(), total_size * sizeof(char), cudaMemcpyHostToDevice));
    timing.copy_to_gpu_time += MPI_Wtime() - copy_start;
    
    // Обмен границами перед началом
    Grid2D w_host = U0;
    ExchangeBuffers exchange_buffers(rows);
    exchange_boundaries_2d(w_host, topo, exchange_buffers, MPI_COMM_WORLD, timing);
    flatten_grid(w_host, flat_w.data());
    CUDA_CHECK(cudaMemcpy(d_w, flat_w.data(), total_size * sizeof(double), cudaMemcpyHostToDevice));
    
    // Вычисляем Aw0 на GPU
    int compute_size = (end_i - start_i) * (end_j - start_j);
    int threads_per_block = 256;
    int num_blocks = (compute_size + threads_per_block - 1) / threads_per_block;
    
    double op_start = MPI_Wtime();
    apply_A_kernel<<<num_blocks, threads_per_block>>>(
        d_Aw, d_w, d_mask, d_border_mask, d_a_coeffs, d_b_coeffs,
        rows, cols, hx, hy, start_i, end_i, start_j, end_j);
    CUDA_CHECK(cudaDeviceSynchronize());
    timing.apply_A_time += MPI_Wtime() - op_start;
    timing.compute_time += MPI_Wtime() - op_start;
    

    op_start = MPI_Wtime();
    init_residual_kernel<<<num_blocks, threads_per_block>>>(
        d_r, d_F, d_Aw, d_inner_mask, rows, cols, start_i, end_i, start_j, end_j);
    CUDA_CHECK(cudaDeviceSynchronize());
    timing.compute_time += MPI_Wtime() - op_start;
    
    op_start = MPI_Wtime();
    solve_D_kernel<<<num_blocks, threads_per_block>>>(
        d_z, d_r, d_mask, d_border_mask, d_a_coeffs, d_b_coeffs,
        rows, cols, hx, hy, start_i, end_i, start_j, end_j);
    CUDA_CHECK(cudaDeviceSynchronize());
    timing.solve_D_time += MPI_Wtime() - op_start;
    timing.compute_time += MPI_Wtime() - op_start;
    
    CUDA_CHECK(cudaMemcpy(d_p, d_z, total_size * sizeof(double), cudaMemcpyDeviceToDevice));
    
    int local_num_unknowns = 0;
    for (int i = start_i; i < end_i; ++i) {
        for (int j = start_j; j < end_j; ++j) {
            if (inner_mask.data[i][j]) local_num_unknowns++;
        }
    }
    int global_num_unknowns = 0;
    MPI_Allreduce(&local_num_unknowns, &global_num_unknowns, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    op_start = MPI_Wtime();
    dot_product_kernel<<<num_blocks, threads_per_block>>>(
        d_r, d_z, d_inner_mask, rows, cols, d_partial_sum, start_i, end_i, start_j, end_j);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    double local_result = thrust::reduce(thrust::device, d_partial_sum, d_partial_sum + compute_size);
    
    // Глобальная редукция через MPI
    double rz_old = 0.0;
    MPI_Allreduce(&local_result, &rz_old, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    timing.dot_product_time += MPI_Wtime() - op_start;
    timing.compute_time += MPI_Wtime() - op_start;
    
    int iterations = 0;
    int rank = topo.rank_y * topo.px + topo.rank_x;
    
    for (int k = 0; k < max_iter && k < global_num_unknowns; ++k) {
        iterations++;
        
        op_start = MPI_Wtime();
        dot_product_kernel<<<num_blocks, threads_per_block>>>(
            d_r, d_r, d_inner_mask, rows, cols, d_partial_sum, start_i, end_i, start_j, end_j);
        CUDA_CHECK(cudaDeviceSynchronize());
        // Редукция на GPU с помощью Thrust
        double current_residual;
        {
            double local_result = thrust::reduce(thrust::device, d_partial_sum, d_partial_sum + compute_size);
            // Глобальная редукция через MPI
            double global_norm_sq = 0.0;
            MPI_Allreduce(&local_result, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            current_residual = std::sqrt(global_norm_sq);
        }
        timing.dot_product_time += MPI_Wtime() - op_start;
        timing.compute_time += MPI_Wtime() - op_start;
        
        if (current_residual < tol) {
            if (rank == 0) {
                std::cout << "Сходимость на итерации " << k 
                          << ", невязка = " << std::scientific << std::setprecision(10) << current_residual << std::endl;
            }
            break;
        }
        
        if (rank == 0 && k % 100 == 0) {
            std::cout << "Итерация " << k << ": невязка = " << std::scientific << std::setprecision(10) << current_residual << std::endl;
        }
        

        copy_start = MPI_Wtime();
        CUDA_CHECK(cudaMemcpy(flat_w.data(), d_p, total_size * sizeof(double), cudaMemcpyDeviceToHost));
        unflatten_grid(flat_w.data(), w_host);
        timing.copy_from_gpu_time += MPI_Wtime() - copy_start;
        
 
        exchange_boundaries_2d(w_host, topo, exchange_buffers, MPI_COMM_WORLD, timing);
        
        copy_start = MPI_Wtime();
        flatten_grid(w_host, flat_w.data());
        CUDA_CHECK(cudaMemcpy(d_p, flat_w.data(), total_size * sizeof(double), cudaMemcpyHostToDevice));
        timing.copy_to_gpu_time += MPI_Wtime() - copy_start;
        
        op_start = MPI_Wtime();
        apply_A_kernel<<<num_blocks, threads_per_block>>>(
            d_Ap, d_p, d_mask, d_border_mask, d_a_coeffs, d_b_coeffs,
            rows, cols, hx, hy, start_i, end_i, start_j, end_j);
        CUDA_CHECK(cudaDeviceSynchronize());
        timing.apply_A_time += MPI_Wtime() - op_start;
        timing.compute_time += MPI_Wtime() - op_start;
        
        op_start = MPI_Wtime();
        dot_product_kernel<<<num_blocks, threads_per_block>>>(
            d_p, d_Ap, d_inner_mask, rows, cols, d_partial_sum, start_i, end_i, start_j, end_j);
        CUDA_CHECK(cudaDeviceSynchronize());
        // Редукция на GPU с помощью Thrust
        double pAp;
        {
            double local_result = thrust::reduce(thrust::device, d_partial_sum, d_partial_sum + compute_size);
            double pAp_temp = 0.0;
            MPI_Allreduce(&local_result, &pAp_temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            pAp = pAp_temp;
        }
        timing.dot_product_time += MPI_Wtime() - op_start;
        timing.compute_time += MPI_Wtime() - op_start;
        
        if (std::fabs(pAp) < 1e-15) {
            if (rank == 0) {
                std::cout << "Матрица вырождена, итерация " << k << std::endl;
            }
            break;
        }
        
        double alpha = rz_old / pAp;
        

        op_start = MPI_Wtime();
        update_vectors_kernel<<<num_blocks, threads_per_block>>>(
            d_w, d_r, d_p, d_Ap, d_inner_mask, alpha, rows, cols, start_i, end_i, start_j, end_j);
        CUDA_CHECK(cudaDeviceSynchronize());
        timing.update_vectors_time += MPI_Wtime() - op_start;
        timing.compute_time += MPI_Wtime() - op_start;
        
        copy_start = MPI_Wtime();
        CUDA_CHECK(cudaMemcpy(flat_w.data(), d_w, total_size * sizeof(double), cudaMemcpyDeviceToHost));
        unflatten_grid(flat_w.data(), w_host);
        timing.copy_from_gpu_time += MPI_Wtime() - copy_start;
        

        exchange_boundaries_2d(w_host, topo, exchange_buffers, MPI_COMM_WORLD, timing);
        

        copy_start = MPI_Wtime();
        flatten_grid(w_host, flat_w.data());
        CUDA_CHECK(cudaMemcpy(d_w, flat_w.data(), total_size * sizeof(double), cudaMemcpyHostToDevice));
        timing.copy_to_gpu_time += MPI_Wtime() - copy_start;
        

        op_start = MPI_Wtime();
        solve_D_kernel<<<num_blocks, threads_per_block>>>(
            d_z, d_r, d_mask, d_border_mask, d_a_coeffs, d_b_coeffs,
            rows, cols, hx, hy, start_i, end_i, start_j, end_j);
        CUDA_CHECK(cudaDeviceSynchronize());
        timing.solve_D_time += MPI_Wtime() - op_start;
        timing.compute_time += MPI_Wtime() - op_start;
        
        op_start = MPI_Wtime();
        dot_product_kernel<<<num_blocks, threads_per_block>>>(
            d_r, d_z, d_inner_mask, rows, cols, d_partial_sum, start_i, end_i, start_j, end_j);
        CUDA_CHECK(cudaDeviceSynchronize());
        double rz_new;
        {
            double local_result = thrust::reduce(thrust::device, d_partial_sum, d_partial_sum + compute_size);
            // Глобальная редукция через MPI
            double rz_new_temp = 0.0;
            MPI_Allreduce(&local_result, &rz_new_temp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            rz_new = rz_new_temp;
        }
        timing.dot_product_time += MPI_Wtime() - op_start;
        timing.compute_time += MPI_Wtime() - op_start;
        
        if (std::fabs(rz_old) < 1e-15) {
            if (rank == 0) {
                std::cout << "Деление на ноль, итерация " << k << std::endl;
            }
            break;
        }
        
        double beta = rz_new / rz_old;
        
        op_start = MPI_Wtime();
        update_p_kernel<<<num_blocks, threads_per_block>>>(
            d_p, d_z, beta, d_inner_mask, rows, cols, start_i, end_i, start_j, end_j);
        CUDA_CHECK(cudaDeviceSynchronize());
        timing.update_p_time += MPI_Wtime() - op_start;
        timing.compute_time += MPI_Wtime() - op_start;
        
        rz_old = rz_new;
    }
    

    copy_start = MPI_Wtime();
    CUDA_CHECK(cudaMemcpy(flat_w.data(), d_w, total_size * sizeof(double), cudaMemcpyDeviceToHost));
    Grid2D result(rows, cols);
    unflatten_grid(flat_w.data(), result);
    timing.copy_from_gpu_time += MPI_Wtime() - copy_start;
    
    // Освобождаем память GPU
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_Aw));
    CUDA_CHECK(cudaFree(d_Ap));
    CUDA_CHECK(cudaFree(d_F));
    CUDA_CHECK(cudaFree(d_a_coeffs));
    CUDA_CHECK(cudaFree(d_b_coeffs));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_border_mask));
    CUDA_CHECK(cudaFree(d_inner_mask));
    CUDA_CHECK(cudaFree(d_partial_sum));
    
    return std::make_pair(result, iterations);
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
    
    std::vector<int> recvcounts(topo.px * topo.py);
    std::vector<int> displs(topo.px * topo.py);
    
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
    
    TimingInfo timing;
    double total_start = MPI_Wtime();
    
    if (argc != 3) {
        if (rank == 0) {
            std::cout << "Использование: mpirun -np <num_procs> " << argv[0] << " <M> <N>" << std::endl;
            std::cout << "Пример: mpirun -np 4 " << argv[0] << " 40 40" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    
    if (M <= 0 || N <= 0) {
        if (rank == 0) {
            std::cout << "Ошибка: все параметры должны быть положительными числами" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Инициализация CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count > 0) {
        int device_id = rank % device_count;
        CUDA_CHECK(cudaSetDevice(device_id));
        if (rank == 0) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
            std::cout << "Используется GPU: " << prop.name << " (Compute Capability " << prop.major << "." << prop.minor << ")" << std::endl;
        }
    } else {
        if (rank == 0) {
            std::cerr << "Ошибка: GPU не найдены!" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    double init_start = MPI_Wtime();
    ProcessTopology2D topo = init_topology_2d(M, N, rank, size);
    timing.init_time = MPI_Wtime() - init_start;
    
    if (rank == 0) {
        std::cout << "Двумерное разбиение: " << topo.px << " x " << topo.py << " процессов" << std::endl;
    }
    
    init_start = MPI_Wtime();
    std::vector<double> x_vals(M + 1), y_vals(N + 1);
    for (int i = 0; i <= M; ++i) x_vals[i] = -3.0 + 6.0 * i / M;
    for (int i = 0; i <= N; ++i) y_vals[i] = 0.0 + 3.0 * i / N;
    
    Grid2D X(topo.local_N_with_ghost, topo.local_M_with_ghost);
    Grid2D Y(topo.local_N_with_ghost, topo.local_M_with_ghost);
    BoolGrid2D mask(topo.local_N_with_ghost, topo.local_M_with_ghost);
    BoolGrid2D border_mask(topo.local_N_with_ghost, topo.local_M_with_ghost);
    
    int local_i_offset = (topo.rank_y > 0) ? -1 : 0;
    int local_j_offset = (topo.rank_x > 0) ? -1 : 0;
    
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
    
    double setup_start = MPI_Wtime();
    setup_fictitious_domain_2d(X, Y, mask, k_coeffs, F_rhs, a_coeffs, b_coeffs, hx, hy, topo);
    timing.setup_time = MPI_Wtime() - setup_start;
    
    Grid2D U(topo.local_N_with_ghost, topo.local_M_with_ghost);
    for (int i = 0; i < topo.local_N_with_ghost; ++i) {
        for (int j = 0; j < topo.local_M_with_ghost; ++j) {
            if (border_mask.data[i][j]) {
                U.data[i][j] = g(X.data[i][j], Y.data[i][j]);
            }
        }
    }
    timing.init_time += MPI_Wtime() - init_start;
    
    MPI_Barrier(MPI_COMM_WORLD);
    double cg_start = MPI_Wtime();
    
    std::pair<Grid2D, int> result = conjugate_gradient_2d_cuda(
        U, F_rhs, mask, border_mask, a_coeffs, b_coeffs, hx, hy,
        (N-1) * (M-1), 1e-8, topo, timing);
    
    MPI_Barrier(MPI_COMM_WORLD);
    timing.cg_time = MPI_Wtime() - cg_start;
    
    double gather_start = MPI_Wtime();
    Grid2D global_U(N + 1, M + 1);
    Grid2D global_X(N + 1, M + 1);
    Grid2D global_Y(N + 1, M + 1);
    BoolGrid2D global_mask(N + 1, M + 1);
    
    gather_solution_2d(result.first, global_U, topo, N + 1, M + 1);
    timing.gather_time = MPI_Wtime() - gather_start;
    
    if (rank == 0) {
        for (int i = 0; i <= N; ++i) {
            for (int j = 0; j <= M; ++j) {
                global_X.data[i][j] = x_vals[j];
                global_Y.data[i][j] = y_vals[i];
                global_mask.data[i][j] = in_trapezoid(global_X.data[i][j], global_Y.data[i][j]);
            }
        }
        
        std::string filename_csv = "solution_" + int_to_string(M) + "_" + int_to_string(N) + "_" + int_to_string(size) + "_mpi_cuda.csv";
        save_solution_csv(global_X, global_Y, global_U, global_mask, filename_csv);
    }
    
    timing.total_time = MPI_Wtime() - total_start;
    
    // Собираем времена со всех процессов
    TimingInfo timing_max;
    MPI_Reduce(&timing.init_time, &timing_max.init_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing.setup_time, &timing_max.setup_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing.cg_time, &timing_max.cg_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing.gather_time, &timing_max.gather_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing.total_time, &timing_max.total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing.copy_to_gpu_time, &timing_max.copy_to_gpu_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing.copy_from_gpu_time, &timing_max.copy_from_gpu_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing.compute_time, &timing_max.compute_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing.comm_time, &timing_max.comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing.apply_A_time, &timing_max.apply_A_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing.solve_D_time, &timing_max.solve_D_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing.dot_product_time, &timing_max.dot_product_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing.update_vectors_time, &timing_max.update_vectors_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timing.update_p_time, &timing_max.update_p_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "\n=== Времена выполнения ===" << std::endl;
        std::cout << "Инициализация: " << timing_max.init_time << " с" << std::endl;
        std::cout << "Setup: " << timing_max.setup_time << " с" << std::endl;
        std::cout << "CG метод: " << timing_max.cg_time << " с" << std::endl;
        std::cout << "  - Копирование на GPU: " << timing_max.copy_to_gpu_time << " с" << std::endl;
        std::cout << "  - Копирование с GPU: " << timing_max.copy_from_gpu_time << " с" << std::endl;
        std::cout << "  - Вычисления на GPU (общее): " << timing_max.compute_time << " с" << std::endl;
        std::cout << "    * apply_A: " << timing_max.apply_A_time << " с" << std::endl;
        std::cout << "    * solve_D: " << timing_max.solve_D_time << " с" << std::endl;
        std::cout << "    * dot_product: " << timing_max.dot_product_time << " с" << std::endl;
        std::cout << "    * update_vectors: " << timing_max.update_vectors_time << " с" << std::endl;
        std::cout << "    * update_p: " << timing_max.update_p_time << " с" << std::endl;
        std::cout << "  - Коммуникации MPI: " << timing_max.comm_time << " с" << std::endl;
        std::cout << "Gather: " << timing_max.gather_time << " с" << std::endl;
        std::cout << "Общее время: " << timing_max.total_time << " с" << std::endl;
        std::cout << "Количество итераций: " << result.second << std::endl;
        
        std::string filename_txt = "solution_" + int_to_string(M) + "_" + int_to_string(N) + "_" + int_to_string(size) + "_mpi_cuda.txt";
        std::ofstream outfile(filename_txt.c_str());
        if (outfile.is_open()) {
            outfile << "Grid_Size: " << M << "x" << N << "\n";
            outfile << "Processes: " << size << " (" << topo.px << "x" << topo.py << ")\n";
            outfile << "Iterations: " << result.second << "\n";
            outfile << "Total_Time(s): " << timing_max.total_time << "\n";
            outfile << "Init_Time(s): " << timing_max.init_time << "\n";
            outfile << "Setup_Time(s): " << timing_max.setup_time << "\n";
            outfile << "CG_Time(s): " << timing_max.cg_time << "\n";
            outfile << "Copy_To_GPU_Time(s): " << timing_max.copy_to_gpu_time << "\n";
            outfile << "Copy_From_GPU_Time(s): " << timing_max.copy_from_gpu_time << "\n";
            outfile << "Compute_Time(s): " << timing_max.compute_time << "\n";
            outfile << "Apply_A_Time(s): " << timing_max.apply_A_time << "\n";
            outfile << "Solve_D_Time(s): " << timing_max.solve_D_time << "\n";
            outfile << "Dot_Product_Time(s): " << timing_max.dot_product_time << "\n";
            outfile << "Update_Vectors_Time(s): " << timing_max.update_vectors_time << "\n";
            outfile << "Update_P_Time(s): " << timing_max.update_p_time << "\n";
            outfile << "Comm_Time(s): " << timing_max.comm_time << "\n";
            outfile << "Gather_Time(s): " << timing_max.gather_time << "\n";
            outfile << "Iterations_per_second: " << result.second / timing_max.total_time << "\n";
            outfile.close();
        }
    }
    
    MPI_Finalize();
    return 0;
}

