#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <limits>
#include <sstream>
#include <sys/time.h>

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

void setup_fictitious_domain(const Grid2D& X, const Grid2D& Y, const BoolGrid2D& mask,
                            Grid2D& k_coeffs, Grid2D& F_rhs, Grid2D& a_coeffs, 
                            Grid2D& b_coeffs, double hx, double hy) {
    double h = std::max(hx, hy);
    double eps = h * h;
    
    int M = X.cols - 1;
    int N = X.rows - 1;
    
    // k_coeffs
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= M; ++j) {
            k_coeffs.data[i][j] = mask.data[i][j] ? 1.0 : 1.0/eps;
        }
    }
    
    // a_coeffs
    for (int i = 1; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            double x_half = X.data[j][i] - hx/2;  // x_{i-1/2}
            double y_j_minus_half = Y.data[j][i] - hy/2;
            double y_j_plus_half = Y.data[j][i] + hy/2;
            
            a_coeffs.data[j][i] = compute_a_ij(x_half, y_j_minus_half, y_j_plus_half, hy, eps);
        }
    }
    
    // b_coeffs
    for (int i = 0; i <= M; ++i) {
        for (int j = 1; j <= N; ++j) {
            double y_half = Y.data[j][i] - hy/2;  // y_{j-1/2}
            double x_i_minus_half = X.data[j][i] - hx/2;
            double x_i_plus_half = X.data[j][i] + hx/2;
            
            b_coeffs.data[j][i] = compute_b_ij(y_half, x_i_minus_half, x_i_plus_half, hx, eps);
        }
    }
    
    // F_rhs
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N; ++j) {
            F_rhs.data[j][i] = compute_F_ij(X.data[j][i], Y.data[j][i], hx, hy);
        }
    }
}

Grid2D apply_A(const Grid2D& w, const BoolGrid2D& mask, const BoolGrid2D& border_mask,
               const Grid2D& a_coeffs, const Grid2D& b_coeffs, double hx, double hy) {
    Grid2D Aw(w.rows, w.cols, 0.0);
    
    for (int i = 1; i < w.rows - 1; ++i) {
        for (int j = 1; j < w.cols - 1; ++j) {
            if (mask.data[i][j] && !border_mask.data[i][j]) {
                double flux_x = (a_coeffs.data[i+1][j] * (w.data[i+1][j] - w.data[i][j]) - 
                               (a_coeffs.data[i][j] * (w.data[i][j] - w.data[i-1][j]))) / (hx * hx);
                
                double flux_y = (b_coeffs.data[i][j+1] * (w.data[i][j+1] - w.data[i][j]) - 
                               (b_coeffs.data[i][j] * (w.data[i][j] - w.data[i][j-1]))) / (hy * hy);
                
                Aw.data[i][j] = -flux_x - flux_y;
            }
        }
    }
    
    return Aw;
}

Grid2D solve_D(const Grid2D& prec, const BoolGrid2D& mask, const BoolGrid2D& border_mask,
               const Grid2D& a_coeffs, const Grid2D& b_coeffs, double hx, double hy) {
    Grid2D z(prec.rows, prec.cols, 0.0);
    
    for (int i = 1; i < prec.rows - 1; ++i) {
        for (int j = 1; j < prec.cols - 1; ++j) {
            if (mask.data[i][j] && !border_mask.data[i][j]) {
                double D_diag = (a_coeffs.data[i][j] + a_coeffs.data[i+1][j]) / (hx * hx) +
                               (b_coeffs.data[i][j] + b_coeffs.data[i][j+1]) / (hy * hy);
                
                if (D_diag > 1e-12) {
                    z.data[i][j] = prec.data[i][j] / D_diag;
                }
            }
        }
    }
    
    return z;
}

double dot_product(const Grid2D& u, const Grid2D& v, const BoolGrid2D& mask) {
    double result = 0.0;
    for (int i = 0; i < u.rows; ++i) {
        for (int j = 0; j < u.cols; ++j) {
            if (mask.data[i][j]) {
                result += u.data[i][j] * v.data[i][j];
            }
        }
    }
    return result;
}

double norm(const Grid2D& u, const BoolGrid2D& mask) {
    return std::sqrt(dot_product(u, u, mask));
}

std::pair<Grid2D, int>  conjugate_gradient(const Grid2D& U0, const Grid2D& F, const BoolGrid2D& mask,
                         const BoolGrid2D& border_mask, const Grid2D& a_coeffs,
                         const Grid2D& b_coeffs, double hx, double hy,
                         int max_iter, double tol) {
    
    Grid2D w = U0;
    
    BoolGrid2D inner_mask(mask.rows, mask.cols);
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            inner_mask.data[i][j] = mask.data[i][j] && !border_mask.data[i][j];
        }
    }
    
    int num_unknowns = 0;
    for (int i = 0; i < inner_mask.rows; ++i) {
        for (int j = 0; j < inner_mask.cols; ++j) {
            if (inner_mask.data[i][j]) num_unknowns++;
        }
    }
    
    Grid2D Aw0 = apply_A(w, mask, border_mask, a_coeffs, b_coeffs, hx, hy);
    Grid2D r(F.rows, F.cols, 0.0);
    
    for (int i = 0; i < r.rows; ++i) {
        for (int j = 0; j < r.cols; ++j) {
            if (inner_mask.data[i][j]) {
                r.data[i][j] = F.data[i][j] - Aw0.data[i][j];
            }
        }
    }
    
    Grid2D z = solve_D(r, mask, border_mask, a_coeffs, b_coeffs, hx, hy);
    Grid2D p = z;
    
    double rz_old = dot_product(r, z, inner_mask);

    int iterations = 0;

    for (int k = 0; k < max_iter && k < num_unknowns; ++k) {
        iterations++;
        double current_residual = norm(r, inner_mask);
        
        if (current_residual < tol) {
            std::cout << "Сходимость на итерации " << k 
                      << ", невязка = " << current_residual << std::endl;
            break;
        }
        
        if (k % 100 == 0) {
            std::cout << "Итерация " << k << ": невязка = " << current_residual << std::endl;
        }
        
        Grid2D Ap = apply_A(p, mask, border_mask, a_coeffs, b_coeffs, hx, hy);
        double pAp = dot_product(p, Ap, inner_mask);
        
        if (std::fabs(pAp) < 1e-15) {
            std::cout << "Матрица вырождена, итерация " << k << std::endl;
            break;
        }
        
        double alpha = rz_old / pAp;
        
        for (int i = 0; i < w.rows; ++i) {
            for (int j = 0; j < w.cols; ++j) {
                if (inner_mask.data[i][j]) {
                    w.data[i][j] += alpha * p.data[i][j];
                    r.data[i][j] -= alpha * Ap.data[i][j];
                }
            }
        }
        
        z = solve_D(r, mask, border_mask, a_coeffs, b_coeffs, hx, hy);
        double rz_new = dot_product(r, z, inner_mask);
        
        if (std::fabs(rz_old) < 1e-15) {
            std::cout << "Деление на ноль, итерация " << k << std::endl;
            break;
        }
        
        double beta = rz_new / rz_old;
        
        for (int i = 0; i < p.rows; ++i) {
            for (int j = 0; j < p.cols; ++j) {
                if (inner_mask.data[i][j]) {
                    p.data[i][j] = z.data[i][j] + beta * p.data[i][j];
                }
            }
        }
        
        rz_old = rz_new;
    }
    
    return std::make_pair(w, iterations);
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

int main() {
    const int num_grids = 3;
    int grid_sizes[num_grids][2] = {{10, 10}, {20, 20}, {40, 40}};
    
    for (int grid_idx = 0; grid_idx < num_grids; ++grid_idx) {
        int M = grid_sizes[grid_idx][0];
        int N = grid_sizes[grid_idx][1];

        std::vector<double> x_vals(M + 1), y_vals(N + 1);
        for (int i = 0; i <= M; ++i) x_vals[i] = -3.0 + 6.0 * i / M;
        for (int i = 0; i <= N; ++i) y_vals[i] = 0.0 + 3.0 * i / N;
        
        Grid2D X(N + 1, M + 1), Y(N + 1, M + 1);
        BoolGrid2D mask(N + 1, M + 1), border_mask(N + 1, M + 1);
        
        for (int i = 0; i <= N; ++i) {
            for (int j = 0; j <= M; ++j) {
                X.data[i][j] = x_vals[j];
                Y.data[i][j] = y_vals[i];
                mask.data[i][j] = in_trapezoid(X.data[i][j], Y.data[i][j]);
                border_mask.data[i][j] = on_border(X.data[i][j], Y.data[i][j]);
            }
        }
        
        double hx = x_vals[1] - x_vals[0];
        double hy = y_vals[1] - y_vals[0];
        
        Grid2D k_coeffs(N + 1, M + 1), F_rhs(N + 1, M + 1);
        Grid2D a_coeffs(N + 1, M + 1), b_coeffs(N + 1, M + 1);
        
        setup_fictitious_domain(X, Y, mask, k_coeffs, F_rhs, a_coeffs, b_coeffs, hx, hy);
        
        Grid2D U(N + 1, M + 1);
        for (int i = 0; i <= N; ++i) {
            for (int j = 0; j <= M; ++j) {
                if (border_mask.data[i][j]) {
                    U.data[i][j] = g(X.data[i][j], Y.data[i][j]);
                }
            }
        }
        
        
        struct timeval start_time, end_time;
        gettimeofday(&start_time, NULL);
        std::pair<Grid2D, int> result = conjugate_gradient(U, F_rhs, mask, border_mask, 
                                           a_coeffs, b_coeffs, hx, hy, (N - 1) * (M - 1), 1e-8);
        gettimeofday(&end_time, NULL);
        double time = (end_time.tv_sec - start_time.tv_sec) +  (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
        
        std::string filename = "solution_" + int_to_string(M) + "x" + int_to_string(N) + ".csv";
        save_solution_csv(X, Y, result.first, mask, filename);
        std::cout << "Результат сохранен в: " << filename << std::endl;
            std::string filename_txt = "solution_" + int_to_string(N) + "_" + int_to_string(M) + ".txt";
        std::ofstream outfile(filename_txt.c_str());  // Используем .c_str() для преобразования string в const char*
        if (outfile.is_open()) {
            outfile << "Grid_Size: " << M << "x" << N << "\n";
            outfile << "Iterations: " << result.second << "\n";
            outfile << "Time(s): " << time << "\n";
            outfile << "Iterations_per_second: " << result.second / time << "\n";
            outfile.close();
        } else {
            std::cerr << "Error: Cannot open file " << filename_txt << " for writing!" << std::endl;
        }
    }
    return 0;
}

