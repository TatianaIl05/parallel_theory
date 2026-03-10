#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <fstream>

using namespace std;

double wtime() {
    return omp_get_wtime();
}

void init_system(vector<double>& A, vector<double>& b, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (i == j) ? 2.0 : 1.0;
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        b[i] = n + 1;
    }
}

int solve_serial(const vector<double>& A, const vector<double>& b, 
                 vector<double>& x, int n, double tau, double eps, int max_iter) {
    int iter = 0;
    double norm;
    vector<double> x_old(n);
    
    do {
        x_old = x;
        norm = 0.0;
        
        for (int i = 0; i < n; i++) {
            double Ax = 0.0;
            for (int j = 0; j < n; j++) {
                Ax += A[i * n + j] * x_old[j];
            }
            double r = b[i] - Ax;
            x[i] = x_old[i] + tau * r;
            norm += r * r;
        }
        norm = sqrt(norm);
        iter++;
    } while (iter < max_iter && norm > eps);
    
    return iter;
}

int solve_v1(const vector<double>& A, const vector<double>& b, 
             vector<double>& x, int n, double tau, double eps, int max_iter, 
             int num_threads) {
    int iter = 0;
    double norm;
    vector<double> x_old(n);
    
    omp_set_num_threads(num_threads);
    
    do {
        x_old = x;
        
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double Ax = 0.0;
            for (int j = 0; j < n; j++) {
                Ax += A[i * n + j] * x_old[j];
            }
            double r = b[i] - Ax;
            x[i] = x_old[i] + tau * r;
        }
        
        norm = 0.0;
        #pragma omp parallel for reduction(+:norm)
        for (int i = 0; i < n; i++) {
            double Ax = 0.0;
            for (int j = 0; j < n; j++) {
                Ax += A[i * n + j] * x[j];
            }
            double r = b[i] - Ax;
            norm += r * r;
        }
        norm = sqrt(norm);
        
        iter++;
    } while (iter < max_iter && norm > eps);
    
    return iter;
}

int solve_v2(const vector<double>& A, const vector<double>& b, 
             vector<double>& x, int n, double tau, double eps, int max_iter, 
             int num_threads) {
    int iter = 0;
    double norm;
    vector<double> x_old(n);
    
    omp_set_num_threads(num_threads);
    
    do {
        x_old = x;
        norm = 0.0;
        
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < n; i++) {
                double Ax = 0.0;
                for (int j = 0; j < n; j++) {
                    Ax += A[i * n + j] * x_old[j];
                }
                double r = b[i] - Ax;
                x[i] = x_old[i] + tau * r;
            }
            
            #pragma omp barrier
            
            double norm_local = 0.0;
            #pragma omp for
            for (int i = 0; i < n; i++) {
                double Ax = 0.0;
                for (int j = 0; j < n; j++) {
                    Ax += A[i * n + j] * x[j];
                }
                double r = b[i] - Ax;
                norm_local += r * r;
            }
            
            #pragma omp atomic
            norm += norm_local;
        }
        
        norm = sqrt(norm);
        iter++;
    } while (iter < max_iter && norm > eps);
    
    return iter;
}

double check_solution(const vector<double>& x, int n) {
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double err = fabs(x[i] - 1.0);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main() {
    int n = 18000;
    double tau = 0.01;
    double eps = 1e-5;
    int max_iter = 50000;
    
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int num_threads = sizeof(threads) / sizeof(threads[0]);
    
    cout << "n = " << n << "\n";
    cout << "Память: " << (n * n * sizeof(double)) / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    cout << "tau = " << tau << ", eps = " << eps << "\n";
    cout << "max_iter = " << max_iter << "\n";
    
    vector<double> A(n * n);
    vector<double> b(n);
    init_system(A, b, n);
    
    cout << "Последовательная версия\n";
    vector<double> x_seq(n, 0.0);
    double t1 = wtime();
    int iter_seq = solve_serial(A, b, x_seq, n, tau, eps, max_iter);
    double time_seq = wtime() - t1;
    double err_seq = check_solution(x_seq, n);
    
    cout << "Итераций: " << iter_seq << "\n";
    cout << "Время: " << fixed << setprecision(2) << time_seq << " сек\n";
    cout << "Ошибка: " << scientific << err_seq << "\n\n";
    
    cout << "Потоки | Вариант 1 (parallel for)        | Вариант 2 (parallel)\n";
    cout << "       | время (сек) ускорение итерации  | время (сек) ускорение итерации\n";

    
    for (int t = 0; t < num_threads; t++) {
        int thr = threads[t];
        
        vector<double> x1(n, 0.0);
        double t1_start = wtime();
        int iter1 = solve_v1(A, b, x1, n, tau, eps, max_iter, thr);
        double time1 = wtime() - t1_start;
        double speedup1 = time_seq / time1;
        double eff1 = speedup1 / thr * 100;
        
        vector<double> x2(n, 0.0);
        double t2_start = wtime();
        int iter2 = solve_v2(A, b, x2, n, tau, eps, max_iter, thr);
        double time2 = wtime() - t2_start;
        double speedup2 = time_seq / time2;
        double eff2 = speedup2 / thr * 100;
        
        cout << setw(4) << thr << "   | "
             << fixed << setprecision(2) << setw(8) << time1 << "  "
             << setprecision(2) << setw(8) << speedup1 << "x "
             << setw(6) << iter1 << "   | "
             << setprecision(2) << setw(8) << time2 << "  "
             << setprecision(2) << setw(8) << speedup2 << "x "
             << setw(6) << iter2 << "\n";
        
    }

    return 0;
}
