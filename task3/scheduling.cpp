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

double test_schedule_v1(const vector<double>& A, const vector<double>& b, int n, int num_threads, const string& schedule) {
    
    vector<double> x(n, 0.0);
    vector<double> x_old(n);
    double tau = 0.01;
    double eps = 1e-5;
    int max_iter = 50000;
    double norm;
    int iter = 0;
    
    omp_set_num_threads(num_threads);
    
    double start = wtime();
    
    do {
        x_old = x;
        
        if (schedule == "static,10") {
            #pragma omp parallel for schedule(static, 10)
            for (int i = 0; i < n; i++) {
                double Ax = 0.0;
                for (int j = 0; j < n; j++) {
                    Ax += A[i * n + j] * x_old[j];
                }
                x[i] = x_old[i] + tau * (b[i] - Ax);
            }
        }
        else if (schedule == "dynamic,10") {
            #pragma omp parallel for schedule(dynamic, 10)
            for (int i = 0; i < n; i++) {
                double Ax = 0.0;
                for (int j = 0; j < n; j++) {
                    Ax += A[i * n + j] * x_old[j];
                }
                x[i] = x_old[i] + tau * (b[i] - Ax);
            }
        }
        else if (schedule == "guided,10") {
            #pragma omp parallel for schedule(guided, 10)
            for (int i = 0; i < n; i++) {
                double Ax = 0.0;
                for (int j = 0; j < n; j++) {
                    Ax += A[i * n + j] * x_old[j];
                }
                x[i] = x_old[i] + tau * (b[i] - Ax);
            }
        }
        else if (schedule == "runtime") {
            #pragma omp parallel for schedule(runtime)
            for (int i = 0; i < n; i++) {
                double Ax = 0.0;
                for (int j = 0; j < n; j++) {
                    Ax += A[i * n + j] * x_old[j];
                }
                x[i] = x_old[i] + tau * (b[i] - Ax);
            }
        }
        else { 
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                double Ax = 0.0;
                for (int j = 0; j < n; j++) {
                    Ax += A[i * n + j] * x_old[j];
                }
                x[i] = x_old[i] + tau * (b[i] - Ax);
            }
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
    
    return wtime() - start;
}

double test_schedule_v2(const vector<double>& A, const vector<double>& b, int n, int num_threads, const string& schedule) {
    
    vector<double> x(n, 0.0);
    vector<double> x_old(n);
    double tau = 0.01;
    double eps = 1e-5;
    int max_iter = 50000;
    double norm;
    int iter = 0;
    
    omp_set_num_threads(num_threads);
    
    double start = wtime();
    
    do {
        x_old = x;
        norm = 0.0;
        
        #pragma omp parallel
        {
            if (schedule == "static,10") {
                #pragma omp for schedule(static, 10)
                for (int i = 0; i < n; i++) {
                    double Ax = 0.0;
                    for (int j = 0; j < n; j++) {
                        Ax += A[i * n + j] * x_old[j];
                    }
                    x[i] = x_old[i] + tau * (b[i] - Ax);
                }
            }
            else if (schedule == "dynamic,10") {
                #pragma omp for schedule(dynamic, 10)
                for (int i = 0; i < n; i++) {
                    double Ax = 0.0;
                    for (int j = 0; j < n; j++) {
                        Ax += A[i * n + j] * x_old[j];
                    }
                    x[i] = x_old[i] + tau * (b[i] - Ax);
                }
            }
            else if (schedule == "guided,10") {
                #pragma omp for schedule(guided, 10)
                for (int i = 0; i < n; i++) {
                    double Ax = 0.0;
                    for (int j = 0; j < n; j++) {
                        Ax += A[i * n + j] * x_old[j];
                    }
                    x[i] = x_old[i] + tau * (b[i] - Ax);
                }
            }
            else if (schedule == "runtime") {
                #pragma omp for schedule(runtime)
                for (int i = 0; i < n; i++) {
                    double Ax = 0.0;
                    for (int j = 0; j < n; j++) {
                        Ax += A[i * n + j] * x_old[j];
                    }
                    x[i] = x_old[i] + tau * (b[i] - Ax);
                }
            }
            else { 
                #pragma omp for
                for (int i = 0; i < n; i++) {
                    double Ax = 0.0;
                    for (int j = 0; j < n; j++) {
                        Ax += A[i * n + j] * x_old[j];
                    }
                    x[i] = x_old[i] + tau * (b[i] - Ax);
                }
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
    
    return wtime() - start;
}

int main() {
    int n = 18000;
    int num_threads = 8; 
    
    cout << "Schedule\n";
    cout << "n = " << n << ", потоки = " << num_threads << "\n";
   
    vector<double> A(n * n);
    vector<double> b(n);
    init_system(A, b, n);
    
    vector<string> schedules = {"static,10", "dynamic,10", "guided,10", "runtime"};
    
    cout << "Var 1:\n";
    cout << setw(15) << "Schedule" << " | " << setw(12) << "Время (сек)" << "\n";
    
    for (const auto& s : schedules) {
        double time = test_schedule_v1(A, b, n, num_threads, s);
        cout << setw(15) << s << " | " << fixed << setprecision(2) << setw(12) << time << "\n";
    }
    
    cout << "\nVar 2:\n";
    cout << setw(15) << "Schedule" << " | " << setw(12) << "Время (сек)" << "\n";
    
    for (const auto& s : schedules) {
        double time = test_schedule_v2(A, b, n, num_threads, s);
        cout << setw(15) << s << " | " << fixed << setprecision(2) << setw(12) << time << "\n";
    }
    
    return 0;
}
