#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>

const double a = -4.0;
const double b = 4.0;

double wtime() {
    return omp_get_wtime();
}

double func(double x) {
    return exp(-x * x);
}

double integrate_omp(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        
        double sumloc = 0.0;
        for (int i = lb; i <= ub; i++) {
            sumloc += func(a + h * (i + 0.5));
        }
        
        #pragma omp atomic
        sum += sumloc;
    }
    
    sum *= h;
    return sum;
}

double integrate_sequential(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    
    for (int i = 0; i < n; i++) {
        sum += func(a + h * (i + 0.5));
    }
    
    return sum * h;
}

int main() {
    int nsteps = 40000000;  
    int threads[] = {2, 4, 7, 8, 16, 20, 40};
    int num_threads = sizeof(threads) / sizeof(threads[0]);
    
    double seq_time = 0.0;
    double par_times[num_threads];
    double speedups[num_threads];
    
    printf("Последовательное выполнение\n");
    omp_set_num_threads(1);
    
    double t_start = wtime();
    double result_seq = integrate_sequential(func, a, b, nsteps);
    seq_time = wtime() - t_start;
    
    printf("Результат: %.10f\n", result_seq);
    printf("Время: %.6f сек\n\n", seq_time);
    
    printf("Параллельное выполнение\n");
    printf("%-10s %-20s %-20s %-20s\n", 
           "Потоки", "Результат", "Время (сек)", "Ускорение");
    
    for (int t = 0; t < num_threads; t++) {
        omp_set_num_threads(threads[t]);
        
        t_start = wtime();
        double result_par = integrate_omp(func, a, b, nsteps);
        par_times[t] = wtime() - t_start;
        speedups[t] = seq_time / par_times[t];
        
        printf("%-10d %-30.20f %-20.6f %-10.3f\n", threads[t], result_par, par_times[t], speedups[t]);
        
    }
    
    return 0;
}
