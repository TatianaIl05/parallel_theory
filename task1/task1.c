#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>

double wtime() {
    return omp_get_wtime();
}

void* xmalloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    return ptr;
}

void parallel_initialize(double *a, double *b, int m, int n) {
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = i + j;
            }
        }
        
        #pragma omp for
        for (int j = 0; j < n; j++) {
            b[j] = j;
        }
    }
}

void matrix_vector_product_sequential(double *a, double *b, double *c, int m, int n) {
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++) {
            c[i] += a[i * n + j] * b[j];
        }
    }
}

void matrix_vector_product_parallel(double *a, double *b, double *c, int m, int n) {
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++) {
            c[i] += a[i * n + j] * b[j];
        }
    }
}

int main() {
    int sizes[] = {20000}; 
    int threads[] = {1, 2, 4, 7, 8, 16, 20, 40};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_threads = sizeof(threads) / sizeof(threads[0]);
    
    for (int s = 0; s < num_sizes; s++) {
        int m = sizes[s];
        int n = sizes[s];
        
        printf("Матрица %dx%d\n", m, n);

        double *a, *b, *c_seq, *c_par;
        
        a = (double*)xmalloc(sizeof(double) * m * n);
        b = (double*)xmalloc(sizeof(double) * n);
        c_seq = (double*)xmalloc(sizeof(double) * m);
        c_par = (double*)xmalloc(sizeof(double) * m);
        
        double mem_size = ((double)(m * n + m + n) * sizeof(double)) / (1024 * 1024 * 1024);
        printf("Используется памяти: %.2f GB\n\n", mem_size);
        
        
        omp_set_num_threads(8);  
        parallel_initialize(a, b, m, n);
        
        omp_set_num_threads(1);
        double t_start = wtime();
        matrix_vector_product_sequential(a, b, c_seq, m, n);
        double seq_time = wtime() - t_start;
        printf("Время выполнения: %.6f сек\n\n", seq_time);
        
        printf("%-10s %-20s %-20s\n", "Потоки", "Время (сек)", "Ускорение");

        for (int t = 0; t < num_threads; t++) {
            omp_set_num_threads(threads[t]);
            
            t_start = wtime();
            matrix_vector_product_parallel(a, b, c_par, m, n);
            double par_time = wtime() - t_start;
            
            double speedup = seq_time / par_time;
            
            printf("%-10d %-20.6f %-20.3f\n", threads[t], par_time, speedup);
        }
        
        free(a);
        free(b);
        free(c_seq);
        free(c_par);
    }
    
    return 0;
}
