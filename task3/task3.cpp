#include <vector>
#include <cmath>
#include <iostream> 
#include <time.h>
#include <omp.h>
#include <cstring>

#define MAX_ITER 100000
#define EPSILON 1e-6

using namespace std;

double cpuSecond()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

void no_parallel(vector<double>& A, vector<double>& b, 
                 vector<double>& x, int N, double tau) { 

    int iter = 0;
    double error = 1.0;
    vector<double> r(N);

    while (iter < MAX_ITER && error > EPSILON) {
        double error_sum = 0.0;

        for (int i = 0; i < N; i++) {
            double Ax = 0.0;
            for (int j = 0; j < N; j++)
                Ax += A[i * N + j] * x[j];
            r[i] = b[i] - Ax;
            error_sum += r[i] * r[i];
        }

        error = sqrt(error_sum);

        for (int i = 0; i < N; i++)
            x[i] += tau * r[i];

        iter++;
    }               
}

void run_no_parallel(vector<double>& A, vector<double>& b, 
                  vector<double>& x, int N, double tau) {

    fill(x.begin(), x.end(), 0.0);

    double t = cpuSecond();
    no_parallel(A, b, x, N, tau);
    t = cpuSecond() - t;

    cout << "Serial: " << t << endl;
}

void first_method(vector<double>& A, vector<double>& b, 
                  vector<double>& x, int N, double tau) {

    int iter = 0;
    double error = 1.0;
    vector<double> r(N);

    while (iter < MAX_ITER && error > EPSILON) {
        double error_sum = 0.0;

        #pragma omp parallel for reduction(+:error_sum)
        for (int i = 0; i < N; i++) {
            double Ax = 0.0;

            for (int j = 0; j < N; j++)
                Ax += A[i * N + j] * x[j];

            r[i] = b[i] - Ax;
            error_sum += r[i] * r[i];
        }

        error = sqrt(error_sum);

        #pragma omp parallel for
        for (int i = 0; i < N; i++)
            x[i] += tau * r[i];

        iter++;
    }
}

void first_method_schedule(vector<double>& A, vector<double>& b, 
                  vector<double>& x, int N, double tau) {

    int iter = 0;
    double error = 1.0;
    vector<double> r(N);

    while (iter < MAX_ITER && error > EPSILON) {
        double error_sum = 0.0;

        #pragma omp parallel for schedule(runtime) reduction(+:error_sum)
        for (int i = 0; i < N; i++) {
            double Ax = 0.0;

            for (int j = 0; j < N; j++)
                Ax += A[i * N + j] * x[j];

            r[i] = b[i] - Ax;
            error_sum += r[i] * r[i];
        }

        error = sqrt(error_sum);

        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < N; i++)
            x[i] += tau * r[i];

        iter++;
    }
}

void run_first_method(vector<double>& A, vector<double>& b, 
                  vector<double>& x, int N, double tau) {

    fill(x.begin(), x.end(), 0.0);

    double t = cpuSecond();
    first_method(A, b, x, N, tau);
    t = cpuSecond() - t;

    cout << "First method: " << t << endl;
}

void run_first_method_schedule(vector<double>& A, vector<double>& b, 
                  vector<double>& x, int N, double tau, const char* sched_type) {
            
    omp_sched_t kind;
    int chunk = 0;
    char sched_name[32];
    sscanf(sched_type, "%31[^,],%d", sched_name, &chunk);

    if (strcmp(sched_name, "static") == 0) kind = omp_sched_static;
    else if (strcmp(sched_name, "dynamic") == 0) kind = omp_sched_dynamic;
    else if (strcmp(sched_name, "guided") == 0) kind = omp_sched_guided;
    else kind = omp_sched_static; 

    omp_set_schedule(kind, chunk); 

    fill(x.begin(), x.end(), 0.0);

    double t = cpuSecond();
    first_method_schedule(A, b, x, N, tau);
    t = cpuSecond() - t;

    cout << "First method with schedule: " << t << endl;
}

void second_method(vector<double>& A, vector<double>& b, 
                         vector<double>& x, int N, double tau) {
    int iter = 0;
    double error = 1.0;
    vector<double> r(N);
    double total_sum = 0.0;
    int flag = 1;  
    
    #pragma omp parallel
    {
        while (1) {
            #pragma omp flush(flag)
            if (!flag) break;
            #pragma omp single
            {
                total_sum = 0.0;
            }
            #pragma omp for
            for (int i = 0; i < N; i++) {
                double Ax = 0.0;
                for (int j = 0; j < N; j++)
                    Ax += A[i * N + j] * x[j];
                r[i] = b[i] - Ax;
                #pragma omp atomic
                total_sum += r[i] * r[i];
            }
            
            #pragma omp for
            for (int i = 0; i < N; i++)
                x[i] += tau * r[i];
            
            #pragma omp single
            {
                error = sqrt(total_sum);
                iter++;
                if (iter >= MAX_ITER || error <= EPSILON)
                    flag = 0;
            }
        }
    }
}

void second_method_schedule(vector<double>& A, vector<double>& b, 
                         vector<double>& x, int N, double tau) {
    int iter = 0;
    double error = 1.0;
    vector<double> r(N);
    double total_sum = 0.0;
    int flag = 1;  
    
    #pragma omp parallel
    {
        while (1) {
            #pragma omp flush(flag)
            if (!flag) break;
            #pragma omp single
            {
                total_sum = 0.0;
            }
            #pragma omp for schedule(runtime)
            for (int i = 0; i < N; i++) {
                double Ax = 0.0;
                for (int j = 0; j < N; j++)
                    Ax += A[i * N + j] * x[j];
                r[i] = b[i] - Ax;
                #pragma omp atomic
                total_sum += r[i] * r[i];
            }
            
            #pragma omp for
            for (int i = 0; i < N; i++)
                x[i] += tau * r[i];
            
            #pragma omp single
            {
                error = sqrt(total_sum);
                iter++;
                if (iter >= MAX_ITER || error <= EPSILON)
                    flag = 0;
            }
        }
    }
}

void run_second_method(vector<double>& A, vector<double>& b, 
                  vector<double>& x, int N, double tau) {

    fill(x.begin(), x.end(), 0.0);
    
    double t = cpuSecond();
    second_method(A, b, x, N, tau);
    t = cpuSecond() - t;

    cout << "Second method: " << t << endl;
}

void run_second_method_schedule(vector<double>& A, vector<double>& b, 
                  vector<double>& x, int N, double tau, const char* sched_type) {

    omp_sched_t kind;
    int chunk = 0;
    char sched_name[32];
    sscanf(sched_type, "%31[^,],%d", sched_name, &chunk);

    if (strcmp(sched_name, "static") == 0) kind = omp_sched_static;
    else if (strcmp(sched_name, "dynamic") == 0) kind = omp_sched_dynamic;
    else if (strcmp(sched_name, "guided") == 0) kind = omp_sched_guided;
    else kind = omp_sched_static; 

    omp_set_schedule(kind, chunk); 

    fill(x.begin(), x.end(), 0.0);

    double t = cpuSecond();
    second_method_schedule(A, b, x, N, tau);
    t = cpuSecond() - t;

    cout << "Second method with schedule: " << t << endl;
}

void check_result(vector<double>& x, int N) {
    double error = 0.0;
    for (int i = 0; i < N; i++) {
        error += fabs(x[i] - 1.0);
    }
    cout << "Average error: " << error / N << endl;
}

int main() {

    int N = 10000;
    double tau = 0.00001;

    vector<double> A(N * N, 1.0);
    vector<double> b(N, N + 1.0);
    vector<double> x(N, 0.0);

    for (int i = 0; i < N; i++) 
        A[i * N + i] = 2.0;

    const char* schedules[] = {
        "static",
        "static,64",
        "dynamic",
        "dynamic,8",
        "guided"
    };

    vector<int> threads = {1, 2, 4, 6, 8, 16, 20};

    run_no_parallel(A, b, x, N, tau);

    for(int num: threads) {
        cout << "\n" << num << " threads:" << endl;

        omp_set_num_threads(num);
        run_first_method(A, b, x, N, tau);
        check_result(x, N);
        
        run_second_method(A, b, x, N, tau);
        check_result(x, N);
        
        for(const char* sched : schedules) {
            run_first_method_schedule(A, b, x, N, tau, sched);
        }

    }

    return 0;
}
