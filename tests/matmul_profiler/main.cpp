#include "lib.h"

#include "cstdlib"
#include "cstdio"
#include "cstring"
#include "ctime"

// extern void static_matmul(float* x, float* y, float* out, int m, int n, int k);


void profile_matmul(int m, int n, int k) {
    float* a = (float*) malloc((m * k) * sizeof(float));
    float* b = (float*) malloc((n * k) * sizeof(float));
    float* out = (float*) malloc((m * n) * sizeof(float));

    for (int i = 0; i < m*k; i++)
        a[i] = ((float)rand()/(float)(RAND_MAX));

    for (int i = 0; i < n*k; i++)
        b[i] = ((float)rand()/(float)(RAND_MAX));

    memset(out, 0, m*n*sizeof(float));

    clock_t start = clock();
    static_matmul(a, b, out, m, n, k);
    clock_t end = clock();

    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("    Time taken: %f seconds\n", duration);
}

int main () {
    srand(0);
    printf("Initiating\n");
    init();

    printf("Testing (lm_head)...\n");
    profile_matmul(1, 32000, 4096);

    printf("Testing (Q, K, V, out projections)...\n");
    profile_matmul(1, 4096, 4096);

    printf("Testing (down_proj)...\n");
    profile_matmul(1, 4096, 11008);

    printf("Testing (up_proj and gate_proj)...\n");
    profile_matmul(1, 11008, 4096);

}