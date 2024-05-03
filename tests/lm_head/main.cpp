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
    // float* out = (float*) malloc((m * k) * sizeof(float));

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

    free(a);
    free(b);
    free(out);
}

int main () {
    srand(0);
    printf("Initiating\n");
    init();

    for (int i = 0; i < 5; i++) {
        printf("[%d] Testing (lm_head)...\n", i+1);
        profile_matmul(1, 32000, 4096);
        printf("[%d] Testing (Q, K, V, out projections)...\n", i+1);
        profile_matmul(1, 4096, 4096);
        printf("[%d] Testing (down_proj)...\n", i+1);
        profile_matmul(1, 4096, 11008);
        printf("[%d] Testing (up_proj and gate_proj)...\n", i+1);
        profile_matmul(1, 11008, 4096);
    }

}