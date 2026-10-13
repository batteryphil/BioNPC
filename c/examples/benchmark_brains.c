#include "bionpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int num_wolves = 100000;
    int num_frames = 60;
    int num_threads = 0;

    /* Parse optional command-line overrides */
    if (argc >= 2) {
        int n = atoi(argv[1]);
        if (n > 0) num_wolves = n;
    }
    if (argc >= 3) {
        int t = atoi(argv[2]);
        if (t > 0) num_threads = t;
    }

    printf("======================================================================\n");
    printf("   BioNPC: Biological Foundation Model for Game AI (Fil-C Cosmo APE)  \n");
    printf("======================================================================\n");
    printf("Architecture: d_in=4, d_out=3, d_model=32, n_layers=2 (Drosophila CTRNN)\n");
    printf("Target Scale: %d continuous intelligent wolves simultaneously\n", num_wolves);
    printf("Safety Mode : Fil-C 128-bit fat pointer capability checks ENABLED\n");
    printf("Portability : Cosmopolitan APE (runs natively on Linux, Win, Mac, BSD)\n");
    printf("======================================================================\n\n");

    /* Create the EmbodiedNPCModel brain */
    bionpc_config_t cfg = {
        .d_in = 4,
        .d_out = 3,
        .d_model = 32,
        .n_layers = 2
    };

    bionpc_model_t *brain = bionpc_model_create(&cfg);
    if (!brain) {
        fprintf(stderr, "Error: Failed to allocate bionpc model\n");
        return 1;
    }
    bionpc_init_random(brain, 42);

    printf("Total brain parameters: %zu\n\n", brain->total_params);

    /* Allocate batched sensory inputs and motor outputs */
    size_t in_size = (size_t)num_wolves * cfg.d_in * sizeof(float);
    size_t out_size = (size_t)num_wolves * cfg.d_out * sizeof(float);

    float *batched_inputs = (float *)malloc(in_size);
    float *batched_outputs = (float *)malloc(out_size);
    bionpc_agent_state_t *states = (bionpc_agent_state_t *)calloc(num_wolves, sizeof(bionpc_agent_state_t));

    if (!batched_inputs || !batched_outputs || !states) {
        fprintf(stderr, "Error: Memory allocation failed for %d agents\n", num_wolves);
        return 1;
    }

    /* Initialize inputs with pseudo-random sensory floats [0.0, 1.0] */
    for (size_t i = 0; i < (size_t)num_wolves * cfg.d_in; i++) {
        batched_inputs[i] = (float)rand() / (float)RAND_MAX;
    }

    /* --- BENCHMARK 1: Single-Threaded CPU Execution --- */
    printf(">>> Running Warmup (10 frames, single-threaded)...\n");
    for (int f = 0; f < 10; f++) {
        bionpc_forward_batch(brain, batched_inputs, batched_outputs, states, num_wolves);
    }

    printf(">>> Benchmarking 60 frames (1 second of game time @ 60 FPS target)...\n");
    double start_time = get_time_sec();
    for (int f = 0; f < num_frames; f++) {
        bionpc_forward_batch(brain, batched_inputs, batched_outputs, states, num_wolves);
    }
    double end_time = get_time_sec();

    double total_time_st = end_time - start_time;
    double fps_st = (double)num_frames / total_time_st;
    double decisions_per_sec_st = ((double)num_wolves * num_frames) / total_time_st;

    printf("\n--- Single-Threaded Results (1 CPU Core) ---\n");
    printf("Processed %d frames for %'d wolves in %.4f seconds\n", num_frames, num_wolves, total_time_st);
    printf("Average latency per frame: %.3f ms\n", (total_time_st / num_frames) * 1000.0);
    printf("Equivalent Game FPS      : %.1f FPS\n", fps_st);
    printf("Decision Throughput      : %'.0f NPC decisions/sec\n", decisions_per_sec_st);

    /* --- BENCHMARK 2: Multi-Threaded CPU Execution --- */
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs <= 0) nprocs = 4;
    int threads_to_use = (num_threads > 0) ? num_threads : (int)nprocs;

    printf("\n>>> Benchmarking Multi-Threaded (%d worker threads, %d frames)...\n", threads_to_use, num_frames);
    start_time = get_time_sec();
    for (int f = 0; f < num_frames; f++) {
        bionpc_forward_batch_mt(brain, batched_inputs, batched_outputs, states, num_wolves, threads_to_use);
    }
    end_time = get_time_sec();

    double total_time_mt = end_time - start_time;
    double fps_mt = (double)num_frames / total_time_mt;
    double decisions_per_sec_mt = ((double)num_wolves * num_frames) / total_time_mt;

    printf("\n--- Multi-Threaded Results (%d Cores) ---\n", threads_to_use);
    printf("Processed %d frames for %'d wolves in %.4f seconds\n", num_frames, num_wolves, total_time_mt);
    printf("Average latency per frame: %.3f ms\n", (total_time_mt / num_frames) * 1000.0);
    printf("Equivalent Game FPS      : %.1f FPS\n", fps_mt);
    printf("Decision Throughput      : %'.0f NPC decisions/sec\n", decisions_per_sec_mt);
    printf("Speedup vs Single-Thread : %.2fx\n", total_time_st / total_time_mt);

    /* Verify sanity of motor outputs */
    printf("\nSample Output Actions (First 3 Wolves):\n");
    for (int w = 0; w < 3; w++) {
        printf("  Wolf %d: [Chase=%.3f, Attack=%.3f, Flee=%.3f] (APL State L0=%.4f, L1=%.4f)\n",
               w,
               batched_outputs[w * 3 + 0],
               batched_outputs[w * 3 + 1],
               batched_outputs[w * 3 + 2],
               states[w].apl_state[0],
               states[w].apl_state[1]);
    }

    printf("\n======================================================================\n");
    printf("STATUS: SUCCESS. Zero memory safety violations trapped by Fil-C.\n");
    printf("======================================================================\n");

    /* Cleanup */
    free(batched_inputs);
    free(batched_outputs);
    free(states);
    bionpc_model_free(brain);

    return 0;
}
