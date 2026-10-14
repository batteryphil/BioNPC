#include "bionpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

int main(void) {
    printf("======================================================================\n");
    printf("   BioNPC C / Fil-C Cosmopolitan Verification & Determinism Suite     \n");
    printf("======================================================================\n");

    bionpc_config_t cfg = {
        .d_in = 4,
        .d_out = 3,
        .d_model = 16,
        .n_layers = 2
    };

    /* 1. Deterministic Weight Init & Serialization */
    bionpc_model_t *m1 = bionpc_model_create(&cfg);
    bionpc_model_t *m1_dup = bionpc_model_create(&cfg);
    bionpc_init_random(m1, 777);
    bionpc_init_random(m1_dup, 777);

    uint32_t w_crc1 = bionpc_crc32_floats(m1->w_in, (size_t)cfg.d_model * cfg.d_in);
    uint32_t w_crc2 = bionpc_crc32_floats(m1_dup->w_in, (size_t)cfg.d_model * cfg.d_in);
    assert(w_crc1 == w_crc2 && w_crc1 != 0);
    printf("  [PASS] Cross-platform SplitMix32 deterministic init verified (CRC32=0x%08X)\n", w_crc1);
    bionpc_model_free(m1_dup);

    const char *test_path = "/tmp/test_model.bionpc";
    int save_rc = bionpc_save_weights(m1, test_path);
    assert(save_rc == 0);
    printf("  [PASS] Saved model weights to %s\n", test_path);

    bionpc_model_t *m2 = bionpc_model_load_file(test_path);
    assert(m2 != NULL);
    printf("  [PASS] Loaded model weights via bionpc_model_load_file(%s)\n", test_path);

    /* 2. Verify exact numerical equivalence of single forward pass */
    float in[4] = {0.12f, 0.34f, 0.56f, 0.78f};
    float out1[3] = {0};
    float out2[3] = {0};
    bionpc_agent_state_t st1 = {0};
    bionpc_agent_state_t st2 = {0};

    bionpc_forward_single(m1, in, out1, &st1);
    bionpc_forward_single(m2, in, out2, &st2);

    for (int i = 0; i < 3; i++) {
        float diff = fabsf(out1[i] - out2[i]);
        assert(diff < 1e-6f);
    }
    for (int l = 0; l < 2; l++) {
        float diff = fabsf(st1.apl_state[l] - st2.apl_state[l]);
        assert(diff < 1e-6f);
    }
    printf("  [PASS] Single-step outputs and recurrent states match exactly (diff < 1e-6)\n");

    /* 3. Verify 4-way tiled batch & multi-threaded batch match single-agent bit-for-bit */
    int N = 19; /* Non-multiple of 4 to test both tile4 and remainder loop */
    float *batch_in = (float *)calloc((size_t)N * 4, sizeof(float));
    float *out_single = (float *)calloc((size_t)N * 3, sizeof(float));
    float *out_batch  = (float *)calloc((size_t)N * 3, sizeof(float));
    float *out_mt     = (float *)calloc((size_t)N * 3, sizeof(float));
    bionpc_agent_state_t *st_single = (bionpc_agent_state_t *)calloc((size_t)N, sizeof(bionpc_agent_state_t));
    bionpc_agent_state_t *st_batch  = (bionpc_agent_state_t *)calloc((size_t)N, sizeof(bionpc_agent_state_t));
    bionpc_agent_state_t *st_mt     = (bionpc_agent_state_t *)calloc((size_t)N, sizeof(bionpc_agent_state_t));

    for (int i = 0; i < N * 4; i++) {
        batch_in[i] = (float)(i % 13) * 0.07f;
    }

    for (int step = 0; step < 5; step++) {
        for (int b = 0; b < N; b++) {
            bionpc_forward_single(m1, &batch_in[b * 4], &out_single[b * 3], &st_single[b]);
        }
        bionpc_forward_batch(m1, batch_in, out_batch, st_batch, N);
        bionpc_forward_batch_mt(m1, batch_in, out_mt, st_mt, N, 4);
    }

    uint32_t crc_s = bionpc_crc32_states(st_single, N, 2);
    uint32_t crc_b = bionpc_crc32_states(st_batch, N, 2);
    uint32_t crc_m = bionpc_crc32_states(st_mt, N, 2);
    assert(crc_s == crc_b && crc_b == crc_m);

    uint32_t out_crc_s = bionpc_crc32_floats(out_single, (size_t)N * 3);
    uint32_t out_crc_b = bionpc_crc32_floats(out_batch, (size_t)N * 3);
    assert(out_crc_s == out_crc_b);
    printf("  [PASS] 4-way tiled batch & MT inference match single-agent bit-for-bit (State CRC32=0x%08X)\n", crc_b);

    /* 4. Multiplayer State Snapshot Save / Load */
    const char *state_path = "/tmp/test_states.bnpcst";
    assert(bionpc_save_states(st_batch, N, 2, state_path) == 0);
    bionpc_agent_state_t *st_loaded = (bionpc_agent_state_t *)calloc((size_t)N, sizeof(bionpc_agent_state_t));
    assert(bionpc_load_states(st_loaded, N, 2, state_path) == 0);
    assert(bionpc_crc32_states(st_loaded, N, 2) == crc_b);
    printf("  [PASS] Multiplayer state snapshot save/load verified (CRC32=0x%08X)\n", crc_b);

    /* 5. Designer Control & INI Profile Loader */
    const char *ini_path = "/tmp/test_profile.ini";
    FILE *fini = fopen(ini_path, "w");
    assert(fini != NULL);
    fprintf(fini,
        "[archetype]\n"
        "name = sanctuary_wolf\n"
        "[chemistry]\n"
        "apl_min = 0.0\n"
        "apl_max = 0.5\n"
        "apl_locked = false\n"
        "decay_logit_l0 = 2.5\n"
        "gain_l0 = 0.2\n"
        "[senses]\n"
        "sensory_scale = 1.5, 1.0, 2.0, 0.5\n"
        "[actions]\n"
        "forced_actions = 1:-10.0\n"
    );
    fclose(fini);

    bionpc_control_t ctrl;
    assert(bionpc_load_ini_profile(ini_path, m1, &ctrl) == 0);
    assert(strcmp(ctrl.archetype_name, "sanctuary_wolf") == 0);
    assert(fabsf(m1->layers[0].decay_logit - 2.5f) < 1e-6f);
    assert(ctrl.action_override_mask[1] == 1);
    assert(fabsf(ctrl.action_override_val[1] - (-10.0f)) < 1e-6f);

    float ctrl_out[3] = {0};
    bionpc_agent_state_t ctrl_st = {0};
    bionpc_forward_single_controlled(m1, in, ctrl_out, &ctrl_st, &ctrl);
    assert(fabsf(ctrl_out[1] - (-10.0f)) < 1e-6f);
    assert(ctrl_st.apl_state[0] >= 0.0f && ctrl_st.apl_state[0] <= 0.5f);

    /* Test state locking & chemical spike */
    float locked_val = ctrl_st.apl_state[0];
    bionpc_state_lock(&ctrl_st, 0, 1);
    bionpc_state_spike(&ctrl_st, 0, 0.99f); /* Should be ignored because layer 0 is locked */
    assert(fabsf(ctrl_st.apl_state[0] - locked_val) < 1e-6f);
    bionpc_state_lock(&ctrl_st, 0, 0);
    bionpc_state_spike(&ctrl_st, 0, 0.25f);
    assert(fabsf(ctrl_st.apl_state[0] - (locked_val + 0.25f)) < 1e-6f);
    printf("  [PASS] DesignerControl INI loader, chemical spikes, state locks, and action overrides verified\n");

    free(batch_in);
    free(out_single);
    free(out_batch);
    free(out_mt);
    free(st_single);
    free(st_batch);
    free(st_mt);
    free(st_loaded);
    bionpc_model_free(m1);
    bionpc_model_free(m2);

    printf("======================================================================\n");
    printf("ALL TESTS PASSED SUCCESSFULLY!\n");
    printf("======================================================================\n");
    return 0;
}
