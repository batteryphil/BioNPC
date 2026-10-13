#include "bionpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

int main() {
    printf("Testing BioNPC Model Serialization and Numerical Invariants...\n");

    bionpc_config_t cfg = {
        .d_in = 4,
        .d_out = 3,
        .d_model = 16,
        .n_layers = 2
    };

    bionpc_model_t *m1 = bionpc_model_create(&cfg);
    bionpc_init_random(m1, 777);

    const char *test_path = "/tmp/test_model.bionpc";
    int save_rc = bionpc_save_weights(m1, test_path);
    assert(save_rc == 0);
    printf("  [PASS] Saved model weights to %s\n", test_path);

    bionpc_model_t *m2 = bionpc_model_create(&cfg);
    int load_rc = bionpc_load_weights(m2, test_path);
    assert(load_rc == 0);
    printf("  [PASS] Loaded model weights from %s\n", test_path);

    /* Verify exact numerical equivalence of forward pass */
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
    printf("  [PASS] Forward inference outputs and recurrent states match exactly (diff < 1e-6)\n");

    bionpc_model_free(m1);
    bionpc_model_free(m2);

    printf("ALL TESTS PASSED!\n");
    return 0;
}
