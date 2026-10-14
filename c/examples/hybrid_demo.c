#include "bionpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

int main(int argc, char **argv) {
    const char *weights_path = (argc >= 2) ? argv[1] : "../archetypes/apex_predator.bionpc";
    const char *ini_path     = (argc >= 3) ? argv[2] : "../archetypes/apex_predator.ini";

    printf("======================================================================\n");
    printf("   BioNPC Hybrid Forebrain + 60 FPS Hindbrain Showcase (Fil-C APE)    \n");
    printf("======================================================================\n");

    bionpc_model_t *brain = bionpc_model_load_file(weights_path);
    if (!brain) {
        bionpc_config_t cfg = { .d_in = 4, .d_out = 3, .d_model = 32, .n_layers = 2 };
        brain = bionpc_model_create(&cfg);
        bionpc_init_random(brain, 1000);
        printf("[Info] Initialized default apex_predator brain in memory.\n");
    } else {
        printf("[Info] Loaded pre-trained archetype weights: %s\n", weights_path);
    }

    bionpc_control_t ctrl;
    if (bionpc_load_ini_profile(ini_path, brain, &ctrl) == 0) {
        printf("[Info] Loaded Designer Profile: %s (archetype='%s')\n\n", ini_path, ctrl.archetype_name);
    } else {
        bionpc_control_init(&ctrl);
        printf("[Info] Initialized default DesignerControl profile.\n\n");
    }

    bionpc_agent_state_t state = {0};
    /* Senses: [prey_sight, threat_sight, hunger, aggression] */
    float senses[4] = { 0.80f, 0.10f, 0.65f, 0.35f };
    float logits[3] = {0};

    printf("Tick | Event / Context                        | Chase  | Attack | Flee   | APL L0 State\n");
    printf("-----+----------------------------------------+--------+--------+--------+-------------\n");

    for (int tick = 1; tick <= 12; tick++) {
        const char *label = "60Hz Hindbrain: Tracking prey";

        if (tick == 4) {
            /* Designer Override: Player steps into Sanctuary Zone -> forbid Attack, lock state */
            label = "[DESIGNER OVERRIDE] Sanctuary Zone!";
            senses[1] = 0.85f; /* Threat is right in front of wolf */
            senses[3] = 0.90f; /* Wolf is aggressive */
            ctrl.action_override_mask[1] = 1;
            ctrl.action_override_val[1] = -10.0f; /* Sigmoid(-10) = 0.0000 */
            bionpc_state_lock(&state, 0, 1);
        } else if (tick == 5 || tick == 6) {
            label = "60Hz Hindbrain: Stalking (Attack Clamped)";
        } else if (tick == 7) {
            /* LLM Forebrain Event: Sanctuary ends & Player threatens pack */
            label = "[LLM FOREBRAIN] 'Burn their den!'";
            ctrl.action_override_mask[1] = 0; /* Release designer clamp */
            bionpc_state_lock(&state, 0, 0);
            bionpc_state_spike(&state, 0, -0.15f); /* Adrenaline disinhibition spike */
            ctrl.sensory_bias[1] = 0.45f; /* Heightened threat perception */
            ctrl.sensory_bias[3] = 0.55f; /* Enraged aggression spike */
        } else if (tick >= 8 && tick <= 10) {
            label = "60Hz Hindbrain: Enraged Combat Reflex";
            /* Natural neurotransmitter bias decay */
            for (int i = 0; i < 4; i++) ctrl.sensory_bias[i] *= 0.85f;
        } else if (tick == 11) {
            label = "[LLM FOREBRAIN] 'Tosses raw meat'";
            senses[0] = 0.85f; /* Meat / food spotted */
            senses[1] = 0.05f; /* Threat gone */
            senses[2] = 0.70f; /* Hungry for the meat */
            senses[3] = 0.15f; /* Calm */
            memset(ctrl.sensory_bias, 0, sizeof(ctrl.sensory_bias));
            ctrl.sensory_bias[0] = 0.25f;
            bionpc_state_spike(&state, 0, 0.15f);
        } else if (tick == 12) {
            label = "60Hz Hindbrain: Approaches offered meat";
        }

        bionpc_forward_single_controlled(brain, senses, logits, &state, &ctrl);

        float p_chase  = sigmoidf(logits[0]) * 100.0f;
        float p_attack = sigmoidf(logits[1]) * 100.0f;
        float p_flee   = sigmoidf(logits[2]) * 100.0f;

        printf(" %2d  | %-38s | %5.1f%% | %5.1f%% | %5.1f%% |   %+.4f\n",
               tick, label, p_chase, p_attack, p_flee, state.apl_state[0]);
    }

    uint32_t state_crc = bionpc_crc32_states(&state, 1, brain->config.n_layers);
    printf("----------------------------------------------------------------------\n");
    printf("Final Multiplayer Lockstep State CRC32: 0x%08X\n", state_crc);
    printf("======================================================================\n");

    bionpc_model_free(brain);
    return 0;
}
