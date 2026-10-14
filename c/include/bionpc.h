#ifndef BIONPC_H
#define BIONPC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BIONPC_MAX_LAYERS 16
#define BIONPC_MAX_DIM    64

/* Configuration defining the neural architecture */
typedef struct {
    int d_in;       /* Sensory input dimension (e.g. 4 for [Sight, Health, Distance, Aggression]) */
    int d_out;      /* Motor output dimension (e.g. 3 for [Move X, Attack, Flee]) */
    int d_model;    /* Internal recurrent representation width (default: 16 or 32) */
    int n_layers;   /* Number of recurrent CTRNN / APL layers (default: 1 or 2) */
} bionpc_config_t;

/* Single CTRNN layer holding FusedAPLNorm parameters and 2-layer MLP weights */
typedef struct {
    /* FusedAPLNorm parameters:
     * decay = sigmoid(decay_logit)
     * gain  = relu(gain)
     */
    float decay_logit;
    float gain;

    /* Precomputed cached scalars for fast streaming */
    float cached_decay;
    float cached_gain;

    /* MLP layer 1: Linear(d_model -> 4 * d_model) */
    float *w_mlp1;   /* shape: (4 * d_model) x d_model */
    float *b_mlp1;   /* shape: 4 * d_model */

    /* MLP layer 2: Linear(4 * d_model -> d_model) */
    float *w_mlp2;   /* shape: d_model x (4 * d_model) */
    float *b_mlp2;   /* shape: d_model */
} bionpc_layer_t;

/* Full Embodied NPC Model */
typedef struct {
    bionpc_config_t config;

    /* proj_in: Linear(d_in -> d_model) */
    float *w_in;     /* shape: d_model x d_in */
    float *b_in;     /* shape: d_model */

    /* Recurrent CTRNN layers */
    bionpc_layer_t *layers;

    /* proj_out: Linear(d_model -> d_out) */
    float *w_out;    /* shape: d_out x d_model */
    float *b_out;    /* shape: d_out */

    /* Total parameter count */
    size_t total_params;
} bionpc_model_t;

/* Continuous biological recurrent state for an individual agent across time ticks */
typedef struct {
    float apl_state[BIONPC_MAX_LAYERS];
    unsigned char apl_locked[BIONPC_MAX_LAYERS];
} bionpc_agent_state_t;

/* Designer Controllability & Chemical Guardrails */
typedef struct {
    /* Sensory input modulation: in_eff[i] = in[i] * sensory_scale[i] + sensory_bias[i] */
    float sensory_scale[BIONPC_MAX_DIM];
    float sensory_bias[BIONPC_MAX_DIM];

    /* Recurrent chemical state clamping & global lock */
    float apl_min;
    float apl_max;
    int   apl_locked;
    float chemical_bias;

    /* Motor output scaling and clamping: out[i] = clamp(out[i] * action_scale[i], action_min[i], action_max[i]) */
    float action_scale[BIONPC_MAX_DIM];
    float action_min[BIONPC_MAX_DIM];
    float action_max[BIONPC_MAX_DIM];

    /* Explicit designer action override (e.g. safe zone disabling attack):
     * if action_override_mask[i] != 0, out[i] = action_override_val[i]
     */
    unsigned char action_override_mask[BIONPC_MAX_DIM];
    float         action_override_val[BIONPC_MAX_DIM];

    /* Optional profile metadata */
    char archetype_name[64];
} bionpc_control_t;

/* Model creation and destruction */
bionpc_model_t *bionpc_model_create(const bionpc_config_t *cfg);
bionpc_model_t *bionpc_model_load_file(const char *path);
void bionpc_model_free(bionpc_model_t *model);

/* Update precomputed decay and gain constants after altering parameters */
void bionpc_model_update_cache(bionpc_model_t *model);

/* Cross-platform deterministic parameter initialization */
void bionpc_init_random(bionpc_model_t *model, unsigned int seed);

/* Binary weights persistence (.bionpc format) */
int bionpc_save_weights(const bionpc_model_t *model, const char *path);
int bionpc_load_weights(bionpc_model_t *model, const char *path);

/* Designer control initialization, chemical state hooks, and INI loader */
void bionpc_control_init(bionpc_control_t *ctrl);
void bionpc_state_spike(bionpc_agent_state_t *state, int layer, float delta);
void bionpc_state_lock(bionpc_agent_state_t *state, int layer, int locked);
int  bionpc_load_ini_profile(const char *path, bionpc_model_t *model, bionpc_control_t *ctrl);

/* Multiplayer determinism: state serialization and IEEE-754 CRC32 checksums */
int      bionpc_save_states(const bionpc_agent_state_t *states, int count, int n_layers, const char *path);
int      bionpc_load_states(bionpc_agent_state_t *states, int count, int n_layers, const char *path);
uint32_t bionpc_crc32_states(const bionpc_agent_state_t *states, int count, int n_layers);
uint32_t bionpc_crc32_floats(const float *data, size_t count);

/* Single agent forward inference:
 * in:    array of length config.d_in
 * out:   array of length config.d_out
 * state: agent recurrent state (can be NULL for stateless/pure feedforward)
 */
void bionpc_forward_single(
    const bionpc_model_t *model,
    const float *in,
    float *out,
    bionpc_agent_state_t *state
);

/* Single agent forward inference with DesignerControl guardrails */
void bionpc_forward_single_controlled(
    const bionpc_model_t *model,
    const float *in,
    float *out,
    bionpc_agent_state_t *state,
    const bionpc_control_t *ctrl
);

/* Batched forward inference (single-threaded, 4-way tiled cache-optimized):
 * inputs:  contiguous array of shape [batch_size, d_in]
 * outputs: contiguous array of shape [batch_size, d_out]
 * states:  array of batch_size states (can be NULL for stateless)
 */
void bionpc_forward_batch(
    const bionpc_model_t *model,
    const float *inputs,
    float *outputs,
    bionpc_agent_state_t *states,
    int batch_size
);

void bionpc_forward_batch_controlled(
    const bionpc_model_t *model,
    const float *inputs,
    float *outputs,
    bionpc_agent_state_t *states,
    const bionpc_control_t *ctrl,
    int batch_size
);

/* Multi-threaded batched forward inference using POSIX threads:
 * Distributes agents across num_threads workers for maximum CPU parallelism.
 */
void bionpc_forward_batch_mt(
    const bionpc_model_t *model,
    const float *inputs,
    float *outputs,
    bionpc_agent_state_t *states,
    int batch_size,
    int num_threads
);

void bionpc_forward_batch_mt_controlled(
    const bionpc_model_t *model,
    const float *inputs,
    float *outputs,
    bionpc_agent_state_t *states,
    const bionpc_control_t *ctrl,
    int batch_size,
    int num_threads
);

#ifdef __cplusplus
}
#endif

#endif /* BIONPC_H */
