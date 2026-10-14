#include "bionpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <pthread.h>

#define BIONPC_MAGIC       "BIONPC01"
#define BIONPC_STATE_MAGIC "BNPCST01"

/* Efficient dot product with 4-way unrolling to facilitate SIMD / FMA vectorization */
static inline float dot_product(const float *a, const float *b, int n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i]     * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* Matrix-vector multiplication: y = W * x + bias */
static inline void matvec_mul(
    const float *W,
    const float *x,
    const float *bias,
    float *out,
    int rows,
    int cols
) {
    for (int i = 0; i < rows; i++) {
        float b = bias ? bias[i] : 0.0f;
        out[i] = b + dot_product(&W[i * cols], x, cols);
    }
}

/* 4-agent tiled matrix-vector multiplication:
 * Reads each weight row W[i * cols ..] ONCE and computes exact 4-way unrolled dot products
 * for 4 agents simultaneously. Reduces memory bandwidth and Fil-C capability checks on W by 4x
 * while preserving 100% bit-for-bit IEEE-754 identity with single-agent matvec_mul.
 */
static inline void matvec_mul_4x(
    const float *W,
    const float *x0, const float *x1, const float *x2, const float *x3,
    const float *bias,
    float *out0, float *out1, float *out2, float *out3,
    int rows,
    int cols
) {
    for (int i = 0; i < rows; i++) {
        const float *w_row = &W[i * cols];
        float b = bias ? bias[i] : 0.0f;

        float s0_0 = 0.0f, s0_1 = 0.0f, s0_2 = 0.0f, s0_3 = 0.0f;
        float s1_0 = 0.0f, s1_1 = 0.0f, s1_2 = 0.0f, s1_3 = 0.0f;
        float s2_0 = 0.0f, s2_1 = 0.0f, s2_2 = 0.0f, s2_3 = 0.0f;
        float s3_0 = 0.0f, s3_1 = 0.0f, s3_2 = 0.0f, s3_3 = 0.0f;

        int j = 0;
        for (; j + 3 < cols; j += 4) {
            float w0 = w_row[j];
            float w1 = w_row[j + 1];
            float w2 = w_row[j + 2];
            float w3 = w_row[j + 3];

            s0_0 += w0 * x0[j];     s0_1 += w1 * x0[j + 1]; s0_2 += w2 * x0[j + 2]; s0_3 += w3 * x0[j + 3];
            s1_0 += w0 * x1[j];     s1_1 += w1 * x1[j + 1]; s1_2 += w2 * x1[j + 2]; s1_3 += w3 * x1[j + 3];
            s2_0 += w0 * x2[j];     s2_1 += w1 * x2[j + 1]; s2_2 += w2 * x2[j + 2]; s2_3 += w3 * x2[j + 3];
            s3_0 += w0 * x3[j];     s3_1 += w1 * x3[j + 1]; s3_2 += w2 * x3[j + 2]; s3_3 += w3 * x3[j + 3];
        }

        float sum0 = (s0_0 + s0_1) + (s0_2 + s0_3);
        float sum1 = (s1_0 + s1_1) + (s1_2 + s1_3);
        float sum2 = (s2_0 + s2_1) + (s2_2 + s2_3);
        float sum3 = (s3_0 + s3_1) + (s3_2 + s3_3);

        for (; j < cols; j++) {
            float w = w_row[j];
            sum0 += w * x0[j];
            sum1 += w * x1[j];
            sum2 += w * x2[j];
            sum3 += w * x3[j];
        }

        out0[i] = b + sum0;
        out1[i] = b + sum1;
        out2[i] = b + sum2;
        out3[i] = b + sum3;
    }
}

/* SiLU activation function: x * sigmoid(x) = x / (1 + exp(-x)) */
static inline float bionpc_silu(float x) {
    if (x < -20.0f) return 0.0f;
    if (x > 20.0f) return x;
    return x / (1.0f + expf(-x));
}

void bionpc_model_update_cache(bionpc_model_t *model) {
    if (!model) return;
    for (int l = 0; l < model->config.n_layers; l++) {
        bionpc_layer_t *layer = &model->layers[l];
        /* decay = sigmoid(decay_logit) */
        layer->cached_decay = 1.0f / (1.0f + expf(-layer->decay_logit));
        /* gain = relu(gain) */
        layer->cached_gain = (layer->gain > 0.0f) ? layer->gain : 0.0f;
    }
}

bionpc_model_t *bionpc_model_create(const bionpc_config_t *cfg) {
    if (!cfg || cfg->d_in <= 0 || cfg->d_out <= 0 || cfg->d_model <= 0 || cfg->n_layers <= 0) {
        return NULL;
    }
    if (cfg->n_layers > BIONPC_MAX_LAYERS || cfg->d_in > BIONPC_MAX_DIM ||
        cfg->d_out > BIONPC_MAX_DIM || cfg->d_model > BIONPC_MAX_DIM) {
        return NULL;
    }

    bionpc_model_t *model = (bionpc_model_t *)calloc(1, sizeof(bionpc_model_t));
    if (!model) return NULL;

    model->config = *cfg;
    int d_in = cfg->d_in;
    int d_out = cfg->d_out;
    int d_model = cfg->d_model;
    int n_layers = cfg->n_layers;

    /* proj_in: d_model x d_in */
    model->w_in = (float *)calloc((size_t)d_model * d_in, sizeof(float));
    model->b_in = (float *)calloc((size_t)d_model, sizeof(float));

    /* Layers */
    model->layers = (bionpc_layer_t *)calloc((size_t)n_layers, sizeof(bionpc_layer_t));

    size_t mlp1_size = (size_t)(4 * d_model) * d_model;
    size_t mlp2_size = (size_t)d_model * (4 * d_model);

    for (int l = 0; l < n_layers; l++) {
        bionpc_layer_t *layer = &model->layers[l];
        layer->decay_logit = 2.0f;
        layer->gain = 0.1f;

        layer->w_mlp1 = (float *)calloc(mlp1_size, sizeof(float));
        layer->b_mlp1 = (float *)calloc((size_t)(4 * d_model), sizeof(float));

        layer->w_mlp2 = (float *)calloc(mlp2_size, sizeof(float));
        layer->b_mlp2 = (float *)calloc((size_t)d_model, sizeof(float));
    }

    /* proj_out: d_out x d_model */
    model->w_out = (float *)calloc((size_t)d_out * d_model, sizeof(float));
    model->b_out = (float *)calloc((size_t)d_out, sizeof(float));

    /* Compute total params */
    model->total_params = (size_t)d_model * d_in + d_model +
                          (size_t)n_layers * (2 + mlp1_size + 4 * d_model + mlp2_size + d_model) +
                          (size_t)d_out * d_model + d_out;

    bionpc_model_update_cache(model);
    return model;
}

void bionpc_model_free(bionpc_model_t *model) {
    if (!model) return;
    free(model->w_in);
    free(model->b_in);
    if (model->layers) {
        for (int l = 0; l < model->config.n_layers; l++) {
            bionpc_layer_t *layer = &model->layers[l];
            free(layer->w_mlp1);
            free(layer->b_mlp1);
            free(layer->w_mlp2);
            free(layer->b_mlp2);
        }
        free(model->layers);
    }
    free(model->w_out);
    free(model->b_out);
    free(model);
}

/* Portable deterministic SplitMix32 PRNG so weights initialized from a seed
 * are 100% identical across Linux, Windows, macOS, and BSD regardless of libc rand().
 */
static inline uint32_t splitmix32_next(uint32_t *state) {
    uint32_t z = (*state += 0x9e3779b9u);
    z = (z ^ (z >> 16)) * 0x85ebca6bu;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    return z ^ (z >> 16);
}

static inline float det_rand_uniform(uint32_t *rng_state, float bound) {
    uint32_t r_int = splitmix32_next(rng_state);
    /* Map 24-bit mantissa to [0.0, 1.0] deterministically */
    float r = (float)(r_int >> 8) * (1.0f / 16777215.0f);
    return (r * 2.0f - 1.0f) * bound;
}

void bionpc_init_random(bionpc_model_t *model, unsigned int seed) {
    if (!model) return;
    uint32_t rng = (seed == 0) ? 1u : (uint32_t)seed;

    int d_in = model->config.d_in;
    int d_out = model->config.d_out;
    int d_model = model->config.d_model;
    int n_layers = model->config.n_layers;

    /* PyTorch nn.Linear uniform initialization: bound = 1 / sqrt(in_features) */
    float bound_in = 1.0f / sqrtf((float)d_in);
    for (int i = 0; i < d_model * d_in; i++) model->w_in[i] = det_rand_uniform(&rng, bound_in);
    for (int i = 0; i < d_model; i++) model->b_in[i] = det_rand_uniform(&rng, bound_in);

    float bound_mlp1 = 1.0f / sqrtf((float)d_model);
    float bound_mlp2 = 1.0f / sqrtf((float)(4 * d_model));

    for (int l = 0; l < n_layers; l++) {
        bionpc_layer_t *layer = &model->layers[l];
        layer->decay_logit = 2.0f;
        layer->gain = 0.1f;

        for (int i = 0; i < (4 * d_model) * d_model; i++)
            layer->w_mlp1[i] = det_rand_uniform(&rng, bound_mlp1);
        for (int i = 0; i < 4 * d_model; i++)
            layer->b_mlp1[i] = det_rand_uniform(&rng, bound_mlp1);

        for (int i = 0; i < d_model * (4 * d_model); i++)
            layer->w_mlp2[i] = det_rand_uniform(&rng, bound_mlp2);
        for (int i = 0; i < d_model; i++)
            layer->b_mlp2[i] = det_rand_uniform(&rng, bound_mlp2);
    }

    float bound_out = 1.0f / sqrtf((float)d_model);
    for (int i = 0; i < d_out * d_model; i++) model->w_out[i] = det_rand_uniform(&rng, bound_out);
    for (int i = 0; i < d_out; i++) model->b_out[i] = det_rand_uniform(&rng, bound_out);

    bionpc_model_update_cache(model);
}

int bionpc_save_weights(const bionpc_model_t *model, const char *path) {
    if (!model || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    if (fwrite(BIONPC_MAGIC, 1, 8, f) != 8) { fclose(f); return -1; }

    int cfg[4] = {
        model->config.d_in,
        model->config.d_out,
        model->config.d_model,
        model->config.n_layers
    };
    if (fwrite(cfg, sizeof(int), 4, f) != 4) { fclose(f); return -1; }

    int d_in = model->config.d_in;
    int d_out = model->config.d_out;
    int d_model = model->config.d_model;
    int n_layers = model->config.n_layers;

    fwrite(model->w_in, sizeof(float), (size_t)d_model * d_in, f);
    fwrite(model->b_in, sizeof(float), (size_t)d_model, f);

    for (int l = 0; l < n_layers; l++) {
        bionpc_layer_t *layer = &model->layers[l];
        fwrite(&layer->decay_logit, sizeof(float), 1, f);
        fwrite(&layer->gain, sizeof(float), 1, f);
        fwrite(layer->w_mlp1, sizeof(float), (size_t)(4 * d_model) * d_model, f);
        fwrite(layer->b_mlp1, sizeof(float), (size_t)(4 * d_model), f);
        fwrite(layer->w_mlp2, sizeof(float), (size_t)d_model * (4 * d_model), f);
        fwrite(layer->b_mlp2, sizeof(float), (size_t)d_model, f);
    }

    fwrite(model->w_out, sizeof(float), (size_t)d_out * d_model, f);
    fwrite(model->b_out, sizeof(float), (size_t)d_out, f);

    fclose(f);
    return 0;
}

int bionpc_load_weights(bionpc_model_t *model, const char *path) {
    if (!model || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, BIONPC_MAGIC, 8) != 0) {
        fclose(f);
        return -1;
    }

    int cfg[4];
    if (fread(cfg, sizeof(int), 4, f) != 4) { fclose(f); return -1; }

    if (cfg[0] != model->config.d_in ||
        cfg[1] != model->config.d_out ||
        cfg[2] != model->config.d_model ||
        cfg[3] != model->config.n_layers) {
        fclose(f);
        return -2;
    }

    int d_in = model->config.d_in;
    int d_out = model->config.d_out;
    int d_model = model->config.d_model;
    int n_layers = model->config.n_layers;

    if (fread(model->w_in, sizeof(float), (size_t)d_model * d_in, f) != (size_t)d_model * d_in ||
        fread(model->b_in, sizeof(float), (size_t)d_model, f) != (size_t)d_model) {
        fclose(f);
        return -3;
    }

    for (int l = 0; l < n_layers; l++) {
        bionpc_layer_t *layer = &model->layers[l];
        if (fread(&layer->decay_logit, sizeof(float), 1, f) != 1 ||
            fread(&layer->gain, sizeof(float), 1, f) != 1 ||
            fread(layer->w_mlp1, sizeof(float), (size_t)(4 * d_model) * d_model, f) != (size_t)(4 * d_model) * d_model ||
            fread(layer->b_mlp1, sizeof(float), (size_t)(4 * d_model), f) != (size_t)(4 * d_model) ||
            fread(layer->w_mlp2, sizeof(float), (size_t)d_model * (4 * d_model), f) != (size_t)d_model * (4 * d_model) ||
            fread(layer->b_mlp2, sizeof(float), (size_t)d_model, f) != (size_t)d_model) {
            fclose(f);
            return -3;
        }
    }

    if (fread(model->w_out, sizeof(float), (size_t)d_out * d_model, f) != (size_t)d_out * d_model ||
        fread(model->b_out, sizeof(float), (size_t)d_out, f) != (size_t)d_out) {
        fclose(f);
        return -3;
    }

    fclose(f);
    bionpc_model_update_cache(model);
    return 0;
}

bionpc_model_t *bionpc_model_load_file(const char *path) {
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    char magic[8];
    int cfg_arr[4];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, BIONPC_MAGIC, 8) != 0 ||
        fread(cfg_arr, sizeof(int), 4, f) != 4) {
        fclose(f);
        return NULL;
    }
    fclose(f);

    bionpc_config_t cfg = {
        .d_in = cfg_arr[0],
        .d_out = cfg_arr[1],
        .d_model = cfg_arr[2],
        .n_layers = cfg_arr[3]
    };
    bionpc_model_t *model = bionpc_model_create(&cfg);
    if (!model) return NULL;

    if (bionpc_load_weights(model, path) != 0) {
        bionpc_model_free(model);
        return NULL;
    }
    return model;
}

/* --- DESIGNER CONTROLLABILITY & INI PROFILE LOADER --- */

void bionpc_control_init(bionpc_control_t *ctrl) {
    if (!ctrl) return;
    memset(ctrl, 0, sizeof(bionpc_control_t));
    for (int i = 0; i < BIONPC_MAX_DIM; i++) {
        ctrl->sensory_scale[i] = 1.0f;
        ctrl->sensory_bias[i] = 0.0f;
        ctrl->action_scale[i] = 1.0f;
        ctrl->action_min[i] = -1e9f;
        ctrl->action_max[i] = 1e9f;
        ctrl->action_override_mask[i] = 0;
        ctrl->action_override_val[i] = 0.0f;
    }
    ctrl->apl_min = -10.0f;
    ctrl->apl_max = 10.0f;
    ctrl->apl_locked = 0;
    ctrl->chemical_bias = 0.0f;
    strncpy(ctrl->archetype_name, "default", sizeof(ctrl->archetype_name) - 1);
}

void bionpc_state_spike(bionpc_agent_state_t *state, int layer, float delta) {
    if (!state) return;
    if (layer >= 0 && layer < BIONPC_MAX_LAYERS) {
        if (!state->apl_locked[layer]) {
            state->apl_state[layer] += delta;
        }
    } else if (layer < 0) {
        /* Apply to all layers */
        for (int l = 0; l < BIONPC_MAX_LAYERS; l++) {
            if (!state->apl_locked[l]) {
                state->apl_state[l] += delta;
            }
        }
    }
}

void bionpc_state_lock(bionpc_agent_state_t *state, int layer, int locked) {
    if (!state) return;
    unsigned char val = locked ? 1 : 0;
    if (layer >= 0 && layer < BIONPC_MAX_LAYERS) {
        state->apl_locked[layer] = val;
    } else if (layer < 0) {
        for (int l = 0; l < BIONPC_MAX_LAYERS; l++) {
            state->apl_locked[l] = val;
        }
    }
}

static char *trim_whitespace(char *str) {
    while (*str && isspace((unsigned char)*str)) str++;
    if (*str == '\0') return str;
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) {
        *end = '\0';
        end--;
    }
    return str;
}

static int parse_csv_floats(const char *str, float *out, int max_count) {
    int count = 0;
    const char *p = str;
    while (*p && count < max_count) {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (!*p) break;
        char *endptr = NULL;
        float val = strtof(p, &endptr);
        if (endptr == p) break;
        out[count++] = val;
        p = endptr;
    }
    return count;
}

int bionpc_load_ini_profile(const char *path, bionpc_model_t *model, bionpc_control_t *ctrl) {
    if (!path) return -1;
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    if (ctrl) bionpc_control_init(ctrl);

    char line[512];
    char section[64] = "";

    while (fgets(line, sizeof(line), f)) {
        char *s = trim_whitespace(line);
        if (*s == '\0' || *s == '#' || *s == ';') continue;

        if (*s == '[') {
            char *close = strchr(s, ']');
            if (close) {
                *close = '\0';
                strncpy(section, s + 1, sizeof(section) - 1);
                section[sizeof(section) - 1] = '\0';
            }
            continue;
        }

        char *eq = strchr(s, '=');
        if (!eq) continue;
        *eq = '\0';
        char *key = trim_whitespace(s);
        char *val = trim_whitespace(eq + 1);

        if (strcmp(section, "archetype") == 0 && ctrl) {
            if (strcmp(key, "name") == 0) {
                strncpy(ctrl->archetype_name, val, sizeof(ctrl->archetype_name) - 1);
            }
        } else if (strcmp(section, "chemistry") == 0) {
            if (ctrl) {
                if (strcmp(key, "apl_min") == 0) ctrl->apl_min = strtof(val, NULL);
                else if (strcmp(key, "apl_max") == 0) ctrl->apl_max = strtof(val, NULL);
                else if (strcmp(key, "apl_locked") == 0) {
                    ctrl->apl_locked = (strcmp(val, "true") == 0 || strcmp(val, "1") == 0);
                } else if (strcmp(key, "chemical_bias") == 0) {
                    ctrl->chemical_bias = strtof(val, NULL);
                }
            }
            if (model) {
                if (strncmp(key, "decay_logit_l", 13) == 0) {
                    int l = atoi(key + 13);
                    if (l >= 0 && l < model->config.n_layers) {
                        model->layers[l].decay_logit = strtof(val, NULL);
                    }
                } else if (strncmp(key, "gain_l", 6) == 0) {
                    int l = atoi(key + 6);
                    if (l >= 0 && l < model->config.n_layers) {
                        model->layers[l].gain = strtof(val, NULL);
                    }
                }
            }
        } else if (strcmp(section, "senses") == 0 && ctrl) {
            if (strcmp(key, "sensory_scale") == 0) {
                parse_csv_floats(val, ctrl->sensory_scale, BIONPC_MAX_DIM);
            } else if (strcmp(key, "sensory_bias") == 0) {
                parse_csv_floats(val, ctrl->sensory_bias, BIONPC_MAX_DIM);
            }
        } else if (strcmp(section, "actions") == 0 && ctrl) {
            if (strcmp(key, "action_scale") == 0) {
                parse_csv_floats(val, ctrl->action_scale, BIONPC_MAX_DIM);
            } else if (strcmp(key, "action_min") == 0) {
                parse_csv_floats(val, ctrl->action_min, BIONPC_MAX_DIM);
            } else if (strcmp(key, "action_max") == 0) {
                parse_csv_floats(val, ctrl->action_max, BIONPC_MAX_DIM);
            } else if (strcmp(key, "forced_actions") == 0) {
                const char *p = val;
                while (*p) {
                    while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
                    if (!*p) break;
                    int idx = atoi(p);
                    const char *colon = strchr(p, ':');
                    if (!colon) break;
                    char *endptr = NULL;
                    float fval = strtof(colon + 1, &endptr);
                    if (idx >= 0 && idx < BIONPC_MAX_DIM) {
                        ctrl->action_override_mask[idx] = 1;
                        ctrl->action_override_val[idx] = fval;
                    }
                    p = endptr ? endptr : (colon + 1);
                }
            }
        }
    }

    fclose(f);
    if (model) bionpc_model_update_cache(model);
    return 0;
}

/* --- MULTIPLAYER DETERMINISM: STATE SNAPSHOTS & CRC32 --- */

static uint32_t crc32_bytes(const uint8_t *buf, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= buf[i];
        for (int b = 0; b < 8; b++) {
            uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

uint32_t bionpc_crc32_floats(const float *data, size_t count) {
    if (!data || count == 0) return 0;
    return crc32_bytes((const uint8_t *)data, count * sizeof(float));
}

uint32_t bionpc_crc32_states(const bionpc_agent_state_t *states, int count, int n_layers) {
    if (!states || count <= 0 || n_layers <= 0) return 0;
    if (n_layers > BIONPC_MAX_LAYERS) n_layers = BIONPC_MAX_LAYERS;

    uint32_t crc = 0xFFFFFFFFu;
    for (int i = 0; i < count; i++) {
        const uint8_t *bytes = (const uint8_t *)states[i].apl_state;
        size_t len = (size_t)n_layers * sizeof(float);
        for (size_t k = 0; k < len; k++) {
            crc ^= bytes[k];
            for (int b = 0; b < 8; b++) {
                uint32_t mask = -(crc & 1u);
                crc = (crc >> 1) ^ (0xEDB88320u & mask);
            }
        }
    }
    return ~crc;
}

int bionpc_save_states(const bionpc_agent_state_t *states, int count, int n_layers, const char *path) {
    if (!states || count <= 0 || n_layers <= 0 || n_layers > BIONPC_MAX_LAYERS || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    if (fwrite(BIONPC_STATE_MAGIC, 1, 8, f) != 8) { fclose(f); return -1; }
    int hdr[2] = { count, n_layers };
    if (fwrite(hdr, sizeof(int), 2, f) != 2) { fclose(f); return -1; }

    for (int i = 0; i < count; i++) {
        if (fwrite(states[i].apl_state, sizeof(float), (size_t)n_layers, f) != (size_t)n_layers) {
            fclose(f);
            return -1;
        }
    }
    for (int i = 0; i < count; i++) {
        if (fwrite(states[i].apl_locked, sizeof(unsigned char), (size_t)n_layers, f) != (size_t)n_layers) {
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    return 0;
}

int bionpc_load_states(bionpc_agent_state_t *states, int count, int n_layers, const char *path) {
    if (!states || count <= 0 || n_layers <= 0 || n_layers > BIONPC_MAX_LAYERS || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[8];
    int hdr[2];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, BIONPC_STATE_MAGIC, 8) != 0 ||
        fread(hdr, sizeof(int), 2, f) != 2) {
        fclose(f);
        return -1;
    }
    if (hdr[0] != count || hdr[1] != n_layers) {
        fclose(f);
        return -2;
    }

    for (int i = 0; i < count; i++) {
        if (fread(states[i].apl_state, sizeof(float), (size_t)n_layers, f) != (size_t)n_layers) {
            fclose(f);
            return -3;
        }
    }
    for (int i = 0; i < count; i++) {
        if (fread(states[i].apl_locked, sizeof(unsigned char), (size_t)n_layers, f) != (size_t)n_layers) {
            fclose(f);
            return -3;
        }
    }

    fclose(f);
    return 0;
}

/* --- FORWARD INFERENCE ENGINE --- */

/* Stack-allocated scratch buffer for inference without heap allocation overhead */
#define BIONPC_SCRATCH_MAX BIONPC_MAX_DIM

void bionpc_forward_single_controlled(
    const bionpc_model_t *model,
    const float *in,
    float *out,
    bionpc_agent_state_t *state,
    const bionpc_control_t *ctrl
) {
    int d_in = model->config.d_in;
    int d_out = model->config.d_out;
    int d_model = model->config.d_model;
    int n_layers = model->config.n_layers;

    float in_mod[BIONPC_MAX_DIM];
    const float *in_ptr = in;
    if (ctrl) {
        for (int i = 0; i < d_in; i++) {
            float v = in[i] * ctrl->sensory_scale[i] + ctrl->sensory_bias[i];
            in_mod[i] = (v < 0.0f) ? 0.0f : v;
        }
        in_ptr = in_mod;
    }

    /* Internal layer activation scratch buffers */
    float x[BIONPC_SCRATCH_MAX];
    float apl_out[BIONPC_SCRATCH_MAX];
    float h1[BIONPC_SCRATCH_MAX * 4];
    float h2[BIONPC_SCRATCH_MAX];

    /* proj_in: x = Linear(in) */
    matvec_mul(model->w_in, in_ptr, model->b_in, x, d_model, d_in);

    /* Recurrent CTRNN layers with FusedAPLNorm */
    for (int l = 0; l < n_layers; l++) {
        const bionpc_layer_t *layer = &model->layers[l];

        /* 1. FusedAPLNorm */
        float sum_abs = 0.0f;
        for (int i = 0; i < d_model; i++) {
            sum_abs += fabsf(x[i]);
        }
        float activity = sum_abs / (float)d_model;

        float s_norm;
        if (state) {
            float prev_s = state->apl_state[l];
            int is_locked = state->apl_locked[l] || (ctrl && ctrl->apl_locked);
            if (is_locked) {
                s_norm = prev_s;
            } else {
                float bias = ctrl ? ctrl->chemical_bias : 0.0f;
                float new_s = prev_s * layer->cached_decay + activity * layer->cached_gain + bias;
                if (ctrl) {
                    if (new_s < ctrl->apl_min) new_s = ctrl->apl_min;
                    if (new_s > ctrl->apl_max) new_s = ctrl->apl_max;
                }
                state->apl_state[l] = new_s;
                s_norm = new_s;
            }
        } else {
            float bias = ctrl ? ctrl->chemical_bias : 0.0f;
            s_norm = activity * layer->cached_gain + bias;
            if (ctrl) {
                if (s_norm < ctrl->apl_min) s_norm = ctrl->apl_min;
                if (s_norm > ctrl->apl_max) s_norm = ctrl->apl_max;
            }
        }

        float factor = 1.0f - s_norm;
        for (int i = 0; i < d_model; i++) {
            apl_out[i] = x[i] * factor;
        }

        /* 2. MLP: Linear(d_model -> 4 * d_model) -> SiLU -> Linear(4 * d_model -> d_model) */
        matvec_mul(layer->w_mlp1, apl_out, layer->b_mlp1, h1, 4 * d_model, d_model);

        int h1_dim = 4 * d_model;
        for (int i = 0; i < h1_dim; i++) {
            h1[i] = bionpc_silu(h1[i]);
        }

        matvec_mul(layer->w_mlp2, h1, layer->b_mlp2, h2, d_model, 4 * d_model);

        /* Residual connection: x = x + MLP(apl(x)) */
        for (int i = 0; i < d_model; i++) {
            x[i] += h2[i];
        }
    }

    /* proj_out: out = Linear(x) */
    matvec_mul(model->w_out, x, model->b_out, out, d_out, d_model);

    if (ctrl) {
        for (int i = 0; i < d_out; i++) {
            if (ctrl->action_override_mask[i]) {
                out[i] = ctrl->action_override_val[i];
            } else {
                float v = out[i] * ctrl->action_scale[i];
                if (v < ctrl->action_min[i]) v = ctrl->action_min[i];
                if (v > ctrl->action_max[i]) v = ctrl->action_max[i];
                out[i] = v;
            }
        }
    }
}

void bionpc_forward_single(
    const bionpc_model_t *model,
    const float *in,
    float *out,
    bionpc_agent_state_t *state
) {
    bionpc_forward_single_controlled(model, in, out, state, NULL);
}

/* 4-agent tiled forward pass: processes 4 agents simultaneously through all layers */
static void bionpc_forward_tile4(
    const bionpc_model_t *model,
    const float *in0, const float *in1, const float *in2, const float *in3,
    float *out0, float *out1, float *out2, float *out3,
    bionpc_agent_state_t *st0, bionpc_agent_state_t *st1,
    bionpc_agent_state_t *st2, bionpc_agent_state_t *st3,
    const bionpc_control_t *ctrl
) {
    int d_in = model->config.d_in;
    int d_out = model->config.d_out;
    int d_model = model->config.d_model;
    int n_layers = model->config.n_layers;

    float mod0[BIONPC_MAX_DIM], mod1[BIONPC_MAX_DIM], mod2[BIONPC_MAX_DIM], mod3[BIONPC_MAX_DIM];
    const float *p0 = in0, *p1 = in1, *p2 = in2, *p3 = in3;
    if (ctrl) {
        for (int i = 0; i < d_in; i++) {
            float sc = ctrl->sensory_scale[i];
            float bi = ctrl->sensory_bias[i];
            float v0 = in0[i] * sc + bi;
            float v1 = in1[i] * sc + bi;
            float v2 = in2[i] * sc + bi;
            float v3 = in3[i] * sc + bi;
            mod0[i] = (v0 < 0.0f) ? 0.0f : v0;
            mod1[i] = (v1 < 0.0f) ? 0.0f : v1;
            mod2[i] = (v2 < 0.0f) ? 0.0f : v2;
            mod3[i] = (v3 < 0.0f) ? 0.0f : v3;
        }
        p0 = mod0; p1 = mod1; p2 = mod2; p3 = mod3;
    }

    float x0[BIONPC_SCRATCH_MAX], x1[BIONPC_SCRATCH_MAX], x2[BIONPC_SCRATCH_MAX], x3[BIONPC_SCRATCH_MAX];
    float a0[BIONPC_SCRATCH_MAX], a1[BIONPC_SCRATCH_MAX], a2[BIONPC_SCRATCH_MAX], a3[BIONPC_SCRATCH_MAX];
    float h1_0[BIONPC_SCRATCH_MAX * 4], h1_1[BIONPC_SCRATCH_MAX * 4];
    float h1_2[BIONPC_SCRATCH_MAX * 4], h1_3[BIONPC_SCRATCH_MAX * 4];
    float h2_0[BIONPC_SCRATCH_MAX], h2_1[BIONPC_SCRATCH_MAX], h2_2[BIONPC_SCRATCH_MAX], h2_3[BIONPC_SCRATCH_MAX];

    matvec_mul_4x(model->w_in, p0, p1, p2, p3, model->b_in, x0, x1, x2, x3, d_model, d_in);

    float inv_d_model = 1.0f / (float)d_model;
    bionpc_agent_state_t *sts[4] = { st0, st1, st2, st3 };
    float *xs[4] = { x0, x1, x2, x3 };
    float *as[4] = { a0, a1, a2, a3 };

    for (int l = 0; l < n_layers; l++) {
        const bionpc_layer_t *layer = &model->layers[l];
        float decay = layer->cached_decay;
        float gain = layer->cached_gain;

        for (int k = 0; k < 4; k++) {
            float *xk = xs[k];
            float *ak = as[k];
            bionpc_agent_state_t *st = sts[k];

            float sum_abs = 0.0f;
            for (int i = 0; i < d_model; i++) sum_abs += fabsf(xk[i]);
            float activity = sum_abs * inv_d_model;

            float s_norm;
            if (st) {
                float prev_s = st->apl_state[l];
                int is_locked = st->apl_locked[l] || (ctrl && ctrl->apl_locked);
                if (is_locked) {
                    s_norm = prev_s;
                } else {
                    float bias = ctrl ? ctrl->chemical_bias : 0.0f;
                    float new_s = prev_s * decay + activity * gain + bias;
                    if (ctrl) {
                        if (new_s < ctrl->apl_min) new_s = ctrl->apl_min;
                        if (new_s > ctrl->apl_max) new_s = ctrl->apl_max;
                    }
                    st->apl_state[l] = new_s;
                    s_norm = new_s;
                }
            } else {
                float bias = ctrl ? ctrl->chemical_bias : 0.0f;
                s_norm = activity * gain + bias;
                if (ctrl) {
                    if (s_norm < ctrl->apl_min) s_norm = ctrl->apl_min;
                    if (s_norm > ctrl->apl_max) s_norm = ctrl->apl_max;
                }
            }

            float factor = 1.0f - s_norm;
            for (int i = 0; i < d_model; i++) ak[i] = xk[i] * factor;
        }

        int h1_dim = 4 * d_model;
        matvec_mul_4x(layer->w_mlp1, a0, a1, a2, a3, layer->b_mlp1, h1_0, h1_1, h1_2, h1_3, h1_dim, d_model);

        for (int i = 0; i < h1_dim; i++) {
            h1_0[i] = bionpc_silu(h1_0[i]);
            h1_1[i] = bionpc_silu(h1_1[i]);
            h1_2[i] = bionpc_silu(h1_2[i]);
            h1_3[i] = bionpc_silu(h1_3[i]);
        }

        matvec_mul_4x(layer->w_mlp2, h1_0, h1_1, h1_2, h1_3, layer->b_mlp2, h2_0, h2_1, h2_2, h2_3, d_model, h1_dim);

        for (int i = 0; i < d_model; i++) {
            x0[i] += h2_0[i];
            x1[i] += h2_1[i];
            x2[i] += h2_2[i];
            x3[i] += h2_3[i];
        }
    }

    matvec_mul_4x(model->w_out, x0, x1, x2, x3, model->b_out, out0, out1, out2, out3, d_out, d_model);

    if (ctrl) {
        float *outs[4] = { out0, out1, out2, out3 };
        for (int k = 0; k < 4; k++) {
            float *ok = outs[k];
            for (int i = 0; i < d_out; i++) {
                if (ctrl->action_override_mask[i]) {
                    ok[i] = ctrl->action_override_val[i];
                } else {
                    float v = ok[i] * ctrl->action_scale[i];
                    if (v < ctrl->action_min[i]) v = ctrl->action_min[i];
                    if (v > ctrl->action_max[i]) v = ctrl->action_max[i];
                    ok[i] = v;
                }
            }
        }
    }
}

void bionpc_forward_batch_controlled(
    const bionpc_model_t *model,
    const float *inputs,
    float *outputs,
    bionpc_agent_state_t *states,
    const bionpc_control_t *ctrl,
    int batch_size
) {
    int d_in = model->config.d_in;
    int d_out = model->config.d_out;

    int b = 0;
    for (; b + 3 < batch_size; b += 4) {
        bionpc_forward_tile4(
            model,
            &inputs[(b + 0) * d_in],
            &inputs[(b + 1) * d_in],
            &inputs[(b + 2) * d_in],
            &inputs[(b + 3) * d_in],
            &outputs[(b + 0) * d_out],
            &outputs[(b + 1) * d_out],
            &outputs[(b + 2) * d_out],
            &outputs[(b + 3) * d_out],
            states ? &states[b + 0] : NULL,
            states ? &states[b + 1] : NULL,
            states ? &states[b + 2] : NULL,
            states ? &states[b + 3] : NULL,
            ctrl
        );
    }
    for (; b < batch_size; b++) {
        const float *in = &inputs[b * d_in];
        float *out = &outputs[b * d_out];
        bionpc_agent_state_t *st = states ? &states[b] : NULL;
        bionpc_forward_single_controlled(model, in, out, st, ctrl);
    }
}

void bionpc_forward_batch(
    const bionpc_model_t *model,
    const float *inputs,
    float *outputs,
    bionpc_agent_state_t *states,
    int batch_size
) {
    bionpc_forward_batch_controlled(model, inputs, outputs, states, NULL, batch_size);
}

/* Worker argument for multi-threaded batch inference */
typedef struct {
    const bionpc_model_t *model;
    const float *inputs;
    float *outputs;
    bionpc_agent_state_t *states;
    const bionpc_control_t *ctrl;
    int start_idx;
    int count;
} bionpc_worker_arg_t;

static void *bionpc_worker_func(void *ptr) {
    bionpc_worker_arg_t *arg = (bionpc_worker_arg_t *)ptr;
    int d_in = arg->model->config.d_in;
    int d_out = arg->model->config.d_out;

    bionpc_forward_batch_controlled(
        arg->model,
        &arg->inputs[arg->start_idx * d_in],
        &arg->outputs[arg->start_idx * d_out],
        arg->states ? &arg->states[arg->start_idx] : NULL,
        arg->ctrl,
        arg->count
    );
    return NULL;
}

void bionpc_forward_batch_mt_controlled(
    const bionpc_model_t *model,
    const float *inputs,
    float *outputs,
    bionpc_agent_state_t *states,
    const bionpc_control_t *ctrl,
    int batch_size,
    int num_threads
) {
    if (num_threads <= 1 || batch_size < 256) {
        bionpc_forward_batch_controlled(model, inputs, outputs, states, ctrl, batch_size);
        return;
    }

    if (num_threads > 64) num_threads = 64;

    pthread_t threads[64];
    bionpc_worker_arg_t args[64];

    int chunk = batch_size / num_threads;
    int remainder = batch_size % num_threads;
    int current_start = 0;

    for (int t = 0; t < num_threads; t++) {
        int count = chunk + (t < remainder ? 1 : 0);
        args[t].model = model;
        args[t].inputs = inputs;
        args[t].outputs = outputs;
        args[t].states = states;
        args[t].ctrl = ctrl;
        args[t].start_idx = current_start;
        args[t].count = count;

        pthread_create(&threads[t], NULL, bionpc_worker_func, &args[t]);
        current_start += count;
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

void bionpc_forward_batch_mt(
    const bionpc_model_t *model,
    const float *inputs,
    float *outputs,
    bionpc_agent_state_t *states,
    int batch_size,
    int num_threads
) {
    bionpc_forward_batch_mt_controlled(model, inputs, outputs, states, NULL, batch_size, num_threads);
}
