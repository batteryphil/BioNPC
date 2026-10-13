#include "bionpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define BIONPC_MAGIC "BIONPC01"

/* Efficient dot product with unrolling to facilitate SIMD / FMA vectorization */
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
    if (cfg->n_layers > BIONPC_MAX_LAYERS) {
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

/* Simple uniform random number generator in [-bound, bound] matching PyTorch init */
static float rand_uniform(float bound) {
    float r = (float)rand() / (float)RAND_MAX; /* [0, 1] */
    return (r * 2.0f - 1.0f) * bound;
}

void bionpc_init_random(bionpc_model_t *model, unsigned int seed) {
    if (!model) return;
    srand(seed);

    int d_in = model->config.d_in;
    int d_out = model->config.d_out;
    int d_model = model->config.d_model;
    int n_layers = model->config.n_layers;

    /* PyTorch nn.Linear uniform initialization: bound = 1 / sqrt(in_features) */
    float bound_in = 1.0f / sqrtf((float)d_in);
    for (int i = 0; i < d_model * d_in; i++) model->w_in[i] = rand_uniform(bound_in);
    for (int i = 0; i < d_model; i++) model->b_in[i] = rand_uniform(bound_in);

    float bound_mlp1 = 1.0f / sqrtf((float)d_model);
    float bound_mlp2 = 1.0f / sqrtf((float)(4 * d_model));

    for (int l = 0; l < n_layers; l++) {
        bionpc_layer_t *layer = &model->layers[l];
        layer->decay_logit = 2.0f;
        layer->gain = 0.1f;

        for (int i = 0; i < (4 * d_model) * d_model; i++)
            layer->w_mlp1[i] = rand_uniform(bound_mlp1);
        for (int i = 0; i < 4 * d_model; i++)
            layer->b_mlp1[i] = rand_uniform(bound_mlp1);

        for (int i = 0; i < d_model * (4 * d_model); i++)
            layer->w_mlp2[i] = rand_uniform(bound_mlp2);
        for (int i = 0; i < d_model; i++)
            layer->b_mlp2[i] = rand_uniform(bound_mlp2);
    }

    float bound_out = 1.0f / sqrtf((float)d_model);
    for (int i = 0; i < d_out * d_model; i++) model->w_out[i] = rand_uniform(bound_out);
    for (int i = 0; i < d_out; i++) model->b_out[i] = rand_uniform(bound_out);

    bionpc_model_update_cache(model);
}

int bionpc_save_weights(const bionpc_model_t *model, const char *path) {
    if (!model || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Magic identifier */
    if (fwrite(BIONPC_MAGIC, 1, 8, f) != 8) { fclose(f); return -1; }

    /* Configuration */
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

    /* proj_in */
    fwrite(model->w_in, sizeof(float), (size_t)d_model * d_in, f);
    fwrite(model->b_in, sizeof(float), (size_t)d_model, f);

    /* Layers */
    for (int l = 0; l < n_layers; l++) {
        bionpc_layer_t *layer = &model->layers[l];
        fwrite(&layer->decay_logit, sizeof(float), 1, f);
        fwrite(&layer->gain, sizeof(float), 1, f);
        fwrite(layer->w_mlp1, sizeof(float), (size_t)(4 * d_model) * d_model, f);
        fwrite(layer->b_mlp1, sizeof(float), (size_t)(4 * d_model), f);
        fwrite(layer->w_mlp2, sizeof(float), (size_t)d_model * (4 * d_model), f);
        fwrite(layer->b_mlp2, sizeof(float), (size_t)d_model, f);
    }

    /* proj_out */
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

    /* Verify configuration matches */
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

    /* proj_in */
    if (fread(model->w_in, sizeof(float), (size_t)d_model * d_in, f) != (size_t)d_model * d_in ||
        fread(model->b_in, sizeof(float), (size_t)d_model, f) != (size_t)d_model) {
        fclose(f);
        return -3;
    }

    /* Layers */
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

    /* proj_out */
    if (fread(model->w_out, sizeof(float), (size_t)d_out * d_model, f) != (size_t)d_out * d_model ||
        fread(model->b_out, sizeof(float), (size_t)d_out, f) != (size_t)d_out) {
        fclose(f);
        return -3;
    }

    fclose(f);
    bionpc_model_update_cache(model);
    return 0;
}

/* Stack-allocated scratch buffer for inference without heap allocation overhead */
#define BIONPC_SCRATCH_MAX 512

void bionpc_forward_single(
    const bionpc_model_t *model,
    const float *in,
    float *out,
    bionpc_agent_state_t *state
) {
    int d_in = model->config.d_in;
    int d_out = model->config.d_out;
    int d_model = model->config.d_model;
    int n_layers = model->config.n_layers;

    /* Internal layer activation scratch buffers */
    float x[BIONPC_SCRATCH_MAX];
    float apl_out[BIONPC_SCRATCH_MAX];
    float h1[BIONPC_SCRATCH_MAX * 4];
    float h2[BIONPC_SCRATCH_MAX];

    /* proj_in: x = Linear(in) */
    matvec_mul(model->w_in, in, model->b_in, x, d_model, d_in);

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
            /* Continuous biological decay across frames */
            float prev_s = state->apl_state[l];
            float new_s = prev_s * layer->cached_decay + activity * layer->cached_gain;
            state->apl_state[l] = new_s;
            s_norm = new_s;
        } else {
            /* Stateless single-step (T=1) */
            s_norm = activity * layer->cached_gain;
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
}

void bionpc_forward_batch(
    const bionpc_model_t *model,
    const float *inputs,
    float *outputs,
    bionpc_agent_state_t *states,
    int batch_size
) {
    int d_in = model->config.d_in;
    int d_out = model->config.d_out;

    for (int b = 0; b < batch_size; b++) {
        const float *in = &inputs[b * d_in];
        float *out = &outputs[b * d_out];
        bionpc_agent_state_t *st = states ? &states[b] : NULL;
        bionpc_forward_single(model, in, out, st);
    }
}

/* Worker argument for multi-threaded batch inference */
typedef struct {
    const bionpc_model_t *model;
    const float *inputs;
    float *outputs;
    bionpc_agent_state_t *states;
    int start_idx;
    int count;
} bionpc_worker_arg_t;

static void *bionpc_worker_func(void *ptr) {
    bionpc_worker_arg_t *arg = (bionpc_worker_arg_t *)ptr;
    int d_in = arg->model->config.d_in;
    int d_out = arg->model->config.d_out;

    for (int i = 0; i < arg->count; i++) {
        int idx = arg->start_idx + i;
        const float *in = &arg->inputs[idx * d_in];
        float *out = &arg->outputs[idx * d_out];
        bionpc_agent_state_t *st = arg->states ? &arg->states[idx] : NULL;
        bionpc_forward_single(arg->model, in, out, st);
    }
    return NULL;
}

void bionpc_forward_batch_mt(
    const bionpc_model_t *model,
    const float *inputs,
    float *outputs,
    bionpc_agent_state_t *states,
    int batch_size,
    int num_threads
) {
    if (num_threads <= 1 || batch_size < 1000) {
        bionpc_forward_batch(model, inputs, outputs, states, batch_size);
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
        args[t].start_idx = current_start;
        args[t].count = count;

        pthread_create(&threads[t], NULL, bionpc_worker_func, &args[t]);
        current_start += count;
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}
