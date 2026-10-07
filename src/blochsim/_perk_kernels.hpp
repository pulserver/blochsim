// The PERK feature map and its regression, fused.
//
//     y = parameter_mean + (scale * cos(W @ x + b) - feature_mean) @ weight.T
//
// Two matrix products with a cosine between, tiled the way a matrix product
// is: a program stages a block of signals and a block of frequencies in shared
// memory, and each thread forms the angles of VOXELS voxels by a block of
// features in registers, so every value read from shared memory feeds several
// multiplies. The cosine and the second product consume the block while it is
// still in registers, so the ``(voxels, features)`` matrix never exists. The
// adjoint forms the angles again rather than keeping them. Every array is
// contiguous and row-major.
//
// Staged blocks are zero past the last feature, parameter or voxel, and a zero
// weight is what removes a padded feature from both products, so the inner
// loops carry no bounds. On a card a program is THREADS threads; on the host
// the same source runs them one after another between barriers.

// Threads per program, and the voxels each holds. ``_perk_gpu`` launches
// programs of ``THREADS`` threads over ``THREADS * VOXELS`` voxels.
constexpr int THREADS = 64;
constexpr int VOXELS = 2;
constexpr int BLOCK_VOXELS = THREADS * VOXELS;
// Contrasts staged at once.
constexpr int CONTRASTS = 32;
// Features formed at once by the forward pass, and parameters it accumulates.
constexpr int FEATURES = 32;
constexpr int PARAMETERS = 8;
// Features formed at once by the adjoint, and contrasts of the gradient it
// accumulates: it holds the angles, their cotangents and the gradient at once.
constexpr int ADJOINT_FEATURES = 16;
constexpr int GRADIENT = 32;
// A row of staged weights is read four at a time, so it is padded to keep each
// row 16-byte aligned and the rows on different banks.
constexpr int PAD = 4;

#if defined(BLOCHSIM_SIMT)
#define PERK_SHARED __shared__
#define PERK_SHARED_ROWS __shared__ __align__(16)
#define PERK_SYNC() __syncthreads()
// The body runs once, as this thread.
#define PERK_EACH_THREAD(t) \
    for (int t = static_cast<int>(threadIdx.x), t##_once = 1; t##_once; t##_once = 0)
// What a thread keeps across a barrier: its own registers.
template <class T>
struct Own {
    T value;
    BSK_HD T& operator[](int) { return value; }
};
#else
#define PERK_SHARED static thread_local
#define PERK_SHARED_ROWS alignas(16) static thread_local
#define PERK_SYNC() static_cast<void>(0)
// The body runs as each thread in turn, so a barrier is the end of the loop.
#define PERK_EACH_THREAD(t) for (int t = 0; t < THREADS; ++t)
template <class T>
struct Own {
    T value[THREADS];
    T& operator[](int t) { return value[t]; }
};
#endif

struct alignas(16) Four {
    float x, y, z, w;
};

BSK_HD const Four& four(const float* address) {
    return *reinterpret_cast<const Four*>(address);
}

BSK_HD int clamp_width(std::int64_t left, int block) {
    return left < block ? static_cast<int>(left) : block;
}

// Signals of voxels ``first`` onward, contrasts ``c0`` onward, ``width`` of
// them, into ``staged[contrast][voxel]``. Consecutive threads copy consecutive
// elements of the rows, which are contiguous in memory.
BSK_HD void _stage_signals(const float* signal, std::int64_t voxels, std::int64_t contrasts,
                           std::int64_t first, std::int64_t c0, int width,
                           float (*staged)[BLOCK_VOXELS + 1], int t) {
    for (int i = t; i < BLOCK_VOXELS * width; i += THREADS) {
        const int v = i / width;
        const int c = i - v * width;
        const std::int64_t voxel = first + v;
        staged[c][v] = voxel < voxels ? signal[voxel * contrasts + c0 + c] : 0.0f;
    }
}

// Frequencies of ``width`` features ``f0`` onward against contrasts ``c0``
// onward, into ``staged[contrast][feature]``, zero past the last feature.
template <int BLOCK>
BSK_HD void _stage_frequencies(const float* frequency, std::int64_t contrasts,
                               std::int64_t f0, int features, std::int64_t c0, int width,
                               float (*staged)[BLOCK + PAD], int t) {
    for (int i = t; i < BLOCK * width; i += THREADS) {
        const int j = i / width;
        const int c = i - j * width;
        staged[c][j] = j < features ? frequency[(f0 + j) * contrasts + c0 + c] : 0.0f;
    }
}

// The angles ``W @ x + b`` of a thread's voxels over a block of features,
// accumulated over the staged contrasts.
template <int BLOCK>
BSK_HD void _accumulate_angles(const float (*signals)[BLOCK_VOXELS + 1],
                               const float (*frequencies)[BLOCK + PAD], int width,
                               float (&angle)[VOXELS][BLOCK], int t) {
    for (int c = 0; c < width; ++c) {
        float x[VOXELS];
#pragma unroll
        for (int r = 0; r < VOXELS; ++r) {
            x[r] = signals[c][t + r * THREADS];
        }
#pragma unroll
        for (int j = 0; j < BLOCK; j += 4) {
            const Four w = four(&frequencies[c][j]);
#pragma unroll
            for (int r = 0; r < VOXELS; ++r) {
                angle[r][j] += x[r] * w.x;
                angle[r][j + 1] += x[r] * w.y;
                angle[r][j + 2] += x[r] * w.z;
                angle[r][j + 3] += x[r] * w.w;
            }
        }
    }
}

// The angles of every voxel of the program over ``width`` features ``f0``
// onward, the phases already staged in ``offset``. Ends at a barrier.
template <int BLOCK>
BSK_HD void _angles(const float* signal, const float* frequency, std::int64_t voxels,
                    std::int64_t contrasts, std::int64_t first, std::int64_t f0, int width,
                    const float* offset, float (*signals)[BLOCK_VOXELS + 1],
                    float (*frequencies)[BLOCK + PAD], Own<float[VOXELS][BLOCK]>& angle) {
    PERK_EACH_THREAD(t) {
#pragma unroll
        for (int r = 0; r < VOXELS; ++r) {
#pragma unroll
            for (int j = 0; j < BLOCK; ++j) {
                angle[t][r][j] = offset[j];
            }
        }
    }
    for (std::int64_t c0 = 0; c0 < contrasts; c0 += CONTRASTS) {
        const int staged = clamp_width(contrasts - c0, CONTRASTS);
        PERK_SYNC();
        PERK_EACH_THREAD(t) {
            _stage_signals(signal, voxels, contrasts, first, c0, staged, signals, t);
            _stage_frequencies<BLOCK>(frequency, contrasts, f0, width, c0, staged, frequencies,
                                      t);
        }
        PERK_SYNC();
        PERK_EACH_THREAD(t) {
            _accumulate_angles<BLOCK>(signals, frequencies, staged, angle[t], t);
        }
    }
    PERK_SYNC();
}

// One program of voxels, from signal to parameters.
BSK_HD void _regress_kernel(const float* signal, const float* frequency, const float* phase,
                            const float* feature_mean, const float* weight,
                            const float* parameter_mean, float* output, std::int64_t voxels,
                            std::int64_t contrasts, std::int64_t features,
                            std::int64_t parameters, float scale, std::int64_t threads) {
    static_cast<void>(threads);
    PERK_SHARED float signals[CONTRASTS][BLOCK_VOXELS + 1];
    PERK_SHARED_ROWS float frequencies[CONTRASTS][FEATURES + PAD];
    PERK_SHARED_ROWS float weights[FEATURES][PARAMETERS];
    PERK_SHARED float offset[FEATURES];
    PERK_SHARED float mean[FEATURES];
    const std::int64_t first = bsk::program_id(0) * BLOCK_VOXELS;
    Own<float[VOXELS][FEATURES]> angle;
    Own<float[VOXELS][PARAMETERS]> total;
    for (std::int64_t p0 = 0; p0 < parameters; p0 += PARAMETERS) {
        const int held = clamp_width(parameters - p0, PARAMETERS);
        PERK_EACH_THREAD(t) {
#pragma unroll
            for (int r = 0; r < VOXELS; ++r) {
#pragma unroll
                for (int k = 0; k < PARAMETERS; ++k) {
                    total[t][r][k] = 0.0f;
                }
            }
        }
        for (std::int64_t f0 = 0; f0 < features; f0 += FEATURES) {
            const int width = clamp_width(features - f0, FEATURES);
            PERK_SYNC();
            PERK_EACH_THREAD(t) {
                for (int j = t; j < FEATURES; j += THREADS) {
                    offset[j] = j < width ? phase[f0 + j] : 0.0f;
                    mean[j] = j < width ? feature_mean[f0 + j] : 0.0f;
                }
                for (int i = t; i < FEATURES * PARAMETERS; i += THREADS) {
                    const int j = i / PARAMETERS;
                    const int k = i - j * PARAMETERS;
                    weights[j][k] =
                        j < width && k < held ? weight[(p0 + k) * features + f0 + j] : 0.0f;
                }
            }
            PERK_SYNC();
            _angles<FEATURES>(signal, frequency, voxels, contrasts, first, f0, width, offset,
                              signals, frequencies, angle);
            PERK_EACH_THREAD(t) {
#pragma unroll
                for (int j = 0; j < FEATURES; ++j) {
                    const float m = mean[j];
#pragma unroll
                    for (int r = 0; r < VOXELS; ++r) {
                        const float mapped = scale * cosf(angle[t][r][j]) - m;
#pragma unroll
                        for (int k = 0; k < PARAMETERS; k += 4) {
                            const Four w = four(&weights[j][k]);
                            total[t][r][k] += mapped * w.x;
                            total[t][r][k + 1] += mapped * w.y;
                            total[t][r][k + 2] += mapped * w.z;
                            total[t][r][k + 3] += mapped * w.w;
                        }
                    }
                }
            }
        }
        PERK_EACH_THREAD(t) {
#pragma unroll
            for (int r = 0; r < VOXELS; ++r) {
                const std::int64_t voxel = first + t + r * THREADS;
#pragma unroll
                for (int k = 0; k < PARAMETERS; ++k) {
                    if (voxel < voxels && k < held) {
                        output[voxel * parameters + p0 + k] =
                            total[t][r][k] + parameter_mean[p0 + k];
                    }
                }
            }
        }
    }
}

// The derivative of one program of voxels with respect to their signals.
BSK_HD void _regress_vjp_kernel(const float* signal, const float* frequency, const float* phase,
                                const float* weight, const float* cotangent, float* output,
                                std::int64_t voxels, std::int64_t contrasts,
                                std::int64_t features, std::int64_t parameters, float scale,
                                std::int64_t threads) {
    static_cast<void>(threads);
    constexpr int BLOCK = ADJOINT_FEATURES;
    PERK_SHARED float signals[CONTRASTS][BLOCK_VOXELS + 1];
    PERK_SHARED_ROWS float frequencies[CONTRASTS][BLOCK + PAD];
    PERK_SHARED_ROWS float back[BLOCK][GRADIENT + PAD];
    PERK_SHARED_ROWS float weights[PARAMETERS][BLOCK];
    PERK_SHARED float offset[BLOCK];
    const std::int64_t first = bsk::program_id(0) * BLOCK_VOXELS;
    Own<float[VOXELS][BLOCK]> angle;
    Own<float[VOXELS][BLOCK]> through;
    Own<float[VOXELS][GRADIENT]> gradient;
    for (std::int64_t g0 = 0; g0 < contrasts; g0 += GRADIENT) {
        const int held = clamp_width(contrasts - g0, GRADIENT);
        PERK_EACH_THREAD(t) {
#pragma unroll
            for (int r = 0; r < VOXELS; ++r) {
#pragma unroll
                for (int c = 0; c < GRADIENT; ++c) {
                    gradient[t][r][c] = 0.0f;
                }
            }
        }
        for (std::int64_t f0 = 0; f0 < features; f0 += BLOCK) {
            const int width = clamp_width(features - f0, BLOCK);
            PERK_SYNC();
            PERK_EACH_THREAD(t) {
                for (int j = t; j < BLOCK; j += THREADS) {
                    offset[j] = j < width ? phase[f0 + j] : 0.0f;
                }
                for (int i = t; i < BLOCK * held; i += THREADS) {
                    const int j = i / held;
                    const int c = i - j * held;
                    back[j][c] = j < width ? frequency[(f0 + j) * contrasts + g0 + c] : 0.0f;
                }
                for (int i = t; i < BLOCK * (GRADIENT - held); i += THREADS) {
                    const int j = i / (GRADIENT - held);
                    back[j][held + i - j * (GRADIENT - held)] = 0.0f;
                }
            }
            PERK_SYNC();
            _angles<BLOCK>(signal, frequency, voxels, contrasts, first, f0, width, offset,
                           signals, frequencies, angle);
            // What reaches each feature from the parameters: cotangent @ weight.
            PERK_EACH_THREAD(t) {
#pragma unroll
                for (int r = 0; r < VOXELS; ++r) {
#pragma unroll
                    for (int j = 0; j < BLOCK; ++j) {
                        through[t][r][j] = 0.0f;
                    }
                }
            }
            for (std::int64_t p0 = 0; p0 < parameters; p0 += PARAMETERS) {
                const int count = clamp_width(parameters - p0, PARAMETERS);
                PERK_SYNC();
                PERK_EACH_THREAD(t) {
                    for (int i = t; i < PARAMETERS * BLOCK; i += THREADS) {
                        const int k = i / BLOCK;
                        const int j = i - k * BLOCK;
                        weights[k][j] =
                            k < count && j < width ? weight[(p0 + k) * features + f0 + j] : 0.0f;
                    }
                }
                PERK_SYNC();
                PERK_EACH_THREAD(t) {
                    for (int k = 0; k < count; ++k) {
#pragma unroll
                        for (int r = 0; r < VOXELS; ++r) {
                            const std::int64_t voxel = first + t + r * THREADS;
                            const float pulled =
                                voxel < voxels ? cotangent[voxel * parameters + p0 + k] : 0.0f;
#pragma unroll
                            for (int j = 0; j < BLOCK; j += 4) {
                                const Four w = four(&weights[k][j]);
                                through[t][r][j] += pulled * w.x;
                                through[t][r][j + 1] += pulled * w.y;
                                through[t][r][j + 2] += pulled * w.z;
                                through[t][r][j + 3] += pulled * w.w;
                            }
                        }
                    }
                }
            }
            PERK_EACH_THREAD(t) {
#pragma unroll
                for (int j = 0; j < BLOCK; ++j) {
#pragma unroll
                    for (int r = 0; r < VOXELS; ++r) {
                        const float slope = -scale * sinf(angle[t][r][j]) * through[t][r][j];
#pragma unroll
                        for (int c = 0; c < GRADIENT; c += 4) {
                            const Four w = four(&back[j][c]);
                            gradient[t][r][c] += slope * w.x;
                            gradient[t][r][c + 1] += slope * w.y;
                            gradient[t][r][c + 2] += slope * w.z;
                            gradient[t][r][c + 3] += slope * w.w;
                        }
                    }
                }
            }
        }
        PERK_EACH_THREAD(t) {
#pragma unroll
            for (int r = 0; r < VOXELS; ++r) {
                const std::int64_t voxel = first + t + r * THREADS;
#pragma unroll
                for (int c = 0; c < GRADIENT; ++c) {
                    if (voxel < voxels && c < held) {
                        output[voxel * contrasts + g0 + c] = gradient[t][r][c];
                    }
                }
            }
        }
    }
}
