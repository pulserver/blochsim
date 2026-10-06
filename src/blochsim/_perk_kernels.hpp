// The PERK feature map and its regression, fused, one voxel per thread.
//
//     y = parameter_mean + (scale * cos(W @ x + b) - feature_mean) @ weight.T
//
// A block of features is formed and consumed into the output accumulator in
// registers, so the ``(voxels, features)`` matrix never exists. The adjoint
// does the same and forms the angles again rather than keeping them. Every
// array is contiguous and row-major.

// Features formed at once, which is how often a voxel's signal is read.
constexpr int FEATURE_BLOCK = 32;
// Parameters accumulated at once by the forward pass.
constexpr int PARAMETER_BLOCK = 16;
// Contrasts accumulated at once by the adjoint.
constexpr int CONTRAST_BLOCK = 32;

// The angles ``W @ x + b`` of features ``first`` onward, as many as there are.
template <class Voxel, class Live>
BSK_HD void _angles(const float* signal, const float* frequency, const float* phase,
                    const Voxel& voxel, const Live& live, std::int64_t contrasts,
                    std::int64_t features, std::int64_t first, bsk::V<float, 1>* angle) {
    for (int j = 0; j < FEATURE_BLOCK; ++j) {
        angle[j] = bsk::V<float, 1>(first + j < features ? phase[first + j] : 0.0f);
    }
    for (std::int64_t contrast = 0; contrast < contrasts; ++contrast) {
        const auto value = bsk::ld(signal + voxel * contrasts + contrast, live, 0.0f);
        for (int j = 0; j < FEATURE_BLOCK; ++j) {
            if (first + j < features) {
                angle[j] = angle[j] + value * frequency[(first + j) * contrasts + contrast];
            }
        }
    }
}

// One block of voxels, from signal to parameters.
BSK_HD void _regress_kernel(const float* signal, const float* frequency, const float* phase,
                            const float* feature_mean, const float* weight,
                            const float* parameter_mean, float* output, std::int64_t voxels,
                            std::int64_t contrasts, std::int64_t features,
                            std::int64_t parameters, float scale, std::int64_t BLOCK_VOXELS) {
    const auto voxel = bsk::program_id(0) * BLOCK_VOXELS + bsk::arange_x();
    const auto live = voxel < voxels;
    bsk::V<float, 1> angle[FEATURE_BLOCK];
    for (std::int64_t base = 0; base < parameters; base += PARAMETER_BLOCK) {
        bsk::V<float, 1> total[PARAMETER_BLOCK];
        for (int k = 0; k < PARAMETER_BLOCK; ++k) {
            total[k] = bsk::V<float, 1>(0.0f);
        }
        for (std::int64_t first = 0; first < features; first += FEATURE_BLOCK) {
            _angles(signal, frequency, phase, voxel, live, contrasts, features, first, angle);
            for (int j = 0; j < FEATURE_BLOCK; ++j) {
                if (first + j >= features) {
                    break;
                }
                const auto mapped = scale * bsk::cos(angle[j]) - feature_mean[first + j];
                for (int k = 0; k < PARAMETER_BLOCK; ++k) {
                    if (base + k < parameters) {
                        total[k] = total[k] + mapped * weight[(base + k) * features + first + j];
                    }
                }
            }
        }
        for (int k = 0; k < PARAMETER_BLOCK; ++k) {
            if (base + k < parameters) {
                bsk::st(output + voxel * parameters + base + k,
                        total[k] + parameter_mean[base + k], live);
            }
        }
    }
}

// The derivative of one block of voxels with respect to their signals.
BSK_HD void _regress_vjp_kernel(const float* signal, const float* frequency, const float* phase,
                                const float* weight, const float* cotangent, float* output,
                                std::int64_t voxels, std::int64_t contrasts,
                                std::int64_t features, std::int64_t parameters, float scale,
                                std::int64_t BLOCK_VOXELS) {
    const auto voxel = bsk::program_id(0) * BLOCK_VOXELS + bsk::arange_x();
    const auto live = voxel < voxels;
    bsk::V<float, 1> angle[FEATURE_BLOCK];
    for (std::int64_t base = 0; base < contrasts; base += CONTRAST_BLOCK) {
        bsk::V<float, 1> gradient[CONTRAST_BLOCK];
        for (int c = 0; c < CONTRAST_BLOCK; ++c) {
            gradient[c] = bsk::V<float, 1>(0.0f);
        }
        for (std::int64_t first = 0; first < features; first += FEATURE_BLOCK) {
            _angles(signal, frequency, phase, voxel, live, contrasts, features, first, angle);
            for (int j = 0; j < FEATURE_BLOCK; ++j) {
                if (first + j >= features) {
                    break;
                }
                bsk::V<float, 1> through(0.0f);
                for (std::int64_t p = 0; p < parameters; ++p) {
                    through = through
                        + bsk::ld(cotangent + voxel * parameters + p, live, 0.0f)
                            * weight[p * features + first + j];
                }
                through = through * (-scale) * bsk::sin(angle[j]);
                for (int c = 0; c < CONTRAST_BLOCK; ++c) {
                    if (base + c < contrasts) {
                        gradient[c] = gradient[c]
                            + through * frequency[(first + j) * contrasts + base + c];
                    }
                }
            }
        }
        for (int c = 0; c < CONTRAST_BLOCK; ++c) {
            if (base + c < contrasts) {
                bsk::st(output + voxel * contrasts + base + c, gradient[c], live);
            }
        }
    }
}
