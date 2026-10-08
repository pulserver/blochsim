// Rows of a program's y axis each thread holds on a card, per kernel: the
// ``Y_LANES`` of _tile.hpp. A kernel's file is compiled with its own, and the
// launcher divides a program's rows by it to size the block.
//
// In the EPG kernels y is the problems a program carries, so rows held in a
// thread share its reading of every event. A kernel whose y is something else
// -- the pools of the pooled kernels -- or that has no y holds one.
#pragma once

#define BLOCHSIM_LANES__three_pool_table_jvp_kernel 1
#define BLOCHSIM_LANES__three_pool_table_kernel 1
#define BLOCHSIM_LANES__epg_vjp_kernel 1
#define BLOCHSIM_LANES__epg_vjp_jvp_kernel 1
#define BLOCHSIM_LANES__epg_real_vjp_jvp_kernel 1
#define BLOCHSIM_LANES__epg_real_vjp_kernel 1
#define BLOCHSIM_LANES__epg_real_kernel 4
#define BLOCHSIM_LANES__epg_real_jvp_kernel 1
#define BLOCHSIM_LANES__epg_kernel 2
#define BLOCHSIM_LANES__epg_jvp_kernel 1
#define BLOCHSIM_LANES__pooled_kernel 1
#define BLOCHSIM_LANES__pooled_adjoint_kernel 1
#define BLOCHSIM_LANES__regress_kernel 1
#define BLOCHSIM_LANES__regress_vjp_kernel 1
