#ifndef DS4_RDMA_TIER_H
#define DS4_RDMA_TIER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Remote-RAM expert tier over one-sided RDMA READ (plan task T2).
 *
 * A second backing store for weight/expert spans, holding the model's leading
 * bytes in a PEER BOX'S RAM and fetching them with RDMA READ striped across two
 * NICs. Measured in T1 at 433 us for a 9.49 MiB span against 1189 us for local
 * NVMe (2.74x), byte-exact.
 *
 * Contract, and it matters: this tier is NEVER authoritative. Every entry point
 * returns 0 rather than failing, and the caller falls through to the model file.
 * A slow read costs latency; a wrong read would cost correct output, so on any
 * doubt -- unverifiable content, a dead QP, a short read -- the tier disables
 * itself for the rest of the process and the model file serves everything.
 *
 * Verbs is dlopen()ed, exactly as ds4_tp.c does it, so linking ds4 never
 * requires libibverbs and a box without RDMA simply reports the tier inactive.
 */

/* Reads DS4_EXPERT_TIER_RDMA and connects. model_fd/model_bytes are used to
 * verify sampled offsets against the real model before the tier is trusted.
 * Returns 1 when the tier is active, 0 otherwise (not configured, unavailable,
 * or failed verification). Safe to call when unconfigured. */
int ds4_rdma_tier_open(int model_fd, uint64_t model_bytes);

/* Fetch [offset, offset+bytes) into dst. Returns 1 on success with dst filled,
 * 0 when the span is not covered or anything at all went wrong. */
int ds4_rdma_tier_read(void *dst, uint64_t offset, uint64_t bytes);

/* Bytes of the model this tier covers from offset 0; 0 when inactive. */
uint64_t ds4_rdma_tier_bytes(void);

/* Cumulative successful reads, bytes, and nanoseconds spent in them. */
void ds4_rdma_tier_stats(uint64_t *reads, uint64_t *bytes, uint64_t *ns);

/* Promotion (plan T4): offer a span that local disk just served, so the peer
 * can hold it for next time. Never blocks and never fails visibly -- a span
 * that is not promoted is simply read from the model file again. */
void ds4_rdma_tier_promote(const void *src, uint64_t offset, uint64_t bytes);
void ds4_rdma_tier_promote_stats(uint64_t *promoted, uint64_t *hits,
                                 uint64_t *skipped, uint32_t *slots);
/* Promoted slots that failed read-back. Nonzero means the peer returned bytes
 * that were not what was promoted; those slots are discarded, never served. */
uint64_t ds4_rdma_tier_promote_corrupt(void);

void ds4_rdma_tier_close(void);

#ifdef __cplusplus
}
#endif

#endif /* DS4_RDMA_TIER_H */
