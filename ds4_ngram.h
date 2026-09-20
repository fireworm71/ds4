/* ds4_ngram.h - High-performance 3-tier N-Gram cache for speculative drafting
 * Supports context-tier (in-memory per session) and dynamic-tier (persisted across runs)
 * using a 4-way set-associative hash table with O(1) lookup.
 */

#ifndef DS4_NGRAM_H
#define DS4_NGRAM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_NGRAM_MIN 1
#define DS4_NGRAM_MAX 4
#define DS4_NGRAM_WAYS 4

#define DS4_NGRAM_DEFAULT_BUCKETS 32768u  /* 32768 * 4 = 131,072 entries (~4.2 MB) */
#define DS4_NGRAM_MAGIC 0x4453344EU       /* 'DS4N' */
#define DS4_NGRAM_VERSION 1u

typedef struct {
    int32_t  tokens[DS4_NGRAM_MAX]; // prefix n-gram tokens
    int32_t  top_token;            // most frequent continuation token
    int32_t  second_token;         // second most frequent continuation token
    uint16_t total_count;          // total occurrences
    uint16_t top_count;            // occurrences of top_token
    uint16_t second_count;         // occurrences of second_token
    uint8_t  gram_len;             // 1, 2, 3, or 4
    uint8_t  tier;                 // 0: context, 1: dynamic, 2: static
} ds4_ngram_entry;

typedef struct {
    ds4_ngram_entry entries[DS4_NGRAM_WAYS];
} ds4_ngram_bucket;

typedef struct {
    ds4_ngram_bucket *buckets;
    uint32_t          num_buckets;  // must be power of two
    uint32_t          bucket_mask;  // num_buckets - 1
    uint32_t          shift;        // 64 - log2(num_buckets)
    uint32_t          total_entries;
    uint8_t           tier;         // 0: context, 1: dynamic, 2: static
} ds4_ngram_table;

/* Initialize an n-gram table with a given number of buckets (power of 2). */
int ds4_ngram_table_init(ds4_ngram_table *table, uint32_t num_buckets, uint8_t tier);

/* Free allocated memory for an n-gram table. */
void ds4_ngram_table_free(ds4_ngram_table *table);

/* Clear all entries without freeing the bucket array. */
void ds4_ngram_table_clear(ds4_ngram_table *table);

/* Update table with a sequence of tokens. nnew specifies how many tokens
 * at the tail of tokens[0 .. n_tokens-1] were newly added. */
void ds4_ngram_table_update(ds4_ngram_table *table,
                            const int32_t *tokens,
                            int n_tokens,
                            int n_new);

/* Query the table for an n-gram continuation.
 * Returns true if found and confidence passes thresholds.
 * out_token: selected continuation token.
 * out_confidence: empirical frequency ratio (0.0 to 1.0). */
bool ds4_ngram_table_lookup(const ds4_ngram_table *table,
                            const int32_t *prefix,
                            int gram_len,
                            int32_t *out_token,
                            float *out_confidence,
                            uint16_t min_count);

/* Speculatively draft up to max_draft tokens chaining across context, static, and dynamic tiers.
 * history: full evaluated history tokens [0 .. n_history-1].
 * current_token: the token just generated or sampled.
 * draft_out: destination buffer for drafted tokens (at least max_draft ints).
 * Returns the number of drafted tokens (0 if no draft produced). */
uint32_t ds4_ngram_draft_3tier(const ds4_ngram_table *context_table,
                               const ds4_ngram_table *static_table,
                               const ds4_ngram_table *dynamic_table,
                               const int32_t *history,
                               int n_history,
                               int32_t current_token,
                               int32_t *draft_out,
                               uint32_t max_draft);

/* Speculatively draft up to max_draft tokens chaining across context and dynamic tiers. */
uint32_t ds4_ngram_draft(const ds4_ngram_table *context_table,
                         const ds4_ngram_table *dynamic_table,
                         const int32_t *history,
                         int n_history,
                         int32_t current_token,
                         int32_t *draft_out,
                         uint32_t max_draft);

struct ds4_session;
typedef struct ds4_session ds4_session;

/* Engine-level draft proposer checking Tier 2 (context) -> Tier 1 (static) -> Tier 3 (dynamic). */
int ds4_ngram_draft_propose(ds4_session *s,
                            const int *history,
                            int n_history,
                            int current_token,
                            int *out_draft,
                            int max_draft);

/* Persist dynamic table to disk. Returns 1 on success, 0 on failure. */
int ds4_ngram_table_save(const ds4_ngram_table *table, const char *filepath);

/* Load persisted dynamic table from disk. Returns 1 on success, 0 on failure. */
int ds4_ngram_table_load(ds4_ngram_table *table, const char *filepath);

/* Get default dynamic cache filepath (under ~/.cache/ds4/ngram_cache.bin). */
const char *ds4_ngram_default_cache_path(void);

#ifdef __cplusplus
}
#endif

#endif /* DS4_NGRAM_H */
