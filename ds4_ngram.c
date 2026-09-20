/* ds4_ngram.c - High-performance 3-tier N-Gram cache for speculative drafting
 * Implements context tier (per session) and dynamic tier (disk persisted)
 * with a 4-way set-associative hash table for O(1) lookup.
 */

#include "ds4_ngram.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <pwd.h>

static inline uint64_t ds4_ngram_hash(const int32_t *tokens, int gram_len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (int i = 0; i < gram_len; i++) {
        h ^= (uint64_t)(uint32_t)tokens[i];
        h *= 1099511628211ULL;
    }
    return h * 11400714819323198485ULL;
}

static inline uint32_t next_power_of_two(uint32_t v) {
    if (v < 2) return 2;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

static inline uint32_t compute_shift(uint32_t num_buckets) {
    uint32_t bits = 0;
    while ((1u << bits) < num_buckets) bits++;
    return 64u - bits;
}

int ds4_ngram_table_init(ds4_ngram_table *table, uint32_t num_buckets, uint8_t tier) {
    if (!table) return 0;
    memset(table, 0, sizeof(*table));
    if (num_buckets == 0) num_buckets = DS4_NGRAM_DEFAULT_BUCKETS;
    num_buckets = next_power_of_two(num_buckets);

    table->buckets = (ds4_ngram_bucket *)calloc(num_buckets, sizeof(ds4_ngram_bucket));
    if (!table->buckets) return 0;

    table->num_buckets = num_buckets;
    table->bucket_mask = num_buckets - 1u;
    table->shift = compute_shift(num_buckets);
    table->total_entries = 0;
    table->tier = tier;
    return 1;
}

void ds4_ngram_table_free(ds4_ngram_table *table) {
    if (!table) return;
    if (table->buckets) {
        free(table->buckets);
        table->buckets = NULL;
    }
    table->num_buckets = 0;
    table->total_entries = 0;
}

void ds4_ngram_table_clear(ds4_ngram_table *table) {
    if (!table || !table->buckets) return;
    memset(table->buckets, 0, (size_t)table->num_buckets * sizeof(ds4_ngram_bucket));
    table->total_entries = 0;
}

static void ngram_insert_or_update(ds4_ngram_table *table,
                                   const int32_t *prefix,
                                   int gram_len,
                                   int32_t next_token) {
    const uint64_t hash = ds4_ngram_hash(prefix, gram_len);
    const uint32_t idx = (uint32_t)(hash >> table->shift) & table->bucket_mask;
    ds4_ngram_bucket *b = &table->buckets[idx];

    int empty_slot = -1;
    int min_slot = 0;
    uint32_t min_count = UINT32_MAX;

    for (int s = 0; s < DS4_NGRAM_WAYS; s++) {
        ds4_ngram_entry *e = &b->entries[s];
        if (e->gram_len == 0) {
            if (empty_slot < 0) empty_slot = s;
            continue;
        }
        if (e->gram_len == gram_len && memcmp(e->tokens, prefix, (size_t)gram_len * sizeof(int32_t)) == 0) {
            /* Match found - update frequency counters */
            if (e->total_count < UINT16_MAX) e->total_count++;
            if (e->top_token == next_token) {
                if (e->top_count < UINT16_MAX) e->top_count++;
            } else if (e->second_token == next_token) {
                if (e->second_count < UINT16_MAX) e->second_count++;
                if (e->second_count > e->top_count) {
                    /* Swap top and second */
                    const int32_t tmp_tok = e->top_token;
                    const uint16_t tmp_cnt = e->top_count;
                    e->top_token = e->second_token;
                    e->top_count = e->second_count;
                    e->second_token = tmp_tok;
                    e->second_count = tmp_cnt;
                }
            } else {
                /* New continuation token */
                if (e->second_count == 0) {
                    e->second_token = next_token;
                    e->second_count = 1;
                }
            }
            return;
        }
        if (e->total_count < min_count) {
            min_count = e->total_count;
            min_slot = s;
        }
    }

    /* Insertion: use empty slot or evict LFU slot */
    const int slot = (empty_slot >= 0) ? empty_slot : min_slot;
    ds4_ngram_entry *e = &b->entries[slot];
    if (e->gram_len == 0) table->total_entries++;

    memset(e->tokens, 0, sizeof(e->tokens));
    memcpy(e->tokens, prefix, (size_t)gram_len * sizeof(int32_t));
    e->gram_len = (uint8_t)gram_len;
    e->top_token = next_token;
    e->top_count = 1;
    e->second_token = -1;
    e->second_count = 0;
    e->total_count = 1;
    e->tier = table->tier;
}

static void ngram_restore_entry(ds4_ngram_table *table, const ds4_ngram_entry *src) {
    if (!table || !table->buckets || !src ||
        src->gram_len < DS4_NGRAM_MIN || src->gram_len > DS4_NGRAM_MAX)
        return;
    const uint64_t hash = ds4_ngram_hash(src->tokens, src->gram_len);
    const uint32_t idx = (uint32_t)(hash >> table->shift) & table->bucket_mask;
    ds4_ngram_bucket *b = &table->buckets[idx];

    int empty_slot = -1;
    int min_slot = 0;
    uint32_t min_count = UINT32_MAX;

    for (int s = 0; s < DS4_NGRAM_WAYS; s++) {
        ds4_ngram_entry *e = &b->entries[s];
        if (e->gram_len == 0) {
            if (empty_slot < 0) empty_slot = s;
            continue;
        }
        if (e->gram_len == src->gram_len &&
            memcmp(e->tokens, src->tokens, (size_t)src->gram_len * sizeof(int32_t)) == 0) {
            /* Match found - merge counts safely */
            uint32_t total = (uint32_t)e->total_count + src->total_count;
            e->total_count = (total > UINT16_MAX) ? UINT16_MAX : (uint16_t)total;
            if (e->top_token == src->top_token) {
                uint32_t cnt = (uint32_t)e->top_count + src->top_count;
                e->top_count = (cnt > UINT16_MAX) ? UINT16_MAX : (uint16_t)cnt;
            } else if (src->top_count > e->top_count) {
                e->second_token = e->top_token;
                e->second_count = e->top_count;
                e->top_token = src->top_token;
                e->top_count = src->top_count;
            } else if (e->second_token == src->top_token) {
                uint32_t cnt = (uint32_t)e->second_count + src->top_count;
                e->second_count = (cnt > UINT16_MAX) ? UINT16_MAX : (uint16_t)cnt;
            }
            return;
        }
        if (e->total_count < min_count) {
            min_count = e->total_count;
            min_slot = s;
        }
    }

    const int slot = (empty_slot >= 0) ? empty_slot : min_slot;
    ds4_ngram_entry *e = &b->entries[slot];
    if (e->gram_len == 0) table->total_entries++;
    *e = *src;
    e->tier = table->tier;
}

void ds4_ngram_table_update(ds4_ngram_table *table,
                            const int32_t *tokens,
                            int n_tokens,
                            int n_new) {
    if (!table || !table->buckets || !tokens || n_tokens <= 0 || n_new <= 0) return;

    for (int gram_len = DS4_NGRAM_MIN; gram_len <= DS4_NGRAM_MAX; gram_len++) {
        int i_start = n_tokens - n_new;
        if (i_start < gram_len) i_start = gram_len;
        for (int i = i_start; i < n_tokens; i++) {
            const int32_t *prefix = &tokens[i - gram_len];
            const int32_t next_tok = tokens[i];
            ngram_insert_or_update(table, prefix, gram_len, next_tok);
        }
    }
}

bool ds4_ngram_table_lookup(const ds4_ngram_table *table,
                            const int32_t *prefix,
                            int gram_len,
                            int32_t *out_token,
                            float *out_confidence,
                            uint16_t min_count) {
    if (!table || !table->buckets || !prefix || gram_len < DS4_NGRAM_MIN || gram_len > DS4_NGRAM_MAX)
        return false;

    const uint64_t hash = ds4_ngram_hash(prefix, gram_len);
    const uint32_t idx = (uint32_t)(hash >> table->shift) & table->bucket_mask;
    const ds4_ngram_bucket *b = &table->buckets[idx];

    for (int s = 0; s < DS4_NGRAM_WAYS; s++) {
        const ds4_ngram_entry *e = &b->entries[s];
        if (e->gram_len == gram_len &&
            memcmp(e->tokens, prefix, (size_t)gram_len * sizeof(int32_t)) == 0) {
            if (e->total_count < min_count) return false;
            if (out_token) *out_token = e->top_token;
            if (out_confidence) {
                *out_confidence = (e->total_count > 0)
                    ? (float)e->top_count / (float)e->total_count
                    : 0.0f;
            }
            return true;
        }
    }
    return false;
}

uint32_t ds4_ngram_draft_3tier(const ds4_ngram_table *context_table,
                               const ds4_ngram_table *static_table,
                               const ds4_ngram_table *dynamic_table,
                               const int32_t *history,
                               int n_history,
                               int32_t current_token,
                               int32_t *draft_out,
                               uint32_t max_draft) {
    if (!draft_out || max_draft == 0) return 0;
    if ((!context_table || !context_table->buckets) &&
        (!static_table || !static_table->buckets) &&
        (!dynamic_table || !dynamic_table->buckets))
        return 0;

    /* Working sequence: copy tail of history, then append current_token and drafts */
    #define MAX_CHAIN_WINDOW 64
    if (max_draft > MAX_CHAIN_WINDOW) max_draft = MAX_CHAIN_WINDOW;

    int32_t chain[MAX_CHAIN_WINDOW + DS4_NGRAM_MAX + 1];
    int prefix_len = 0;

    /* Gather up to DS4_NGRAM_MAX tokens before current_token from history */
    const int need = DS4_NGRAM_MAX;
    if (history && n_history > 0) {
        int avail = n_history;
        if (avail > need) avail = need;
        for (int i = 0; i < avail; i++) {
            chain[prefix_len++] = history[n_history - avail + i];
        }
    }
    /* Append current token */
    chain[prefix_len++] = current_token;
    int curr_len = prefix_len;

    uint32_t drafted = 0;

    while (drafted < max_draft) {
        int32_t next_tok = -1;
        float conf = 0.0f;
        bool found = false;

        /* Fallback order: Tier 2 (context) -> Tier 1 (static) -> Tier 3 (dynamic)
         * In each tier: trying 3-gram key, fallback to 2-gram, fallback to 1-gram */

        /* Tier 2: Context cache (in-memory, immediate session) */
        if (context_table && context_table->buckets) {
            for (int g = (curr_len >= DS4_NGRAM_MAX ? DS4_NGRAM_MAX : curr_len); g >= DS4_NGRAM_MIN; g--) {
                const int32_t *prefix = &chain[curr_len - g];
                const uint16_t min_cnt = (g >= 2) ? 1 : 2;
                if (ds4_ngram_table_lookup(context_table, prefix, g, &next_tok, &conf, min_cnt)) {
                    if (conf >= 0.60f || (g >= 3 && conf >= 0.50f)) {
                        found = true;
                        break;
                    }
                }
            }
        }

        /* Tier 1: Static cache (precomputed corpus) */
        if (!found && static_table && static_table->buckets) {
            for (int g = (curr_len >= DS4_NGRAM_MAX ? DS4_NGRAM_MAX : curr_len); g >= DS4_NGRAM_MIN; g--) {
                const int32_t *prefix = &chain[curr_len - g];
                const uint16_t min_cnt = (g >= 3) ? 1 : 2;
                if (ds4_ngram_table_lookup(static_table, prefix, g, &next_tok, &conf, min_cnt)) {
                    if (conf >= 0.40f) {
                        found = true;
                        break;
                    }
                }
            }
        }

        /* Tier 3: Dynamic cache (persisted across runs) */
        if (!found && dynamic_table && dynamic_table->buckets) {
            for (int g = (curr_len >= DS4_NGRAM_MAX ? DS4_NGRAM_MAX : curr_len); g >= DS4_NGRAM_MIN; g--) {
                const int32_t *prefix = &chain[curr_len - g];
                const uint16_t min_cnt = (g >= 3) ? 1 : 2;
                if (ds4_ngram_table_lookup(dynamic_table, prefix, g, &next_tok, &conf, min_cnt)) {
                    if (conf >= 0.50f) {
                        found = true;
                        break;
                    }
                }
            }
        }

        if (!found || next_tok <= 0 || next_tok > 200000) break;

        /* Stop draft if token looks like EOS/EOT */
        if (next_tok == 0) break;

        draft_out[drafted++] = next_tok;
        chain[curr_len++] = next_tok;

        /* Confidence-aware chain truncation: if continuation confidence is moderate
         * (0.50 - 0.70), limit further speculative lookahead to 2 additional tokens
         * to avoid high verification penalties on divergent branches. */
        if (conf < 0.70f && drafted + 2 < max_draft) {
            max_draft = drafted + 2;
        }
    }

    return drafted;
}

uint32_t ds4_ngram_draft(const ds4_ngram_table *context_table,
                         const ds4_ngram_table *dynamic_table,
                         const int32_t *history,
                         int n_history,
                         int32_t current_token,
                         int32_t *draft_out,
                         uint32_t max_draft) {
    return ds4_ngram_draft_3tier(context_table, NULL, dynamic_table,
                                 history, n_history, current_token,
                                 draft_out, max_draft);
}

int ds4_ngram_table_save(const ds4_ngram_table *table, const char *filepath) {
    if (!table || !table->buckets || !filepath) return 0;

    FILE *f = fopen(filepath, "wb");
    if (!f) return 0;

    uint32_t magic = DS4_NGRAM_MAGIC;
    uint32_t version = DS4_NGRAM_VERSION;
    uint32_t num_buckets = table->num_buckets;
    uint32_t total_entries = table->total_entries;

    if (fwrite(&magic, sizeof(magic), 1, f) != 1 ||
        fwrite(&version, sizeof(version), 1, f) != 1 ||
        fwrite(&num_buckets, sizeof(num_buckets), 1, f) != 1 ||
        fwrite(&total_entries, sizeof(total_entries), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    for (uint32_t i = 0; i < table->num_buckets; i++) {
        const ds4_ngram_bucket *b = &table->buckets[i];
        for (int s = 0; s < DS4_NGRAM_WAYS; s++) {
            const ds4_ngram_entry *e = &b->entries[s];
            if (e->gram_len > 0) {
                if (fwrite(e, sizeof(ds4_ngram_entry), 1, f) != 1) {
                    fclose(f);
                    return 0;
                }
            }
        }
    }

    fclose(f);
    return 1;
}

int ds4_ngram_table_load(ds4_ngram_table *table, const char *filepath) {
    if (!table || !filepath) return 0;

    FILE *f = fopen(filepath, "rb");
    if (!f) return 0;

    uint32_t magic = 0, version = 0, num_buckets = 0, total_entries = 0;
    if (fread(&magic, sizeof(magic), 1, f) != 1 ||
        fread(&version, sizeof(version), 1, f) != 1 ||
        fread(&num_buckets, sizeof(num_buckets), 1, f) != 1 ||
        fread(&total_entries, sizeof(total_entries), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    if (magic != DS4_NGRAM_MAGIC || version != DS4_NGRAM_VERSION) {
        fclose(f);
        return 0;
    }

    const uint8_t tier = table->tier;
    if (!table->buckets || table->num_buckets != num_buckets) {
        ds4_ngram_table_free(table);
        if (!ds4_ngram_table_init(table, num_buckets, tier)) {
            fclose(f);
            return 0;
        }
    }

    ds4_ngram_entry e;
    for (uint32_t i = 0; i < total_entries; i++) {
        if (fread(&e, sizeof(ds4_ngram_entry), 1, f) != 1) break;
        if (e.gram_len >= DS4_NGRAM_MIN && e.gram_len <= DS4_NGRAM_MAX) {
            ngram_restore_entry(table, &e);
        }
    }

    fclose(f);
    return 1;
}

const char *ds4_ngram_default_cache_path(void) {
    static char path[512];
    const char *env = getenv("DS4_NGRAM_CACHE_PATH");
    if (env && env[0]) return env;

    const char *home = getenv("HOME");
    if (!home) {
        struct passwd *pw = getpwuid(getuid());
        if (pw) home = pw->pw_dir;
    }
    if (!home) home = "/tmp";

    snprintf(path, sizeof(path), "%s/.cache/ds4", home);
    mkdir(path, 0755);

    /* Check if legacy ngram_dynamic.bin exists when ngram_cache.bin does not yet */
    char legacy_path[512];
    snprintf(legacy_path, sizeof(legacy_path), "%s/.cache/ds4/ngram_dynamic.bin", home);
    snprintf(path, sizeof(path), "%s/.cache/ds4/ngram_cache.bin", home);
    if (access(path, F_OK) != 0 && access(legacy_path, F_OK) == 0) {
        return strncpy(path, legacy_path, sizeof(path));
    }
    return path;
}
