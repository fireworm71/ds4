import re, json
src = open("/home/jason/ds4-41flash/.claude/worktrees/q3-tp2/ds4.c").read()

def grab(name, count):
    i = src.index(f"{name}[{count}] = {{")
    j = src.index("};", i)
    body = src[src.index("{", i) + 1:j]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    vals = [v.strip() for v in body.split(",") if v.strip()]
    out = [int(v, 0) for v in vals]
    assert len(out) == count, (name, len(out), count)
    return out

tables = {
    "kmask_iq2xs": grab("kmask_iq2xs", 8),
    "ksigns_iq2xs": grab("ksigns_iq2xs", 128),
    "iq2xxs_grid": grab("iq2xxs_grid", 256),
}
json.dump(tables, open("/home/jason/.claude/jobs/882780ed/tmp/iq2_tables.json", "w"))
print({k: (len(v), hex(v[0]), hex(v[-1])) for k, v in tables.items()})
