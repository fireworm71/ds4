# T1 prototype -- one-sided RDMA READ expert transport (plan task T1).
# Standalone: links libibverbs directly. ds4 itself dlopen()s verbs instead
# (ds4_tp.c), which a prototype has no reason to imitate.
CC      ?= cc
CFLAGS  ?= -O2 -g -Wall -Wextra -Wno-unused-parameter
LDLIBS   = -libverbs

all: t1_server t1_client

t1_server: t1_server.c t1_common.h
	$(CC) $(CFLAGS) -o $@ t1_server.c $(LDLIBS)

t1_client: t1_client.c t1_common.h
	$(CC) $(CFLAGS) -o $@ t1_client.c $(LDLIBS)

clean:
	rm -f t1_server t1_client
.PHONY: all clean
