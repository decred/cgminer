#ifndef BLAKE_H
#define BLAKE_H

#include "miner.h"

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8; 

typedef struct { 
  u32 h[8], s[4], t[2];
  int buflen, nullt;
  u8  buf[64];
} state __attribute__ ((aligned (16)));

extern int blake256_test(unsigned char *pdata, const unsigned char *ptarget, uint32_t nonce);
extern void blake256_regenhash(struct work *work);

void blake256_init(state *S);
void blake256_update(state *S, const u8 *data, u64 datalen, u8 revinput);

#endif /* BLAKE_H */
