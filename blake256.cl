// Blake-256 14-round kernel for Decred mining
// By Wolf (Wolf0 aka Wolf9466)

#pragma OPENCL EXTENSION cl_amd_media_ops : enable

#ifndef WORKSIZE
#define WORKSIZE						256
#endif

static const __constant uchar sigma[10][16] =
{
	{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
	{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
	{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
	{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
	{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
	{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
	{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
	{  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
	{ 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
};

static const __constant uint cst[16] =
{
	0x243F6A88U, 0x85A308D3U, 0x13198A2EU, 0x03707344U,
	0xA4093822U, 0x299F31D0U, 0x082EFA98U, 0xEC4E6C89U,
	0x452821E6U, 0x38D01377U, 0xBE5466CFU, 0x34E90C6CU,
	0xC0AC29B7U, 0xC97C50DDU, 0x3F84D5B5U, 0xB5470917U
};

#define BLAKE_MIX_V(A, B, C, D, m0, m1, c0, c1) do { \
	A += B + (m0 ^ c1); \
	D = amd_bytealign(D ^ A, D ^ A, 2U); \
	C += D; \
	B = amd_bitalign(B ^ C, B ^ C, 12U); \
	A += B + (m1 ^ c0); \
	D = amd_bytealign(D ^ A, D ^ A, 1U); \
	C += D; \
	B = amd_bitalign(B ^ C, B ^ C, 7U); \
} while(0)

#define BLAKE_RND_V(rnd)	do { \
	BLAKE_MIX_V(V[0], V[1], V[2], V[3], (uint4)(M[sigma[rnd][0]], M[sigma[rnd][2]], M[sigma[rnd][4]], M[sigma[rnd][6]]), (uint4)(M[sigma[rnd][1]], M[sigma[rnd][3]], M[sigma[rnd][5]], M[sigma[rnd][7]]), (uint4)(cst[sigma[rnd][0]], cst[sigma[rnd][2]], cst[sigma[rnd][4]], cst[sigma[rnd][6]]), (uint4)(cst[sigma[rnd][1]], cst[sigma[rnd][3]], cst[sigma[rnd][5]], cst[sigma[rnd][7]])); \
	BLAKE_MIX_V(V[0], V[1].s1230, V[2].s2301, V[3].s3012, (uint4)(M[sigma[rnd][8]], M[sigma[rnd][10]], M[sigma[rnd][12]], M[sigma[rnd][14]]), (uint4)(M[sigma[rnd][9]], M[sigma[rnd][11]], M[sigma[rnd][13]], M[sigma[rnd][15]]), (uint4)(cst[sigma[rnd][8]], cst[sigma[rnd][0xA]], cst[sigma[rnd][0xC]], cst[sigma[rnd][0xE]]), (uint4)(cst[sigma[rnd][9]], cst[sigma[rnd][0xB]], cst[sigma[rnd][0xD]], cst[sigma[rnd][0xF]])); \
} while(0)

#define BLAKE_FIRST_MIX_V(A, B, C, D, m0, m1, c0, c1) do { \
	A.s1 += (m1 ^ c0); \
	D.s1 = amd_bytealign(D.s1 ^ A.s1, D.s1 ^ A.s1, 1U); \
	C.s1 += D.s1; \
	B.s1 = amd_bitalign(B.s1 ^ C.s1, B.s1 ^ C.s1, 7U); \
} while(0)

#define BLAKE_FIRST_RND_V(rnd)	do { \
	BLAKE_FIRST_MIX_V(V[0], V[1], V[2], V[3], M[sigma[rnd][2]], M[sigma[rnd][3]], cst[sigma[rnd][2]], cst[sigma[rnd][3]]); \
	BLAKE_MIX_V(V[0], V[1].s1230, V[2].s2301, V[3].s3012, (uint4)(M[sigma[rnd][8]], M[sigma[rnd][10]], M[sigma[rnd][12]], M[sigma[rnd][14]]), (uint4)(M[sigma[rnd][9]], M[sigma[rnd][11]], M[sigma[rnd][13]], M[sigma[rnd][15]]), (uint4)(cst[sigma[rnd][8]], cst[sigma[rnd][0xA]], cst[sigma[rnd][0xC]], cst[sigma[rnd][0xE]]), (uint4)(cst[sigma[rnd][9]], cst[sigma[rnd][0xB]], cst[sigma[rnd][0xD]], cst[sigma[rnd][0xF]])); \
} while(0)

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(__global uint *restrict Output, __global const uint *restrict Midstate, __global const uint *restrict BlockHdr, const ulong DCRPrev7, const ulong Target)
{
	uint M[16] = { 0U };
	uint4 V[4];
	
	uint nonce = get_global_id(0);
	
	((uint16 *)V)[0] = vload16(0, Midstate);

	// Load the block header and padding		
	((uint8 *)M)[0] = vload8(0, BlockHdr);
	M[3] = nonce;
	M[8] = BlockHdr[8];
	M[9] = BlockHdr[9];
	M[10] = BlockHdr[10];
	M[11] = BlockHdr[11];
	M[12] = BlockHdr[12];
	M[13] = 0x80000001U;
	M[14] = 0x00000000U;
	M[15] = 0x000005A0U;
	
	BLAKE_FIRST_RND_V(0);
	BLAKE_RND_V(1);
	BLAKE_RND_V(2);
	BLAKE_RND_V(3);
	BLAKE_RND_V(4);
	BLAKE_RND_V(5);
	BLAKE_RND_V(6);
	BLAKE_RND_V(7);
	BLAKE_RND_V(8);
	BLAKE_RND_V(9);
	BLAKE_RND_V(0);
	BLAKE_RND_V(1);
	BLAKE_RND_V(2);
	BLAKE_RND_V(3);
	
	if((DCRPrev7 ^ as_ulong(V[1].s23 ^ V[3].s23)) <= Target) Output[atomic_inc(Output+0xFF)] = nonce;
}
