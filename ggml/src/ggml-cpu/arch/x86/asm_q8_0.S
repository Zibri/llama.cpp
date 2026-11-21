/* q8_0 x q8_0 dot product, AVX-512/VNNI, System V AMD64.
 * Args: edi=n, rsi=s, rdx=bs, rcx=vx, r8=bx, r9=vy, [rsp+8]=by, [rsp+16]=nrc.
 */
    .intel_syntax noprefix
    .text
    .globl ggml_vec_dot_q8_0_q8_0_asm
    .type  ggml_vec_dot_q8_0_q8_0_asm, @function
    .align 64
ggml_vec_dot_q8_0_q8_0_asm:
    mov     rax, rdi
    shr     rax, 5
    jz      .Lq8_zero
    vmovaps zmm28, ZMMWORD PTR [rip+.Lperm01]
    vmovaps zmm29, ZMMWORD PTR [rip+.Lperm23]
    vmovaps zmm30, ZMMWORD PTR [rip+.Lperm45]
    vmovaps zmm31, ZMMWORD PTR [rip+.Lperm67]
    vpxor   xmm0, xmm0, xmm0
    vpxor   xmm1, xmm1, xmm1
    vpxor   xmm3, xmm3, xmm3        /* scalar tail accumulation */
    mov     r10, rcx
    mov     r11, r9
    mov     rdx, rax
    shr     rdx, 3
    jz      .Lq8_tail

    .align 32
.Lq8_main:
    /* first four blocks */
    vmovdqu ymm2, YMMWORD PTR [r10+2]
    vmovdqu ymm3, YMMWORD PTR [r10+36]
    vmovdqu ymm4, YMMWORD PTR [r10+70]
    vmovdqu ymm5, YMMWORD PTR [r10+104]
    vmovdqu ymm6, YMMWORD PTR [r11+2]
    vmovdqu ymm7, YMMWORD PTR [r11+36]
    vmovdqu ymm8, YMMWORD PTR [r11+70]
    vmovdqu ymm9, YMMWORD PTR [r11+104]
    vpsignb ymm10, ymm2, ymm2
    vpsignb ymm11, ymm6, ymm2
    vpxor   ymm12, ymm12, ymm12
    vpdpbusd ymm12, ymm10, ymm11
    vpsignb ymm10, ymm3, ymm3
    vpsignb ymm11, ymm7, ymm3
    vpxor   ymm13, ymm13, ymm13
    vpdpbusd ymm13, ymm10, ymm11
    vpsignb ymm10, ymm4, ymm4
    vpsignb ymm11, ymm8, ymm4
    vpxor   ymm14, ymm14, ymm14
    vpdpbusd ymm14, ymm10, ymm11
    vpsignb ymm10, ymm5, ymm5
    vpsignb ymm11, ymm9, ymm5
    vpxor   ymm15, ymm15, ymm15
    vpdpbusd ymm15, ymm10, ymm11
    vinserti64x4 zmm22, zmm12, ymm13, 1
    vinserti64x4 zmm23, zmm14, ymm15, 1
    vcvtdq2ps zmm22, zmm22
    vcvtdq2ps zmm23, zmm23

    /* next four blocks */
    vmovdqu ymm2, YMMWORD PTR [r10+138]
    vmovdqu ymm3, YMMWORD PTR [r10+172]
    vmovdqu ymm4, YMMWORD PTR [r10+206]
    vmovdqu ymm5, YMMWORD PTR [r10+240]
    vmovdqu ymm6, YMMWORD PTR [r11+138]
    vmovdqu ymm7, YMMWORD PTR [r11+172]
    vmovdqu ymm8, YMMWORD PTR [r11+206]
    vmovdqu ymm9, YMMWORD PTR [r11+240]
    vpsignb ymm10, ymm2, ymm2
    vpsignb ymm11, ymm6, ymm2
    vpxor   ymm12, ymm12, ymm12
    vpdpbusd ymm12, ymm10, ymm11
    vpsignb ymm10, ymm3, ymm3
    vpsignb ymm11, ymm7, ymm3
    vpxor   ymm13, ymm13, ymm13
    vpdpbusd ymm13, ymm10, ymm11
    vpsignb ymm10, ymm4, ymm4
    vpsignb ymm11, ymm8, ymm4
    vpxor   ymm14, ymm14, ymm14
    vpdpbusd ymm14, ymm10, ymm11
    vpsignb ymm10, ymm5, ymm5
    vpsignb ymm11, ymm9, ymm5
    vpxor   ymm15, ymm15, ymm15
    vpdpbusd ymm15, ymm10, ymm11
    vinserti64x4 zmm24, zmm12, ymm13, 1
    vinserti64x4 zmm25, zmm14, ymm15, 1
    vcvtdq2ps zmm24, zmm24
    vcvtdq2ps zmm25, zmm25

    /* eight fp16 scales -> eight floats */
    vmovd   xmm2, DWORD PTR [r10]
    vpinsrw xmm2, xmm2, WORD PTR [r10+34],  1
    vpinsrw xmm2, xmm2, WORD PTR [r10+68],  2
    vpinsrw xmm2, xmm2, WORD PTR [r10+102], 3
    vpinsrw xmm2, xmm2, WORD PTR [r10+136], 4
    vpinsrw xmm2, xmm2, WORD PTR [r10+170], 5
    vpinsrw xmm2, xmm2, WORD PTR [r10+204], 6
    vpinsrw xmm2, xmm2, WORD PTR [r10+238], 7
    vmovd   xmm3, DWORD PTR [r11]
    vpinsrw xmm3, xmm3, WORD PTR [r11+34],  1
    vpinsrw xmm3, xmm3, WORD PTR [r11+68],  2
    vpinsrw xmm3, xmm3, WORD PTR [r11+102], 3
    vpinsrw xmm3, xmm3, WORD PTR [r11+136], 4
    vpinsrw xmm3, xmm3, WORD PTR [r11+170], 5
    vpinsrw xmm3, xmm3, WORD PTR [r11+204], 6
    vpinsrw xmm3, xmm3, WORD PTR [r11+238], 7
    vcvtph2ps ymm10, xmm2
    vcvtph2ps ymm11, xmm3
    vmulps  ymm10, ymm10, ymm11

    vpermps zmm18, zmm28, zmm10
    vpermps zmm19, zmm29, zmm10
    vpermps zmm20, zmm30, zmm10
    vpermps zmm21, zmm31, zmm10
    vfmadd231ps zmm0, zmm22, zmm18
    vfmadd231ps zmm1, zmm23, zmm19
    vfmadd231ps zmm0, zmm24, zmm20
    vfmadd231ps zmm1, zmm25, zmm21

    add     r10, 272
    add     r11, 272
    dec     rdx
    jnz     .Lq8_main

.Lq8_tail:
    and     rax, 7
    jz      .Lq8_finish
    .align 16
.Lq8_tail_loop:
    vmovdqu ymm4, YMMWORD PTR [r10+2]
    vmovdqu ymm5, YMMWORD PTR [r11+2]
    vpsignb ymm6, ymm4, ymm4
    vpsignb ymm7, ymm5, ymm4
    vpxor   ymm8, ymm8, ymm8
    vpdpbusd ymm8, ymm6, ymm7
    vcvtdq2ps ymm8, ymm8
    vmovd   xmm9,  DWORD PTR [r10]
    vmovd   xmm10, DWORD PTR [r11]
    vcvtph2ps xmm9, xmm9
    vcvtph2ps xmm10, xmm10
    vmulss  xmm9, xmm9, xmm10
    vbroadcastss ymm9, xmm9
    vmulps  ymm8, ymm8, ymm9
    vextractf128 xmm10, ymm8, 1
    vaddps  xmm8, xmm8, xmm10
    vmovhlps xmm10, xmm8, xmm8
    vaddps  xmm8, xmm8, xmm10
    vshufps xmm10, xmm8, xmm8, 0x55
    vaddss  xmm3, xmm3, xmm8
    vaddss  xmm3, xmm3, xmm10
    add     r10, 34
    add     r11, 34
    dec     rax
    jnz     .Lq8_tail_loop

.Lq8_finish:
    vaddps  zmm0, zmm0, zmm1
    vextractf64x4 ymm2, zmm0, 1
    vaddps  ymm0, ymm0, ymm2
    vextractf128 xmm2, ymm0, 1
    vaddps  xmm0, xmm0, xmm2
    vmovhlps xmm2, xmm0, xmm0
    vaddps  xmm0, xmm0, xmm2
    vshufps xmm2, xmm0, xmm0, 0x55
    vaddss  xmm0, xmm0, xmm2
    vaddss  xmm0, xmm0, xmm3
    vmovss  DWORD PTR [rsi], xmm0
    vzeroupper
    ret
.Lq8_zero:
    vpxor   xmm0, xmm0, xmm0
    vmovss  DWORD PTR [rsi], xmm0
    ret
    .size ggml_vec_dot_q8_0_q8_0_asm, .-ggml_vec_dot_q8_0_q8_0_asm

    .section .rodata
    .align 64
.Lperm01: .long 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1
.Lperm23: .long 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3
.Lperm45: .long 4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5
.Lperm67: .long 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7
