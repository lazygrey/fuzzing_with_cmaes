#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>


static int OVER_MAX_INPUT_SIZE = 3;
static int ASSUME = 10;
static int ERROR = 100;

// ssize_t _read(void *p, size_t n);

unsigned int parse_signed(unsigned int x, int bits) {
    return x / (1U << (32 - bits)) - (1U << (bits - 1));
}

unsigned int parse_unsigned(unsigned int x, int bits) {
    return x / (1U << (32 - bits));
}

unsigned int _read(size_t n, int is_signed) {
    unsigned int x;
    read(0, &x, sizeof(x));
    // printf("real: %d\n", x); // test
    // printf("bytes: %x\n", x); // test
    // printf("size: %ld\n", n); // test
    if (is_signed) {
        return parse_signed(x, 8*n);
    } else {
        return parse_unsigned(x, 8*n);
    }   
}

unsigned long parse_signed_long(unsigned long x, long bits) {
    return x / (1L << (64 - bits * 8)) - (1L << (bits * 8 - 1));
}

unsigned long parse_unsigned_long(unsigned long x, int bits) {
    return x / (1L << (64 - bits * 8));
}

unsigned long _read2(size_t n, int is_signed) {
    unsigned long x;
    read(0, &x, sizeof(x));
    if (is_signed) {
        return parse_signed_long(x, 8*n);
    } else {
        return parse_unsigned_long(x, 8*n);
    }
}

void __VERIFIER_error() {
    exit(ERROR);
}

// ssize_t _read(void *p, size_t n) {
//     long dump = 0;
//     read(0, &dump, 8 - n);
//     read(0, p, n);
//     // char inputs[n];
//     // long dump;
//     // read(0, &dump, 8 - n); // ignore first (8 - n)  bytes
//     // char input;
//     // // read(0, input, n);
//     // size_t i;
//     // for (i = 0; i < n; i++){
//     //     read(0, &inputs[i], 1);
//     // }
//     // // read(0, p, n);

//     // memcpy(p, &inputs, n);
//     return n;
// }

// void _read2(void *buf, size_t n) {
//     // char* inputs;
//     long dump;
//     read(0, &dump, 8 - n); // ignore first (8 - n)  bytes
//     // read(0, input, n);
//     size_t i;
//     read(0, buf, n);
//     // printf("")
//     // for (i = 0; i < n; i++){
//     // }
// }

_Bool __VERIFIER_nondet_bool() {
    _Bool x = 0;
    x = _read(sizeof(x), 1);
    // printf("  <input type=\"bool\">%d</input>%ld</input size>\n", x, sizeof(x));
    return x;
}

char __VERIFIER_nondet_char() {
    char x = 0;
    x = _read(sizeof(x), 1);
    // printf("  <input type=\"char\">%d</input>%ld</input size>\n", x, sizeof(x));
    return x;
}

unsigned char __VERIFIER_nondet_uchar() {
    unsigned char x = 0;
    x = _read(sizeof(x), 0);
    // printf("  <input type=\"unsigned char\">%u</input>\n", x);
    return x;
}

short __VERIFIER_nondet_short() {
    short x = 0;
    x = _read(sizeof(x), 1);
    // scanf("%hd",&x);
    // printf("  <input type=\"short\">%hi</input>\n", x);
    return x;
}

unsigned short __VERIFIER_nondet_ushort() {
    unsigned short x = 0;
    x = _read(sizeof(x), 0);
    // printf("  <input type=\"unsigned short\">%hu</input>\n", x);
    return x;
}

unsigned long __VERIFIER_nondet_unsigned_long() {
    unsigned long x = 0;
    x = _read2(sizeof(x), 0);
    // printf("  <input type=\"unsigned long\">%lu</input>\n", x);
    return x;
}

// void * __VERIFIER_nondet_pointer() {
//     unsigned long x = 0;
//   
    // x = _read(sizeof(x), 1);
    // printf("  <input type=\"unsigned long\">%lu</input>\n", x);
//     return (void *) x;
// }

long __VERIFIER_nondet_long() {
    long x = 0;
    x = _read2(sizeof(x), 1);
    // printf("  <input type=\"long\">%li</input>\n", x);
    return x;
}

unsigned int __VERIFIER_nondet_uint() {
    unsigned int x = 0;
    x = _read(sizeof(x), 0);
    // printf("  <input type=\"unsigned int\">%u</input>\n", x);
    return x;
}

int __VERIFIER_nondet_int() {
    int x = 0;
    x = _read(sizeof(x), 1);
    // printf("  <input type=\"int\">%d</input>\n", x);
    return x;
}

unsigned __VERIFIER_nondet_unsigned() {
    unsigned x = 0;
    x = _read(sizeof(x), 0);
    // printf("  <input type=\"unsigned\">%d</input>\n", x);
    return x;
}

unsigned long __VERIFIER_nondet_ulong() {
    unsigned long x = 0;
    x = _read2(sizeof(x), 0);
    // printf("  <input type=\"unsigned long\">%lu</input>\n", x);
    return x;
}

float __VERIFIER_nondet_float() {
    float x = 0.0;
    x = _read(sizeof(x), 1);
    // printf("  <input type=\"float\">%f</input>\n", x);
    return x;
}

double __VERIFIER_nondet_double() {
    double x = 0.0;
    x = _read2(sizeof(x), 1);
    // printf("  <input type=\"double\">%lf</input>\n", x);
    return x;
}


// int __VERIFIER_nondet_const_char_pointer() {

// }

//int __VERIFIER_nondet_S8() {
// How many bytes in S8?
//	return __my_read_int(8);
//}

void __VERIFIER_assume(int arg) {
    if (!arg) {
        // printf("!!!arg: %d\n", arg);
        // printf("!!!!!!!!!!!!assume error\n");
        // __VERIFIER_assume(arg);
        exit(ASSUME);
    }
}
