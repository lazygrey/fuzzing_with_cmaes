#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
// #include <assert.h>


static int ERROR = 100;
static int ASSUME = 101;
static int OVER_MAX_INPUT_SIZE = 102;
static int OVER_TOTAL_INPUT_SIZE = 104;

static int MAX_INPUT_SIZE = 1000;

static int input_size;

void _print_input_size(){
    printf("n%d",input_size);
}

unsigned int _read(size_t n, int signed_) {
    if (input_size == 0) {
        atexit(_print_input_size);
    }
    input_size += 1;
    unsigned int x;
    if (!read(0, &x, sizeof(x))){
        return 0;
    }
    return (x >> (32 - 8*n)) - signed_ * (1U<<(8*n-1));
}

unsigned long _read2(size_t n, int signed_) {
    if (input_size == 0) {
        atexit(_print_input_size);
    }
    input_size += 2;
    unsigned long x;
    if (!read(0, &x, sizeof(x))){
        return 0;
    }
    return (x >> (64 - 8*n)) - signed_ * (1UL<<(8*n-1));
}

void __VERIFIER_error() {
    exit(ERROR);
}

char __VERIFIER_nondet_char() {
    char x = 0;
    x = _read(sizeof(x), 1);
    // printf("  <input type=\"char\">%d</input>%ld</input size>\n", x, sizeof(x));
    return x;
}

_Bool __VERIFIER_nondet_bool() {
    // _Bool x = 0;
    // x = _read(sizeof(x), 0);
    // // printf("  <input type=\"bool\">%d</input>%ld</input size>\n", x, sizeof(x));
    // return x;
    char x = __VERIFIER_nondet_char();
    if (x < 0) {
        return 0;
    } else {
        return 1;
    }
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
    float f = 0.0;
    unsigned int x;
    x = _read(sizeof(x), 0);
    f = *(float*) &x;
    // printf("  <input type=\"float\">%f</input>\n", f);
    return f;
}

double __VERIFIER_nondet_double() {
    double d = 0.0;
    long x;
    x = _read2(sizeof(x), 0);
    d = *(double*) &x;
    // printf("  <input type=\"double\">%f</input>\n", d);
    return d;
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
