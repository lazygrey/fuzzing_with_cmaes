#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
// #include <assert.h>


static int ERROR = 100;
static int ASSUME = 101;
static int OVER_MAX_INPUT_SIZE = 102;

static int MAX_INPUT_SIZE = 1000;

static int input_size;

void __VERIFIER_error() {
    exit(ERROR);
}

void _print_input_size(){
    printf("n%d",input_size);
}

ssize_t _read (void * p, size_t n) {
    if (input_size == 0) {
        atexit(_print_input_size);
    }
    input_size += n;
    return read(0, p, n);
}

char __VERIFIER_nondet_char() {
    char x = 0;
    _read(&x, sizeof(x));
    // printf("  <input type=\"char\">%d</input>%ld</input size>\n", x, sizeof(x));
    return x;
}

_Bool __VERIFIER_nondet_bool() {
    // _Bool x = 0;
    // _read(&x, sizeof(x));
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
    _read(&x, sizeof(x));
    // printf("  <input type=\"unsigned char\">%u</input>\n", x);
    return x;
}

short __VERIFIER_nondet_short() {
    short x = 0;
    _read(&x, sizeof(x));
    // printf("  <input type=\"short\">%hi</input>\n", x);
    return x;
}

unsigned short __VERIFIER_nondet_ushort() {
    unsigned short x = 0;
    _read(&x, sizeof(x));
    // printf("  <input type=\"unsigned short\">%hu</input>\n", x);
    return x;
}

unsigned long __VERIFIER_nondet_unsigned_long() {
    unsigned long x = 0;
    _read(&x, sizeof(x));
    // printf("  <input type=\"unsigned long\">%lu</input>\n", x);
    return x;
}

// void * __VERIFIER_nondet_pointer() {
//     unsigned long x = 0;
//     _read(&x, sizeof(x));
    // printf("  <input type=\"unsigned long\">%lu</input>\n", x);
//     return (void *) x;
// }

long __VERIFIER_nondet_long() {
    long x = 0;
    _read(&x, sizeof(x));
    // printf("  <input type=\"long\">%li</input>\n", x);
    return x;
}

unsigned int __VERIFIER_nondet_uint() {
    unsigned int x = 0;
    _read(&x, sizeof(x));
    // printf("  <input type=\"unsigned int\">%u</input>\n", x);
    return x;
}

int __VERIFIER_nondet_int() {
    int x = 0;
    _read(&x, sizeof(x));
    // printf("  <input type=\"int\">%d</input>\n", x);
    return x;
}

unsigned __VERIFIER_nondet_unsigned() {
    unsigned x = 0;
    _read(&x, sizeof(x));
    // printf("  <input type=\"unsigned\">%d</input>\n", x);
    return x;
}

unsigned long __VERIFIER_nondet_ulong() {
    unsigned long x = 0;
    _read(&x, sizeof(x));
    // printf("  <input type=\"unsigned long\">%lu</input>\n", x);
    return x;
}

float __VERIFIER_nondet_float() {
    float x = 0.0;
    _read(&x, sizeof(x));
    // printf("  <input type=\"float\">%f</input>\n", x);
    return x;
}

double __VERIFIER_nondet_double() {
    double x = 0.0;
    _read(&x, sizeof(x));
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
