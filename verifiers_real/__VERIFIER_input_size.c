#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

/* both variables are implicitly initialized to 0 */
static int initialized;
static int finalized;
static int total_input_size;

static int seed = 123;
static int MAX_INPUT_SIZE = 1000;

static int ERROR = 100;
static int ASSUME = 101;
static int OVER_MAX_INPUT_SIZE = 102;
static int INPUT_SIZE_EXECUTED = 103;

void _initialize();
void _finalize();
size_t _read(void *p, size_t n);

void _initialize() {
    if(initialized)
        return;
    initialized = 1;
    read(0, &seed, sizeof(seed));
    // fflush(stdout);

    // srand(time(NULL));
    // struct timeval t;
    // gettimeofday(&t, NULL);
    // srand(t.tv_usec * t.tv_sec);
    srand(seed);
    atexit(_finalize);
}

void _finalize() {
    if (finalized) {
        return;
    }
    finalized = 1;
    printf("n%d", total_input_size);
    total_input_size = 0;
    exit(INPUT_SIZE_EXECUTED);
    // FILE * file;
    /* open the file for writing*/
    // file = fopen("inputsize.txt","w");

    /* write total_input_size into the file stream*/
    // total_input_size *= 2;
    // fprintf (file, "%d",total_input_size);

    /* close the file*/  
    // fclose (file);
    // printf("%d\n", total_input_size);
    // fflush(stdout);
}

size_t _read(void *p, size_t n) {
    unsigned char *x = (unsigned char*)p;
    size_t i;
    total_input_size++;
    if (n > 4) {
        total_input_size++;
    }

    if(total_input_size >= MAX_INPUT_SIZE){
        _finalize();
    }
    x[0] = rand()%5;
    return n;
}

void __VERIFIER_error() {
    _finalize();
    // error_counter++;
    // exit(ERROR);
}

_Bool __VERIFIER_nondet_bool() {
    _initialize();
    _Bool x = 0;
    _read(&x, sizeof(x));
    return x;
}

char __VERIFIER_nondet_char() {
    _initialize();
    char x = 0;
    _read(&x, sizeof(x));
    return x;
}

unsigned char __VERIFIER_nondet_uchar() {
    _initialize();
    unsigned char x = 0;
    _read(&x, sizeof(x));
    return x;
}

short __VERIFIER_nondet_short() {
    _initialize();
    short x = 0;
    _read(&x, sizeof(x));
    return x;
}

unsigned short __VERIFIER_nondet_ushort() {
    _initialize();
    unsigned short x = 0;
    _read(&x, sizeof(x));
    return x;
}

unsigned long __VERIFIER_nondet_unsigned_long() {
    _initialize();
    unsigned long x = 0;
    _read(&x, sizeof(x));
    return x;
}

long __VERIFIER_nondet_long() {
    _initialize();
    long x = 0;
    _read(&x, sizeof(x));
    return x;
}

unsigned int __VERIFIER_nondet_uint() {
    _initialize();
    unsigned int x = 0;
    _read(&x, sizeof(x));
    return x;
}

int __VERIFIER_nondet_int() {
    _initialize();
    int x = 0;
    _read(&x, sizeof(x));
    return x;
}

unsigned __VERIFIER_nondet_unsigned() {
    _initialize();
    unsigned x = 0;
    _read(&x, sizeof(x));
    return x;
}

unsigned long __VERIFIER_nondet_ulong() {
    _initialize();
    unsigned long x = 0;
    _read(&x, sizeof(x));
    return x;
}

float __VERIFIER_nondet_float() {
    _initialize();
    float x = 0.0;
    _read(&x, sizeof(x));
    return x;
}

double __VERIFIER_nondet_double() {
    _initialize();
    double x = 0.0;
    _read(&x, sizeof(x));
    return x;
}

void __VERIFIER_assume(int arg) {
    // if (!arg) {
    //     // printf("!!!arg: %d\n", arg);
    //     // __VERIFIER_assume(arg);
    //     exit(10);
    // }
    _finalize();
}

// int main() {

//     return 0;
// }