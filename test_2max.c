#include <stdio.h>
#include <unistd.h>
#include "__VERIFIER.h"

// extern char __VERIFIER_nondet_char();

int main() {
    short x = __VERIFIER_nondet_short();
    // read(0, &x, sizeof(x));
    short y = -32768;
    short z = 8192;
    if (x < y + z) {
        x += 0;
    }
    y += z;
    if (y <= x && x < y + z) {
        printf("1st max ");
        x += 0;
        x += 0;
        x += 0;
        x += 0;
        x += 0;
        x += 0;
        x += 0;
    }
    y += z;
    if (y <= x && x < y + z) {
        x += 0;
        x += 0;
        x += 0;
        x += 0;
    }
    y += z;
    if (y <= x && x < y + z) {
        x += 0;
        x += 0;
        x += 0;
        x += 0;
    }
    y += z;
    if (y <= x && x < y + z) {
        x += 0;
        x += 0;
    }
    y += z;
    if (y <= x && x < y + z) {
        printf("2nd max ");
        x += 0;
        x += 0;
        x += 0;
        x += 0;
        x += 0;
    }
    y += z;
    if (y <= x && x < y + z) {
        x += 0;
        x += 0;
    }
    y += z;
    if (y <= x) {
        x += 0;
    }
    return 0;
}