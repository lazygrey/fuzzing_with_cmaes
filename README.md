# Fuzzing with CMA-ES
A coverage-based Fuzzer with [cmaes](https://en.wikipedia.org/wiki/CMA-ES) for C-programs.

## Requirements
Operating System :: Linux only

1. NumPy 
    * Install example:
        ```
        pip3 install numpy
        ```


2. pycma (cmaes in python)
    * Install example: more detail at [pycma](https://github.com/CMA-ES/pycma)

        ```
        python3 -m pip3 install cma
        ```
3. Python 3 with version with at least Python 3.7
4. GCC (GCC Compiler)


## Arguments
In a Terminal:

```
python3 fuzzer.py [-h] [-od OUTPUT_DIR] [-ld LOG_DIR] [-ip INIT_POPSIZE]
                 [-mp MAX_POPSIZE] [-m MODE] [-me MAX_EVALUATIONS] [-s SEED]
                 [-t TIMEOUT] [-o OBJECTIVE] [-hrt HOT_RESTART_THRESHOLD]
                 [-nr] [-hr] [-si] [-ll]
                 program_path [program_path ...]

```
POSITIONAL:
```
program_path          relative program path to test (only last argument will be regarded as program path)
```

OPTIONAL:
```
-h, --help            show this help message and exit
-od OUTPUT_DIR, --output_dir OUTPUT_DIR
                    directory for complied and executable programs
-ld LOG_DIR, --log_dir LOG_DIR
                    directory for logs
-ip INIT_POPSIZE, --init_popsize INIT_POPSIZE
                    initial population size for CMA-ES to start with
-mp MAX_POPSIZE, --max_popsize MAX_POPSIZE
                    maximum population size for CMA-ES
-me MAX_EVALUATIONS, --max_evaluations MAX_EVALUATIONS
                    maximum evaluations for CMA-ES-Fuzzer
-s SEED, --seed SEED  seed to control the randomness
-st SAMPLE_TYPE, --sample_type SAMPLE_TYPE
                    type of samples that CMA-ES-Fuzzer work with
-t TIMEOUT, --timeout TIMEOUT
                    timeout in seconds
-ct COVERAGE_TYPE, --coverage_type COVERAGE_TYPE
                    type of coverage for obejctive function for CMA-ES-Fuzzer
-hrt HOT_RESTART_THRESHOLD, --hot_restart_threshold HOT_RESTART_THRESHOLD
                    threshold for the optimized sigma vector to decide whether their components, among mean vector components, are reset to default for the hot restart.
-nr, --no_reset       deactivate reset after not optimized (this is for the most basic version)
-hr, --hot_restart    activate hot restart while optimizing samples
-si, --save_interesting
                    save interesting paths while optimizing
--strategy STRATEGY   strategy label for log and csv
-ll, --live_logs      write logs as txt file in log files whenever it changes
```


## Fuzzing Examples
Example 1: timout in 10 seconds
```bash
python3 fuzzer.py examples/test.c -t 10
```

Example 2: hot restart
```bash
python3 fuzzer.py user_program_dir/user_program.c --hot_restart
```

Example 3: maximum population size is 100
```bash
python3 fuzzer.py user_program_dir/user_program.c -mp 100
```

## Log Examples:
Example 1: timout in 10 seconds
```
fuzzer args:
fuzzer.py examples/test.c
program_path: examples/test.c
initial parameters:
no_reset  hot_restart  save_interesting  mode  coverage_type  input_size  max_popsize  popsize_scale  max_gens  max_eval  timeout  seed  strategy  
False     False        False             bytes branch         2           1000         10             1000      100000    840      700   None      

-------------------------------------------------------------------------------------------------------------------------------------------
logs:
fuzzer_state   optimized   popsize   current_testcase   total_testcase   generations   current_coverage   total_coverage   evaluations   time   CMA_ES_seed   
optimizing     -           10        1                  1                0             39.29              39.29            1             0.15   700           
optimizing     -           10        1                  1                1             50.0               50.0             12            0.24   700           
done           True        10        1                  1                9             50.0               50.0             99            0.89   700           
optimizing     -           10        2                  2                0             53.57              53.57            100           0.9    701           
optimizing     -           10        2                  2                0             78.57              78.57            102           0.92   701           
done           True        10        2                  2                7             78.57              78.57            176           1.45   701           
optimizing     -           10        3                  3                0             82.14              82.14            177           1.46   702           
done           True        10        3                  3                6             82.14              82.14            242           1.93   702           
optimizing     -           10        4                  4                0             85.71              85.71            244           1.95   703           
done           True        10        4                  4                8             85.71              85.71            330           2.55   703           
optimizing     -           10        5                  5                0             89.29              89.29            332           2.57   704           
done           True        10        5                  5                10            89.29              89.29            440           3.36   704           
optimizing     -           10        6                  6                0             92.86              92.86            443           3.38   705           
done           True        10        6                  6                10            92.86              92.86            550           4.14   705           
optimizing     -           10        7                  7                0             96.43              96.43            552           4.16   706           
done           True        10        7                  7                8             96.43              96.43            638           4.79   706           
optimizing     -           10        8                  8                0             100.0              100.0            644           4.84   707           
done           True        10        8                  8                0             100.0              100.0            644           4.84   707           

-------------------------------------------------------------------------------------------------------------------------------------------
final report:
total_testcase        total_coverage        stop_reason        testcase_statuses
      8               100.0               total coverage is 100%               ['SAFE', 'SAFE', 'SAFE', 'SAFE', 'SAFE', 'ERROR', 'SAFE', 'SAFE']         
execution time for each method:
stop    get_executed_paths  _encode_bytes  cal_branches  ask     _delete_gcda  tell    _gcov   get_branches  _run    objective  optimize_sample  
0.0039  0.0117              0.0199         0.028         0.0423  0.0484        0.0701  2.0414  2.1231        2.2194  4.3818     4.6987           

-------------------------------------------------------------------------------------------------------------------------------------------
total sample len: 8
total samples: [array([ 31.13450664, 115.03648683]), array([193.08750047, 135.77202659]), array([91.40600764, 64.95261197]), array([135.28334219, 161.639187  ]), array([112.72864089,  23.40044769]), array([142.87455394,  37.00855194]), array([135.458315  , 216.68228025]), array([ 13.41027645, 250.72958518])]
total input vectors: [bytearray(b'\x1fs'), bytearray(b'\xc1\x87'), bytearray(b'[@'), bytearray(b'\x87\xa1'), bytearray(b'p\x17'), bytearray(b'\x8e%'), bytearray(b'\x87\xd8'), bytearray(b'\r\xfa')]
line_coverage: 1.0
branch_coverage: 1.0
total_eval: 644
seed: 700
```