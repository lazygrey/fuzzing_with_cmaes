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
-nr, --no_reset       deactivate reset after not optimized (only for testing)
-hr, --hot_restart    activate hot restart while optimizing samples
-si, --save_interesting
                    save interesting coverage item ids while optimizing
-is INPUT_SIZE, --input_size INPUT_SIZE
                    fixed input size for CMA-ES
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
Example 1:
```
fuzzer args:
fuzzer.py examples/test.c
program_path: examples/test.c
initial parameters:
no_reset  hot_restart  save_interesting  sample_type  coverage_type  input_size  max_popsize  popsize_scale  max_gens  max_eval  timeout  seed  strategy  
False     False        False             bytes        branch         2           1000         10             1000      100000    840      700   None      

-----------------------------------------------------------------------------------------------------------------------------------------------------------
logs:
fuzzer_state   optimized   popsize   current_testcase   total_testcase   generations   current_coverage   total_coverage   evaluations   time   CMA_ES_seed   
optimizing     -           10        1                  1                0             39.2857            39.2857          1             0.3    700           
optimizing     -           10        1                  1                1             50.0               50.0             11            0.38   700           
done           True        10        1                  1                9             50.0               50.0             90            1.03   700           
optimizing     -           10        2                  2                0             53.5714            53.5714          91            1.04   701           
optimizing     -           10        2                  2                0             78.5714            78.5714          93            1.06   701           
done           True        10        2                  2                7             78.5714            78.5714          160           1.6    701           
optimizing     -           10        3                  3                0             82.1429            82.1429          161           1.6    702           
done           True        10        3                  3                6             82.1429            82.1429          220           2.07   702           
optimizing     -           10        4                  4                0             85.7143            85.7143          222           2.09   703           
done           True        10        4                  4                8             85.7143            85.7143          300           2.72   703           
optimizing     -           10        5                  5                0             89.2857            89.2857          302           2.74   704           
done           True        10        5                  5                10            89.2857            89.2857          400           3.53   704           
optimizing     -           10        6                  6                0             92.8571            92.8571          403           3.56   705           
done           True        10        6                  6                10            92.8571            92.8571          500           4.33   705           
optimizing     -           10        7                  7                0             96.4286            96.4286          502           4.35   706           
done           True        10        7                  7                8             96.4286            96.4286          580           4.99   706           
optimizing     -           10        8                  8                0             100.0              100.0            586           5.04   707           
done           True        10        8                  8                0             100.0              100.0            586           5.04   707           

-----------------------------------------------------------------------------------------------------------------------------------------------------------
final report:
total_testcase        total_coverage        stop_reason        testcase_statuses
      8               100.0               total coverage is 100%               ['SAFE', 'SAFE', 'SAFE', 'SAFE', 'SAFE', 'ERROR', 'SAFE', 'SAFE']         
execution time for each method:
get_line_and_branch_coverages  stop    get_executed_coverage_item_ids  cal_branches  ask     _delete_gcda  _encode_bytes  tell    _gcov   _run    objective  optimize_sample  
0.0034                         0.0042  0.0096                          0.0287        0.0399  0.0433        0.0444         0.071   2.1345  2.3055  4.57       4.7523           

-----------------------------------------------------------------------------------------------------------------------------------------------------------
total sample len: 8
line_coverage: 1.0
branch_coverage: 1.0
total_eval: 586
seed: 700
```