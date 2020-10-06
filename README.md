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
program_path          relative program path to test (only last argument will be regarded as program path)

OPTIONAL:

-h, --help            show this help message and exit
-od OUTPUT_DIR, --output_dir OUTPUT_DIR
                    directory for complied and executable programs
-ld LOG_DIR, --log_dir LOG_DIR
                    directory for logs
-ip INIT_POPSIZE, --init_popsize INIT_POPSIZE
                    initial population size for CMA-ES to start with
-mp MAX_POPSIZE, --max_popsize MAX_POPSIZE
                    maximum population size for CMA-ES
-m MODE, --mode MODE  type of samples that CMA-ES-Fuzzer work with
-me MAX_EVALUATIONS, --max_evaluations MAX_EVALUATIONS
                    maximum evaluations for CMA-ES-Fuzzer
-s SEED, --seed SEED  seed to control the randomness
-t TIMEOUT, --timeout TIMEOUT
                    timeout in seconds
-o OBJECTIVE, --objective OBJECTIVE
                    type of objective function for CMA-ES-Fuzzer
-hrt HOT_RESTART_THRESHOLD, --hot_restart_threshold HOT_RESTART_THRESHOLD
                    threshold for the optimized sigma vector to decide whether their components, among mean vector components, are reset to default for the hot restart.
-nr, --no_reset       deactivate reset after not optimized (this is for the most basic version)
-hr, --hot_restart    activate hot restart while optimizing samples
-si, --save_interesting
                    save interesting paths while optimizing
-ll, --live_logs      write logs as txt file in log files whenever it changes



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
fuzzer.py examples/test.c -t 10
program_path: examples/test.c
initial parameters:
no_reset          hot_restart       save_interesting  mode              objective         input_size        max_popsize       popsize_scale     max_gens          max_eval          timeout           
False             False             False             bytes             _f_branch         2                 1000              10                1000              100000            10                

-------------------------------------------------------------------------------------------------------------------------------------------------------
logs:
fuzzer_state  optimized  popsize  current_testcase  total_testcase  generations  current_coverage  total_coverage  evaluations  time  seed  
optimizing    -          10       1                 1               0            39.29             39.29           1            0.14  884   
optimizing    -          10       1                 1               0            50.0              50.0            3            0.16  884   
done          True       10       1                 1               10           50.0              50.0            110          0.91  884   
optimizing    -          10       2                 2               0            57.14             57.14           111          0.92  885   
optimizing    -          10       2                 2               0            60.71             60.71           114          0.94  885   
optimizing    -          10       2                 2               0            71.43             71.43           115          0.95  885   
optimizing    -          10       2                 2               1            75.0              75.0            127          1.03  885   
done          True       10       2                 2               27           75.0              75.0            407          2.96  885   
optimizing    -          10       3                 3               0            78.57             78.57           408          2.97  886   
optimizing    -          10       3                 3               0            82.14             82.14           415          3.02  886   
done          True       10       3                 3               9            82.14             82.14           506          3.64  886   
optimizing    -          10       4                 4               0            85.71             85.71           507          3.65  887   
done          True       10       4                 4               6            85.71             85.71           572          4.1   887   
optimizing    -          10       5                 5               0            89.29             89.29           577          4.14  888   
done          True       10       5                 5               10           89.29             89.29           682          4.86  888   
optimizing    -          10       6                 6               0            92.86             92.86           684          4.88  889   
done          True       10       6                 6               10           92.86             92.86           792          5.62  889   
optimizing    -          10       7                 7               0            96.43             96.43           794          5.63  890   
done          True       10       7                 7               30           96.43             96.43           1122         7.88  890   
optimizing    -          10       8                 8               0            100.0             100.0           1126         7.91  891   
done          True       10       8                 8               1            100.0             100.0           1133         7.96  891   

-------------------------------------------------------------------------------------------------------------------------------------------------------
final report:
total_testcase        total_coverage        stop_reason        testcase_statuses
      8               100.0               coverage is 100%               ['SAFE', 'SAFE', 'SAFE', 'SAFE', 'ERROR', 'SAFE', 'SAFE', 'SAFE']         
execution time for each method:
stop     ask     _run     _gcov     cal_branches     _delete_gcda     get_branches     get_executed_paths     tell     optimize_sample     optimize_samples     
0.0069   0.0723   3.727   3.3609   0.0537   0.0658   3.4989   0.014   0.1178   7.8247   7.8252   

-------------------------------------------------------------------------------------------------------------------------------------------------------
total sample len: 8
total samples: [array([17.13105069, 89.31149773]), array([225.13862945, 142.57292903]), array([219.35965406, 119.31659362]), array([  4.65913897, 189.44266223]), array([135.69827014,  43.52198338]), array([ 84.82877474, 241.75931575]), array([255.9928346 , 220.71963015]), array([221.74850515,   7.6530482 ])]
total input vectors: [bytearray(b'\x11Y'), bytearray(b'\xe1\x8e'), bytearray(b'\xdbw'), bytearray(b'\x04\xbd'), bytearray(b'\x87+'), bytearray(b'T\xf1'), bytearray(b'\xff\xdc'), bytearray(b'\xdd\x07')]
line_coverage: 1.0
branch_coverage: 1.0
total_eval: 1133
```