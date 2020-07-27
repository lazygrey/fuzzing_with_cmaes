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
python3 fuzzer.py <program_path> [-od <output_dir>] [-ld <log_dir>] [-mp <max_popsize>] [-t timeout]
```
MANDATORY:

<program_path> := path to a C-program.


OPTIONAL:


-od <output_dir> := directory in where executable C programs will be stored after compiled. (default: "output/")

-ld <log_dir> := directory in where log information will be stored. (default: "logs/")

-mp <max_popsize> := maximum population size while optimizing an input with CMA-ES. (default: 1000)

-t \<timeout> := time in seconds for timeout. (default: 900)


## Fuzzing Examples
Example 1: timout in 10 seconds
```
python3 fuzzer.py examples/test.c -t 10
testsuite: [array([ 58.51964139, 191.83170435]), array([159.68674647,  53.4558355 ]), array([ 90.96958387, 233.69458071]), array([146.11062388,  31.59472821]), array([248.33267195, 210.30889049]), array([209.01992056,  72.52722745])]
File 'examples/test.c'
Lines executed:95.65% of 46
Creating 'test.c.gcov'


```
Example 2:
```bash
python3 fuzzer.py user_program_dir/user_program.c -od user_output_dir/ -ld user_log_dir/
```

Example 3: maximum population size is 100
```bash
python3 fuzzer.py user_program_dir/user_program.c -mp 100
```

## Log Examples:
Example 1: timout in 10 seconds
```
program_path: examples/test.c
initial parameters:
input_size       max_popsize       popsize_scale_factor       max_gens      timeout
   2              1000              2              1000              10           
-----------------------------------------------------------------------------------
logs:
testcase   popsize   generations   coverage   optimized   time   
     1          7          10          58.7          True          0.5     
     2          7          33          69.57          True          2.39     
     3          7          10          78.26          True          3.12     
     4          7          10          82.61          True          4.03     
     5          7          50          91.3          True          9.55     
     6          7          3          95.65          True          10.0     

-----------------------------------------------------------------------------------
final report:
total_testcase        total_coverage        stop_reason        testcase_statuses
      6               95.65               TimeoutExpired               ['SAFE', 'ERROR', 'SAFE', 'SAFE', 'SAFE', 'SAFE']         
-----------------------------------------------------------------------------------
execution time for each method:
stop     ask     _run     _gcov     _reset     tell     optimize_sample     optimize_testsuite     
0.0142   0.0766   7.564   1.8844   0.0459   0.1205   9.8635   9.9407   
```