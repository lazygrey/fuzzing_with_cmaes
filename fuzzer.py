import cma
import numpy as np
import subprocess
import os
import time
import sys
import random
import argparse
import csv

_init_time = time.time()
_time_log = {}

def _timeit(f):
        def newf(*args, **kwargs):
            startTime = time.time()
            output = f(*args, **kwargs)
            elapsedTime = time.time() - startTime

            if not f.__name__ in _time_log:
                _time_log[f.__name__] = 0.
            _time_log[f.__name__] += elapsedTime

            return output
        return newf


class FuzzerLogger:
    N_INITIAL = 18

    def __init__(self, live = False):
        self._fuzzer = None
        self._log = dict(fuzzer_state = '', optimized = False, popsize = 0, current_testcase = 0, total_testcase = 0, generations = 0, current_coverage = 0, total_coverage = 0, evaluations = 0 ,time = 0, seed = None)
        self._log_message_lines = []
        self._csv_lines = []
        self._live = live

    def format_pretty(self, info, n):
        return "{: <{n}}".format(str(info), n=n)

    def resister(self, fuzzer):
        self._fuzzer = fuzzer
        self._log_path = fuzzer._program.log_dir
        self._filename = self._log_path + fuzzer._program.pname +'.txt'
        self._csvname = self._log_path + fuzzer._program.pname + '.csv'
        
        initial_parameter_keys = ['no_reset', 'hot_restart', 'save_interesting', 'mode', 'objective', 'input_size', 'max_popsize', 'popsize_scale', 'max_gens', 'max_eval', 'timeout', 'seed']
        initial_parameter_values = [fuzzer.no_reset, fuzzer.hot_restart, fuzzer.save_interesting, fuzzer._cma_es.mode['name'], fuzzer.objective.__name__, fuzzer._cma_es._input_size, fuzzer._cma_es._max_popsize, fuzzer._cma_es._popsize_scale, fuzzer._cma_es._max_gens, fuzzer._cma_es.max_evaluations, fuzzer._timeout, fuzzer.seed]

        self._csv_lines.append(list(self._log.keys()))

        self._log_message_lines.append('\nfuzzer args:\n' + ' '.join(sys.argv))
        self._log_message_lines.append('program_path: ' + fuzzer._program.path)
        self._log_message_lines.append('initial parameters:')
        self._log_message_lines.append(''.join([self.format_pretty(key, self.N_INITIAL) for key in initial_parameter_keys]))
        self._log_message_lines.append(''.join([self.format_pretty(value, self.N_INITIAL) for value in initial_parameter_values]))
        self._log_message_lines.append('\n-------------------------------------------------------------------------------------------------------------------------------------------------------')
        self._log_message_lines.append('logs:')
        self._log_message_lines.append(''.join([self.format_pretty(key, len(key) + 2) for key in self._log]))

        if self._live:
            with open(self._filename, 'w') as f:
                f.write('fuzzer args:\n' + ' '.join(sys.argv) + '\n')
                f.write('program_path: ' + fuzzer._program.path + '\n')
                f.write('initial parameters:\n')
                f.writelines(self.format_pretty(key, self.N_INITIAL) for key in initial_parameter_keys)
                f.write('\n')
                f.writelines(self.format_pretty(param, self.N_INITIAL) for param in initial_parameter_values)
                f.write('\n-------------------------------------------------------------------------------------------------------------------------------------------------------\n')
                f.write('logs:\n')
                f.writelines(self.format_pretty(key, len(key) + 2) for key in self._log)
                f.write('\n')

        return self

    def report_changes(self, optimized, state):
        if not self._fuzzer:
            return

        self._log['fuzzer_state'] = state
        self._log['optimized'] = optimized
        self._log.update(self._fuzzer.get_current_state())
        self._log['time'] = round(self._fuzzer.time(), 2)

        self._csv_lines.append(list(self._log.values()))

        self._log_message_lines.append(''.join([self.format_pretty(value, len(key) + 2) for key, value in self._log.items()]))

        if self._live:
            with open(self._filename, 'a') as f:
                f.writelines(self.format_pretty(value, len(key) + 2) for key, value in self._log.items())
                f.write('\n')
    
    def report_final(self):
        final_report = [self._fuzzer._samplecollector.get_total_size(), self._fuzzer.get_total_coverage(), self._fuzzer._stop_reason, self._fuzzer._statuses]

        self._log_message_lines.append('\n-------------------------------------------------------------------------------------------------------------------------------------------------------')
        self._log_message_lines.append('final report:')
        self._log_message_lines.append('total_testcase        total_coverage        stop_reason        testcase_statuses')
        self._log_message_lines.append(''.join(['      %s         ' % str(total) for total in final_report]))

        if self._live:
            with open(self._filename, 'a') as f:
                f.write('\n-------------------------------------------------------------------------------------------------------------------------------------------------------\n')
                f.write('final report:\n')
                f.write('total_testcase        total_coverage        stop_reason        testcase_statuses\n')
                f.writelines('      %s         ' % str(total) for total in final_report)

    def report_time_log(self):
        self._log_message_lines.append('execution time for each method:')
        self._log_message_lines.append(''.join(['%s     ' % str(key) for key in _time_log]))
        self._log_message_lines.append(''.join(['%s   ' % str(round(value,4)) for value in _time_log.values()]))
        self._log_message_lines.append('\n-------------------------------------------------------------------------------------------------------------------------------------------------------')

        if self._live:
            with open(self._filename, 'a') as f:
                f.write('execution time for each method:\n')
                f.writelines('%s     ' % str(key) for key in _time_log)
                f.write('\n')
                f.writelines('%s   ' % str(round(item, _Program.COV_DIGITS)) for item in _time_log.values())
                f.write('\n------------------------------------------------------------------------------------------------------------------------------\n')

    def write_logs(self):
        with open(self._filename, 'w') as f:
            f.writelines('%s\n' % message for message in self._log_message_lines)

    def write_csv(self):
        with open(self._csvname, 'w+', newline='') as f:
           csv.writer(f).writerows(self._csv_lines) 

    def print_logs(self):
        print(*self._log_message_lines, sep='\n')

class _Program:
    # return codes
    SAFE = 0
    COMPILER_ERROR = 1
    OVER_MAX_INPUT_SIZE = 3
    ASSUME = 10
    ERROR = 100
    SEGMENTATION_FAULT = -11

    MIN_INPUT_SIZE = 2

    COV_DIGITS = 2

    DEFAULT_DIRS = {'log' : 'logs/', 'output' : 'output/'}

    def __init__(self, path, verifier_path = '/__VERIFIER.c', verifier_input_path = '/__VERIFIER_input_size.c', output_dir = DEFAULT_DIRS['output'], log_dir = DEFAULT_DIRS['log'], timeout = None, mode = ''):
        self.path = path
        vdir = 'verifiers'
        if mode == 'real':
            vdir += '_real'
        self.verifier_path = vdir+verifier_path
        self.verifier_input_path = vdir+verifier_input_path
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.codelines = {}
        self.current_input_bytes = None
        self._total_lines = 0
        self.pname = path[:-2].rsplit('/', 1)[-1]
        self._state = _Program.SAFE
        self._timeout = timeout
        self._init_dirs()

    def _init_dirs(self):
        if self.output_dir[-1:] != '/':
            self.output_dir += '/'
        if self.log_dir[-1:] != '/':
            self.log_dir += '/'

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)


    def timeout(self):
        if self._timeout:
            return self._timeout - time.time() + _init_time
        return None

    def _compile_program(self):
        return subprocess.run(['gcc',self.path , self.verifier_path, '-o', self.output_dir + self.pname, '--coverage']).returncode

    def _compile_input_size(self):
        return subprocess.run(['gcc', self.path, self.verifier_input_path, '-o', self.output_dir + self.pname + '_input_size']).returncode

    def cal_input_size(self):
        output = subprocess.run(self.output_dir + self.pname + '_input_size', capture_output=True)
        returncode = output.returncode
        output = output.stdout.decode()
        input_size = max(int(output[output.rfind('n') + 1:]), self.MIN_INPUT_SIZE)

        return input_size, returncode        

    @_timeit
    def _delete_gcda(self):
        if os.path.isfile(self.pname + '.gcda'):
            os.remove(self.pname+'.gcda')

    @_timeit
    def _run(self, input_bytes):
        self.current_input_bytes = input_bytes
        return subprocess.run(self.output_dir + self.pname, input = input_bytes, timeout=self.timeout()).returncode

    @_timeit
    def _gcov(self, *args):
        return subprocess.run(['gcov', self.pname + '.gcda', *args], capture_output = True, timeout=self.timeout()).stdout.decode()

    def _coverage(self, output):
        if len(output) == 0:
            return 0

    @_timeit
    def cal_lines(self, gcov):
        output_lines = set()
        if len(gcov) == 0:
            return output_lines
        
        not_executed = 0
        # parse string lines to a set
        lines = gcov.split('\n')
        for line in lines:
            if line == '':
                break
            #    24:  244: ...
            parts = line.split(':', 2)
            if '#' in parts[0]:
                not_executed += 1
            elif not '-' in parts[0]:
                # line 244
                output_lines.add(int(parts[1]))
        total = not_executed + len(output_lines)
        # self._reset()
        return output_lines, total

    @_timeit
    def cal_branches(self, gcov):
        output_branches = set()
        if len(gcov) == 0:
            return output_branches
        
        total = 0
        lines = gcov.split('\n')
        for i, line in enumerate(lines):
            if line == '':
                break
            if line.startswith('b'):
                total += 1
                if 'taken' == line[10:15]:
                    if int(line[15:17]) > 0:
                        output_branches.add(i)

        return output_branches, total
    
    @_timeit
    def get_lines(self):
        gcov = self._gcov('-t')
        lines, total = self.cal_lines(gcov)
        self._delete_gcda()
        return lines, total

    @_timeit
    def get_branches(self):
        gcov = self._gcov('-b', '-c', '-t')
        branches, total = self.cal_branches(gcov)
        self._delete_gcda()
        return branches, total

    def get_line_and_branch_coverages(self):
        gcov = self._gcov('-b', '-c')
        gcov_lines = gcov.split('\n')
        index = -1
        line_offset = 15
        branch_offset = 20
        line_coverage, branch_coverage = 0,0
        for i, line in enumerate(gcov_lines):
            if line.startswith("File '" + self.path):
                index = i
                break
        if index > -1:
            line_index = index + 1
            branch_index = index + 3
            line_coverage = float(gcov_lines[line_index][line_offset:].split('%')[0])
            branch_coverage = float(gcov_lines[branch_index][branch_offset:].split('%')[0])
        self._delete_gcda()
        return line_coverage, branch_coverage


class CMA_ES:
    DEFAULTS = {'seed' : None, 'mode' : 'bytes', 'init_popsize' : 10, 'max_popsize' : 1000, 'max_gens' : 1000, 'popsize_scale' : 10, 'max_evaluations' : 10 ** 5}
    MIN_INPUT_SIZE = 2
    MODES = {'real': {'name': 'real', 'x0' : [(2**32)/2], 'sigma0' : 0.3*(2**32), 'bounds' : [0,2**32]},
     'bytes' : {'name': 'bytes', 'x0' : [128], 'sigma0' : 0.3*256, 'bounds' : [0, 256]}}

    def __init__(self, input_size = MIN_INPUT_SIZE, seed = DEFAULTS['seed'], mode = 'bytes', init_popsize = DEFAULTS['init_popsize'], max_popsize = DEFAULTS['max_popsize'],
    max_gens = DEFAULTS['max_gens'], popsize_scale = DEFAULTS['popsize_scale'], max_evaluations = DEFAULTS['max_evaluations']):
        self._input_size = input_size
        self.mode = self.MODES['bytes']
        self._options = dict(popsize = init_popsize, verb_disp = 0, seed = self.init_seed(seed), bounds = [self.mode['bounds'][0], self.mode['bounds'][1]])
        self._args = dict(x0 = self.mode['x0'] * self._input_size, sigma0 = self.mode['sigma0'])
        self._max_popsize = max_popsize
        self._max_gens = max_gens
        self._es = None 
        self._init_popsize = init_popsize
        self._popsize_scale = popsize_scale
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.result = None
        """A results tuple from `CMAEvolutionStrategy` property ``result``.

        This tuple contains in the given position and as attribute

        - 0 ``xbest`` best solution evaluated
        - 1 ``fbest`` objective function value of best solution
        - 2 ``evals_best`` evaluation count when ``xbest`` was evaluated
        - 3 ``evaluations`` evaluations overall done
        - 4 ``iterations``
        - 5 ``xfavorite`` distribution mean in "phenotype" space, to be
          considered as current best estimate of the optimum
        - 6 ``stds`` effective standard deviations, can be used to
          compute a lower bound on the expected coordinate-wise distance
          to the true optimum, which is (very) approximately stds[i] *
          dimension**0.5 / min(mueff, dimension) / 1.5 / 5 ~ std_i *
          dimension**0.5 / min(popsize / 2, dimension) / 5, where
          dimension = CMAEvolutionStrategy.N and mueff =
          CMAEvolutionStrategy.sp.weights.mueff ~ 0.3 * popsize.
        - 7 ``stop`` termination conditions in a dictionary

        The penalized best solution of the last completed iteration can be
        accessed via attribute ``pop_sorted[0]`` of `CMAEvolutionStrategy`
        and the respective objective function value via ``fit.fit[0]``.

        Details:

        - This class is of purely declarative nature and for providing
          this docstring. It does not provide any further functionality.
        - ``list(fit.fit).find(0)`` is the index of the first sampled
          solution of the last completed iteration in ``pop_sorted``.

        """

    def init_seed(self, seed):
        if seed is None:
            return random.randint(10, 1000)
        return seed - 1

    def init_cmaes(self, mean = None, sigma = None, sigmas = None, fixed_variables = None):
        self._options['seed'] += 1

        if mean is  None:
            self._args['x0'] = self.mode['x0'] * self._input_size
            # lowerbound, upperbound = self.mode['bounds']
            # random.seed(self._options['seed'])
            # self._args['x0'] = [random.randrange(lowerbound, upperbound) for _ in range(self._input_size)]
        else:
            self._args['x0'] = mean
        if sigma is None:
            self._args['sigma0'] = self.mode['sigma0']
        else:
            self._args['sigma0'] = sigma

        if sigmas is None:
            self._options['CMA_stds'] = None
        else:
            self._options['CMA_stds'] = sigmas
            self._args['sigma0'] = 1
        if fixed_variables is None:
            self._options['fixed_variables'] = None
        else:
            self._options['fixed_variables'] = fixed_variables
        
        self._es = cma.CMAEvolutionStrategy(**self._args, inopts=self._options)
        self.result = self._es.result
        return self

    @_timeit
    def stop(self):
        return self._es.stop() or self._es.result.iterations >= self._max_gens or self.result.fbest == -100.0 or self.evaluations > self.max_evaluations

    @_timeit
    def ask(self, **kwargs):
        return self._es.ask(**kwargs)

    @_timeit
    def tell(self, solutions, values):
        return self._es.tell(solutions, values)

    def update(self):
        self.evaluations += 1
        self.result = self._es.result

    def _reset_popsize(self):
        self._options['popsize'] = self._init_popsize


    def _increase_popsize(self):
        self._options['popsize'] *= self._popsize_scale
        return self._options['popsize'] <= self._max_popsize


class _SampleHolder:
    def __init__(self, sample = None, path = set(), score = -1, stds = []):
        self.sample = sample
        self.path = path
        self.score = score
        self.stds = stds
        
    def update(self, sample, path, score):
        optimized = score > self.score
        if optimized:
            self.path = path
            self.sample = sample
            self.score = score
        return optimized

    def clear(self):
        self.sample = None
        self.path = set()
        self.score = 0


class SampleCollector:
    def __init__(self, save_interesting = False):
        self.total_path_size = 0
        self.total_sample_holders = [] 
        self.total_paths = set()
        self.optimized_sample_holders = []
        self.optimized_paths = set()
        self.best_sample_holder = _SampleHolder()
        self.current_coverage = 0
        self.common_path = set()
        self.save_interesting = save_interesting

    def update(self, sample, current_path, score, total_path_size):
        self.total_path_size = total_path_size
        sample_holder = self.best_sample_holder
        if sample_holder.update(sample, current_path, score) or self.save_interesting:
            self.current_coverage = round(100 * sample_holder.score / self.total_path_size, _Program.COV_DIGITS)
            self.total_coverage = round(100 * len(current_path | self.total_paths) / self.total_path_size, _Program.COV_DIGITS)

    @_timeit
    def get_executed_paths(self, sample, current_path, total_path_size):
        if self.save_interesting:
            self.check_interesting(sample, current_path)

        output_path = self.optimized_paths | current_path
        self.update(sample, current_path, len(output_path), total_path_size)

        return output_path        

    # @_timeit
    def check_interesting(self, sample, current_path):
        pre_score = len(self.total_paths)
        self.total_paths.update(current_path)
        current_score = len(self.total_paths)
        is_interesting = pre_score < current_score

        if is_interesting:
            self.total_sample_holders.append(_SampleHolder(sample, current_path))

    def add_best(self, sample, stds):
        sample = self.best_sample_holder.sample
        path = self.best_sample_holder.path
            
        pre_score = len(self.optimized_paths)
        self.optimized_paths.update(path)
        optimized = pre_score < len(self.optimized_paths)

        if optimized:
            self.optimized_sample_holders.append(_SampleHolder(sample, path, stds = stds))
            if not self.save_interesting:
                self.total_sample_holders.append(_SampleHolder(sample, path, stds = stds))
                self.total_paths.update(path)
        
        self.best_sample_holder.clear()

        return optimized

    def remove_common_paths(self):
        if len(self.optimized_paths) == 0:
            return
        common_paths = set.intersection(*[sample.path for sample in self.optimized_sample_holders])
        self.optimized_paths -= common_paths

    def reset_optimized(self):
        self.optimized_sample_holders = []
        self.optimized_paths = set()
        self.best_sample_holder = _SampleHolder()
        self.current_coverage = 0

    def pop_first_optimum_holder(self):
        if len(self.optimized_sample_holders) == 0:
            return None
        
        sample_holder = self.optimized_sample_holders[0]
        self.optimized_sample_holders = self.optimized_sample_holders[1:]
        # self.optimized_paths -= sample_holder.path
        self.optimized_paths = set()
        for holder in self.optimized_sample_holders:
            self.optimized_paths.update(holder.path)
        return sample_holder

    def get_current_coverage(self):
        return self.current_coverage

    def get_total_coverage(self):
        if self.total_path_size == 0:
            return 0

        return self.total_coverage

    def get_optimized_samples(self):
        return [s.sample for s in self.optimized_sample_holders]

    def get_total_samples(self):
        return [s.sample for s in self.total_sample_holders]

    def cal_size(self, size):
        if self.best_sample_holder.sample is not None:
            size += 1
        return size

    def get_current_size(self):
        return self.cal_size(len(self.optimized_sample_holders))

    def get_total_size(self):
        size = len(self.total_sample_holders)
        if self.save_interesting:
            return size
        return self.cal_size(size)



class Fuzzer:
    DEFAULTS = {'timeout' : 14 * 60, 'mode' : 'bytes', 'objective' : 'branch', 'hot_restart_threshold' : 25}
    VERIFIER_ERROS = {_Program.SAFE : 'SAFE', _Program.ERROR : 'ERROR', _Program.ASSUME : 'ASSUME_ERROR'}
    PARSING_SCALE = 2 ** (32 - 8)
    UNSIGNED_INT_MIN = 0
    UNSIGNED_INT_MAX = 2 ** 32 - 1

    def __init__(self, program_path, no_reset = False, live_logs = False, hot_restart = False, save_interesting = False, mode = DEFAULTS['mode'], objective = DEFAULTS['objective'],  timeout = DEFAULTS['timeout'],  hot_restart_threshold = DEFAULTS['hot_restart_threshold'],
    output_dir = _Program.DEFAULT_DIRS['output'], log_dir = _Program.DEFAULT_DIRS['log'], seed = CMA_ES.DEFAULTS['seed'], init_popsize = CMA_ES.DEFAULTS['init_popsize'],
    max_popsize = CMA_ES.DEFAULTS['max_popsize'], max_gens = CMA_ES.DEFAULTS['max_gens'], max_evaluations = CMA_ES.DEFAULTS['max_evaluations'], popsize_scale = CMA_ES.DEFAULTS['popsize_scale']):
        """Fuzzer with cmaes.

        Args:
        sampe_type = the type of a sample to optimize
        function = score function to optimize
        """
        self._timeout = timeout
        self._interrupted = ''
        self._stop_reason = ''
        self._statuses = []

        self.no_reset = no_reset
        self.hot_restart = hot_restart
        self.save_interesting = save_interesting
        self.hot_restart_threshold = hot_restart_threshold
        self.seed = seed

        self.objective = self._select_obejctive(objective)
        self.encode = self._select_encode(mode)
        self._program = _Program(program_path, output_dir = output_dir, log_dir = log_dir, timeout=timeout, mode = mode)
        self._cma_es = CMA_ES(seed = seed, init_popsize= init_popsize ,input_size = self.cal_input_size(), max_popsize = max_popsize, max_gens= max_gens, popsize_scale = popsize_scale,mode = mode, max_evaluations=max_evaluations)
        self._samplecollector = SampleCollector(save_interesting)
        self._logger = FuzzerLogger(live_logs).resister(self)

    def _select_obejctive(self, objective):
        if objective == 'line':
            return self._f_line
        elif objective == 'branch':
            return self._f_branch
        else:
            return self._f_branch

    def _select_encode(self, mode):
        if mode == 'real':
            return self._encode_real
        elif mode == 'bytes':
            return self._encode_bytes
        else:
            return self._encode_bytes

    def _check_compile_error(self, returncode):
        if returncode == _Program.COMPILER_ERROR:
            exit('ERROR: Compliler Error!')

    def _check_runtime_error(self, returncode):
        if returncode == _Program.OVER_MAX_INPUT_SIZE:
            exit('ERROR: The input for "' + self._program.path + '" requires more than 1000 bytes!')
        elif returncode == _Program.SEGMENTATION_FAULT:
            exit('ERROR: Segmentation Fault for "' + self._program.path +'"!')

    def _check_verifier_error(self, returncode):
        state = 'UNKOWN: ' + str(returncode)
        if returncode in Fuzzer.VERIFIER_ERROS:
            state = Fuzzer.VERIFIER_ERROS[returncode]

        self._statuses.append(state)

    def cal_input_size(self):
        self._check_compile_error(self._program._compile_input_size())
        input_size, returncode = self._program.cal_input_size()
        self._check_runtime_error(returncode)
        if input_size < CMA_ES.MIN_INPUT_SIZE:
            exit('ERROR: input_size: ' + str(input_size) + ', Input size must be greater than ' + str(CMA_ES.MIN_INPUT_SIZE) + '!')
        return input_size

    def get_current_coverage(self):
        return self._samplecollector.get_current_coverage()

    def get_total_coverage(self):
        return self._samplecollector.get_total_coverage()

    def get_optimized_samples(self):
        return self._samplecollector.get_optimized_samples()

    def get_total_samples(self):
        return self._samplecollector.get_total_samples()

    def _reset(self):
        self._samplecollector.reset_optimized()

    @_timeit
    def _encode_real(self, sample):
        parse_to_feasible = lambda x: int(min(max(x * self.PARSING_SCALE, self.UNSIGNED_INT_MIN), self.UNSIGNED_INT_MAX))
        out = bytearray()
        for sample_comp in sample:
            out += parse_to_feasible(sample_comp).to_bytes(4, 'little', signed = False)
        return out

    @_timeit
    def _encode_bytes(self, sample: np.ndarray):
        lowerbound, upperbound = self._cma_es.mode['bounds']
        parse_to_feasible = lambda x: int(min(max(x, lowerbound), upperbound))
        out = bytearray(np.frompyfunc(parse_to_feasible, 1, 1)(sample).tolist())
        return out

    @_timeit
    def _run_sample(self, sample, returncode_check = False):
        if sample is None:
            return
        returncode = self._program._run(self.encode(sample))
        if returncode_check:
            self._check_verifier_error(returncode)

    def _run_samples(self, samples, returncode_check = False):
        for sample in samples:
            self._run_sample(sample, returncode_check)
    
    def penalize(self, input_vector):
        penalty = 0
        minimum, upperbound = self._cma_es.mode['bounds']
        for input_component in input_vector:
            if input_component < minimum:
                penalty += abs(input_component)
            if input_component >= upperbound:
                penalty += input_component - upperbound + 1

        return 1 - 1/(int(penalty) + 1)

    @_timeit
    def _f_line(self, sample):
        self._run_sample(sample)
        lines, n = self._program.get_lines() # independently executed lines from sample, total number of lines
        # penalty = self.penalize(sample)
        lines = self._samplecollector.get_executed_paths(sample, lines, n)

        return -round(100 * len(lines)/n, _Program.COV_DIGITS)

    @_timeit
    def _f_branch(self, sample):
        self._run_sample(sample)
        branches, n = self._program.get_branches() # independently executed branches from sample, total number of branches
        # penalty = self.penalize(sample)
        branches = self._samplecollector.get_executed_paths(sample, branches, n)

        return -round(100 * len(branches)/n, _Program.COV_DIGITS)
    
    def get_current_state(self):
        return dict(current_testcase = self._samplecollector.get_current_size(), total_testcase =  self._samplecollector.get_total_size(),
         current_coverage = self.get_current_coverage(), total_coverage = self.get_total_coverage(),
         seed = self._cma_es._options['seed'],popsize = self._cma_es._options['popsize'], generations = self._cma_es.result.iterations, evaluations = self._cma_es.evaluations)

    def _stop(self):
        if self._interrupted:
            self._stop_reason = self._interrupted
        elif self.get_total_coverage() == 100 or self.get_current_coverage() == 100:
            self._stop_reason = 'coverage is 100%'
        elif self._cma_es.evaluations > self._cma_es.max_evaluations:
            self._stop_reason = 'evaluations is over max_evaluations' 
        return self._stop_reason

    def time(self):
        return time.time() - _init_time

    def check_optimized(self, sample, check):
        try:
            prev_current_cov = self.get_current_coverage()
            prev_total_cov = self.get_total_coverage()
            value = self.objective(sample)
        except (subprocess.TimeoutExpired,  KeyboardInterrupt) as e:
            raise e
        finally:
            # for logs
            self._cma_es.update()
            if check and (prev_current_cov < self.get_current_coverage() or self.save_interesting and prev_total_cov < self.get_total_coverage()):
                self._logger.report_changes('-', state = 'optimizing')

        return value

    @_timeit
    def sample_until_interesting_found(self, number, score, check):
        es = self._cma_es
        samples = []
        values = []

        while len(samples) <= number and score >= self._samplecollector.best_sample_holder.score:
            extra_samples = es.ask()
            samples += extra_samples
            values += [self.check_optimized(sample, check) for sample in extra_samples]
            if self._cma_es.evaluations > self._cma_es.max_evaluations:
                break

        return samples, values

    @_timeit
    def optimize_sample(self, number = 0, score = 0, mean = None, sigma = None, sigmas = None, fixed_variables = None, check = True):
        # mute = True
        es = self._cma_es.init_cmaes(mean, sigma, sigmas, fixed_variables)
        while not es.stop():
            try:
                samples = es.ask()
                values = [self.check_optimized(sample, check) for sample in samples]
                # extra_samples, extra_values = self.sample_until_interesting_found(number, score, check = check)
                es.tell(samples, values)
                es.update()

                # if mute:
                #     continue
                # print('----------------------------------')
                # print('samples:\ns', samples)
                # print('iter:\n', es.result.iterations)
                # print('evals:\n', es.result.evaluations)
                # print('values:\n', values)
                # print('means:\n',es.result.xfavorite)
                # print('bestx:\n',es.result.xbest)
                # print('bestf:\n',es.result.fbest)
                # print('stds:\n',es.result.stds)

            except (subprocess.TimeoutExpired,  KeyboardInterrupt) as e:
                self._interrupted = e.__class__.__name__
                self._program._delete_gcda()
                break

        return self._samplecollector.add_best(es.result.xbest, es.result.stds)
    
    def extract_mean_sigmas_for_hot_restart(self, sample_holder):
        mean = []
        sigmas = []
        # stds_mean = sum(sample_holder.stds)/len(sample_holder.stds)
        lowerbound, upperbound = self._cma_es.mode['bounds']

        for i, std in enumerate(sample_holder.stds):
            # if std >= upperbound or std > stds_mean:
            if std >= upperbound or std > self.hot_restart_threshold:
                mean.append(self._cma_es.mode['x0'][0])
                # mean.append(random.randrange(lowerbound, upperbound))
                sigmas.append(self._cma_es.mode['sigma0'])
            else:
                mean.append(sample_holder.sample[i])
                sigmas.append(std)

        return mean, sigmas

    @_timeit
    def optimize_samples_with_hot_restart(self):
        number_of_hot_restarts = len(self._samplecollector.optimized_sample_holders)
        optimized = False
        pre_seed = self._cma_es._options['seed']
        while number_of_hot_restarts > 0 and not self._stop():
            if not optimized:
                mean, sigmas = self.extract_mean_sigmas_for_hot_restart(self._samplecollector.pop_first_optimum_holder())

            optimized = self.optimize_sample(mean = mean, sigmas = sigmas, check = False)
            # optimized = self.optimize_sample(mean = mean, sigmas = sigmas, number=1000, score=self.get_total_coverage())
            self._logger.report_changes(optimized, state = 'hot_restart')

            # count the number down only if not optimized, otherwise try to optimize with the previous mean and sigmas
            if not optimized:
                number_of_hot_restarts -= 1

        # to observe the independent effect of hot restart
        self._cma_es._options['seed'] = pre_seed

    # @_timeit
    def optimize_samples(self):
        optimized = False
        while not self._stop():
            # optimized = self.optimize_sample(number = 1000, score = self.get_current_coverage())
            # optimized = self.optimize_sample(number = 1000, score = self.get_total_coverage())
            prev_optimized = optimized
            optimized = self.optimize_sample()
            self._logger.report_changes(optimized, state = 'done')

            if not optimized:
                if self.hot_restart and prev_optimized:
                    self.optimize_samples_with_hot_restart()
                if not self._cma_es._increase_popsize():
                    self._cma_es._reset_popsize()
                if not self.no_reset:
                    self._reset()

    def generate_testsuite(self):
        self._program._compile_program()
        self.optimize_samples()
        self._program._timeout = None
        # return self.get_total_samples()
        return self.parse_total_samples_to_input_vectors()

    def parse_total_samples_to_input_vectors(self):
        return [self.encode(sample) for sample in self.get_total_samples()]            

    def last_report(self):
        if os.path.isfile(self._program.pname+'.gcda'):
            os.remove(self._program.pname+'.gcda')

        total_samples = self.get_total_samples()
        self._run_samples(total_samples, returncode_check=True)
        line, branch = self._program.get_line_and_branch_coverages()

        self._logger.report_final()
        self._logger.report_time_log()
        self._logger.print_logs()
        self._logger.write_csv()

        print('total sample len:', len(total_samples))
        print('total samples:', total_samples)
        print('total input vectors:', self.parse_total_samples_to_input_vectors())
        print('line_coverage:', round(line/100, _Program.COV_DIGITS))
        print('branch_coverage:', round(branch/100,_Program.COV_DIGITS))
        print('total_eval:', self._cma_es.evaluations)

def parse_argv_to_fuzzer_kwargs():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-od', '--output_dir', type = str, default =_Program.DEFAULT_DIRS['output'], 
        help = 'directory for complied and executable programs')
    arg_parser.add_argument('-ld', '--log_dir', type = str, default =_Program.DEFAULT_DIRS['log'],
        help = 'directory for logs')
    arg_parser.add_argument('-ip', '--init_popsize', type = int, default = CMA_ES.DEFAULTS['init_popsize'],
        help = 'initial population size for CMA-ES to start with')
    arg_parser.add_argument('-mp', '--max_popsize', type = int, default = CMA_ES.DEFAULTS['max_popsize'],
        help = 'maximum population size for CMA-ES')
    arg_parser.add_argument('-m', '--mode', type = str, default = CMA_ES.DEFAULTS['mode'],
        help = 'type of samples that CMA-ES-Fuzzer work with')
    arg_parser.add_argument('-me', '--max_evaluations', type = int, default = CMA_ES.DEFAULTS['max_evaluations'],
        help = 'maximum evaluations for CMA-ES-Fuzzer')
    arg_parser.add_argument('-s', '--seed', type = int, default = CMA_ES.DEFAULTS['seed'],
        help = 'seed to control the randomness')
    arg_parser.add_argument('-t', '--timeout', type = int, default = Fuzzer.DEFAULTS['timeout'],
        help = 'timeout in seconds')
    arg_parser.add_argument('-o', '--objective', type = str, default = Fuzzer.DEFAULTS['objective'],
        help = 'type of objective function for CMA-ES-Fuzzer')
    arg_parser.add_argument('-hrt', '--hot_restart_threshold', type = int, default = Fuzzer.DEFAULTS['hot_restart_threshold'],
        help = 'threshold for the optimized sigma vector to decide whether their components, among mean vector components, are reset to default for the hot restart.')
    arg_parser.add_argument('-nr', '--no_reset', action = 'store_true',
        help = 'deactivate reset after not optimized (this is for the most basic version)')
    arg_parser.add_argument('-hr', '--hot_restart', action = 'store_true',
        help = 'activate hot restart while optimizing samples')
    arg_parser.add_argument('-si', '--save_interesting', action = 'store_true',
        help = 'save interesting paths while optimizing')
    arg_parser.add_argument('-ll', '--live_logs', action = 'store_true',
        help = 'write logs as txt file in log files whenever it changes')
    arg_parser.add_argument('program_path', nargs = '+' ,type = str,
        help = 'relative program path to test (only last argument will be regarded as program path)')

    args= arg_parser.parse_known_args()[0]
    args.program_path = args.program_path[-1]

    return vars(args)

def main():
    kwargs = parse_argv_to_fuzzer_kwargs()
    fuzzer = Fuzzer(**kwargs)
    t = fuzzer.generate_testsuite()
    fuzzer.last_report()

if __name__ == "__main__":
    main()
