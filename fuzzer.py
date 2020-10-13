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

    def __init__(self, strategy, live):
        self._fuzzer = None
        self._log = dict(fuzzer_state = '', optimized = False, popsize = 0, current_testcase = 0, total_testcase = 0, generations = 0, current_coverage = 0, total_coverage = 0, evaluations = 0 ,time = 0, CMA_ES_seed = None)
        self._log_message_lines = []
        self._strategy_name = str(strategy)
        self._csv_lines = [['strategy', self._strategy_name]]
        self._live = live

    def format_pretty(self, info, n):
        return "{: <{n}}".format(str(info), n=n)

    def resister(self, fuzzer):
        self._fuzzer = fuzzer
        self._log_path = fuzzer._program.log_dir
        self._filename = self._log_path + fuzzer._program.pname +'.txt'
        self._csvname = self._log_path + fuzzer._program.pname + '_' + self._strategy_name +'.csv'
        
        initial_parameter_keys = ['no_reset', 'hot_restart', 'save_interesting', 'sample_type', 'coverage_type', 'input_size', 'max_popsize', 'popsize_scale', 'max_gens', 'max_eval', 'timeout', 'seed', 'strategy']
        initial_parameter_values = [fuzzer.no_reset, fuzzer.hot_restart, fuzzer.save_interesting, fuzzer.sample_type, fuzzer._program.coverage_type, fuzzer.cma_es.input_size, fuzzer.cma_es._max_popsize, fuzzer.cma_es._popsize_scale, fuzzer.cma_es._max_gens, fuzzer.cma_es.max_evaluations, fuzzer._timeout, fuzzer.cma_es.seed, self._strategy_name]

        self._csv_lines.append(list(self._log.keys()))

        self._log_message_lines.append('\nfuzzer args:\n' + ' '.join(sys.argv))
        self._log_message_lines.append('program_path: ' + fuzzer._program.path)
        self._log_message_lines.append('initial parameters:')
        self._log_message_lines.append(''.join([self.format_pretty(key, len(key) + 2) for key in initial_parameter_keys]))
        self._log_message_lines.append(''.join([self.format_pretty(value, len(initial_parameter_keys[i]) + 2) for i, value in enumerate(initial_parameter_values)]))
        self._log_message_lines.append('\n-----------------------------------------------------------------------------------------------------------------------------------------------------------')
        self._log_message_lines.append('logs:')
        self._log_message_lines.append(''.join([self.format_pretty(key, len(key) + 3) for key in self._log]))

        if self._live:
            with open(self._filename, 'w') as f:
                f.write('fuzzer args:\n' + ' '.join(sys.argv) + '\n')
                f.write('program_path: ' + fuzzer._program.path + '\n')
                f.write('initial parameters:\n')
                f.writelines(self.format_pretty(key, len(key) + 2) for key in initial_parameter_keys)
                f.write('\n')
                f.writelines(self.format_pretty(value, len(initial_parameter_keys[i]) + 2) for i, value in enumerate(initial_parameter_values))
                f.write('\n-----------------------------------------------------------------------------------------------------------------------------------------------------------\n')
                f.write('logs:\n')
                f.writelines(self.format_pretty(key, len(key) + 3) for key in self._log)
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

        self._log_message_lines.append(''.join([self.format_pretty(value, len(key) + 3) for key, value in self._log.items()]))

        if self._live:
            with open(self._filename, 'a') as f:
                f.writelines(self.format_pretty(value, len(key) + 3) for key, value in self._log.items())
                f.write('\n')
    
    def report_final(self):
        final_report = [self._fuzzer._samplecollector.get_total_size(), round(self._fuzzer.get_total_coverage(),Program.COV_DIGITS), self._fuzzer._stop_reason, self._fuzzer._statuses]

        self._log_message_lines.append('\n-----------------------------------------------------------------------------------------------------------------------------------------------------------')
        self._log_message_lines.append('final report:')
        self._log_message_lines.append('total_testcase        total_coverage        stop_reason        testcase_statuses')
        self._log_message_lines.append(''.join(['      %s         ' % str(total) for total in final_report]))

        if self._live:
            with open(self._filename, 'a') as f:
                f.write('\n-----------------------------------------------------------------------------------------------------------------------------------------------------------\n')
                f.write('final report:\n')
                f.write('total_testcase        total_coverage        stop_reason        testcase_statuses\n')
                f.writelines('      %s         ' % str(total) for total in final_report)

    def report_time_log(self):
        time_log = {k: v for k, v in sorted(_time_log.items(), key=lambda item: item[1])}

        self._log_message_lines.append('execution time for each method:')
        self._log_message_lines.append(''.join([self.format_pretty(key, max(len(key),6) + 2) for key, value in time_log.items()]))
        self._log_message_lines.append(''.join([self.format_pretty(round(value,4), max(len(key),6) + 2) for key, value in time_log.items()]))
        self._log_message_lines.append('\n-----------------------------------------------------------------------------------------------------------------------------------------------------------')

        if self._live:
            with open(self._filename, 'a') as f:
                f.write('execution time for each method:\n')
                f.writelines(self.format_pretty(key, max(len(key), 6) + 2) for key, value in time_log.items())
                f.write('\n')
                f.writelines(self.format_pretty(round(value,4), max(len(key), 6) + 2) for key, value in time_log.items())
                f.write('\n-----------------------------------------------------------------------------------------------------------------------------------------------------------\n')

    def write_logs(self):
        with open(self._filename, 'w') as f:
            f.writelines('%s\n' % message for message in self._log_message_lines)

    def write_csv(self):
        with open(self._csvname, 'w+', newline='') as f:
           csv.writer(f).writerows(self._csv_lines) 

    def print_logs(self):
        print(*self._log_message_lines, sep='\n')

class Program:
    DEFAULTS = {'coverage_type' : 'branch'}

    # return codes
    SAFE = 0
    COMPILER_ERROR = 1
    ERROR = 100
    ASSUME = 101
    OVER_MAX_INPUT_SIZE = 102
    INPUT_SIZE_EXECUTED = 103
    SEGMENTATION_FAULT = -11

    MIN_INPUT_SIZE = 2
    MAX_INPUT_SIZE = 1000

    COV_DIGITS = 2

    DEFAULT_DIRS = {'log' : 'logs/', 'output' : 'output/', 'verifiers': 'verifiers/'}
    def __init__(self, path, output_dir, log_dir, timeout, sample_type, coverage_type, verifier_path = '/__VERIFIER.c', verifier_input_size_path = '/__VERIFIER_input_size.c'):
        self.path = path
        self.output_dir = output_dir
        self.log_dir = log_dir
        verifier_dir = 'verifiers_' + sample_type
        self.verifier_path = verifier_dir + verifier_path
        self.verifier_input_size_path = verifier_dir + verifier_input_size_path
        self.codelines = {}
        self.pname = path[:-2].rsplit('/', 1)[-1]
        self._total_lines = 0
        self._state = Program.SAFE
        self._timeout = timeout
        self._init_dirs()
        self.coverage_type = coverage_type
        self.get_coverage_item_ids = self._select_coverage_item_type()

    def _init_dirs(self):
        if self.output_dir[-1:] != '/':
            self.output_dir += '/'
        if self.log_dir[-1:] != '/':
            self.log_dir += '/'

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

    def _select_coverage_item_type(self):
        if self.coverage_type == 'line':
            return self.get_line_ids
        elif self.coverage_type == 'branch':
            return self.get_branche_ids

        exit('ERROR: No such coverage type is supported!')

    def _cal_timeout(self):
        if self._timeout is None:
            return None
        return self._timeout - time.time() + _init_time

    @_timeit
    def _compile_program(self):
        return subprocess.run(['gcc',self.path , self.verifier_path, '-o', self.output_dir + self.pname, '--coverage']).returncode

    @_timeit
    def _compile_input_size(self):
        return subprocess.run(['gcc', self.path, self.verifier_input_size_path, '-o', self.output_dir + self.pname + '_input_size', '--coverage']).returncode

    def cal_input_size(self):
        output = subprocess.run(self.output_dir + self.pname + '_input_size', capture_output=True)
        returncode = output.returncode
        if returncode != self.INPUT_SIZE_EXECUTED:
            return 0, returncode

        output = output.stdout.decode()
        input_size = max(int(output[output.rfind('n') + 1:]), self.MIN_INPUT_SIZE)
        return input_size, returncode

    def cal_coverage_item_size(self):
        gcov = self._gcov('-b', '-c')
        self._delete_gcda()
        gcov_lines = gcov.split('\n')
        if self.coverage_type == 'line':
            offset = 1
        else:
            offset = 2
            
        index = -1
        for i, line in enumerate(gcov_lines):
            if line.startswith("File '" + self.path):
                index = i
                break
            
        if gcov_lines[index + 2].startswith('No'):
            return 0
        else:
            text = gcov_lines[index + offset]
            return int(text[text.rfind('f')+1:])

    @_timeit
    def _delete_gcda(self):
        if os.path.isfile(self.pname + '.gcda'):
            os.remove(self.pname+'.gcda')
        if os.path.isfile('__VERIFIER.gcda'):
            os.remove('__VERIFIER.gcda')

    @_timeit
    def _run(self, input_bytes):
        return subprocess.run(self.output_dir + self.pname, input = input_bytes, timeout=self._cal_timeout()).returncode

    @_timeit
    def _gcov(self, *args):
        return subprocess.run(['gcov', self.pname + '.gcda', *args], capture_output = True, timeout=self._cal_timeout()).stdout.decode()

    def _coverage(self, output):
        if len(output) == 0:
            return 0

    @staticmethod
    @_timeit
    def cal_lines(gcov):
        output_lines = set()
        if len(gcov) == 0:
            return output_lines
        
        lines = gcov.split('\n')
        for i, line in enumerate(lines):
            if line == '':
                break
            if line[0] == ' ' and line[8] != '-' and line[8] != '#':
                output_lines.add(i)
        return output_lines

    @staticmethod
    @_timeit
    def cal_branches(gcov):
        output_branches = set()
        if len(gcov) == 0:
            return output_branches
        
        lines = gcov.split('\n')
        for i, line in enumerate(lines):
            if line == '':
                break
            if line[0] == 'b' and line[10] == 't' and int(line[15:17]) > 0:
                output_branches.add(i)

        return output_branches
    
    # @_timeit
    def get_line_ids(self):
        gcov = self._gcov('-t')
        self._delete_gcda()
        return self.cal_lines(gcov)

    # @_timeit
    def get_branche_ids(self):
        gcov = self._gcov('-b', '-c', '-t')
        self._delete_gcda()
        return self.cal_branches(gcov)

    @_timeit
    def get_line_and_branch_coverages(self):
        gcov = self._gcov('-b', '-c')
        self._delete_gcda()
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
            branch_index = index + 2
            if not gcov_lines[line_index].startswith('No'):
                line_coverage = float(gcov_lines[line_index][line_offset:].split('%')[0])
            if not gcov_lines[branch_index].startswith('No'):
                branch_index += 1 # for 'Taken'
                branch_coverage = float(gcov_lines[branch_index][branch_offset:].split('%')[0])
        return line_coverage, branch_coverage


class CMA_ES:
    DEFAULTS = {'seed' : None, 'init_popsize' : 10, 'max_popsize' : 1000, 'max_gens' : 1000, 'popsize_scale' : 10, 'max_evaluations' : 10 ** 5, 'x0' : [128], 'sigma0' : 0.3*256, 'bounds' : [0, 256]}

    def __init__(self, seed, input_size, init_popsize, max_popsize, max_gens, popsize_scale, max_evaluations):
        self.seed = self.init_seed(seed)
        self.input_size = input_size
        self._options = dict(popsize = init_popsize, verb_disp = 0, seed = self.seed - 1, bounds = [self.DEFAULTS['bounds'][0], self.DEFAULTS['bounds'][1]])
        self._args = dict(x0 = self.DEFAULTS['x0'] * self.input_size, sigma0 = self.DEFAULTS['sigma0'])
        self._max_popsize = max_popsize
        self._max_gens = max_gens
        self._es = None 
        self._init_popsize = init_popsize
        self._popsize_scale = popsize_scale
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.result = None

    def init_seed(self, seed):
        if seed is None:
            return random.randint(10, 1000)
        return seed

    def init_cmaes(self, mean = None, sigma = None, sigmas = None, fixed_variables = None):
        self._options['seed'] += 1

        if mean is None:
            self._args['x0'] = self.DEFAULTS['x0'] * self.input_size
            # lowerbound, upperbound = self.DEFAULTS['bounds']
            # random.seed(self._options['seed'])
            # self._args['x0'] = [random.randrange(lowerbound, upperbound) for _ in range(self.input_size)]
        else:
            self._args['x0'] = mean
        if sigma is None:
            self._args['sigma0'] = self.DEFAULTS['sigma0']
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

    def get_bounds(self):
        return self.DEFAULTS['bounds']

    @_timeit
    def stop(self):
        return self._es.stop() or self._es.result.iterations >= self._max_gens or self.result.fbest == -100.0 or self.evaluations > self.max_evaluations

    @_timeit
    def ask(self, **kwargs):
        return self._es.ask(**kwargs)

    @_timeit
    def tell(self, solutions, values):
        return self._es.tell(solutions, values)

    def update_evals(self):
        self.evaluations += 1

    def update_result(self):
        self.result = self._es.result

    def _reset_popsize(self):
        self._options['popsize'] = self._init_popsize


    def _increase_popsize(self):
        self._options['popsize'] *= self._popsize_scale
        return self._options['popsize'] <= self._max_popsize


class SampleHolder:
    def __init__(self, sample = None, coverage_item_ids = set(), score = -1, stds = []):
        self.sample = sample
        self.coverage_item_ids = coverage_item_ids
        self.score = score
        self.stds = stds
        
    def update(self, sample, coverage_item_ids, score):
        optimized = score > self.score
        if optimized:
            self.coverage_item_ids = coverage_item_ids
            self.sample = sample
            self.score = score
        return optimized

    def clear(self):
        self.sample = None
        self.coverage_item_ids = set()
        self.score = 0


class SampleCollector:
    def __init__(self, save_interesting, coverage_item_size):
        self.total_sample_holders = [] 
        self.total_coverage_item_ids = set()
        self.coverage_item_size = coverage_item_size
        self.optimized_sample_holders = []
        self.optimized_coverage_item_ids = set()
        self.best_sample_holder = SampleHolder()
        self.current_score = 0
        self.total_score = 0
        self.save_interesting = save_interesting

    def update(self, sample, current_coverage_item_ids, score):
        sample_holder = self.best_sample_holder
        if sample_holder.update(sample, current_coverage_item_ids, score) and not self.coverage_item_size == 0:
            self.current_score = sample_holder.score
            if not self.save_interesting:
                self.total_score = len(current_coverage_item_ids | self.total_coverage_item_ids)

    @_timeit
    def get_executed_coverage_item_ids(self, sample, current_coverage_item_ids):
        if self.save_interesting:
            self.check_interesting(sample, current_coverage_item_ids)

        output_ids = self.optimized_coverage_item_ids | current_coverage_item_ids
        self.update(sample, current_coverage_item_ids, len(output_ids))

        return output_ids        

    # @_timeit
    def check_interesting(self, sample, current_coverage_item_ids):
        pre_score = len(self.total_coverage_item_ids)
        self.total_coverage_item_ids.update(current_coverage_item_ids)
        self.total_score = len(self.total_coverage_item_ids)
        is_interesting = pre_score < self.total_score

        if is_interesting:
            self.total_sample_holders.append(SampleHolder(sample, current_coverage_item_ids))

    def add_best(self, sample, stds):
        sample = self.best_sample_holder.sample
        coverage_item_ids = self.best_sample_holder.coverage_item_ids
            
        pre_score = len(self.optimized_coverage_item_ids)
        self.optimized_coverage_item_ids.update(coverage_item_ids)
        optimized = pre_score < len(self.optimized_coverage_item_ids)

        if optimized or len(self.total_sample_holders) == 0:
            self.optimized_sample_holders.append(SampleHolder(sample, coverage_item_ids, stds = stds))
            if not self.save_interesting:
                self.total_sample_holders.append(SampleHolder(sample, coverage_item_ids, stds = stds))
                self.total_coverage_item_ids.update(coverage_item_ids)
        
        self.best_sample_holder.clear()

        return optimized

    def remove_common_coverage_item_ids(self):
        if len(self.optimized_coverage_item_ids) == 0:
            return
        common_ids = set.intersection(*[sample.coverage_item_ids for sample in self.optimized_sample_holders])
        self.optimized_coverage_item_ids -= common_ids

    def reset_optimized(self):
        self.optimized_sample_holders = []
        self.optimized_coverage_item_ids = set()
        self.best_sample_holder = SampleHolder()
        self.current_coverage = 0
        self.current_score = 0

    def pop_first_optimum_holder(self):
        if len(self.optimized_sample_holders) == 0:
            return None
        
        sample_holder = self.optimized_sample_holders[0]
        self.optimized_sample_holders = self.optimized_sample_holders[1:]
        # self.optimized_coverage_item_ids -= sample_holder.coverage_item_ids
        self.optimized_coverage_item_ids = set()
        for holder in self.optimized_sample_holders:
            self.optimized_coverage_item_ids.update(holder.coverage_item_ids)

        return sample_holder

    def get_optimized_samples(self):
        return [s.sample for s in self.optimized_sample_holders]

    def get_total_samples(self):
        return [s.sample for s in self.total_sample_holders]

    def cal_size(self, size):
        if self.best_sample_holder.sample is not None:
            size += 1
        return size

    def get_current_score(self):
        return self.current_score
    
    def get_total_score(self):
        return self.total_score

    def get_current_size(self):
        return self.cal_size(len(self.optimized_sample_holders))

    def get_total_size(self):
        size = len(self.total_sample_holders)
        if self.save_interesting:
            return size
        return self.cal_size(size)


class Fuzzer:
    DEFAULTS = {'timeout' : 14 * 60, 'sample_type' : 'bytes', 'hot_restart_threshold' : 0.5*0.3*256}
    VERIFIER_ERROS = {Program.SAFE : 'SAFE', Program.ERROR : 'ERROR', Program.ASSUME : 'ASSUME_ERROR', Program.OVER_MAX_INPUT_SIZE: 'OVER_MAX_INPUT_SIZE'}
    PARSING_SCALE = 2 ** (32 - 8)
    UNSIGNED_INT_MIN = 0
    UNSIGNED_INT_MAX = 2 ** 32 - 1

    FULL_COVERAGE = 'total coverage is 100%'
    OVER_MAX_EVAL = 'evaluations are over max evaluations'
    NO_INTERESTING_BRANCHES = 'the given program has no interesting branches'
    NO_INPUT = 'the given program takes no inputs'

    def __init__(self, program_path, no_reset = False, live_logs = False, hot_restart = False, save_interesting = False, strategy = None, input_size = None,
    sample_type = DEFAULTS['sample_type'], timeout = DEFAULTS['timeout'],  hot_restart_threshold = DEFAULTS['hot_restart_threshold'], coverage_type = Program.DEFAULTS['coverage_type'],
    output_dir = Program.DEFAULT_DIRS['output'], log_dir = Program.DEFAULT_DIRS['log'], seed = CMA_ES.DEFAULTS['seed'], init_popsize = CMA_ES.DEFAULTS['init_popsize'],
    max_popsize = CMA_ES.DEFAULTS['max_popsize'], max_gens = CMA_ES.DEFAULTS['max_gens'], max_evaluations = CMA_ES.DEFAULTS['max_evaluations'], popsize_scale = CMA_ES.DEFAULTS['popsize_scale']):

        self._timeout = timeout
        self._interrupted = None
        self._stop_reason = ''
        self._statuses = []

        self.no_reset = no_reset
        self.hot_restart = hot_restart
        self.save_interesting = save_interesting
        self.hot_restart_threshold = hot_restart_threshold
        self.sample_type = sample_type

        self.encode = self._select_encode(sample_type)
        self._program = Program(program_path, output_dir, log_dir, timeout, sample_type, coverage_type)
        self._check_compile_error(self._program._compile_input_size())
        self.cma_es = CMA_ES(seed, self._cal_input_size(input_size), init_popsize, max_popsize, max_gens, popsize_scale, max_evaluations)
        self._samplecollector = SampleCollector(save_interesting, self._program.cal_coverage_item_size())
        self._logger = FuzzerLogger(strategy, live_logs).resister(self)

    def _select_encode(self, sample_type):
        if sample_type == 'real':
            self.sample_type = 'real'
            return self._encode_real
        elif sample_type == 'bytes':
            self.sample_type = 'bytes'
            return self._encode_bytes
        exit('ERROR: Unknown sample type!')

    def _check_compile_error(self, returncode):
        if returncode == Program.COMPILER_ERROR:
            exit('ERROR: Compliler Error!')

    def _check_runtime_error(self, returncode):
        if returncode == Program.OVER_MAX_INPUT_SIZE:
            exit('ERROR: The input for "' + self._program.path + '" requires more than 1000 input size!')
        elif returncode == Program.SEGMENTATION_FAULT:
            exit('ERROR: Segmentation Fault for "' + self._program.path +'"!')

    def _check_verifier_error(self, returncode):
        if returncode in Fuzzer.VERIFIER_ERROS:
            state = Fuzzer.VERIFIER_ERROS[returncode]
        else:
            state = 'UNKOWN: ' + str(returncode)

        self._statuses.append(state)

    def _cal_input_size(self, input_size):
        if input_size is None:
            input_size, returncode = self._program.cal_input_size()
            self._check_runtime_error(returncode)
        else:
            input_size = min(max(input_size, Program.MIN_INPUT_SIZE), Program.MAX_INPUT_SIZE)
        return input_size

    def get_current_coverage(self):
        if self._samplecollector.coverage_item_size == 0:
            return 0

        return 100 * self._samplecollector.get_current_score() / self._samplecollector.coverage_item_size

    def get_total_coverage(self):
        if self._samplecollector.coverage_item_size == 0:
            return 0

        return 100 * self._samplecollector.get_total_score() / self._samplecollector.coverage_item_size

    def get_optimized_samples(self):
        return self._samplecollector.get_optimized_samples()

    def get_total_samples(self):
        return self._samplecollector.get_total_samples()

    def _reset(self):
        self._samplecollector.reset_optimized()

    @_timeit
    def _encode_real(self, sample):
        if sample is None:
            return None
        parse_to_feasible = lambda x: int(min(max(x * self.PARSING_SCALE, self.UNSIGNED_INT_MIN), self.UNSIGNED_INT_MAX))
        out = bytearray()
        for sample_comp in sample:
            out += parse_to_feasible(sample_comp).to_bytes(4, 'little', signed = False)
        return out

    @_timeit
    def _encode_bytes(self, sample):
        if sample is None:
            return None

        lowerbound, upperbound = self.cma_es.get_bounds()
        parse_to_feasible = lambda x: int(min(max(x, lowerbound), upperbound - 1))
        out = bytearray(list(np.frompyfunc(parse_to_feasible, 1, 1)(sample)))
        return out

    # @_timeit
    def _run_sample(self, sample, returncode_check = False):
        if sample is None:
            return
        returncode = self._program._run(self.encode(sample))
        if returncode_check:
            self._check_verifier_error(returncode)

    def save_random_sample(self):
        lowerbound, upperbound = self.cma_es.get_bounds()
        sample = np.array([random.randrange(lowerbound,upperbound) for _ in range(self.cma_es.input_size)])
        self._samplecollector.total_sample_holders.append(SampleHolder(sample))

    def _run_samples(self, samples, returncode_check = False):
        for sample in samples:
            self._run_sample(sample, returncode_check)
    
    def penalize(self, input_vector):
        penalty = 0
        minimum, upperbound = self.cma_es.get_bounds()
        for input_component in input_vector:
            if input_component < minimum:
                penalty += abs(input_component)
            if input_component >= upperbound:
                penalty += input_component - upperbound + 1

        return 1 - 1/(int(penalty) + 1)

    @_timeit
    def objective(self, sample):
        self._run_sample(sample)
        coverage_item_ids = self._program.get_coverage_item_ids()
        # penalty = self.penalize(sample)
        executed_coverage_item_ids = self._samplecollector.get_executed_coverage_item_ids(sample, coverage_item_ids)
        
        return -len(executed_coverage_item_ids)
    
    def get_current_state(self):
        return dict(current_testcase = self._samplecollector.get_current_size(), total_testcase =  self._samplecollector.get_total_size(),
         current_coverage = round(self.get_current_coverage(), 4), total_coverage = round(self.get_total_coverage(), 4),
         CMA_ES_seed = self.cma_es._options['seed'],popsize = self.cma_es._options['popsize'], generations = self.cma_es.result.iterations, evaluations = self.cma_es.evaluations)

    def _stop(self):
        if self._interrupted is not None:
            if self._interrupted.__class__ != StopIteration:
                self._stop_reason = self._interrupted.__class__.__name__
            else:
                self._stop_reason = self._interrupted.args[0]
        elif self.get_total_coverage() == 100 or self.get_current_coverage() == 100:
            self._stop_reason = self.FULL_COVERAGE
        elif self.cma_es.evaluations > self.cma_es.max_evaluations:
            self._stop_reason = self.OVER_MAX_EVAL
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
            self.cma_es.update_evals()
            if check and (prev_current_cov < self.get_current_coverage() or self.save_interesting and prev_total_cov < self.get_total_coverage()):
                self._logger.report_changes('-', state = 'optimizing')

            if self.cma_es.evaluations >= self.cma_es.max_evaluations:
                raise StopIteration(self.OVER_MAX_EVAL)

            if self.get_total_coverage() == 100:
                raise StopIteration(self.FULL_COVERAGE)

        return value

    @_timeit
    def sample_until_interesting_found(self, number, score, check):
        es = self.cma_es
        samples = []
        values = []

        while len(samples) <= number and score >= self._samplecollector.best_sample_holder.score and not es.stop():
            extra_samples = es.ask()
            samples += extra_samples
            values += [self.check_optimized(sample, check) for sample in extra_samples]

        return samples, values

    @_timeit
    def optimize_sample(self, number = 0, score = 0, mean = None, sigma = None, sigmas = None, fixed_variables = None, check = True):
        es = self.cma_es.init_cmaes(mean, sigma, sigmas, fixed_variables)
        while not es.stop():
            try:
                samples = es.ask()
                values = [self.check_optimized(sample, check) for sample in samples]
                # extra_samples, extra_values = self.sample_until_interesting_found(number, score, check = check)
                es.tell(samples, values)
                es.update_result()

                # print('----------------------------------')
                # print('iter:\n', es.result.iterations)
                # print('evals:\n', es.result.evaluations)
                # print('values:\n', values)
                # print('samples:\n', samples)
                # print('means:\n',es.result.xfavorite)
                # print('bestx:\n',es.result.xbest)
                # print('bestf:\n',es.result.fbest)
                # print('stds:\n',es.result.stds)
            except (subprocess.TimeoutExpired,  KeyboardInterrupt, StopIteration) as e:
                self._interrupted = e
                break

        return self._samplecollector.add_best(es.result.xbest, es.result.stds)
    
    def extract_mean_sigmas_for_hot_restart(self, sample_holder):
        mean = []
        sigmas = []
        # stds_mean = sum(sample_holder.stds)/len(sample_holder.stds)
        lowerbound, upperbound = self.cma_es.get_bounds()

        for i, std in enumerate(sample_holder.stds):
            # if std >= upperbound or std > stds_mean:
            if std >= upperbound or std > self.hot_restart_threshold:
                mean.append(CMA_ES.DEFAULTS['x0'][0])
                # mean.append(random.randrange(lowerbound, upperbound))
                sigmas.append(CMA_ES.DEFAULTS['sigma0'])
            else:
                mean.append(sample_holder.sample[i])
                sigmas.append(std)

        return mean, sigmas

    @_timeit
    def optimize_samples_with_hot_restart(self):
        number_of_hot_restarts = len(self._samplecollector.optimized_sample_holders)
        optimized = False
        prev_seed = self.cma_es._options['seed']
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
        self.cma_es._options['seed'] = prev_seed

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
                if not self.cma_es._increase_popsize():
                    self.cma_es._reset_popsize()
                if not self.no_reset:
                    self._reset()

    def check_no_early_stop(self):
        if self.cma_es.input_size == 0:
            self._stop_reason = self.NO_INPUT
            return False

        if self._samplecollector.coverage_item_size == 0:
            self.save_random_sample()
            self._stop_reason = self.NO_INTERESTING_BRANCHES
            return False

        return True

    def generate_testsuite(self):
        try:
            self._program._compile_program()
            if self.check_no_early_stop():
                self.optimize_samples()
        except:
            self._interrupted = sys.exc_info()[0]
        finally:
            self._program._delete_gcda()
            self._program._timeout = None

        return self.parse_total_samples_to_input_vectors()

    def parse_total_samples_to_input_vectors(self):
        return [self.encode(sample) for sample in self.get_total_samples()]            

    def last_report(self):
        self._program._compile_program()

        total_samples = self.get_total_samples()
        if self.cma_es.input_size == 0:
            self._program._run(None)
        else:
            self._run_samples(total_samples, returncode_check=True)
        
        line, branch = self._program.get_line_and_branch_coverages()

        self._logger.report_final()
        self._logger.report_time_log()
        self._logger.print_logs()
        self._logger.write_csv()

        print('total sample len:', len(total_samples))
        print('total samples:', total_samples)
        print('total input vectors:', self.parse_total_samples_to_input_vectors())
        print('line_coverage:', 0.01 * line)
        print('branch_coverage:', 0.01 * branch)
        print('total_eval:', self.cma_es.evaluations)
        print('seed:', self.cma_es.seed)

def parse_argv_to_fuzzer_kwargs():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-od', '--output_dir', type = str, default =Program.DEFAULT_DIRS['output'], 
        help = 'directory for complied and executable programs')
    arg_parser.add_argument('-ld', '--log_dir', type = str, default =Program.DEFAULT_DIRS['log'],
        help = 'directory for logs')
    arg_parser.add_argument('-ip', '--init_popsize', type = int, default = CMA_ES.DEFAULTS['init_popsize'],
        help = 'initial population size for CMA-ES to start with')
    arg_parser.add_argument('-mp', '--max_popsize', type = int, default = CMA_ES.DEFAULTS['max_popsize'],
        help = 'maximum population size for CMA-ES')
    arg_parser.add_argument('-me', '--max_evaluations', type = int, default = CMA_ES.DEFAULTS['max_evaluations'],
        help = 'maximum evaluations for CMA-ES-Fuzzer')
    arg_parser.add_argument('-s', '--seed', type = int, default = CMA_ES.DEFAULTS['seed'],
        help = 'seed to control the randomness')
    arg_parser.add_argument('-st', '--sample_type', type = str, default = Fuzzer.DEFAULTS['sample_type'],
        help = 'type of samples that CMA-ES-Fuzzer work with')
    arg_parser.add_argument('-t', '--timeout', type = int, default = Fuzzer.DEFAULTS['timeout'],
        help = 'timeout in seconds')
    arg_parser.add_argument('-ct', '--coverage_type', type = str, default = Program.DEFAULTS['coverage_type'],
        help = 'type of coverage for obejctive function for CMA-ES-Fuzzer')
    arg_parser.add_argument('-hrt', '--hot_restart_threshold', type = int, default = Fuzzer.DEFAULTS['hot_restart_threshold'],
        help = 'threshold for the optimized sigma vector to decide whether their components, among mean vector components, are reset to default for the hot restart.')
    arg_parser.add_argument('-nr', '--no_reset', action = 'store_true',
        help = 'deactivate reset after not optimized (this is for the most basic version)')
    arg_parser.add_argument('-hr', '--hot_restart', action = 'store_true',
        help = 'activate hot restart while optimizing samples')
    arg_parser.add_argument('-si', '--save_interesting', action = 'store_true',
        help = 'save interesting coverage item ids while optimizing')
    arg_parser.add_argument('-is', '--input_size', type = int,
        help = 'fixed input size for CMA-ES')
    arg_parser.add_argument('--strategy', type = str,
        help = 'strategy label for log and csv')
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
