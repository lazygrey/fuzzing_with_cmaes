import cma
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import subprocess
import os
import time
import sys
import random

# random.seed(0)

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
            # try:
            # except subprocess.TimeoutExpired as timeout:
            #     print('stopped optimizesample with time',_time_log['optimize_sample'])
            #     raise timeout
            # finally:
                # print(elapsedTime)

            return output

            # print('function [{}] finished in {} ms'.format(
            #     f.__name__, int(elapsedTime * 1000)))
        return newf


    

class _Program:

    SAFE = 0
    COMPILER_ERROR = 1
    OVER_MAX_INPUT_SIZE = 3
    ASSUME = 10
    ERROR = 100
    SEGMENTATION_FAULT = -11

    COV_DIGITS = 4

    DEFAULT_DIRS = {'log' : 'logs/', 'output' : 'output/'}
    # GCOV = {'File'}

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
        """ codelines:
            #####:  239:    mode3 = (_Bool)1;
                -:  240:  }
            #####:  241:  return;
                -:  242:}
                -:  243:}
                -:  244:void (*nodes[3])(void)  = {      & node1,      & node2,      & node3};
                2:  245:int init(void) 
                -:  246:{ 
            this should be:
        245 : 2
        
        """

        self.pname = [] # *.c   
        for c in reversed(path[:-2]):
            if c == '/':
                break
            self.pname.insert(0, c)
        self.pname = ''.join(self.pname)
        self._state = _Program.SAFE
        self._timeout = timeout
        self._init_dirs()
        """
        subprocess.run(args, timeout = timer.timeout())
        timer.timeout():
                        10s           ?1.2        ?0.2
          return maxrunningtime - time.time() + self._inittime 
        """

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
        # initialize inputsize.txt
        output = ''
        with open('inputsize.txt', 'w') as f:
            f.write(output)
        returncode = subprocess.run(self.output_dir + self.pname + '_input_size').returncode
        with open('inputsize.txt', 'r') as f:
            output = f.read()

        # output = process.stdout.decode()
        # print('inputsize output',output)
        # TODO: think what it means for the output to be ''
        if output == '':
            output = 0
        output = int(output)
        if output < 2:
            output = 2

        return output, returncode        

    @_timeit
    def _reset(self):
        if os.path.isfile(self.pname + '.gcda'):
            os.remove(self.pname+'.gcda')
        else:
            print('WARNING: No gcda file at %ss!' % round(_init_time - time.time()))
            # maybe program reached error?

    @_timeit
    def _run(self, input_bytes):
        self.current_input_bytes = input_bytes
        # outputs/test <- input_bytes
        return subprocess.run(self.output_dir + self.pname, input = input_bytes, timeout=self.timeout()).returncode

    @_timeit
    def _gcov(self, t = ''):
        # gcov test
        return subprocess.run(['gcov', t, self.pname + '.gcda'], capture_output = True, timeout=self.timeout()).stdout.decode()

    # def _cov(self, output):
    #     output.split()
    #     ''.split()
    #     if len(output) == 0:
    #         return 0
        

    def _coverage(self, output):
        if len(output) == 0:
            return 0

        

    @_timeit
    def _cov(self, output):
        if len(output) == 0:
            return 0.0

        cov_path = ''
        while self.path != cov_path:
            cov_path ,output = output.split("File '", 1)[1].split("'", 1)
            # print('reseult!',cov_path, output)

        # print('last',cov_path, output)
            
        start, end = 0 , 0
        for i in range(len(output)):
            if output[i] == ':':
                start = i + 1
            elif output[i] == '%':
                end = i
                break
        return float(output[start:end])


    def get_coverage(self):
        output = self._gcov()
        self._reset()
        return self._cov(output)

    @_timeit
    def cal_lines(self):
        gcov = self._gcov('-t')
        output_lines = set()
        # print(gcov)
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
        self._reset()
        return output_lines, total

    @_timeit
    def get_coverage_lines(self):
        gcov = self._gcov('-t')
        # print(gcov)
        if len(gcov) == 0:
            return 0, set()
        
        outputset = set()
        counter = 0
        lines = gcov.split('\n')
        for line in lines:
            if line == '':
                break
            #    24:  244: ...
            parts = line.split(':', 2)
            if '#' in parts[0]:
                counter += 1
            elif not '-' in parts[0]:
                # line 244
                outputset.add(int(parts[1]))
        self._total_lines = counter + len(outputset)
        coverage = round(100* len(outputset) / (counter + len(outputset)), self.COV_DIGITS)
        # outputset.issubset(self.codelines)
        # cov = self._gcov()
        # print('outputset len, total:' , len(outputset), total)
        # print(cov)
        # print('linecov, cov:', coverage, self._cov(self._gcov()))
        self._reset()
        return coverage, outputset


class CMAES_Builder:
    MIN_INPUT_SIZE = 2
    MODES = {'real': {'x0' : [(2**32 - 1)/2], 'sigma0' : (2**32 - 1)/3, 'bounds' : [0,2**32-1]},
     'bytes' : {'x0' : [127.5], 'sigma0' : 85, 'bounds' : [0, 255]}}

    def __init__(self, mean = [128], sigma = 64, input_size = 1000, init_popsize = 7, max_popsize = 1000, max_gens = 1000, popsize_scale = 2, mode = 'bytes'):
        # print('inputdim =', input_size)
        self._input_size = input_size
        self.mode = self.MODES[mode]
        self._options = dict(bounds = self.mode['bounds'], popsize = init_popsize, verb_disp = 0, seed = 100)
        self._args = dict(x0 = self.mode['x0'] * self._input_size, sigma0 = self.mode['sigma0'])
        # if real:
        #     # mean = [2**63]
        #     mean = [(2**64 -1) / 2] * input_size
        #     sigma = (2**64-1) /3
        #     self._options = dict(bounds = [0, 2**64-1],popsize = 10, verb_disp = 0, seed = 123)
        #     self._args = dict(x0 = mean, sigma0 = sigma)
        # else:
        #     # original
        #     # self._options = dict(bounds = [0, 255.99], popsize = init_popsize, verb_disp = 0, seed = 123)
        #     # self._args = dict(x0 = input_size * mean, sigma0 = sigma, inopts = self._options)
        #     #
        #     # Test with mean: change bounds [0, 256] to [-256, 256] and later in _f modulo them into [0,255]
        #     mean = [127.5] * input_size
        #     # meanx = [-23.58940515,-8.68696602, -18.15140697,-8.73820749, 4.89701663,20.02202807, 3.21376194,-1.29155978]
        #     sigma = 85
        #     # sigma =  [58.07498712,58.69641609,57.94491933,52.05122605,75.67440815,62.63755384,53.15353417, 51.18854103]
        #     self._options = dict(bounds = [None, None], popsize = 200, verb_disp = 0, seed = 123)
        #     self._args = dict(x0 = mean, sigma0 = sigma)
            # self._args = dict(x0 = meanx, sigma0 = sigma, inopts = self._options)
        #
        self._max_popsize = max_popsize
        self._max_gens = max_gens
        self._es = None #
        self._fbest = 0
        self._prev_fbest = 0
        self._optimized = False
        self._init_popsize = init_popsize
        # self._potential_popsize = init_popsize
        self._popsize_scale = popsize_scale
        # self.result = {'generations': 0, }
        # self._generations = 0
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


    def init_cmaes(self, mean = None, sigma = None):
        # self._es = cma.CMAEvolutionStrategy(**self._args)
        # self._options['popsize'] = self._potential_popsize
        # if sigma and not self.real:
            # self._args['sigma0'] = sigma
        self._options['seed'] += 1
        self._optimized = False
        # self._generations = 0
        self._es = cma.CMAEvolutionStrategy(**self._args, inopts=self._options)
        self.result = self._es.result
        # return self

    @_timeit
    def stop(self):
        return self._es.stop() or self._es.result.iterations >= self._max_gens or self.result.fbest == -100

    @_timeit
    def stop_lines(self):
        # return self._es.stop() or self._es.result.iterations >= self._max_gens or self.result.fbest == -100
        return self._es.stop() or self._es.result.iterations >= self._max_gens or self.result.fbest == -100
 

    def reset_stop_conditions(self):
        self._es.stop().clear()

    @_timeit
    def ask(self):
        # return self._es.ask(self._init_popsize)
        return self._es.ask()
    @_timeit
    def tell(self, solutions, values):
        return self._es.tell(solutions, values)

    def update(self):
        self.result = self._es.result

    def is_optimized(self):
        return self._optimized

    def _update_fbest(self):
        fbest = self._es.result.fbest
        self._optimized = self._fbest > fbest
        if self._optimized:
            self._fbest = fbest

    def _reset_fbest(self):
        self._fbest = 0

    def _reset_popsize(self):
        # self._potential_popsize = self._options['popsize']
        self._options['popsize'] = self._init_popsize


    def _increase_popsize(self):
        # self._potential_popsize *= self._popsize_scale
        self._options['popsize'] *= self._popsize_scale
        # print('increase popsize to:', self._potential_popsize)
        return self._options['popsize'] <= self._max_popsize
    
    def _is_over_threshold(self):
        # _stop = 
        # # _stop = self._potential_popsize > self._max_popsize or self._es.result.iterations >= self._max_gens
        # print('_is_over_threshold:', _stop)
        return self._options['popsize'] > self._max_popsize

    def get_sample(self):
        return self._es.result.xbest

    def get_coverage(self):
        return -self._es.result.fbest

    def get_generations(self):
        return self._es.result.iterations

    def get_xbest(self):
        return self._es.result.xbest

    def get_fbest(self):
        return self._fbest

    def _reset(self):
        self._optimized = False
        # self._potential_popsize = self._options['popsize']
        # self._potential_popsize = self._init_popsize
        # maybe reset dim as well

    # def _reset

    def _current_state(self):
        # if 
        # return dict(generations = self._es.result.iterations, popsize = self._current_popsize, optimized = self.is_optimized())
        # return dict(popsize = self._current_popsize, optimized = self.is_optimized())
        # return dict(popsize = self._options['popsize'], optimized = self.is_optimized())
        return dict(seed = self._options['seed'],popsize = self._options['popsize'], generations = self.result.iterations)
        #  generations = self._generations)
        # return dict(input_size = self._input_size, popsize = self._current_popsize, optimized = self._optimized)

class FuzzerLogger:
    def __init__(self):
        self._fuzzer = None
        self._log = dict(seed = None, testcase = 0, popsize = 0, generations = 0, coverage = 0, optimized = False, time = 0)
        self._number = 1

        """
        program_path: programs/test.c
        initial parameters:
        input_size       max_popsize       popsize_scale_factor       max_gens      timeout
        91              1000              2              1000              900           
        -----------------------------------------------------------------------------------
        logs:
        testcase   popsize   generations   coverage   optimized   time   
            1          7          1          100.0          True          0.25     

        -----------------------------------------------------------------------------------
        final report:
        total_testcase        total_coverage        stop_reason        testcase_statuses
            1               100.0               coverage is 100%               ['SAFE']         
        -----------------------------------------------------------------------------------
        execution time for each method:
        stop     ask     _run     _gcov     _reset     tell     optimize_sample     optimize_testsuite     
        0.0012   0.0008   0.0244   0.0258   0.0005   0.0013   0.0483   0.1714   
        """
    # def time(self):
    #     return self._log['time']

    def resister(self, fuzzer):
        self._fuzzer = fuzzer
        self._log_path = fuzzer._program.log_dir
        # example: logs/test.txt
        self._filename = self._log_path + fuzzer._program.pname +'.txt'
        # write initial parameters of Fuzzer.
        # program_path, max_sample_size, max_gen, max_popsize, timout, 
        initial_parameters = [fuzzer._cmaesbuilder._input_size, fuzzer._cmaesbuilder._max_popsize, fuzzer._cmaesbuilder._popsize_scale, fuzzer._cmaesbuilder._max_gens, fuzzer._timeout]
        with open(self._filename, 'w') as f:
            f.write('program_path: ' + fuzzer._program.path + '\n')
            f.write('initial parameters:\n')
            f.write('input_size       max_popsize       popsize_scale_factor       max_gens      timeout\n')
            f.writelines('   %s           ' % key for key in initial_parameters)
            f.write('\n-----------------------------------------------------------------------------------\n')
            f.write('logs:\n')

        # write input_size samplesize generations popsize coverage optimized time 
        with open(self._filename, 'a') as f:
            f.writelines("%s   " % key for key in self._log)
            f.write('\n')
        return self

    def report_changes(self, optimized):
        if not self._fuzzer:
            return
        # precov = self._log['coverage']
        # if precov != self._fuzzer.get_coverage()
        self._log['optimized'] = optimized
        self._log.update(self._fuzzer._cmaesbuilder._current_state())
        self._log.update(self._fuzzer._current_state())
        # self._log['samplesize'] = len(self._fuzzer._samples)
        # self._log['coverage'] = self._fuzzer._coverage
        # self._log['coverage'] = cov
        # print(log)
        # if _time_log:

        # print('@@@@@@@@@@@@@@@@@@@@@@time log',_time_log)
        self._log['time'] = round(self._fuzzer.time(),2)
        # if self._fuzzer._samples:
        #     self._log['sample'] = self._fuzzer._samples[-1]
        with open(self._filename, 'a') as f:
            # f.write(str(self._log)+'\n')
            f.writelines("%s     " % str(item) for item in self._log.values())
            f.write('\n')
        # self._mode = 'a'
    
    def report_time_log(self):
        with open(self._filename, 'a') as f:
            f.write('\n-----------------------------------------------------------------------------------\n')
            f.write('execution time for each method:\n')
            f.writelines('%s     ' % str(key) for key in _time_log)
            f.write('\n')
            f.writelines('%s   ' % str(round(item,4)) for item in _time_log.values())
            # f.writelines('%s   ' )

    def report_final(self):
        final_report = [len(self._fuzzer._total_samples), self._fuzzer._total_coverage, self._fuzzer._stop_reason, self._fuzzer._statuses]
        with open(self._filename, 'a') as f:
            f.write('\n-----------------------------------------------------------------------------------\n')
            f.write('final report:\n')
            f.write('total_testcase        total_coverage        stop_reason        testcase_statuses\n')
            f.writelines('      %s         ' % str(total) for total in final_report)


class _SampleHolder:
    def __init__(self, sample = None, path = set(), coverage = 0):
        self.sample = sample
        self.path = path
        self.coverage = 0
        
    def update_path(self, sample, path):
        updated = len(path) > len(self.path)
        if updated:
            self.path = path
            self.sample = sample
        return updated

    def update_coverage(self, sample, coverage):
        updated = coverage > self.coverage
        if updated:
            self.coverage = coverage
            self.sample = sample
        return updated

    def clear(self):
        # self.sample = None
        self.path = set()
        self.coverage = 0


class SampleCollector:
    def __init__(self):
        self._total_path_length = 0
        self.total_samples = [] # [sample holder1, sample holder 2, ...
        self.interesting_paths = set()
        self.optimized_samples = [] # [sample holder1, sample holder 2, ...]
        self.optimized_paths = set()
        self.best_sample = _SampleHolder()
        self.current_coverage = 0

    # def test_dup(self,path):
    #     for i, s1 in enumerate(self.total_samples):
    #         for j,s2 in enumerate(self.total_samples):
    #             if i != j and (s1.path.issubset(s2.path) or s2.path.issubset(s1.path)):
    #                 print([s.path for s in self.total_samples])
    #                 print(path)
    #                 raise KeyboardInterrupt

    @_timeit
    def check_interesting(self, sample, path, total):
        self._total_path_length = total
        output_path = self.optimized_paths | path
        self.best_sample.update_path(sample, output_path)
        # check if the given path covers another new path
        if len(output_path) > len(self.optimized_paths):
        # if not path.issubset(self.optimized_paths):
            interesting = True
            reduced_samples = []
            # eliminate all samples with path that are real subset of the given path / minimize interesting samples
            for s in self.total_samples:
                # if len(s.path) > len(output_path) or len(s.path) == len(output_path) and not output_path.issubset(s.path) or not s.path.issubset(output_path):
                # if output_path == s.path or not s.path.issubset(output_path):
                if len(s.path) >= len(path) or not s.path.issubset(path) :
                    reduced_samples.append(s)
                # if not s.path.issubset(output_path):
                    # reduced_samples.append(s)
                    if interesting and path.issubset(s.path):
                        interesting = False
            # if len(self.total_samples) > len(reduced_samples):
            #     print('!!!!!reduced', len(self.total_samples), len(reduced_samples))
            self.total_samples = reduced_samples

            # add the given sample only if the given sample was interesting
            if interesting:
                self.total_samples.append(_SampleHolder(sample, path))
                # print('path:')
                # for s in self.total_samples:
                #     print(s.path) 
        self.interesting_paths.update(path)

        return output_path


    def update_optimized(self, sample, path): # updated == True <=> optimized
        pre_len = len(self.optimized_paths)
        self.optimized_paths.update(path)
        optimized = pre_len < len(self.optimized_paths)

        if optimized:
            self.optimized_samples.append(_SampleHolder(sample, path))
        # self.best_sample.clear()
        
        return optimized

    def update_coverage(self, sample, coverage):        
        optimized = coverage > self.current_coverage

        if optimized:
            s = _SampleHolder(sample, coverage = coverage)
            self.optimized_samples.append(s)
            self.total_samples.append(s)
            self.current_coverage = coverage
            
        return optimized


    # def check_optimized(self, sample):
    #     pre_len = len(self.optimized_paths)


    def reset_optimized(self):
        self.optimized_samples = []
        self.current_coverage = 0
        self.optimized_paths = set()
        self.best_sample.clear()

    def coverage(self):
        if self.current_coverage > 0:
            return self.current_coverage
            
        if self._total_path_length == 0:
            return 0
        # print('op lenth:', len(self.optimized_paths))
        return  round(100 * len(self.optimized_paths)/ self._total_path_length,4)
        # return 0

    def total_coverage(self):
        if self._total_path_length == 0:
            return 0
        # print('interesting lenth:', len(self.interesting_paths))
        return  round(100 * len(self.interesting_paths)/ self._total_path_length,4)

    def get_optimized_samples(self):
        return [s.sample for s in self.optimized_samples]

    def get_total_samples(self):
        return [s.sample for s in self.total_samples]

    def get_total_size(self):
        return len(self.total_samples)


class Fuzzer:

    _VERIFIER_ERROS = {_Program.SAFE : 'SAFE', _Program.ERROR : 'ERROR', _Program.ASSUME : 'ASSUME_ERROR'}


    # def __init__(self, function, mean, sigma, options, program_path = 'test.c', sample_size = 1, max_sample_size = 10, resetable = True, max_popsize = 1000, input_size = 1000):
    def __init__(self, program_path, output_dir = _Program.DEFAULT_DIRS['output'], log_dir = _Program.DEFAULT_DIRS['log'], max_test = 10, init_popsize = 7 ,max_popsize = 1000, max_gens = 1000 ,resetable = True, timeout = 15 * 60, testsuitesize = 1, mode = 'bytes', objective = ''):
        """Fuzzer with cmaes.

        Args:
        sampe_type = the type of a sample to optimize
        function = score function to optimize
        """
        # self._timer = _Timer()
        self._samples = []
        self._total_samples = []
        self._resetable = resetable
        self._max_test = max_test
        self._timeout = timeout
        self._coverage = 0
        self._total_coverage = 0
        self._generations = 0
        self._current_sample = None
        self._total_executed_lines = set()
        self._executed_line_sample_set = [] # {1: (block1, sample1),}
        self._interrupted = ''
        self._stop_reason = ''
        self._statuses = []
        self._samples_map = {}
        self._best_sample_index = (-1,0)
        # self._testsuitesize = self._calculate_testsuitesize()
        self._testsuitesize = testsuitesize
        self.objective = self._select_obejctive(objective)
        self.encode = self._select_encode(mode)
        self._program = _Program(program_path, output_dir = output_dir, log_dir = log_dir, timeout=timeout, mode = mode)
        self._cmaesbuilder = CMAES_Builder(init_popsize= init_popsize ,input_size = self._generate_input_size() * self._testsuitesize, max_popsize = max_popsize, max_gens= max_gens, mode = mode) # maybe parameter as dict
        self._samplecollector = SampleCollector()
        self._logger = FuzzerLogger().resister(self)

    def _select_obejctive(self, objective):
        if objective == 'lines':
            return self._f_lines
        elif objective == 'testsuite':
            return self._f_testsuite
        else:
            return self._f

    def _select_encode(self, mode):
        if mode == 'real':
            return self._encode_real
        elif mode == 'bytes':
            return self._encode
        else:
            return self._encode

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
        if returncode in Fuzzer._VERIFIER_ERROS:
            state = Fuzzer._VERIFIER_ERROS[returncode]

        self._statuses.append(state)

    def _generate_input_size(self):
        self._check_compile_error(self._program._compile_input_size())
        input_size, returncode = self._program.cal_input_size()
        self._check_runtime_error(returncode)
        #TODO: check input_szie < 2
        if input_size < CMAES_Builder.MIN_INPUT_SIZE:
            exit('ERROR: input_size: ' + str(input_size) + ', Input size must be greater than ' + str(CMAES_Builder.MIN_INPUT_SIZE) + '!')
        return input_size

    def _calculate_testsuitesize(self):
        return 8


    def get_coverage(self):
        # return self._cmaesbuilder.get_coverage()
        return self._coverage

    def _reset(self):
        if self._resetable:
            self._generations = 0
            self._samplecollector.reset_optimized()


    def _reset_samples(self):
        # self._logger.resister(self)
        if self._resetable:
            self._generations = 0
            self._samples = []
            # self._sample_map = {}
            # self._cmaesbuilder._reset()

    def _encode_real(self, sample):
        # a * 100.0=> long_max
        # a * 50 => lomg_max/2
        # a * 0 => 0
        # a = log_max / 100
        # a, b = sample
        # # a = int(a)
        # # a = a / (1 << 16)
        # print('a: ',a)
        # print('b: ',b)

        out = b''
        for input_comp in sample:
            out += (int(input_comp)).to_bytes(4, 'little', signed = False)
            # print(out)
        # print(out)
        return out

    def _encode(self, sample: np.ndarray):
        # if self.mode:
            # out = b''
            # for b in [abs(int(input_comp)).to_bytes(8,'little', signed = False) for input_comp in sample.tolist()]:
            #     # print('input_comp:',b)
            #     out += b
            # return out
        # else:
            # original
            # return bytes(sample.astype(int).tolist())
            #
            # Test
            # out = bytes(np.vectorize(lambda x: abs(int(x))%256)(sample).tolist())
        out = bytes(np.vectorize(lambda x: int(x))(sample).tolist())
        # print('bytes:',out)
        return out
            #


    def _run_sample(self, sample, returncode_check = False):
        returncode = self._program._run(self.encode(sample))
        if returncode_check:
            self._check_verifier_error(returncode)

    def _run_samples(self, samples, returncode_check = False):
        for sample in samples:
            self._run_sample(sample, returncode_check)
            # returncode = self._program._run(self._encode(sample))
            # if returncode_check:
            #     self._check_verifier_error(returncode)
            # print("RETURNCODE!!!:",returncode)

    def check_lines(self, lines, sample, cov):
        # if the new sample executes new lines, update the total executed lines
        # if not lines.issubset(self._total_executed_lines):
        if not lines.issubset(self._total_executed_lines):
            self._total_executed_lines.update(lines)
            self._executed_line_sample_set.append((lines, sample))


    def _f(self, sample:np.ndarray):
        # self._run_all_samples()
        # self._samplecollector.get_optimized_samples()
        self._run_samples(self._samplecollector.get_optimized_samples() + [sample])
        # cov = self._program.get_coverage()
        cov = self._program.get_coverage()
        # cov, lines = self._program.get_coverage_lines()
        # self.check_lines(lines, sample, cov)
        return -cov

    def _f_lines(self, sample):
        self._run_sample(sample)
        lines, total = self._program.cal_lines()
        lines = self._samplecollector.check_interesting(sample, lines, total)
        return -round(100 * len(lines)/total, 4)

    def _f_testsuite(self, testsuite):
        # for sample in np.split(testsuite, self._testsuitesize):
        self._run_samples(np.split(testsuite, self._testsuitesize))
        cov, lines = self._program.get_coverage_lines()
        for sample in np.split(testsuite, self._testsuitesize):
            self.check_lines(lines, sample, cov)
        # cov = round(100* len(self._total_executed_lines) / self._program._total_lines, 4)
        return -cov
        # cov, lines = self._program.get_coverage_lines()
        # self.check_lines(lines, sample, cov)

    # def _f_real(self, input_vector):
        

    
    def _current_state(self):
        # if self._samplecollector:
        # return dict(testcase = len(self._samples), coverage = self._coverage, generations = self._generations)
        return dict(testcase = (len(self._samplecollector.optimized_samples), self._samplecollector.get_total_size()),
         coverage = (self._samplecollector.coverage(), self._samplecollector.total_coverage()))
        #  generations = self._generations)
        # return dict(testcase = len(self._samples), coverage = self._coverage)


    def _stop(self):
        if self._interrupted:
            self._stop_reason = self._interrupted
        elif self._samplecollector.total_coverage() == 100 or self._cmaesbuilder._fbest == -100:
            self._stop_reason = 'coverage is 100%'
        elif self._cmaesbuilder._options['popsize'] > self._cmaesbuilder._max_popsize:
            self._stop_reason = 'max popsize is reached'

        return self._stop_reason
    def time(self):
        return time.time() - _init_time

    def optimize_sample_lines(self):
        es = self._cmaesbuilder
        # while self._samplecollector.total_coverage() < 100 and not es.stop_lines():
        while not es.stop_lines():
            try:
                solutions = es.ask()
                values = [self.objective(x) for x in solutions]
                es.tell(solutions, values)
                es.update()


                # print('!!!!!!!!best:\n',es._es.result.xbest)
                # print("!!!!!!!!optimizingsample== xbest", self._samplecollector.optimizing_sample.sample == es._es.result.xbest)
                # print('!!!!!!!!means:\n',es._es.result.xfavorite)
                # print('!!!!!!!!stds:\n',es._es.result.stds)
                # if not any(self._samplecollector.optimizing_sample.sample == es._es.result.xbest):
                # if test == 4:
                #     print('sol',solutions)
                #     print('val', values)
                    # exit()
                    # raise KeyboardInterrupt
                # print('!!!!!!!!values:\n',values)

            except (subprocess.TimeoutExpired, KeyboardInterrupt) as e:
                self._interrupted = e.__class__.__name__
                self._program._reset()
                break

        return self._samplecollector.update_optimized()

    def optimize_testsuite_lines(self):
        self._program._compile_program()

        # what is the first good sigma and mean?
        # what is good bounds?
        # sigma = 148
        while not self._stop():
            self._cmaesbuilder.init_cmaes()
            optimized = self.optimize_sample_lines()
            self._logger.report_changes(optimized)
            if not optimized:
                if self._cmaesbuilder._increase_popsize():
                    self._reset()

    @_timeit
    def optimize_sample(self):
        es = self._cmaesbuilder
        while not es.stop():
            try:
                solutions = es.ask()
                values = [self.objective(x) for x in solutions]
                es.tell(solutions, values)
                es.update()
                
                # print('!!!!!!!!solutions:\n', solutions)
                # print('!!!!!!!!best:\n',es._es.result.xbest)
                # print('!!!!!!!!means:\n',es._es.result.xfavorite)
                # print('!!!!!!!!stds:\n',es._es.result.stds)
                # print('!!!!!!!!solutions:\ns', values)

                # print('!!!!!!!!means:\n',es._es.result.xfavorite)
                # print('!!!!!!!!stds:\n',es._es.result.stds)

            except (subprocess.TimeoutExpired,  KeyboardInterrupt) as e:
                self._interrupted = e.__class__.__name__
                self._program._reset()
                break

        return self._samplecollector.update_coverage(es.result.xbest, -es.result.fbest)

    @_timeit
    def optimize_testsuite(self):
        self._program._compile_program()
        
        while not self._stop():
            self._cmaesbuilder.init_cmaes()
            optimized = self.optimize_sample()
            self._logger.report_changes(optimized)

            if not optimized:
                if self._cmaesbuilder._increase_popsize():
                    self._reset()


    def optimize_testsuite2(self):
        self._program._compile_program()
        count = 0
        self._dists = []
        mean = None

        # self._cmaesbuilder.init_cmaes()
        # self.optimize_sample()
        # self._samples = [sample for (i, (_, sample)) in self._executed_line_sample_set.items() if i != self._best_sample_index[0]]
        
        
        while not self._stop() :
            self._cmaesbuilder.init_cmaes()
            optimized = self.optimize_sample()
            if optimized:
                self._current_sample = self._cmaesbuilder.get_xbest()
                # if len(self._samples) == 0:
                self._samples.append(self._current_sample)
                self._total_samples.append(self._current_sample)
                mean = self._cmaesbuilder._es.result.xfavorite
                self._dists.append(mean)

                self._logger.report_changes()
                # self._samples = [sample for (lines, sample) in self._executed_line_sample_set.values() if not 359 in lines]
                # self._samples = [sample for (i, (_, sample)) in self._executed_line_sample_set.items() if i != self._best_sample_index[0]]
                # self._run_samples(self._samples)
                # self._cmaesbuilder._fbest = -self._program.get_coverage()
                # self._coverage = -self._cmaesbuilder._fbest
                # self._program.get_coverage_lines()
                # self._samples = [sample for (i, (_, sample)) in self._executed_line_sample_set.items()]

                # print('current_sample:', self._current_sample)
                # self._sample_map[str(sample)] = self.get_coverage()
                count = 0
                # maybe better if we increase if it was not optimized
                # self._cmaesbuilder._reset_popsize()
            else:
                self._logger.report_changes()
                mean = None
                # coverage is the same or lower than -fbest
                count += 1


            # # idea 2 : delete first sample if it failed
            # if count > 0 and len(self._samples) > 0:
            #     # maybe rset fbest as wel
            #     self._cmaesbuilder._reset_fbest()
            #     del self._samples[0]
                # del self._sample_map.keys()[-1]

            # if optimized:

                

            if count > 0:
                # self._total_samples += self._samples
                # self._coverage = 0
                self._cmaesbuilder._reset_fbest()
                self._samples_map = {self._cmaesbuilder._potential_popsize : self._samples}
                if self._cmaesbuilder._increase_popsize():
                    # self._cmaesbuilder._reset_fbest()
                    self._reset_samples()
                    # pass
            # else:

                
            # if not self._cmaesbuilder._is_over_threshold():
            # else:
                # break


        return self._samples

    def optimize_testsuite3(self):
        self._program._compile_program()
        es = self._cmaesbuilder
        while not self._stop() :
            es.init_cmaes()
            # optimized = self.optimize_sample() try:
            try:
                while not es.stop():
                    solutions = es.ask()
                    # print(len(solutions))
                    values = [self._f_testsuite(x) for x in solutions]
                    # print(len(values))
                    es.tell(solutions, values)
                    # self._update()
                    self._generations = self._cmaesbuilder.get_generations()
                    self._coverage = -es.get_fbest()
                    # print('values:',values )
                    # print('f')

                    print('!!!!!!!!best:\n',es._es.result.xbest)
                    print('!!!!!!!!means:\n',es._es.result.xfavorite)
                    print('!!!!!!!!stds:\n',es._es.result.stds)
                    # print('iterations:', self._cmaesbuilder.get_generations())
                    # print('fbest:',self._cmaesbuilder._es.result.fbest)
                    # print('evaluations:', es.result.evals_best)
                    # es.tell(solutions, values)
                    # print('\n')
            except (subprocess.TimeoutExpired,  KeyboardInterrupt) as e:
                self._interrupted = e.__class__.__name__
                self._program._reset()

            es.reset_stop_conditions()
            es._update_fbest()
            self._coverage = -es.get_fbest()
            
            # if es.is_optimized():
            #     self._current_sample = self._cmaesbuilder.get_xbest()
            #     # if len(self._samples) == 0:
            #     self._samples.append(self._current_sample)
            #     self._total_samples.append(self._current_sample)
            #     mean = self._cmaesbuilder._es.result.xfavorite
            #     self._dists.append(mean)
            self._samples = np.split(es.get_xbest(), self._testsuitesize)
            # self._total_samples = self._executed_line_sample_set

    # def optimize_testsuite_real(self):
        # self._program._compile_pro

    # def _check_testsuite_minimization(self):
    #     safe = True
    #     for i, (lines1, _) in enumerate(self._executed_line_sample_set):
    #         for j, (lines2, _) in enumerate(self._executed_line_sample_set):
    #             if i != j:
    #                 safe = safe and not lines1.issubset(lines2) and not lines2.issubset(lines1)
        
    #     print('minimization is ', safe)


    def generate_testsuite_lines(self):
        self.optimize_testsuite_lines()
        self._program._timeout = None
        # self._check_testsuite_minimization()
        # self._total_samples = [s.sample for s in self._samplecollector.total_samples]
        self._total_samples = self._samplecollector.get_total_samples()
        # return self._total_samples, self._samples_map, self._dists
        return self._total_samples


    def generate_testsuite(self):
        self.optimize_testsuite()
        self._program._timeout = None
        self._total_samples = [s.sample for s in self._samplecollector.total_samples]
        return self._total_samples

    def generate_testsuite2(self):
        self.optimize_testsuite3()
        self._total_samples = [sample for (_, sample) in self._executed_line_sample_set]
        self._program_timeout = None
        return self._total_samples
        

    

    # def generate_testsuite_real(self):
        # self.optimize_testsuite_real()
        # return self._total_samples
    # def gcov(self):
    #     self._run_samples(self._samples)
    #     print(self._program._gcov())

    # def cal_potential_dim(self):
        # program = _Program(verifier_path='programs/__VERIFIER_input_size.c')

        # subprocess.run()

    # def get_timelog(self):
    #     # log = _Timer.log
    #     # _Timer.reset()
    #     # log = _time_log
    #     # _time_log = {}
    #     return _time_log

    # def report(self):
    #     print('reporting')
    #     print(self.get_timelog())
    #     _time_log = {}

    def last_report(self):
        # self._program._compiled = False
        # self._program.verifier_path = 'programs/__VERIFIER_with_error.c'
        # self._program._compile()
        # if os.path.isfile('__VERIF
        # m.pname+'.gcda')
        # self._program._reset()
        print('total sample len:', len(self._total_samples))
        self._run_samples(self._total_samples, returncode_check=True)
        gcov = self._program._gcov() 
        print(gcov)
        self._total_coverage = self._program._cov(gcov)
        self._logger.report_final()
        self._logger.report_time_log()
        if os.path.isfile('__VERIFIER.gcda'):
            os.remove('__VERIFIER.gcda')
        if os.path.isfile(self._program.pname+'.gcda'):
            os.remove(self._program.pname+'.gcda')


# def bytes_to_int(bytes: np.ndarray) -> int:
#     # assert len(bytes) == 4, "Integer should have 4 bytes"
#     # check
#     # if np.any( bytes < 0) or np.any(bytes > 255):
#     # check
#     # print("bytes before: ", bytes)
#     # bytes = np.where(bytes < 0, 0., bytes)
#     # bytes = np.where(bytes > 255, 255., bytes)
#     # print("bytes after: ", bytes)

#     result = int.from_bytes(bytes.astype(int).tolist(), 'little', signed = True)
#     # print('bytes:', bytes)
#     # print('int:', result)
#     return result




# def program(input : int):
#     coverage = 0
#     x = input

#     if x > 0:
#         if x > 1000:
#             if x < 100000:
#                 return 4

#             else:
#                 return 3

#         else:
#             return 2
#     else:
#         return 1


# def function(sample: np.ndarray):
#     coverage = program(bytes_to_int(sample))
#     return -coverage


# def function1(sample : np.ndarray):
#     # any input has same coverage
#     return 50


# def function2(sample : np.ndarray):
#     # check if x3 < 50 will be found

#     x1, x2, x3 = sample
#     if x1 > 128:
#         if x2 < 128:
#             if x3 < 50:
#                 return 0.
#             else:
#                 return 1.
#         else:
#             return 2.
#     else:
#         return 3.


# def function3(sample: np.ndarray):
#     # check if x3 == 128 will be found

#     x1, x2, x3 = sample
#     if 128 <= x1 < 129 and 128 <= x2 < 129:
#         return 0.
#     else:
#         return 3.

# def function4(sample: np.ndarray):
#     # check if more 0 or more 255
#     x1, x2, x3 = sample
#     if x1 > 128:
#         if x2 > 128:
#             if 0 <= x3 < 1 or 255 < x3 <= 256:
#                 return 0.
#             else:
#                 return 1.
#         else:
#             return 2.
#     else:
#         return 3.

# def function5(sample: np.ndarray):
#     # check if more 100 or more 200
#     x1, x2, x3 = sample
#     if x1 > 128:
#         if x2 > 128:
#             if x3 < 101 or 200 < x3 <= 201:
#                 return 0.
#             else:
#                 return 1.
#         else:
#             return 2.
#     else:
#         return 3.

# def function6(sample: np.ndarray):
#     # check if f4 with 256 (max) and f6 with 255 different

#     x1, x2, x3 = sample
#     if x1 > 128:
#         if x2 > 128:
#             if 254 <= x3 < 256 or 0 <= x3 < 1 :
#                 return 0.
#             else:
#                 return 1.
#         else:
#             return 2.
#     else:
#         return 3.

# def function7(sample: np.ndarray):
#     # check if minimum or maximum will be found more often

#     x1, x2, x3 = sample
#     if 0 <= x1 <= 1 or  255 <= x1 <= 256:
#        #   or \
#        # 85 <= x3 < 86 or 170 <= x3 < 171:
#          return 0.
#     else:
#         return 3.

# def function8(sample: np.ndarray):
#     # check if minimum or maximum will be found more often

#     x1, x2, x3 = sample
#     if 0 <= x1 <= 10 or  246 <= x1 <= 256:
#        #   or \
#        # 85 <= x3 < 86 or 170 <= x3 < 171:
#          return 0.
#     else:
#         return 3.


# # def parse_bytes_to_sample(sample_type: SampleType):
#     # maybe a general method or class to parse byte to sample in a right type
#     # pass



# def plot_samples(xs, ys, zs):
#     fig = plt.figure()
#     ax = plt.axes(projection = '3d')
#     ax.scatter3D(xs, ys, zs, cmap='Greens')
#     plt.show()

# def dist_samples(xs, ys, zs, kwargs):
#     # hist_samples = plt.figure()
#     # plt.figure()
#     fig, (axes1, axes2) = plt.subplots(2, 3, figsize=(10,5), dpi=100, sharex=True, sharey=False)
#     data = [xs, ys, zs]
#     colors = ['g', 'b', 'r']
#     labels = ['x1', 'x2', 'x3']
#     # kwargs = dict(alpha = 0.5, bins = 100)
#     kkwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

#     # # density
#     for i, (ax1, ax2) in enumerate(zip(axes1.flatten(), axes2.flatten())):
#         sns.distplot(data[i], color = colors[i], axlabel = labels[i], ax = ax1, **kkwargs)
#         ax2.hist(data[i], **kwargs, color = colors[i])

#     # hist
#     # for i, ax in enumerate(axes.flatten()):

#     plt.tight_layout()

# def dist_samples_one(sample):
#     # hist_samples = plt.figure()
#     # plt.figure()
#     # fig, axes = plt.subplots(1, 3, figsize=(10,2.5), dpi=100, sharex=True, sharey=False)
#     kwargs = dict(alpha = 0.5, bins = 50)

#     # sns.distplot(samples, color = 'g', label = 'Input')
#     plt.hist(sample, **kwargs, color='g')

#     # plt.tight_layout()
#     plt.legend()

# def dist_values(values, kwargs):
#     hist_values = plt.figure()
#     # kwargs = dict(alpha = 0.5, bins = 50)
#     plt.hist(values, **kwargs, color = 'g')

# def visualize_results(samples, values, fname, i, sample_size):

#     xs, ys, zs = [],[],[]
#     for sample in samples:
#         x, y, z = sample
#         xs.append(x)
#         ys.append(y)
#         zs.append(z)
#     xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

#     kwargs = dict(alpha = 0.5, bins = sample_size)

#     plt.figure(2*i - 1)
#     dist_samples(xs, ys, zs, kwargs)
#     plt.suptitle(fname)
#     # dist_samples_one(np.array(samples))
#     plt.figure(2*i)
#     dist_values(np.array(values), kwargs)
#     plt.suptitle(fname)

# def visualize_results1(samples, coverages, pname):
#     plt.plot(samples, coverages)
#     plt.grid(True)
#     plt.suptitle(pname)
#     plt.xlabel('sample')
#     plt.ylabel('coverage')
#     plt.show()


# def main1():
#     mean = 3 * [128]
#     sigma = 64
#     options = dict(bounds = [0, 256], popsize = 10000)
#     # functions = [function1, function2, function3, function7, function8]
#     functions = [function3]
#     sample_size = 3
#     # functions = [function]

#     for i, function in enumerate(functions, 1):
#         fuzzer = Fuzzer(function, mean, sigma, options)
#         samples = []
#         values = []
#         stds=[]
#         for _ in range(sample_size):
#             sample, std, value= fuzzer.get_sample()
#             samples.append(sample)
#             # values.append(function(sample))
#             values.append(value)
#             stds.append(std)
#         # print(samples)
#         # print(values)
#         # print(stds)
#         visualize_results(samples, values, function.__name__, i, sample_size)

#     plt.show()


# def get_coverage(program, input):
#     gcc = 'gcc'
#     path = '../test/'
#     program = 'test.c'

#     subprocess.run('./test', input = input)
#     output = subprocess.run(['gcov', program], stdout = subprocess.PIPE).stdout.decode('utf-8')

#     start, end = 0 , 0
#     for i in range(len(output)):
#         if output[i] == ':':
#             start = i + 1
#         elif output[i] == '%':
#             end = i
#             break

#     return float(output[start:end])

# def main2():
#     # subprocess.run('gcc ./test.c -o ./test --coverage')
#     #
#     # input = np.zeros(4).astype(int).tolist()
#     #
#     # program = './test.c'
#     # coverage = get_coverage(program, bytes(input))
#     # print(coverage)
#     # print(get_coverage(program, '1'._encode('utf-8')))

#     # fuzzer = Fuzzer()
#     # subprocess.run('del *.gcda', shell = True)

#     # subprocess.run('gcc test1.c -o test1')
#     # for i in range(256):
#     #     input = bytes([i,0,0,0])
#     #     output = subprocess.run('./test1', stdout=subprocess.PIPE, input = input)
#     #     print('generated output:',output.stdout)
        
#     # print(input)
#     subprocess.run('gcc testingsize.c __VERIFIER.c -o testingsize --coverage')
#     for i in range(1):
#         input = b'\x1a\x00\x00\x00\x1a\x00\x00\x00'
#         output = subprocess.run(['./testingsize'], capture_output = True, input = input)
#         print(output.stdout.decode())
#         subprocess.run('gcov testingsize')
#         os.remove('testingsize.gcda')

# def main3():
#     mean = 4 * [128]
#     sigma = 64
#     options = dict(bounds = [0, 256], popsize = 10, verb_disp = 0)
#     generations = 4
#     programs = ['./test.c']
#     max_popsize = 100

#     for i, program in enumerate(programs, 1):
#         samples = []
#         values = []
#         stds=[]
#         fuzzer = Fuzzer(None, mean, sigma, options, program_path = program, sample_size = 1, max_popsize = max_popsize)
#         for i in range(generations):
#             # sample, std, value = fuzzer.get_sample2()
#             # subprocess.run('del test.gcda', shell = True)
#             # for x in np.split(sample, 1):
#             #     print(x)
#             #     samples.append(bytes_to_int(x))c
#             # values.append(value)
#             # stds.append(std)
#             # for sample in fuzzer.get_samples():
#             #     samples.append(bytes_to_int(sample))
#             print('generation:', i)
#             print('coverage:', fuzzer.get_coverage())
#             for sample in fuzzer.get_samples2():
#                 samples.append(bytes_to_int(sample))
#         # visualize_results(samples, values, function.__name__, i, sample_size)
#     print('last samples:', samples)
#     print('last coverage:',fuzzer.get_coverage())
#     # print(values)
#     # print(stds)

#     # plt.show()

# def test0(sample: np.ndarray):
#     x = bytes_to_int(sample)
#     if x > 0:
#         return 87.74
#         if x < 100:
#             return 100.0
        
#     return 67.14

# def main4():
#     mean = 2 * [128]
#     sigma = 64
#     options = dict(bounds = [0, 256], popsize = 10, verb_disp = 0)
#     programs = ['./test_2max.c']
#     sample_size = 8

#     for program in programs:
#         fuzzer = Fuzzer(None, mean, sigma, options, program_path = program)
#         samples = {}
#         inputs = {}
#         bs = []
#         for i in range(sample_size):
#             sample = fuzzer.get_sample()
#             samples[bytes_to_int(sample)] = fuzzer.get_coverage()
#             inputs[str(sample)] = fuzzer.get_coverage()
#             bs.append(sample)

#     for b in bs:
#         subprocess.run('test_2max', input = bytes(b.astype(int).tolist()))
#     subprocess.run(['gcov', 'test_2max'])
#     print(fuzzer.get_timelog())
#     print('samples:', samples)
#     print('inputs:', inputs)

# def main5():
#     program = _Program('./test_2max.c')
#     program.compile()
#     samples = []
#     inputs = []
#     coverages = []
#     s = 2 ** 0
#     for i in range(int(256/s)):
#         print(i)
#         for j in range(int(256/s)):
#             input = bytes([s*j,s*i])
#             sample = int.from_bytes(input, 'little', signed = True)
#             cov = program.get_coverage(input)
#             inputs.append(input)
#             samples.append(sample)
#             coverages.append(cov)
#     # visualize_results1(samples, coverages, 'test1.c')
#     samples, coverages = zip(*sorted(zip(samples, coverages)))
#     print(*zip(samples,coverages))
#     for input in inputs:
#         subprocess.run('./test_2max', input = input)
#     subprocess.run('gcov ./test_2max')
#     visualize_results1(samples, coverages, 'test1.c')

# def simple_run(program, input):
#     # subprocess.run(['gcc', program, '-o', program[:-2]])
#     subprocess.run(program, input = input)

# def test2(x):
#     y = -32768
#     z = 8192
#     if (x < y + z):
#         return 43.75
#     y += z
#     if (y <= x and x < y + z):
#         return 58.33
#     y += z
#     if (y <= x and x < y + z):
#         return 50.0
#     y += z
#     if (y <= x and x < y + z):
#         return 50.0
#     y += z
#     if (y <= x and x < y + z):
#         return 45.83
#     y += z
#     if (y <= x and x < y + z):
#         return 54.17
    
#     y += z
#     if (y <= x and x < y + z):
#         return 45.83
    
#     y += z
#     if (y <= x):
#         return 43.75

#     return 0

# def main6():
#     samples = []
#     coverages = []
#     s = 2 ** 0
#     for i in range(int(256/s)):
#         print(i)
#         for j in range(int(256/s)):
#             input = bytes([s*j,s*i])
#             sample = int.from_bytes(input, 'little', signed = True)
#             cov = test2(sample)
#             samples.append(sample)
#             coverages.append(cov)
#     # visualize_results1(samples, coverages, 'test1.c')
#     samples, coverages = zip(*sorted(zip(samples, coverages)))
#     visualize_results1(samples, coverages, 'test2.c')


# def main_sv():
#     mean = 100 * 1 * [128]
#     sigma = 64
#     options = dict(bounds = [0, 255.9], popsize = 100, verb_disp = 0)
#     pname = 'programs/pals_STARTPALS_ActiveStandby.ufo.BOUNDED-10.pals'
#     # pname = './pals_lcr-var-start-time.3.ufo.BOUNDED-6.pals'
#     # pname = './cdaudio_simpl1.cil-2'
#     # pname = './s3_clnt_1.cil-1'
#     # pname = './pals_floodmax.3.ufo.BOUNDED-6.pals' # True
#     # pname = './data_structures_set_multi_proc_ground-2'
#     # pname = './pals_STARTPALS_ActiveStandby.1.ufo.BOUNDED-10.pals'
#     # pname = './pals_opt-floodmax.4.ufo.BOUNDED-8.pals' # almost infinitely optimizable
#     # pname = 'programs/test_2max'
#     # programs = ['./data_structures_set_multi_proc_ground-1.c']
#     # programs = ['./data_structures_set_multi_proc_ground-2.c'] #true
#     # programs = ['standard_init1_ground-1.c'] 
#     # programs = ['standard_copy1_ground-1.c'] # True
#     # programs = ['standard_copy1_ground-2.c']
#     # programs = ['relax-2.c'] # True
#     # programs = ['pals_lcr-var-start-time.3.ufo.BOUNDED-6.pals.c'] # True
#     programs = [pname+'.c']

#     sample_size = 2
#     max_popsize = 1000

#     for program in programs:
#         fuzzer = Fuzzer(None, mean, sigma, options, sample_size = sample_size, program_path = program, max_popsize=max_popsize)
#         print(fuzzer.get_samples())
#         print(fuzzer.get_timelog())
#         fuzzer.gcov()
    

#     # for program in programs:
#     #     fuzzer = Fuzzer(None, mean, sigma, options, program_path = program)
#     #     samples = {}
#     #     inputs = {}
#     #     bs = []
#     #     for i in range(sample_size):
#     #         sample = fuzzer.get_sample()
#     #         samples[bytes_to_int(sample)] = fuzzer.get_coverage()
#     #         inputs[str(sample)] = fuzzer.get_coverage()
#     #         bs.append(sample)

#     # for b in bs:
#     #     subprocess.run(pname, input = bytes(b.astype(int).tolist()))
#     # subprocess.run(['gcov', pname+'.gcno'])
#     # print('samples:', samples)
#     # print('inputs:', inputs)

# def main_sv2():

#     path = 'programs/'

#     programs = [path+'pals_STARTPALS_ActiveStandby.ufo.BOUNDED-10.pals.c']
#     # programs = [path+'relax-2.c'] # True
#     # programs = [path+'test_2max.c']

#     sample_size = 10

#     for program in programs:
#         fuzzer = Fuzzer(None, [0], None, None, program_path = program, max_sample_size = sample_size)
#         t = fuzzer.generate_testsuite()
#         # print('testsuite:', t)
#         fuzzer.gcov()
#         fuzzer.update()


def main_sv3():
    max_popsize = 1000
    # sample_size = 10
    timeout = 15 * 60
    path = 'programs/pals_STARTPALS_ActiveStandby.ufo.BOUNDED-10.pals.c' # ~ 83%
    # path = 'programs/pals_floodmax.5.1.ufo.BOUNDED-10.pals.c' # ~ 95%
    # path = 'programs/xor5.c' # ~100%

    argsize = len(sys.argv)
    if argsize > 1:
        path = sys.argv[1]
        if argsize > 2:
            max_popsize = int(sys.argv[2])
            if argsize > 3:
                timeout = int(sys.argv[3])


    programs = [path]
    
    
    # programs = [path+'test.c']


    for program in programs:
        # fuzzer = Fuzzer(None, [0], None, None, program_path = program, max_sample_size = sample_size, input_size = input_size)
        fuzzer = Fuzzer(program_path = program, max_popsize = max_popsize, timeout=timeout)
        t = fuzzer.generate_testsuite()
        print('testsuite:', t)
        fuzzer.last_report()
        # fuzzer.gcov()
        # fuzzer.report()

def main_test():
    # subprocess.run('gcc programs/test.c programs/__VERIFIER_input_size.c -o test')

    #            1,2,3,4,5,6,7,8,9,0,11,12,13,14,15,16,17,18,19
    testsuite = [np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
        -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
         81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
         95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
        141.18546982,  139.96057095,   64.2624934 ]), np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
        -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
         81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
         95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
        141.18546982,  139.96057095,   64.2624934 ]), np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
        -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
         81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
         95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
        141.18546982,  139.96057095,   64.2624934 ]), np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
        -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
         81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
         95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
        141.18546982,  139.96057095,   64.2624934 ]), np.array([ 2.31112248e+02,  7.88875081e+01, -1.00272686e+02, -1.64848604e+01,
       -1.07149203e-01, -2.23708581e+01,  1.88348397e+02,  5.34317338e-01,
        1.21936163e+02,  2.09374535e+02,  1.06511931e+02,  1.25130298e+02,
       -1.87479582e+02, -3.67562446e+01, -7.15702679e+01, -1.72612956e+02,
        6.98037430e+01,  1.98282392e+02, -1.16470826e+02]), np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
        -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
         81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
         95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
        141.18546982,  139.96057095,   64.2624934 ]), np.array([  54.1248795 ,  -71.6752295 ,  -22.99511084, -103.02129871,
          0.8684931 , -113.55180125,  -76.8893616 ,   70.16172702,
         55.10751883,  -97.30580052,  -28.63692408,   29.664043  ,
         25.12036177, -104.14226332,   16.64123275,  -38.90410403,
         76.71797608,  -12.81207861,   22.99548734]), np.array([  54.1248795 ,  -71.6752295 ,  -22.99511084, -103.02129871,
          0.8684931 , -113.55180125,  -76.8893616 ,   70.16172702,
         55.10751883,  -97.30580052,  -28.63692408,   29.664043  ,
         25.12036177, -104.14226332,   16.64123275,  -38.90410403,
         76.71797608,  -12.81207861,   22.99548734]), np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
        -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
         81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
         95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
        141.18546982,  139.96057095,   64.2624934 ]), np.array([  54.1248795 ,  -71.6752295 ,  -22.99511084, -103.02129871,
          0.8684931 , -113.55180125,  -76.8893616 ,   70.16172702,
         55.10751883,  -97.30580052,  -28.63692408,   29.664043  ,
         25.12036177, -104.14226332,   16.64123275,  -38.90410403,
         76.71797608,  -12.81207861,   22.99548734]), np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
        -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
         81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
         95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
        141.18546982,  139.96057095,   64.2624934 ]), np.array([  54.1248795 ,  -71.6752295 ,  -22.99511084, -103.02129871,
          0.8684931 , -113.55180125,  -76.8893616 ,   70.16172702,
         55.10751883,  -97.30580052,  -28.63692408,   29.664043  ,
         25.12036177, -104.14226332,   16.64123275,  -38.90410403,
         76.71797608,  -12.81207861,   22.99548734]), np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
        -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
         81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
         95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
        141.18546982,  139.96057095,   64.2624934 ])]

    for sample in testsuite:
        subprocess.run('output/pals_STARTPALS_ActiveStandby.ufo.BOUNDED-10.pals', input = bytes(np.vectorize(lambda x: int(x)%256)(sample).tolist()))
    # inp = bytes([0,0,0,0,0,0,0,0,0,0,254,254,254,0,0,0,0,0,113])
    # subprocess.run('output/pals_STARTPALS_ActiveStandby.ufo.BOUNDED-10.pals', input = inp)

    # time_with_none = 0
    # time_with_19dim = 0
    # for _ in range(10):
    #     init_time = time.time()
    #     runtime += time.time() - init_time
    # print("tme")

    # init_time = time.time()

def main_test_41cov():
    array = np.array
    testsuite= [array([ 8.17945369e-02,  2.41488359e+00,  6.06120964e+00, -6.20581374e-01,
       -4.15943706e+00,  8.35842643e+01,  2.02186960e+01,  8.51386345e+01,
        9.25300293e+01,  1.51778777e+00,  4.21060736e+01,  3.61547523e+01,
       -5.48409122e+01,  9.68681958e+01,  6.87742054e+01, -6.95826083e+00,
       -5.51625389e+01, -5.35698789e+01, -1.80691723e+01,  6.94381889e+01,
        1.79509374e+01, -1.70465721e+01, -2.73731169e+00, -1.19490791e+02,
       -6.79847997e-01, -4.98916785e+01,  7.53100010e+01])]



    for sample in testsuite:
        subprocess.run('output/pals_floodmax.3.ufo.BOUNDED-6.pals', input = bytes(np.vectorize(lambda x: int(x)%256)(sample).tolist()))


def parse_argv_to_fuzzer_kwargs():
    argvsize = len(sys.argv)
    example = 'e.g.: python3 fuzzer.py <program_path> [-od <output_dir>] [-ld <log_dir>] [-ip <init_popsize>] [-mp <max_popsize>] [-mg <max_gens>] [-t <timeout>] [-ts <testsuitesize>] [-m <mode>] [-o <objective>]'
    if argvsize == 1:
        exit('ERROR: No program_path is given!\n' + example)

    commands = {'-od' : 'output_dir', '-ld' : 'log_dir', '-ip' : 'init_popsize', '-mp' : 'max_popsize', '-mg' : 'max_gens','-t' : 'timeout', '-ts' : 'testsuitesize', '-m' : 'mode', '-o' : 'objective'}
    command_type_dict = {'output_dir' : str, 'log_dir' : str, 'max_popsize' : int, 'timeout' : float, 'init_popsize' : int, 'max_gens' : int, 'testsuitesize' : int, 'mode' : str, 'objective' : str}
    current_command = None
    fuzzer_kwargs = {}
    if argvsize > 1:
        fuzzer_kwargs['program_path'] = sys.argv[1]
        for arg in sys.argv[2:]:
            if arg == '-help':
                exit(example)
            if arg in commands:
                current_command = commands[arg]
                if current_command in fuzzer_kwargs:
                    exit('ERROR: <' + current_command + '> is already given!')
            elif current_command:
                fuzzer_kwargs[current_command] = command_type_dict[current_command](arg)
                current_command = None
            else:
                exit('ERROR: Unkown command "' + arg + '"!\n' + example)

    return fuzzer_kwargs

def main():
    fuzzer = Fuzzer(**parse_argv_to_fuzzer_kwargs())
    t = fuzzer.generate_testsuite()
    print('testsuite:\n', t)
    fuzzer.last_report()
    # for program in programs:
        # fuzzer = Fuzzer(None, [0], None, None, program_path = program, max_sample_size = sample_size, input_size = input_size)

def main_testsuit():
    fuzzer = Fuzzer(**parse_argv_to_fuzzer_kwargs())
    t = fuzzer.generate_testsuite2()
    print('testsuite:\n', t)
    print('coverage:\n', fuzzer.get_coverage())
    fuzzer.last_report()
    fuzzer._total_samples = fuzzer._samples
    fuzzer.last_report()


def test_input_with_type():
    program_path = 'e.c'
    # mean = []
    fuzzer = Fuzzer(program_path=program_path)
    fuzzer._cmaesbuilder._es

def test_real():
    # subprocess.run("gcc test.c verifiers/__VERIFIER_real.c -o test")
    # inp = bytes([1,0,0,0,1,0,2,0]*2)
    v = 12345
    print(v)
    inp = int(v).to_bytes(8, 'little', signed=False)
    # inp = int(1).to_bytes(2,'little', signed=False) + b'\x00' * 6
    # inp = inp
    print(inp)
    # subprocess.run('gcc ./parse_to_int.c -o ./parse_to_int')
    subprocess.run("./parse_to_int", input = inp)
    # inp = bytes([255,2,255,4,255,255,7,8]*2)
    # inp = int(777*2^(64-16)).to_bytes(8, 'big', signed=False)
    # inp = b'\x00' * 6 + int(1).to_bytes(2,'big', signed=False)
    # inp = inp * 2
    # print(inp)
    # subprocess.run("./test", input = inp)
    # subprocess.run("./test", input = inp)

def main_real():
    fuzzer = Fuzzer(**parse_argv_to_fuzzer_kwargs(), real = True)
    t, tm, dists = fuzzer.generate_testsuite()
    print('testsuite:\n', t)
    print('testsuite map:\n', tm)
    print('distributions:\n', dists)
    fuzzer.last_report()

def test2():
    # inp = bytes([0,0])
    # subprocess.run('./test2', input = inp)
    # inp = bytes([127,128])
    # subprocess.run('./test2', input = inp)
    # inp = bytes([128,128])
    # subprocess.run('./test2', input = inp)
    # inp = bytes([255,255])
    # subprocess.run('./test2', input = inp)
    inp = bytes(np.vectorize(lambda x: int(x)%256)(np.array([58.51964139, 191.83170435])).tolist())
    subprocess.run('./test2', input = inp)
    

def main_lines():
    fuzzer = Fuzzer(**parse_argv_to_fuzzer_kwargs(), objective='lines')
    t= fuzzer.generate_testsuite_lines()
    print('testsuite:\n', t)
    fuzzer.last_report()
    

if __name__ == "__main__":
    # main_sv3()
    main()
    # test_real()
    # main_lines()
    # test2()
    # main_testsuit()
    # main_real()
    # test_real()
    # main_test_41cov()
    # main_test()
    # main_test_inputsize()
    # main_sv3()
    # main2()
    # simple_run('test2', bytes([79, 185]))
    # simple_run('test2', bytes([141,249]))
    
