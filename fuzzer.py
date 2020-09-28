import cma
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import subprocess
import os
import time
import sys
import random
import argparse

# from scipy.optimize import minimize, fmin_l_bfgs_b as minimize_l_bfgs_b

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

    COV_DIGITS = 2

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

    @_timeit
    def _gcov_branch(self, arg = ''):
        return subprocess.run(['gcov', self.pname + '.gcda', '-b', '-c', arg], capture_output = True, timeout=self.timeout()).stdout.decode()

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
    def cal_lines(self, gcov):
        # gcov = self._gcov('-t')
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
        # self._reset()
        return output_lines, total

    @_timeit
    def cal_branches(self, gcov):
        output_branches = set()
        if len(gcov) == 0:
            return output_branches
        
        executed = 0
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
                        # if int(line[6:9]) % 2 == 0:
                        #     # total += 1
                        #     output_branches.add(i+0.5)

                    #     executed += 1


            # elif line.startswith('c'):
            #     if 'returned' == line[10:18]:
            #         if int(line[18:20]) > 0:
            #             executed += 1
            #             output_branches.add(i)
            # else:
            #     parts = line.split(':', 2)
            #     if not '#' in parts[0] and not '-' in parts[0]:
            #         executed += 1
            #         # line 244
            #         output_branches.add(int(parts[1]))


        return output_branches, total
    
    @_timeit
    def get_lines_total(self):
        gcov = self._gcov('-t')
        lines, total = self.cal_lines(gcov)
        self._reset()
        return lines, total

    @_timeit
    def get_branches_total(self):
        gcov = self._gcov_branch('-t')
        branches, total = self.cal_branches(gcov)
        self._reset()
        return branches, total

    def get_last_coverages(self):
        gcov = self._gcov_branch()
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
            # print('line_coverage:', float(gcov_lines[line_index][line_offset:line_offset+5])/100)
            # print('branch_coverage:', float(gcov_lines[branch_index][branch_offset:branch_offset+5])/100)
        self._reset()
        return line_coverage, branch_coverage

    @_timeit
    def get_coverage_lines(self):
        gcov = self._gcov('-t')
        outputset, total = self.cal_lines(gcov)
        # print(gcov)
        # if len(gcov) == 0:
        #     return 0, set()
        
        # outputset = set()
        # counter = 0
        # lines = gcov.split('\n')
        # for line in lines:
        #     if line == '':
        #         break
        #     #    24:  244: ...
        #     parts = line.split(':', 2)
        #     if '#' in parts[0]:
        #         counter += 1
        #     elif not '-' in parts[0]:
        #         # line 244
        #         outputset.add(int(parts[1]))
        # self._total_lines = counter + len(outputset)
        coverage = round(100* len(outputset) / total, self.COV_DIGITS)
        # outputset.issubset(self.codelines)
        # cov = self._gcov()
        # print('outputset len, total:' , len(outputset), total)
        # print(cov)
        # print('linecov, cov:', coverage, self._cov(self._gcov()))
        self._reset()
        return coverage, outputset


class CMAES_Builder:
    DEFAULTS = {'init_popsize' : 10, 'max_popsize' : 1000, 'max_gens' : 1000, 'mode' : 'bytes', 'max_evaluations' : 10 ** 5}
    MIN_INPUT_SIZE = 2
    MODES = {'real': {'x0' : [(2**32)/2], 'sigma0' : 0.3*(2**32), 'bounds' : [0,2**32]},
     'bytes' : {'x0' : [128], 'sigma0' : 0.3*256, 'bounds' : [0, 256]}}

    def __init__(self, input_size = 1000, init_popsize = 10, max_popsize = 1000, max_gens = 1000, popsize_scale = 10, mode = 'bytes', max_evaluations = 10**5):
        # print('inputdim =', input_size)
        self._input_size = input_size
        self.mode = self.MODES['bytes']
        """ copy bounds before, if bounds in options is used """
        self.seed = 100
        # random.seed(self.seed)
        self._options = dict(popsize = init_popsize, verb_disp = 0, seed = self.seed, bounds = [self.mode['bounds'][0], self.mode['bounds'][1]])
        self._options['seed'] = random.randint(10, 1000) # 157
        self._args = dict(x0 = self.mode['x0'] * self._input_size, sigma0 = self.mode['sigma0'])
        # self.init_mean()
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
        self.max_evaluations = max_evaluations
        self.evaluations = 0
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
    def init_mean(self):
        minimum, upperbound = self.mode['bounds']
        x0 = [random.randint(minimum, upperbound - 1) for _ in range(self._input_size)]
        print(x0)
        self._args['x0'] = x0

    def init_cmaes(self, mean = None, sigma = None, sigmas = None, fixed_variables = None):
        if mean is None:
            self._args['x0'] = self.mode['x0'] * self._input_size
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
        

        self._options['seed'] += 1
        self._optimized = False
        # self._generations = 0
        self._es = cma.CMAEvolutionStrategy(**self._args, inopts=self._options)
        self.result = self._es.result
        return self
    
    def init_cmaes_no_bounds(self, mean = None, sigma = None):
        if mean is None:
            self._options['bounds'] = [None, None]
        else:
            self._options['bounds'] = [self.mode['bounds'][0],self.mode['bounds'][1]]
        self.init_cmaes(mean, sigma)

    @_timeit
    def stop(self):
        return self._es.stop() or self._es.result.iterations >= self._max_gens or self.result.fbest == -100.0 or self.evaluations > self.max_evaluations

    @_timeit
    def stop_lines(self):
        # return self._es.stop() or self._es.result.iterations >= self._max_gens or self.result.fbest == -100
        return self._es.stop() or self._es.result.iterations >= self._max_gens

    def reset_stop_conditions(self):
        self._es.stop().clear()

    @_timeit
    def ask(self, **kwargs):
        self.evaluations += self._options['popsize']
        return self._es.ask(**kwargs)
        # return self.resample_until_feasible()
        # return self.filterInfeasible(self._es.ask(), [])
        # return self.parseToFeasible(self._es.ask())

    # def parseToFeasible(self, solutions):
    #     minimum, upperbound = self.mode['bounds']
    #     for solution in solutions:
    #         if any(solution < minimum) or any(solution >= upperbound):
    #             print('\n\nnot feasible:\n')
    #             break
    #     return np.vectorize(lambda x: int(x)%256)(solutions)
    def resample_until_feasible(self):
        minimum, upperbound = self.mode['bounds']
        feasible_input_vectors = []
        popsize = self._options['popsize']
        while(len(feasible_input_vectors) < self._options['popsize']):
            unchecked_input_vectors = self._es.ask(popsize)
            print(unchecked_input_vectors)
            for input_vector in unchecked_input_vectors:
                if np.all(input_vector >= minimum) and np.all(input_vector < upperbound):
                    feasible_input_vectors.append(input_vector)
        
            print('NOT FEASIBLE WITH',len(feasible_input_vectors))
            popsize *= 2
        #  if len(feasible_input_vectors) < self._options['popsize']:
                # return self.resample_until_feasible(self._es.ask(), feasible_input_vectors)
        return feasible_input_vectors
        
    def filterInfeasible(self, unchecked_solutions, feasible_solutions):
        # print('begining of filter')
        # print(len(unchecked_solutions))
        minimum, upperbound = self.mode['bounds']
        for solution in unchecked_solutions:
            if not (any(solution < minimum) or any(solution >= upperbound)):
                feasible_solutions.append(solution)

        # print(unchecked_solutions)
        
        # print(len(feasible_solutions))
        if len(feasible_solutions) == 0:
            feasible_solutions = self.filterInfeasible(self._es.ask(), feasible_solutions)
        if len(feasible_solutions) < self._options['popsize']:
            feasible_solutions *= int(self._options['popsize'] / len(feasible_solutions)) + 1

        return feasible_solutions
        


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
        # self._options['popsize'] *= self._popsize_scale
        self._options['popsize'] += 100
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
        return dict(seed = self._options['seed'],popsize = self._options['popsize'], generations = self.result.iterations, evaluations = self.evaluations)
        #  generations = self._generations)
        # return dict(input_size = self._input_size, popsize = self._current_popsize, optimized = self._optimized)

class FuzzerLogger:
    def __init__(self):
        self._fuzzer = None
        self._log = dict(seed = None, testcase = 0, popsize = 0, generations = 0, coverage = 0, optimized = False, evaluations = 0 ,time = 0)
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
        initial_parameters = [fuzzer._cmaesbuilder._input_size, fuzzer._cmaesbuilder._max_popsize, fuzzer._cmaesbuilder._popsize_scale, fuzzer._cmaesbuilder._max_gens, fuzzer._cmaesbuilder.max_evaluations, fuzzer._timeout]
        with open(self._filename, 'w') as f:
            f.write('program_path: ' + fuzzer._program.path + '\n')
            f.write('initial parameters:\n')
            f.write('input_size       max_popsize       popsize_scale_factor       max_gens      max_eval      timeout\n')
            f.writelines('   %s           ' % key for key in initial_parameters)
            f.write('\n-----------------------------------------------------------------------------------------\n')
            f.write('logs:\n')

        # write input_size samplesize generations popsize coverage optimized evaluations time 
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

    def print_logs(self):
        with open(self._filename, 'r') as f:
            print(f.read())



class _SampleHolder:
    def __init__(self, sample = None, path = set(), coverage = 0, score = -1, stds = []):
        self.sample = sample
        self.path = path
        self.score = score
        self.coverage = 0
        self.stds = stds
        
    def update(self, sample, path, score):
        optimized = score > self.score
        if optimized:
            self.path = path
            self.sample = sample
            self.score = score
        if sample is None:
            print(sample)
            print(path)
            # print(penalty)
            exit('sample is never optimized')
        return optimized

    def update_coverage(self, sample, coverage):
        updated = coverage > self.coverage
        if updated:
            self.coverage = coverage
            self.sample = sample
        return updated

    def clear(self):
        self.sample = None
        self.path = set()
        self.coverage = 0
        self.score = 0


class SampleCollector:
    def __init__(self):
        self._total_path_length = 0
        self.total_samples = [] # [sample holder1, sample holder 2, ...
        self.total_paths = set()
        self.optimized_sample_holders = [] # [sample holder1, sample holder 2, ...]
        self.optimized_paths = set()
        self.interesting_samples = []
        self.interesting_paths = set()
        self.best_sample_holder = _SampleHolder()
        self.current_coverage = 0
        self.common_path = set()

    # def test_dup(self,path):
    #     for i, s1 in enumerate(self.total_samples):
    #         for j,s2 in enumerate(self.total_samples):
    #             if i != j and (s1.path.issubset(s2.path) or s2.path.issubset(s1.path)):
    #                 print([s.path for s in self.total_samples])
    #                 print(path)
    #                 raise KeyboardInterrupt

    @_timeit
    def get_executed_lines(self, sample, current_path, total, check_interesting = False):
        self._total_path_length = total
        output_path = self.optimized_paths | current_path
        self.best_sample_holder.update(sample, current_path, len(output_path))
        # if len(output_path) > 48:
        #     print(len(output_path))
        #     print(self.optimized)
        #     exit('something is wrong')
        # check if the given path covers another new path
        if check_interesting:
            self.check_interesting(sample, current_path)
        return output_path        

    @_timeit
    def check_interesting(self, sample, current_path): # TODO: refactor total -> interesting
        self.interesting_paths.update(current_path)
        pre_total_path = len(self.total_paths)
        # TODO: compare optimized paths with current path? => lesser total samples
        if pre_total_path < len(self.total_paths):
            reduced_samples = []
            for prev_sample in self.interesting_samples:
                if len(prev_sample.path) >= len(current_path) or not prev_sample.path.issubset(current_path):
                    if not prev_sample.path.issubset(self.optimized_paths):
                        reduced_samples.append(prev_sample)

                    # if interesting and current_path.issubset(prev_sample.path):
                    #     interesting = False
            # if interesting:
            reduced_samples.append(_SampleHolder(sample, current_path))
            self.interesting_samples = reduced_samples
                
            
        # if len(output_path) > len(self.optimized_paths):
            # # if not output_path.issubset(self.optimized_paths):
            #     interesting = True
            #     reduced_samples = []
            #     # eliminate all samples with path that are real subset of the given path / minimize interesting samples
            #     for prev_sample in self.total_samples:
            #         # if len(prev_sample.path) > len(output_path) or len(prev_sample.path) == len(output_path) and not output_path.issubset(prev_sample.path) or not prev_sample.path.issubset(output_path):
            #         # if output_path == prev_sample.path or not prev_sample.path.issubset(output_path):
            #         if len(prev_sample.path) >= len(output_path) or not prev_sample.path.issubset(output_path):
            #             reduced_samples.append(prev_sample)
            #         # if not prev_sample.path.issubset(output_path):
            #             # reduced_samples.append(prev_sample)
            #             if interesting and output_path.issubset(prev_sample.path):
            #                 interesting = False
            #     # if len(self.total_samples) > len(reduced_samples):
            #     #     print('!!!!!reduced', len(self.total_samples), len(reduced_samples))
            #     self.total_samples = reduced_samples

            #     # add the given sample only if the given sample was interesting
            #     if interesting:
            #         self.total_samples.append(_SampleHolder(sample, output_path))
            #         # print('path:')
            #         # for prev_sample in self.total_samples:
            #         #     print(prev_sample.path)
        # self.total_paths.update(current_path)

    def is_optimized(self):
        return len(self.optimized_paths) < len(self.best_sample_holder.path)

    def update_best(self, sample, stds, interrupted):
        print('updating the best sample', self.best_sample_holder.sample)
        if not interrupted and (np.any(sample != self.best_sample_holder.sample)):
            print(sample, self.best_sample_holder.sample)
            exit('given sample is not the same as best sample')

        sample = self.best_sample_holder.sample
        path = self.best_sample_holder.path
            
        # score
        pre_len = len(self.optimized_paths)
        self.optimized_paths.update(path)
        self.total_paths.update(path)
        optimized = pre_len < len(self.optimized_paths)

        if interrupted or optimized:
            self.optimized_sample_holders.append(_SampleHolder(sample, path, stds))
            self.total_samples.append(_SampleHolder(sample, path, stds))
        
        # self.best_sample_holder = _SampleHolder()
        self.best_sample_holder.clear()
        # self.optimized = False # no need?

        return optimized

    def remove_common_path(self):
        if len(self.optimized_paths) == 0:
            return
        common_path = set.intersection(*[sample.path for sample in self.optimized_sample_holders])
        self.optimized_paths -= common_path

    # def update_optimized(self, sample = None, score = 0, interrupted = False): # updated == True <=> optimized
    #     return self.update_best(sample, interrupted)
        # if self._total_path_length > 0:
        
        # coverage = -score
        
        # optimized = coverage > self.current_coverage

        # if optimized:
        #     s = _SampleHolder(sample, coverage = coverage)
        #     self.optimized_sample_holders.append(s)
        #     self.total_samples.append(s)
        #     self.current_coverage = coverage
            
        # return optimized

    # def update_coverage(self, sample, coverage):        
    #     optimized = coverage > self.current_coverage

    #     if optimized:
    #         s = _SampleHolder(sample, coverage = coverage)
    #         self.optimized_sample_holders.append(s)
    #         self.total_samples.append(s)
    #         self.current_coverage = coverage
            
    #     return optimized


    # def check_optimized(self, sample):
    #     pre_len = len(self.optimized_paths)


    def reset_optimized(self):
        self.optimized_sample_holders = []
        self.optimized_paths = set()
        self.current_coverage = 0
        self.best_sample_holder = _SampleHolder()

    def pop_first_optimum_holder(self):
        if len(self.optimized_sample_holders) == 0:
            return None
        
        sample_holder = self.optimized_sample_holders[0]
        self.optimized_sample_holders = self.optimized_sample_holders[1:]
        self.optimized_paths -= sample_holder.path
        return sample_holder



    def coverage(self):
        if self.current_coverage > 0:
            return self.current_coverage
            
        if self._total_path_length == 0:
            return 0
        # print('op lenth:', len(self.optimized_paths))
        return round(100 * len(self.optimized_paths) / self._total_path_length, 4)
        # return 0

    def total_coverage(self):
        if len(self.total_paths) == 0:
            return 0
        # print('interesting lenth:', len(self.total_paths))
        return round(100 * len(self.total_paths)/ self._total_path_length,4)

    def get_optimized_sample_holders(self):
        return [s.sample for s in self.optimized_sample_holders]

    def get_total_samples(self):
        # if len(self.total_samples) == 0:
        #     self.total_samples = self.optimized_sample_holders
        if len(self.interesting_samples) > 0:
            self.total_samples = self.interesting_samples
        return [s.sample for s in self.total_samples]

    def get_total_size(self):
        return len(self.total_samples)


class Fuzzer:
    VERIFIER_ERROS = {_Program.SAFE : 'SAFE', _Program.ERROR : 'ERROR', _Program.ASSUME : 'ASSUME_ERROR'}


    # def __init__(self, function, mean, sigma, options, program_path = 'test.c', sample_size = 1, max_sample_size = 10, resetable = True, max_popsize = 1000, input_size = 1000):
    def __init__(self, program_path, output_dir = _Program.DEFAULT_DIRS['output'], log_dir = _Program.DEFAULT_DIRS['log'], max_test = 10, init_popsize = 10 ,max_popsize = 10000, max_gens = 10000 ,resetable = True, timeout = 15 * 60, testsuitesize = 1, mode = 'bytes', objective = '', max_evaluations = 10**5, popsize_scale = 10, hot_restart = False):
        """Fuzzer with cmaes.

        Args:
        sampe_type = the type of a sample to optimize
        function = score function to optimize
        """
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
        self.hot_restart = hot_restart
        # self.optimize_testsuite = self._select_optimize_teststuie(hot_restart)

        self._program = _Program(program_path, output_dir = output_dir, log_dir = log_dir, timeout=timeout, mode = mode)
        self._cmaesbuilder = CMAES_Builder(init_popsize= init_popsize ,input_size = self._generate_input_size() * self._testsuitesize, max_popsize = max_popsize, max_gens= max_gens, popsize_scale = popsize_scale,mode = mode, max_evaluations=max_evaluations) # maybe parameter as dict
        self._samplecollector = SampleCollector()
        self._logger = FuzzerLogger().resister(self)

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
        if returncode in Fuzzer.VERIFIER_ERROS:
            state = Fuzzer.VERIFIER_ERROS[returncode]

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
        # def parseToFeasible(x):
        #     # return int(x)
        #     return int(min(max(x,0),255) * ((2**32 - 1) / (2 ** 8 - 1)))

        def parseToFeasible(x):
            result = int(min(max(x,0),256) * ((2**32) / (2 ** 8)))
            if result >= 2**32:
                result -=1
            return result

        # print()
        # sample = [0,0]


        
            # exit('not feasible')
            # return x
        # sample = np.array([127.99759276, 127.99378568])
        # for s in sample:
        #     print(parseToFeasible(s) >> 16)
        # exit(sample)


        # 32895
        # sample = [2**32 - 1, 2**32 - 1]
        # sample = [127.99759276, 128]
        # a * 100.0=> long_max
        # a * 50 => lomg_max/2
        # a * 0 => 0
        # a = log_max / 100
        # a, b = sample
        # # a = int(a)
        # # a = a / (1 << 16)
        # print('a: ',a)
        # print('b: ',b)
        # try :
        out = b''
        for input_comp in sample:
            out += parseToFeasible(input_comp).to_bytes(4, 'little', signed = False)
        # except:
        # print(sample)
        # exit(out)
        # out = bytes([128,128])
            # print(out)
        # print(out)
        # if out[0:4] == b'\xff'*4: 
        #     print(len(out))
        return out

    def _encode(self, sample: np.ndarray):
        def parseToFeasible(x):
            # print('parseToFeasible ?',x)
            if x < 0:
                # print('parseToFeasible < 0',x)
                return 0
            if x >=256:
                # print('parseToFeasible >= 256',x)
                return 255
            return int(x)
            
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
        # print('sampel',sample)
        out = bytes(np.frompyfunc(parseToFeasible,1,1)(sample).tolist())
        # out = bytes([127,128])
        # if out[0:1] == b'\x00'*1:
        #     print(len(out))
        #     exit(out)
        # print('bytes:',out)
        return out
            #


    def _run_sample(self, sample, returncode_check = False):
        if sample is None:
            return
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


    def checkSampleOverBounds(self, sample):
        penalty = 0
        count = 0
        for s in sample:
            if s < 0:
                penalty += abs(s) + 1
                # penalty = penalty
                count += 1
            elif s >= 256:
                penalty += s - 255
                # penalty *= penalty/
                count += 1
        # if count > 0:
        #     penalty = penalty /
        return penalty / (256)
    
    def penalize(self, input_vector):
        penalty = 0
        minimum, upperbound = self._cmaesbuilder.mode['bounds']
        # print(self._cmaesbuilder.mode['bounds'])
        # print(type(minimum))
        # print(minimum)
        # print(type(upperbound))
        # exit()
        for input_component in input_vector:
            if input_component < minimum:
                penalty += abs(input_component)
            if input_component >= upperbound:
                penalty += input_component - upperbound + 1

        return 1 - 1/(int(penalty) + 1)


    def _f(self, sample:np.ndarray):
        # penalty = self.checkSampleOverBounds(sample)
        # print('sample:', sample)
        # print('penalty',penalty)
        # if check:
        #     return check
        # self._run_all_samples()
        # self._samplecollector.get_optimized_sample_holders()
        # self._run_samples(self._samplecollector.get_optimized_sample_holders() + [sample])
        # # cov = self._program.get_coverage()
        # cov = self._program.get_coverage()
        # # cov, lines = self._program.get_coverage_lines()
        # # self.check_lines(lines, sample, cov)
        # return -cov

        # return self._f_line(sample, False)
        return self._f_branch(sample, False)

    def _f_line(self, sample, interesting = False):
        # samples = self._samplecollector.get_optimized_sample_holders()
        # self._run_samples(samples + [sample])
        # cov = self._program.get_coverage()
        self._run_sample(sample)
        lines, total = self._program.get_lines_total() # independently executed lines from sample, total number of lines
        # penalty = self.penalize(sample)
        lines = self._samplecollector.get_executed_lines(sample, lines, total, interesting)

        # cov2 = round(100 * len(lines)/total, _Program.COV_DIGITS) # coverage
        # if cov != cov2:
        #     print(samples + [sample]) 
        #     print(cov, cov2)
        #     exit('wrong cov')
        # return -cov
        # return -len(lines) + penalty
        return -100 * len(lines) / total

    def _f_branch(self, sample, interesting = False):
        # samples = self._samplecollector.get_optimized_sample_holders()
        # self._run_samples(samples + [sample])
        # cov = self._program.get_coverage()
        self._run_sample(sample)
        branches, total = self._program.get_branches_total() # independently executed lines from sample, total number of lines
        # penalty = 0
        branches = self._samplecollector.get_executed_lines(sample, branches, total, interesting)

        # cov2 = round(100 * len(lines)/total, _Program.COV_DIGITS) # coverage
        # if cov != cov2:
        #     print(samples + [sample]) 
        #     print(cov, cov2)
        #     exit('wrong cov')
        # return -cov
        return -100 * len(branches) / total

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
        return dict(testcase = (len(self._samplecollector.optimized_sample_holders), self._samplecollector.get_total_size()),
         coverage = (self._samplecollector.coverage(), self._samplecollector.total_coverage()))
        #  generations = self._generations)
        # return dict(testcase = len(self._samples), coverage = self._coverage)


    def _stop(self):
        if self._interrupted:
            self._stop_reason = self._interrupted
        elif self._samplecollector.total_coverage() == 100 or self._samplecollector.coverage() == 100:
            self._stop_reason = 'coverage is 100%'
        # elif self._cmaesbuilder._options['popsize'] > self._cmaesbuilder._max_popsize:
        #     self._stop_reason = 'max popsize is reached'
        elif self._cmaesbuilder.evaluations > self._cmaesbuilder.max_evaluations:
            self._stop_reason = 'evaluations is over max_evaluations' 

        return self._stop_reason
    def time(self):
        return time.time() - _init_time

    # def optimize_sample_lines(self):
    #     es = self._cmaesbuilder
    #     # while self._samplecollector.total_coverage() < 100 and not es.stop_lines():
    #     while not es.stop_lines() or self._samplecollector.total_coverage == 100:
    #         try:
    #             solutions = es.ask()
    #             values = [self.objective(x) for x in solutions]
    #             es.tell(solutions, values)
    #             es.update()


    #             # print('!!!!!!!!best:\n',es._es.result.xbest)
    #             # print("!!!!!!!!optimizingsample== xbest", self._samplecollector.optimizing_sample.sample == es._es.result.xbest)
    #             # print('!!!!!!!!means:\n',es._es.result.xfavorite)
    #             # print('!!!!!!!!stds:\n',es._es.result.stds)
    #             # if not any(self._samplecollector.optimizing_sample.sample == es._es.result.xbest):
    #             # if test == 4:
    #             #     print('sol',solutions)
    #             #     print('val', values)
    #                 # exit()
    #                 # raise KeyboardInterrupt
    #             # print('!!!!!!!!values:\n',values)

    #         except (subprocess.TimeoutExpired, KeyboardInterrupt) as e:
    #             self._interrupted = e.__class__.__name__
    #             self._program._reset()
    #             break

    #     return self._samplecollector.update_best()

    # def optimize_testsuite_lines(self):
    #     self._program._compile_program()

    #     # what is the first good sigma and mean?
    #     # what is good bounds?
    #     # sigma = 148
    #     while not self._stop():
    #         self._cmaesbuilder.init_cmaes()
    #         optimized = self.optimize_sample_lines()
    #         self._logger.report_changes(optimized)
    #         if not optimized:
    #             if self._cmaesbuilder._increase_popsize():
    #                 self._reset()

    # @_timeit
    # def optimize_sample(self):
    #     es = self._cmaesbuilder._es

    # def optimize_samle_nelder_mean(self):
    #     coverage = 0
    #     inputsize, returncode = self._program.cal_input_size()
    #     self._check_runtime_error(returncode)
    #     self._program._compile_program()
    #     seed = 100
    #     options = {'maxiter':1000, 'maxfev':1000 } # ?
    #     mininum, maximum = self._cmaesbuilder.mode['bounds']

    #     while coverage != 100:
    #         # random.seed(seed)
    #         x0 = [random.randint(mininum, maximum) for _ in range(inputsize)]
    #         # res = minimize(self.objective, x0, method='Nelder-Mead', options=options)
    #         res = minimize(self.objective, x0, method='L-BFGS-B', options=options)
    #         # res = minimize(self.objective, x0, method='BFGS', options=options)
    #         # res = minim
    #         x, f, d = res.x, res.fun, res
    #         # x, f,d = minimize_l_bfgs_b(self.objective, x0 * inputsize)
    #         # res = minimize(self.objective, np.array([128] * inputsize), method='powell', options=options)
    #         self._samplecollector.update_best(x, f)
    #         print(x)
    #         print(d)
    #         print(self._samplecollector.coverage())
    #         # seed += 1
        
    #     exit('finished')
        

    @_timeit
    def optimize_sample(self, number = 0, score = 0, mean = None, sigma = None, sigmas = None, fixed_variables = None):
        mute = False
        es = self._cmaesbuilder.init_cmaes(mean, sigma, sigmas, fixed_variables)
        while not es.stop():
            try:
                solutions = es.ask()
                values = [self.objective(x) for x in solutions]
                sample_score = -es.result.fbest

                while len(solutions) <= number and score >= sample_score:
                    extra_solutions = es.ask()
                    solutions += extra_solutions
                    values += [self.objective(x) for x in extra_solutions]
                    if self._cmaesbuilder.evaluations > self._cmaesbuilder.max_evaluations:
                        break
                es.tell(solutions, values)
                es.update()
                if mute:
                    continue
                # print('!!!!!!!!solutions:\n', len(solutions),':', solutions)
                print('!!!!!!!!evals:\n', es.result.evaluations)
                print('!!!!!!!!values:\n', values)
                print('!!!!!!!!means:\n',es.result.xfavorite)
                print('!!!!!!!!bestx:\n',es.result.xbest)
                print('!!!!!!!!bestf:\n',es.result.fbest)
                print('!!!!!!!!stds:\n',es.result.stds)
                # print('!!!!!!!!solutions:\ns', values)

                # print('!!!!!!!!stds:\n',es._es.result.stds)
                # if len(solutions) >= es._max_popsize:
                    # break

            except (subprocess.TimeoutExpired,  KeyboardInterrupt) as e:
                self._interrupted = e.__class__.__name__
                self._program._reset()
                break
        print('!!!!!!!!means:\n',es.result.xfavorite)
        print('!!!!!!!!bestx:\n',es.result.xbest)
        print('!!!!!!!!stds:\n',es.result.stds)
        print()
        # return self._samplecollector.update_optimized(es.result.xbest, es.result.fbest, self._interrupted)
        return self._samplecollector.update_best(es.result.xbest, es.result.stds, self._interrupted)

    # def test_common(self):
    #     s1 =  np.array(
    #         [102.75671481,  26.63636618,  94.47393305,  33.56735494,
    #         29.28751084,  26.80879269,  34.8711325 ,  67.13575287,
    #        102.96109645,  34.5255833 ,  54.36791905,  75.98214846,
    #         28.88815514,  21.03463968,  97.04708168,  19.85266509,
    #        120.70947056,  79.47848498, 110.79477557,   9.1637533 ,
    #        133.9869082 , 128.10528068,   2.26041838,  64.93894455,
    #        226.97495209, 198.92743639,  49.01880846, 255.70323267,
    #      132.07062739, 220.56084855,  29.92003258, 254.88536194,
    #        126.88849826, 128.4624836 , 162.47884841, 138.30642983,
    #       170.31666436, 116.13308629,  19.61037188, 169.05482272,
    #       248.0148994 , 161.94005658, 194.41374147, 189.2994771 ,
    #       28.17991729, 139.31869597, 182.92244041,  45.05804462,
    #      221.65410569,  90.08819226, 167.21709601, 133.83862774,
    #       25.48075068,  39.79170264, 163.54862276])
    #     s2 = np.array(
    #     [ 64.6809964 ,  48.71498551,  47.09119842,  45.26976222,
    #      14.98816626, 65.80294799,  30.02042244, 139.25355972, 
    #      96.05001331,  93.63831569, 16.40221445,  13.52853336, 
    #      109.05214518,  17.593332  ,   8.99886378, 123.48681681,
    #       63.16384872, 160.92284867,  20.5618866 , 100.22302972,
    #       251.18257562, 128.4192702 ,  44.30358106, 221.71068409,
    #      148.84156664, 164.53285966, 170.22957364, 162.76826926,
    #       164.13174457, 131.86923248, 65.60047602,  92.9603752 ,
    #        15.80015087,   1.71521537, 215.16112035, 78.73563629,
    #         93.23983325,  82.8811455 ,  13.92916725,  88.95266951,
    #       236.01335762, 203.66693295,  12.77767757,  16.35838728, 
    #       104.99083298, 235.99119454,  62.16267656, 168.248867,
    #        199.34691557, 118.03089781,8.4161071 , 114.69835301,
    #         216.90169468, 254.28780186,  92.97366035])
    #     s3 = np.array(
    #     [ 18.65322533,  80.40061165,   2.04429758,  82.14610063,
    #    106.66741914,  30.28164531,  82.24226197, 144.96135124,
    #     25.88355373,   3.66452424, 191.18958659,  12.65470353,
    #     62.46424648,  91.66130013,  34.56421264,  62.2101505 ,
    #     39.03121456, 170.88016823,  34.27276814,  11.33784554,
    #    236.0865206 , 128.4123899 ,  93.17838845, 104.4959291 ,
    #    141.89752355, 255.5046319 , 111.11403182, 255.44980522,
    #    243.96506298, 178.63691203, 129.00476459,   8.62993016,
    #     80.48237179,  30.21425315, 217.45772681, 237.66591283,
    #    208.35091954,  56.51213531,  30.24043833, 206.5382521 ,
    #    255.73940749, 163.92330634,  76.77436575,   3.96608757,
    #     71.25322338, 133.68815137, 253.10435   , 255.00120457,
    #    240.08815969,  42.53529033,  51.22576896, 226.45178585,
    #    148.20826458, 154.66012873,  83.5145776 ])
    #     ss = [s1, s2, s3]
    #     ss.append(np.array([ 86.85258768,  53.53366438, 117.23287095,  53.60493577,
    #     41.70536335,  33.41108747,  67.30943053, 189.05857728,
    #     70.68290642,  74.56180471,  10.03271839, 183.12944736,
    #     40.06427786,   3.77842413,  87.67639573, 180.64817536,
    #     24.45859222,   7.81719945,  64.2765345 , 110.46391949,
    #    138.68695592, 133.86887144,  32.69583676, 204.77014984,
    #    222.66017808, 186.39158128,  70.76827872, 206.13125041,
    #    255.50342375, 153.74418896,  11.67641972, 176.66445305,
    #    102.16659556,  53.35165504, 145.66769474, 129.66494188,
    #    252.86538838,  29.87518832,  83.64538054, 206.8805602 ,
    #    255.94802055, 176.69286777, 135.71211361, 138.0847658 ,
    #     19.99783718, 125.66928883, 253.98402695, 141.63184362,
    #    147.15752377,  76.88141082,  43.83974931, 217.21980636,
    #    250.74783184, 152.43113559, 139.51643838]))
    #     s = sum(ss)/len(ss)
    #     print(s)
    #     # exit()
    #     return s

    # def cal_mean_best(self, bests, best):
    #     bests.append(best)
    #     return sum(bests)/len(bests)

    
    # def optimize_testsuite_with_fixed(self):
    #     self._cmaesbuilder._options['seed'] = 122
    #     while not self._stop():
    #         list_of_fixed_variables = []
    #         while not self._stop():
    #             optimized = self.optimize_sample()
    #             self._logger.report_changes(optimized)
    #             if not optimized:
    #                 break
    #             optimum = self._cmaesbuilder.result.xbest
    #             stds = self._cmaesbuilder.result.stds
    #             stds_mean = sum(stds)/len(stds)
    #             fixed_variables = {}
    #             for i, std in enumerate(stds):
    #                 if std < stds_mean:
    #                     fixed_variables[i] = optimum[i]
                
    #             list_of_fixed_variables.append(fixed_variables)
    #             # score = -self._cmaesbuilder.result.fbest
    #             # break

    #         print(list_of_fixed_variables)

    #         for fixed_variables in list_of_fixed_variables:
    #             self._samplecollector.pop_first_optimum_holder()
    #             optimized = self.optimize_sample(number =1000, score = len(self._samplecollector.optimized_paths), fixed_variables=fixed_variables)
    #             self._logger.report_changes('fixed '+str(optimized))
    #             # if not optimized:
    #             #     break
            
    #         return
    #         # if pre_score >= len(self._samplecollector.optimized_paths):
    #         #     print('reseting')
    #         #     if not self._cmaesbuilder._increase_popsize():
    #         #         self._cmaesbuilder._reset_popsize()
    #         #     self._reset()


    @_timeit
    def optimize_testsuite_with_hot_restart(self):
        number_of_hot_restarts = len(self._samplecollector.optimized_sample_holders)
        optimized = False
        while number_of_hot_restarts > 0 and not self._stop():
            if not optimized:
                sample_holder = self._samplecollector.pop_first_optimum_holder()
                initial_mean = sample_holder.sample
                initial_sigmas = sample_holder.stds
                stds_median = np.median(initial_sigmas)

            # reset components with larger standard deviations to default
            for i, std in enumerate(sample_holder.stds):
                if std >= stds_median:
                    initial_mean[i] = self._cmaesbuilder.mode['x0'][0]
                    initial_sigmas[i] = self._cmaesbuilder.mode['sigma0']

            self._cmaesbuilder.init_cmaes(initial_mean, 1, initial_sigmas)
            optimized = self.optimize_sample()
            self._logger.report_changes(str('hot')+str(optimized))

            # count the number down only if not optimized, otherwise try to optimize with the previous initial mean and sigmas
            if not optimized:
                number_of_hot_restarts -= 1

    # def optimize_testsuite_with_hot_restart_old(self):
    #     # self.optimize_testsuite_with_fixed()
    #     # return
    #     # self._cmaesbuilder._options['seed'] = 656
    #     self._cmaesbuilder._options['seed'] = 72
    #     while not self._stop():
    #         optimums = []
    #         stds = []
    #         initial_mean = None
    #         initial_sigma = None
    #         count = 0
    #         while not self._stop():
    #             self._cmaesbuilder.init_cmaes()
    #             # self._cmaesbuilder.init_cmaes(initial_mean, initial_sigma)
    #             # optimized = self.optimize_sample(score = len(self._samplecollector.optimized_paths), number = self._cmaesbuilder._options['popsize'] * 10)
    #             length = len(self._samplecollector.optimized_sample_holders)
    #             number = 1000
    #             if length > 0:
    #                 number = max(number,self._cmaesbuilder.evaluations/length)
    #             optimized = self.optimize_sample(score = len(self._samplecollector.optimized_paths), number = number)
    #             # optimized = self.optimize_sample()
    #             self._logger.report_changes(optimized)

    #             if not optimized:
    #                 # print(self._cmaesbuilder.evaluations)
    #                 # if self._cmaesbuilder.evaluations > len(self._samplecollector.optimized_sample_holders) * 10**3:
    #                 break
                        
    #             else:
    #                 xbest = self._cmaesbuilder.result.xbest
    #                 xstds = self._cmaesbuilder.result.stds
    #                 stds_mean = sum(xstds)/len(xstds)
    #                 for i,std in enumerate(self._cmaesbuilder.result.stds):
    #                     if std > stds_mean:
    #                         xbest[i] = 128
    #                         xstds[i] = 76.8
                            
    #                 optimums.append(xbest)
    #                 stds.append(xstds)

    #         pre_score = len(self._samplecollector.optimized_paths)
    #         score = pre_score
    #         # self._samplecollector.remove_common_path()
    #         # initial_mean = [128,0]
    #         # print(optimums)

    #         hot_optimums = []
    #         hot_stds = []

    #         while len(optimums) > 0 and not self._stop():
    #             # break
    #             # self._samplecollector.remove_common_path()

    #             if not optimized:
    #                 sample =  self._samplecollector.pop_first_optimum_holder()
    #                 # initial_mean = optimums[count]
    #                 if sample.sample is not optimums[count]:
    #                     exit('wrong first optimum')
    #                 initial_mean = sample.sample
    #                 initial_sigma = 1
    #                 initial_sigmas = stds[count]

    #             # if not optimized:
    #             # print(initial_mean)
    #             self._cmaesbuilder.init_cmaes(initial_mean, initial_sigma, initial_sigmas)
    #             optimized = self.optimize_sample()
    #             # optimized = self.optimize_sample(score = score, number = 1000)
    #             self._logger.report_changes(str('hot')+str(optimized))
    #             if not optimized:
    #                 # self._samplecollector.optimized_sample_holders.append(sample)
    #                 # self._samplecollector.optimized_paths.update(sample.path)
    #                 count += 1
    #             else:
    #                 hot_optimums.append(self._cmaesbuilder.result.xbest)
    #                 hot_stds.append(self._cmaesbuilder.result.stds)

    #             if count >= len(optimums):
    #                 if score >= len(self._samplecollector.optimized_paths):
    #                     break
    #                 score = len(self._samplecollector.total_paths)
    #                 optimums = hot_optimums
    #                 stds = hot_stds
    #                 hot_optimums = []
    #                 hot_stds = []
    #                 count = 0

    #         if pre_score >= len(self._samplecollector.optimized_paths):
    #             print('reseting')
    #             if not self._cmaesbuilder._increase_popsize():
    #                 self._cmaesbuilder._reset_popsize()
    #             self._reset()

            # if not optimized:
            #     first = self._samplecollector.pop_first_optimum_holder()
            #     optimums.append(first)

        # initial_mean = self._samplecollector.pop_firtestingst_optimum()
        # initial_sigma = 256/5 # ?

        # # self._cmaesbuilder._increase_popsize()
        # self._samplecollector.remove_common_path()
        # self._cmaesbuilder._options['popsize'] *=1

        # while not self._stop():
        #     self._cmaesbuilder.init_cmaes(initial_mean, initial_sigma)
        #     # self._cmaesbuilder.init_cmaes()
        #     optimized = self.optimize_sample()
        #     self._logger.report_changes(optimized)
        #     count += 1
        #     if count > 100: 
        #         break
        #     self._samplecollector.remove_common_path()
        #     if not optimized: 
        #         initial_mean = self._samplecollector.pop_first_optimum_holder()
        #         initial_mean = None
        #         initial_sigma = None
        #         # initial_sigma = 256/5
        #     else:
        #         initial_mean = None
        #         initial_sigma = None


    # def optimize_testsuite_test(self):
    #     optimums = []
    #     initial_mean = None
    #     initial_sigma = None
    #     number = None
    #     count = 0
    #     # best = init
    #     while not self._stop():
    #         self._cmaesbuilder.init_cmaes(initial_mean, initial_sigma)
    #         optimized = self.optimize_sample()
    #         self._logger.report_changes(optimized)
    #         count += 1
    #         # number = 1000
    #         initial_sigma = 256/2

    #         if count > 10:
    #             print("optimums:\n",optimums)
    #             print("mean:\n",initial_mean)
    #             print("last optimum:\n",self._samplecollector.pop_first_optimum_holder())
    #             count = 0
    #             self._cmaesbuilder._increase_popsize()
    #             # continue
    #             break

    #         best = self._samplecollector.pop_first_optimum_holder()
    #         optimums.append(best)
    #         # initial_mean = sum(optimums)/len(optimums)
    #         initial_mean = best

    #         self._reset()


    @_timeit
    def optimize_testsuite(self):
        while not self._stop():
            optimized = self.optimize_sample()
            self._logger.report_changes(optimized)
            if not optimized:
                if self.hot_restart:
                    self.optimize_testsuite_with_hot_restart()
                if not self._cmaesbuilder._increase_popsize():
                    self._cmaesbuilder._reset_popsize()
                self._reset()
            

    # @_timeit
    # def optimize_testsuite_default(self):
    #     # self.optimize_testsuite_test()
    #     # return
    #     # break # testing
    #     while not self._stop():
    #         self._cmaesbuilder.init_cmaes()
    #         optimized = self.optimize_sample()
    #         self._logger.report_changes(optimized)
    #         if not optimized:
    #             if not self._cmaesbuilder._increase_popsize():
    #                 self._cmaesbuilder._reset_popsize()
    #             self._reset()


    # def optimize_testsuite2(self):
    #     self._program._compile_program()
    #     count = 0
    #     self._dists = []
    #     mean = None

    #     # self._cmaesbuilder.init_cmaes()
    #     # self.optimize_sample()
    #     # self._samples = [sample for (i, (_, sample)) in self._executed_line_sample_set.items() if i != self._best_sample_index[0]]
        
        
    #     while not self._stop() :
    #         self._cmaesbuilder.init_cmaes()
    #         optimized = self.optimize_sample()
    #         if optimized:
    #             self._current_sample = self._cmaesbuilder.get_xbest()
    #             # if len(self._samples) == 0:
    #             self._samples.append(self._current_sample)
    #             self._total_samples.append(self._current_sample)
    #             mean = self._cmaesbuilder._es.result.xfavorite
    #             self._dists.append(mean)

    #             self._logger.report_changes()
    #             # self._samples = [sample for (lines, sample) in self._executed_line_sample_set.values() if not 359 in lines]
    #             # self._samples = [sample for (i, (_, sample)) in self._executed_line_sample_set.items() if i != self._best_sample_index[0]]
    #             # self._run_samples(self._samples)
    #             # self._cmaesbuilder._fbest = -self._program.get_coverage()
    #             # self._coverage = -self._cmaesbuilder._fbest
    #             # self._program.get_coverage_lines()
    #             # self._samples = [sample for (i, (_, sample)) in self._executed_line_sample_set.items()]

    #             # print('current_sample:', self._current_sample)
    #             # self._sample_map[str(sample)] = self.get_coverage()
    #             count = 0
    #             # maybe better if we increase if it was not optimized
    #             # self._cmaesbuilder._reset_popsize()
    #         else:
    #             self._logger.report_changes()
    #             mean = None
    #             # coverage is the same or lower than -fbest
    #             count += 1


    #         # # idea 2 : delete first sample if it failed
    #         # if count > 0 and len(self._samples) > 0:
    #         #     # maybe rset fbest as wel
    #         #     self._cmaesbuilder._reset_fbest()
    #         #     del self._samples[0]
    #             # del self._sample_map.keys()[-1]

    #         # if optimized:

                

    #         if count > 0:
    #             # self._total_samples += self._samples
    #             # self._coverage = 0
    #             self._cmaesbuilder._reset_fbest()
    #             self._samples_map = {self._cmaesbuilder._potential_popsize : self._samples}
    #             if self._cmaesbuilder._increase_popsize():
    #                 # self._cmaesbuilder._reset_fbest()
    #                 self._reset_samples()
    #                 # pass
    #         # else:

                
    #         # if not self._cmaesbuilder._is_over_threshold():
    #         # else:
    #             # break


    #     return self._samples

    # def optimize_testsuite3(self):
    #     self._program._compile_program()
    #     es = self._cmaesbuilder
    #     while not self._stop() :
    #         es.init_cmaes()
    #         # optimized = self.optimize_sample() try:
    #         try:
    #             while not es.stop():
    #                 solutions = es.ask()
    #                 # print(len(solutions))
    #                 values = [self._f_testsuite(x) for x in solutions]
    #                 # print(len(values))
    #                 es.tell(solutions, values)
    #                 # self._update()
    #                 self._generations = self._cmaesbuilder.get_generations()
    #                 self._coverage = -es.get_fbest()
    #                 # print('values:',values )
    #                 # print('f')

    #                 print('!!!!!!!!best:\n',es._es.result.xbest)
    #                 print('!!!!!!!!means:\n',es._es.result.xfavorite)
    #                 print('!!!!!!!!stds:\n',es._es.result.stds)
    #                 # print('iterations:', self._cmaesbuilder.get_generations())
    #                 # print('fbest:',self._cmaesbuilder._es.result.fbest)
    #                 # print('evaluations:', es.result.evals_best)
    #                 # es.tell(solutions, values)
    #                 # print('\n')
    #         except (subprocess.TimeoutExpired,  KeyboardInterrupt) as e:
    #             self._interrupted = e.__class__.__name__
    #             self._program._reset()

    #         es.reset_stop_conditions()
    #         es._update_fbest()
    #         self._coverage = -es.get_fbest()
            
    #         # if es.is_optimized():
    #         #     self._current_sample = self._cmaesbuilder.get_xbest()
    #         #     # if len(self._samples) == 0:
    #         #     self._samples.append(self._current_sample)
    #         #     self._total_samples.append(self._current_sample)
    #         #     mean = self._cmaesbuilder._es.result.xfavorite
    #         #     self._dists.append(mean)
    #         self._samples = np.split(es.get_xbest(), self._testsuitesize)
    #         # self._total_samples = self._executed_line_sample_set

    # def optimize_testsuite_real(self):
        # self._program._compile_pro

    # def _check_testsuite_minimization(self):
    #     safe = True
    #     for i, (lines1, _) in enumerate(self._executed_line_sample_set):
    #         for j, (lines2, _) in enumerate(self._executed_line_sample_set):
    #             if i != j:
    #                 safe = safe and not lines1.issubset(lines2) and not lines2.issubset(lines1)
        
    #     print('minimization is ', safe)


    # def generate_testsuite_lines(self):
    #     self.optimize_testsuite_lines()
    #     self._program._timeout = None
    #     # self._check_testsuite_minimization()
    #     # self._total_samples = [s.sample for s in self._samplecollector.total_samples]
    #     self._total_samples = self._samplecollector.get_total_samples()
    #     # return self._total_samples, self._samples_map, self._dists
    #     return self._total_samples


    def generate_testsuite(self):
        self._program._compile_program()
        self.optimize_testsuite()
        self._program._timeout = None
        self._total_samples = self._samplecollector.get_total_samples()
        return self._total_samples

    # def generate_testsuite2(self):
    #     self.optimize_testsuite3()
    #     self._total_samples = [sample for (_, sample) in self._executed_line_sample_set]
    #     self._program_timeout = None
    #     return self._total_samples
        

    

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
        # if os.path.isfile('__VERIFIER.gcda'):
        #     os.remove('__VERIFIER.gcda')
        if os.path.isfile(self._program.pname+'.gcda'):
            os.remove(self._program.pname+'.gcda')
        self._run_samples(self._total_samples, returncode_check=True)
        line, branch = self._program.get_last_coverages()
        # gcov = self._program._gcov_branch()
        # print(gcov)
        self._total_coverage = branch
        self._logger.report_final()
        self._logger.report_time_log()
        self._logger.print_logs()

        print('total sample len:', len(self._total_samples))
        print('line_coverage:', round(line/100, 4))
        print('branch_coverage:', round(branch/100,4))
        print('total_eval:', self._cmaesbuilder.evaluations)


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


# def main_sv3():
#     max_popsize = 1000
#     # sample_size = 10
#     timeout = 15 * 60
#     path = 'programs/pals_STARTPALS_ActiveStandby.ufo.BOUNDED-10.pals.c' # ~ 83%
#     # path = 'programs/pals_floodmax.5.1.ufo.BOUNDED-10.pals.c' # ~ 95%
#     # path = 'programs/xor5.c' # ~100%

#     argsize = len(sys.argv)
#     if argsize > 1:
#         path = sys.argv[1]
#         if argsize > 2:
#             max_popsize = int(sys.argv[2])
#             if argsize > 3:
#                 timeout = int(sys.argv[3])


#     programs = [path]
    
    
#     # programs = [path+'test.c']


#     for program in programs:
#         # fuzzer = Fuzzer(None, [0], None, None, program_path = program, max_sample_size = sample_size, input_size = input_size)
#         fuzzer = Fuzzer(program_path = program, max_popsize = max_popsize, timeout=timeout)
#         t = fuzzer.generate_testsuite()
#         print('testsuite:', t)
#         fuzzer.last_report()
#         # fuzzer.gcov()
#         # fuzzer.report()

# def main_test():
#     # subprocess.run('gcc programs/test.c programs/__VERIFIER_input_size.c -o test')

#     #            1,2,3,4,5,6,7,8,9,0,11,12,13,14,15,16,17,18,19
#     testsuite = [np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
#         -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
#          81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
#          95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
#         141.18546982,  139.96057095,   64.2624934 ]), np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
#         -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
#          81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
#          95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
#         141.18546982,  139.96057095,   64.2624934 ]), np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
#         -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
#          81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
#          95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
#         141.18546982,  139.96057095,   64.2624934 ]), np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
#         -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
#          81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
#          95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
#         141.18546982,  139.96057095,   64.2624934 ]), np.array([ 2.31112248e+02,  7.88875081e+01, -1.00272686e+02, -1.64848604e+01,
#        -1.07149203e-01, -2.23708581e+01,  1.88348397e+02,  5.34317338e-01,
#         1.21936163e+02,  2.09374535e+02,  1.06511931e+02,  1.25130298e+02,
#        -1.87479582e+02, -3.67562446e+01, -7.15702679e+01, -1.72612956e+02,
#         6.98037430e+01,  1.98282392e+02, -1.16470826e+02]), np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
#         -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
#          81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
#          95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
#         141.18546982,  139.96057095,   64.2624934 ]), np.array([  54.1248795 ,  -71.6752295 ,  -22.99511084, -103.02129871,
#           0.8684931 , -113.55180125,  -76.8893616 ,   70.16172702,
#          55.10751883,  -97.30580052,  -28.63692408,   29.664043  ,
#          25.12036177, -104.14226332,   16.64123275,  -38.90410403,
#          76.71797608,  -12.81207861,   22.99548734]), np.array([  54.1248795 ,  -71.6752295 ,  -22.99511084, -103.02129871,
#           0.8684931 , -113.55180125,  -76.8893616 ,   70.16172702,
#          55.10751883,  -97.30580052,  -28.63692408,   29.664043  ,
#          25.12036177, -104.14226332,   16.64123275,  -38.90410403,
#          76.71797608,  -12.81207861,   22.99548734]), np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
#         -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
#          81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
#          95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
#         141.18546982,  139.96057095,   64.2624934 ]), np.array([  54.1248795 ,  -71.6752295 ,  -22.99511084, -103.02129871,
#           0.8684931 , -113.55180125,  -76.8893616 ,   70.16172702,
#          55.10751883,  -97.30580052,  -28.63692408,   29.664043  ,
#          25.12036177, -104.14226332,   16.64123275,  -38.90410403,
#          76.71797608,  -12.81207861,   22.99548734]), np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
#         -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
#          81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
#          95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
#         141.18546982,  139.96057095,   64.2624934 ]), np.array([  54.1248795 ,  -71.6752295 ,  -22.99511084, -103.02129871,
#           0.8684931 , -113.55180125,  -76.8893616 ,   70.16172702,
#          55.10751883,  -97.30580052,  -28.63692408,   29.664043  ,
#          25.12036177, -104.14226332,   16.64123275,  -38.90410403,
#          76.71797608,  -12.81207861,   22.99548734]), np.array([ -69.48035861,   63.83027656,   18.11071919,  -96.40362277,
#         -37.03080592,  105.69332907, -155.30992382,  -27.45091392,
#          81.02162626,  -55.47269956,  -43.44985711,   -6.06154948,
#          95.4519503 ,  -40.89112667,  -28.4158923 ,  -27.79957897,
#         141.18546982,  139.96057095,   64.2624934 ])]

#     for sample in testsuite:
#         subprocess.run('output/pals_STARTPALS_ActiveStandby.ufo.BOUNDED-10.pals', input = bytes(np.vectorize(lambda x: int(x)%256)(sample).tolist()))
#     # inp = bytes([0,0,0,0,0,0,0,0,0,0,254,254,254,0,0,0,0,0,113])
#     # subprocess.run('output/pals_STARTPALS_ActiveStandby.ufo.BOUNDED-10.pals', input = inp)

    # time_with_none = 0
    # time_with_19dim = 0
    # for _ in range(10):
    #     init_time = time.time()
    #     runtime += time.time() - init_time
    # print("tme")

    # init_time = time.time()

# def main_test_41cov():
#     array = np.array
#     testsuite= [array([ 8.17945369e-02,  2.41488359e+00,  6.06120964e+00, -6.20581374e-01,
#        -4.15943706e+00,  8.35842643e+01,  2.02186960e+01,  8.51386345e+01,
#         9.25300293e+01,  1.51778777e+00,  4.21060736e+01,  3.61547523e+01,
#        -5.48409122e+01,  9.68681958e+01,  6.87742054e+01, -6.95826083e+00,
#        -5.51625389e+01, -5.35698789e+01, -1.80691723e+01,  6.94381889e+01,
#         1.79509374e+01, -1.70465721e+01, -2.73731169e+00, -1.19490791e+02,
#        -6.79847997e-01, -4.98916785e+01,  7.53100010e+01])]



#     for sample in testsuite:
#         subprocess.run('output/pals_floodmax.3.ufo.BOUNDED-6.pals', input = bytes(np.vectorize(lambda x: int(x)%256)(sample).tolist()))


def parse_argv_to_fuzzer_kwargs():
    arg_parser = argparse.ArgumentParser(add_help=False)
    # arg_parser.add_argument('-h', '--help', type = str, default=argparse.SUPPRESS, he)
    arg_parser.add_argument('-od', '--output_dir', type = str, default =_Program.DEFAULT_DIRS['output'])
    arg_parser.add_argument('-ld', '--log_dir', type = str, default =_Program.DEFAULT_DIRS['log'])
    arg_parser.add_argument('-ip', '--init_popsize', type = int, default = CMAES_Builder.DEFAULTS['init_popsize'])
    arg_parser.add_argument('-mp', '--max_popsize', type = int, default = CMAES_Builder.DEFAULTS['max_popsize'])
    arg_parser.add_argument('-m', '--mode', type = str, default = CMAES_Builder.DEFAULTS['mode'])
    arg_parser.add_argument('-me', '--max_evaluations', type = int, default = CMAES_Builder.DEFAULTS['max_evaluations'])
    arg_parser.add_argument('-o', '--objective', type = str)
    arg_parser.add_argument('-hr', '--hot_restart', action='store_true')
    arg_parser.add_argument('-t', '--timeout', type = int, default = 60*15 - 30)
    arg_parser.add_argument('program_path', type = str)
    args= arg_parser.parse_known_args()
    print(args[0])
    return vars(args[0])

# def parse_argv_to_fuzzer_kwargs2():
#     argvsize = len(sys.argv)
#     example = 'e.g.: python3 fuzzer.py [-od <output_dir>] [-ld <log_dir>] [-ip <init_popsize>] [-mp <max_popsize>] [-mg <max_gens>] [-ps <popsize_scale>] [-t <timeout>] [-m <mode>] [-o <objective>] [-me <max_evaluations>] [-hr <hot_restart>] <program_path>'
#     if argvsize == 1:
#         exit('ERROR: No program_path is given!\n' + example)

#     commands = {'-od' : 'output_dir', '-ld' : 'log_dir', '-ip' : 'init_popsize', '-mp' : 'max_popsize', '-mg' : 'max_gens', '-ps' : 'popsize_scale','-t' : 'timeout', '-ts' : 'testsuitesize', '-m' : 'mode', '-o' : 'objective', '-me' : 'max_evaluations', '-hr' : 'hot_restart'}
#     command_type_dict = {'output_dir' : str, 'log_dir' : str, 'max_popsize' : int, 'popsize_scale' : int ,'timeout' : float, 'init_popsize' : int, 'max_gens' : int, 'testsuitesize' : int, 'mode' : str, 'objective' : str, 'max_evaluations' : int, 'hot_restart': int}
#     current_command = None
#     fuzzer_kwargs = {}
#     if argvsize > 1:
#         fuzzer_kwargs['program_path'] = sys.argv[argvsize - 1]
#         for arg in sys.argv[1:argvsize - 1]:
#             if arg == '-help':
#                 exit(example)
#             if current_command is not None:
#                 fuzzer_kwargs[current_command] = command_type_dict[current_command](arg)
#                 current_command = None
#             elif arg in commands:
#                 current_command = commands[arg]
#                 if current_command in fuzzer_kwargs:
#                     exit('ERROR: <' + current_command + '> is already given!')
#             # else:
#             #     exit('ERROR: Unkown command "' + arg + '"!\n' + example)

#     return fuzzer_kwargs

def main():
    kwargs = parse_argv_to_fuzzer_kwargs()
    fuzzer = Fuzzer(**kwargs)
    t = fuzzer.generate_testsuite()
    print('testsuite:\n', t)
    fuzzer.last_report()
    # for program in programs:
        # fuzzer = Fuzzer(None, [0], None, None, program_path = program, max_sample_size = sample_size, input_size = input_size)

# def main_testsuit():
#     fuzzer = Fuzzer(**parse_argv_to_fuzzer_kwargs())
#     t = fuzzer.generate_testsuite2()
#     print('testsuite:\n', t)
#     print('coverage:\n', fuzzer.get_coverage())
#     fuzzer.last_report()
#     fuzzer._total_samples = fuzzer._samples
#     fuzzer.last_report()


# def test_input_with_type():
#     program_path = 'e.c'
#     # mean = []
#     fuzzer = Fuzzer(program_path=program_path)
#     fuzzer._cmaesbuilder._es

# def test_real():
#     # subprocess.run("gcc test.c verifiers/__VERIFIER_real.c -o test")
#     # inp = bytes([1,0,0,0,1,0,2,0]*2)
#     v = 12345
#     print(v)
#     inp = int(v).to_bytes(8, 'little', signed=False)
#     # inp = int(1).to_bytes(2,'little', signed=False) + b'\x00' * 6
#     # inp = inp
#     print(inp)
#     # subprocess.run('gcc ./parse_to_int.c -o ./parse_to_int')
#     subprocess.run("./parse_to_int", input = inp)
    # inp = bytes([255,2,255,4,255,255,7,8]*2)
    # inp = int(777*2^(64-16)).to_bytes(8, 'big', signed=False)
    # inp = b'\x00' * 6 + int(1).to_bytes(2,'big', signed=False)
    # inp = inp * 2
    # print(inp)
    # subprocess.run("./test", input = inp)
    # subprocess.run("./test", input = inp)

# def main_real():
#     fuzzer = Fuzzer(**parse_argv_to_fuzzer_kwargs(), real = True)
#     t, tm, dists = fuzzer.generate_testsuite()
#     print('testsuite:\n', t)
#     print('testsuite map:\n', tm)
#     print('distributions:\n', dists)
#     fuzzer.last_report()

# def test2():
#     # inp = bytes([0,0])
#     # subprocess.run('./test2', input = inp)
#     # inp = bytes([127,128])
#     # subprocess.run('./test2', input = inp)
#     # inp = bytes([128,128])
#     # subprocess.run('./test2', input = inp)
#     # inp = bytes([255,255])
#     # subprocess.run('./test2', input = inp)
#     # inp = bytes(np.vectorize(lambda x: int(x)%256)(np.array(128,])).tolist())
#     inp = bytes([128,128])
#     subprocess.run('./test2', input = inp)

# def test_remove_first_optimum():
#     solutions = [np.array([101.33312033,  11.29165343,  11.52148982,  16.87401891,
#         42.53967197,  48.44160791,  93.7532028 ,  40.61020174,
#        111.2249071 ,  18.61674362,  16.13200654,  51.02304202,
#         60.49335106, 252.33382306,   5.7729167 ,  49.52820804,
#        113.50488143, 118.98392453, 151.53710859,  51.09176335,
#        186.31487407, 128.74908729, 250.83292827,  62.90310826,
#        157.561685  ,  58.15126176,  25.41158411, 254.57758489,
#        147.35688646, 172.65215605,  33.39885449, 224.50543796,
#          9.92183905,  14.52607897, 230.06725474,  84.51320683,
#        197.87437075, 227.30271545, 164.28393415, 222.27231247,
#        169.7446412 , 131.91500415, 207.5530416 , 115.69282328,
#         25.78978152, 105.17185581,  84.90476823, 255.29454268,
#        223.45796836, 211.30025663,  76.98417421,  52.69355979,
#         20.72239426, 148.87453535,  25.56883154]), np.array([ 10.24809068,  86.23372563, 214.62735704, 229.03427964,
#         79.13982086, 101.5318461 , 158.92598805, 133.46136759,
#         64.99866589,  79.70726999, 255.81373004,   1.30034085,
#          3.93326653,  33.59328689,  54.08589478, 254.0695396 ,
#        209.28814752,   3.91981824, 127.1088321 , 185.86122345,
#        127.08549598, 185.21203362,  21.22267772, 133.42592324,
#        255.24363325, 130.50392065, 187.31649619, 252.83636213,
#         66.85107392,  43.4710487 ,  74.27741788, 202.73664569,
#        195.52234147, 237.74983213,  13.78071452,  89.66426606,
#        250.76589271, 201.06914583, 179.90185398, 250.76791649,
#        129.21061473,  98.33566143,  31.40272408, 209.7520885 ,
#        134.22067591,  87.09646861, 119.62994921, 254.66908914,
#         65.3938889 , 205.34116212, 160.43111343, 169.68639933,
#        238.03123917, 159.20693961,  70.91725352]), np.array([200.74400151, 153.29868639,  21.6691219 ,  60.76423563,
#        178.84019594, 187.90489485,  70.76759337,   3.12113381,
#        192.04486771, 149.10540855,  11.02845425,  25.49936306,
#        197.73330669,  87.30691738,  97.47218685,  93.69908304,
#          4.41484108, 255.09008166, 113.91048734, 114.74218668,
#        255.98776692, 198.38795566, 244.66804517, 243.43157168,
#        186.03749619, 238.13573689,  68.39499054,  13.43177487,
#        197.65377693,  90.09690123, 159.17822696, 127.64930659,
#        121.36470726, 234.21659924, 154.11493135,  16.43998449,
#        252.60193836,  37.74557728, 254.03159103, 235.71307542,
#        216.90566212,  78.15685104,  50.21582545, 114.03442915,
#        129.90408232,  70.63296149, 222.08284248, 128.61012268,
#         92.03783927, 102.25624114, 234.19962863, 123.90283368,
#        246.12862086, 255.53848867, 230.73635543]), np.array([ 85.48221287,  56.87885083, 233.22940468,  37.47150427,
#         13.73121337, 212.09770593, 105.20771271, 195.41978043,
#         62.43842341,  78.91276279,  67.16639915, 252.77772911,
#         76.48500808,  79.05737698, 172.72470942,  10.30122488,
#         29.4593597 , 219.57125883, 247.03290161, 255.78239626,
#        167.60371977, 212.74792622,  23.32689147, 147.94545544,
#        134.74876499,  82.0033332 , 240.2117739 ,  82.66304932,
#        109.95274514, 103.15055297,  30.37562491,  64.47963493,
#         81.79902451, 170.26070382, 239.85441802,  29.61678751,
#         80.95766295,  35.75152018, 253.17345822, 109.8065477 ,
#        104.34336803, 166.74468629,  98.65284583,  82.50593625,
#         78.79719396,  94.36250276, 154.5578537 ,  48.77159632,
#        141.77959182, 177.4754604 ,  79.9264924 , 157.9228598 ,
#        138.13176997, 175.86514188,  26.98563198]), np.array([ 77.95262141,  91.18412608, 109.56271599, 224.44070194,
#        203.99582537,  59.86509909,  36.94554799,  60.95339593,
#        251.27254188,  13.38961255,  39.64077154, 104.21991017,
#         20.90786439,   8.8603106 , 253.75689118,  39.07647288,
#        159.66676366, 194.89930803,  48.78683047,  21.27103024,
#        113.30421872, 139.67492664, 175.78644323, 209.75336579,
#         13.29985611,  42.52094788, 188.44978418,  26.44702542,
#        100.10548995, 146.45179094,  75.37115378, 234.45570295,
#        199.75312776,  72.30423208,   2.51954345, 185.08124489,
#        243.2086509 ,  11.9973729 , 254.82059332,  87.20223481,
#        147.59271533,  62.55362623, 196.7465814 , 249.3488582 ,
#        151.85056681, 195.81948327,  49.96891938, 199.91924503,
#        242.55266693, 255.59033969,   9.50965073, 239.73663578,
#        254.1011131 ,  26.68665062, 183.5366961 ]), np.array([1.48041650e+01, 1.98478534e+02, 6.78591131e+01, 5.56507087e+01,
#        2.37713959e+02, 2.55188809e+02, 2.26866456e+02, 1.68169294e+01,
#        4.12060086e+01, 2.49047060e+02, 1.93852753e+02, 4.20496596e+01,
#        1.80480730e+02, 3.60192670e-01, 7.67810770e+01, 1.18202456e+01,
#        2.22340926e+02, 1.71010541e+02, 6.92628506e+01, 3.78269561e+01,
#        1.49428036e+02, 6.90531644e+01, 2.55237950e+02, 1.25569764e-01,
#        1.98846988e+02, 5.83231361e+01, 2.51417707e+02, 5.86384565e+01,
#        3.49599987e+01, 2.07782987e+02, 2.30859828e+02, 5.23754016e+01,
#        2.52258147e+02, 4.89799294e+01, 1.90536156e+02, 1.03100925e+02,
#        2.55138631e+02, 4.78904991e+01, 1.85527851e+02, 4.90068456e+01,
#        1.37700340e+00, 2.34431865e+02, 2.55505953e+02, 1.17137074e+02,
#        1.33844014e+02, 1.67877836e+02, 1.86927703e+02, 1.59800913e+02,
#        5.51508087e+01, 1.02474950e+02, 8.25070599e+01, 2.33714466e+02,
#        1.03940156e+02, 2.02325638e+02, 2.25959205e+01])]

#     removed_solutions = solutions[1:]
#     return removed_solutions

    

# # def main_nelder_mead():
#     fuzzer = Fuzzer(**parse_argv_to_fuzzer_kwargs())
#     fuzzer.optimize_samle_nelder_mean()
    
# def test_byte_add():
#     s =  [np.array([2.39493474e-03, 2.06662978e+01])]


# def test_argparse():
#     # parser = argparse.ArgumentParser()
#     parser = argparse.ArgumentParser(add_help=False)

#     # parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0', help="Show program's version number and exit.")
#     # parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and exit.')
#     # parser = argparse.ArgumentParser(add_help=False)
#     # zz = {'--c', '-b', 'program)path'}
#     # # parser.add_argument_group(zz)
#     parser.add_argument('-c', '--cype', type = int, help='heling --c')
#     parser.add_argument('-b', help= 'helping -b',action='store_true')
#     parser.add_argument('program_path', help= 'heping program')

#     args = parser.parse_known_args()
#     print('args:\n',args[0])


if __name__ == "__main__":
    # print('branch_coverage:', 1)
    # print('line_coverage:', 1)
    # main_sv3()
    # test_remove_first_optimum()
    main()
    # test_argparse()
    # main_nelder_mead()
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
    
