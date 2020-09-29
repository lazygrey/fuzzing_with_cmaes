import cma
import numpy as np
import subprocess
import os
import time
import sys
import random
import argparse

_init_time = time.time()
_time_log = {}
_test = False

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
    def __init__(self, live = False):
        self._fuzzer = None
        self._log = dict(seed = None, testcase = 0, popsize = 0, generations = 0, coverage = 0, optimized = False, evaluations = 0 ,time = 0)
        self._log_message_lines = []
        self._live = live
        
        """
        program_path: examples/test2.c
        initial parameters:
        input_size       max_popsize       popsize_scale_factor       max_gens      max_eval      timeout
        2              1000              10              10000              100000              870           

        -----------------------------------------------------------------------------------------
        logs:
        seed   testcase   popsize   generations   coverage   optimized   evaluations   time   
        559     (1, 1)     100     2     (20.0, 20.0)     True     300     1.83     

        -----------------------------------------------------------------------------------
        final report:
        total_testcase        total_coverage        stop_reason        testcase_statuses
            1               20.0               KeyboardInterrupt               ['SAFE']         

        -----------------------------------------------------------------------------------
        execution time for each method:
        stop     ask     _run     _gcov_branch     cal_branches     _reset     get_branches_total     get_executed_paths     tell     optimize_sample     optimize_testsuite     
        0.0002   0.0206   0.8224   0.7462   0.0133   0.0141   0.7737   0.0019   0.0105   1.6536   1.6538   
        """

    def resister(self, fuzzer):
        self._fuzzer = fuzzer
        self._log_path = fuzzer._program.log_dir
        self._filename = self._log_path + fuzzer._program.pname +'.txt'
        initial_parameters = [fuzzer._cmaesbuilder._input_size, fuzzer._cmaesbuilder._max_popsize, fuzzer._cmaesbuilder._popsize_scale, fuzzer._cmaesbuilder._max_gens, fuzzer._cmaesbuilder.max_evaluations, fuzzer._timeout]
    
        self._log_message_lines.append('program_path: ' + fuzzer._program.path)
        self._log_message_lines.append('initial parameters:')
        self._log_message_lines.append('input_size       max_popsize       popsize_scale_factor       max_gens      max_eval      timeout')
        self._log_message_lines.append(''.join(['   %s           ' % param for param in initial_parameters]))
        self._log_message_lines.append('\n-----------------------------------------------------------------------------------------')
        self._log_message_lines.append('logs:')
        self._log_message_lines.append(''.join(['%s   ' % key for key in self._log]))

        if self._live:
            with open(self._filename, 'w') as f:
                f.write('program_path: ' + fuzzer._program.path + '\n')
                f.write('initial parameters:\n')
                f.write('input_size       max_popsize       popsize_scale_factor       max_gens      max_eval      timeout\n')
                f.writelines('   %s           ' % key for key in initial_parameters)
                f.write('\n-----------------------------------------------------------------------------------------\n')
                f.write('logs:\n')
                f.writelines("%s   " % key for key in self._log)
                f.write('\n')

        return self

    def report_changes(self, optimized):
        if not self._fuzzer:
            return

        self._log['optimized'] = optimized
        self._log.update(self._fuzzer._cmaesbuilder._current_state())
        self._log.update(self._fuzzer._current_state())
        self._log['time'] = round(self._fuzzer.time(), 2)

        self._log_message_lines.append(''.join(['%s     ' % str(value) for value in self._log.values()]))

        if self._live:
            with open(self._filename, 'a') as f:
                f.writelines("%s     " % str(item) for item in self._log.values())
                f.write('\n')
    
    def report_time_log(self):
        self._log_message_lines.append('\n-----------------------------------------------------------------------------------')
        self._log_message_lines.append('execution time for each method:')
        self._log_message_lines.append(''.join(['%s     ' % str(key) for key in _time_log]))
        self._log_message_lines.append(''.join(['%s   ' % str(round(value,4)) for value in _time_log.values()]))

        if self._live:
            with open(self._filename, 'a') as f:
                f.write('\n-----------------------------------------------------------------------------------\n')
                f.write('execution time for each method:\n')
                f.writelines('%s     ' % str(key) for key in _time_log)
                f.write('\n')
                f.writelines('%s   ' % str(round(item,4)) for item in _time_log.values())

    def report_final(self):
        final_report = [len(self._fuzzer._total_samples), self._fuzzer._samplecollector.total_coverage(), self._fuzzer._stop_reason, self._fuzzer._statuses]

        self._log_message_lines.append('\n-----------------------------------------------------------------------------------')
        self._log_message_lines.append('final report:')
        self._log_message_lines.append('total_testcase        total_coverage        stop_reason        testcase_statuses')
        self._log_message_lines.append(''.join(['      %s         ' % str(total) for total in final_report]))

        if self._live:
            with open(self._filename, 'a') as f:
                f.write('\n-----------------------------------------------------------------------------------\n')
                f.write('final report:\n')
                f.write('total_testcase        total_coverage        stop_reason        testcase_statuses\n')
                f.writelines('      %s         ' % str(total) for total in final_report)

    def write_logs(self):
        with open(self._filename, 'w') as f:
            f.writelines('%s\n' % message for message in self._log_message_lines)

    def print_logs(self):
        print(*self._log_message_lines, sep='\n')

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
        self.pname = path[:-2].rsplit('/', 1)[-1]
        # self.pname = [] # *.c   
        # for c in reversed(path[:-2]):
        #     if c == '/':
        #         break
        #     self.pname.insert(0, c)
        # self.pname = ''.join(self.pname)
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
    def _gcov(self, *args):
        # gcov test
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
    def get_lines_total(self):
        gcov = self._gcov('-t')
        lines, total = self.cal_lines(gcov)
        self._reset()
        return lines, total

    @_timeit
    def get_branches_total(self):
        gcov = self._gcov('-b', '-c', '-t')
        branches, total = self.cal_branches(gcov)
        self._reset()
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
        self._reset()
        return line_coverage, branch_coverage


class CMAES_Builder:
    DEFAULTS = {'seed' : 100, 'mode' : 'bytes', 'init_popsize' : 10, 'max_popsize' : 1000, 'max_gens' : 1000, 'popsize_scale' : 2, 'max_evaluations' : 10 ** 5}
    MIN_INPUT_SIZE = 2
    MODES = {'real': {'x0' : [(2**32)/2], 'sigma0' : 0.3*(2**32), 'bounds' : [0,2**32]},
     'bytes' : {'x0' : [128], 'sigma0' : 0.3*256, 'bounds' : [0, 256]}}

    def __init__(self, input_size = MIN_INPUT_SIZE, seed = DEFAULTS['seed'], mode = 'bytes', init_popsize = DEFAULTS['init_popsize'], max_popsize = DEFAULTS['max_popsize'],
    max_gens = DEFAULTS['max_gens'], popsize_scale = DEFAULTS['popsize_scale'], max_evaluations = DEFAULTS['max_evaluations']):
        # print('inputdim =', input_size)
        self._input_size = input_size
        self.mode = self.MODES['bytes']
        self.seed = seed - 1
        # random.seed(self.seed)
        self._options = dict(popsize = init_popsize, verb_disp = 0, seed = self.seed, bounds = [self.mode['bounds'][0], self.mode['bounds'][1]])
        self._options['seed'] = random.randint(10, 1000) # 157
        self._args = dict(x0 = self.mode['x0'] * self._input_size, sigma0 = self.mode['sigma0'])
        # self.init_mean()
        self._max_popsize = max_popsize
        self._max_gens = max_gens
        self._es = None #
        # self._fbest = 0
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
    
    # def init_cmaes_no_bounds(self, mean = None, sigma = None):
    #     if mean is None:
    #         self._options['bounds'] = [None, None]
    #     else:
    #         self._options['bounds'] = [self.mode['bounds'][0],self.mode['bounds'][1]]
    #     self.init_cmaes(mean, sigma)

    @_timeit
    def stop(self):
        return self._es.stop() or self._es.result.iterations >= self._max_gens or self.result.fbest == -100.0 or self.evaluations > self.max_evaluations

    # @_timeit
    # def stop_lines(self):
    #     # return self._es.stop() or self._es.result.iterations >= self._max_gens or self.result.fbest == -100
    #     return self._es.stop() or self._es.result.iterations >= self._max_gens

    # def reset_stop_conditions(self):
    #     self._es.stop().clear()

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

    # def resample_until_feasible(self):
    #     minimum, upperbound = self.mode['bounds']
    #     feasible_input_vectors = []
    #     popsize = self._options['popsize']
    #     while(len(feasible_input_vectors) < self._options['popsize']):
    #         unchecked_input_vectors = self._es.ask(popsize)
    #         print(unchecked_input_vectors)
    #         for input_vector in unchecked_input_vectors:
    #             if np.all(input_vector >= minimum) and np.all(input_vector < upperbound):
    #                 feasible_input_vectors.append(input_vector)
        
    #         print('NOT FEASIBLE WITH',len(feasible_input_vectors))
    #         popsize *= 2
    #     #  if len(feasible_input_vectors) < self._options['popsize']:
    #             # return self.resample_until_feasible(self._es.ask(), feasible_input_vectors)
    #     return feasible_input_vectors
        
    # def filterInfeasible(self, unchecked_solutions, feasible_solutions):
    #     # print('begining of filter')
    #     # print(len(unchecked_solutions))
    #     minimum, upperbound = self.mode['bounds']
    #     for solution in unchecked_solutions:
    #         if not (any(solution < minimum) or any(solution >= upperbound)):
    #             feasible_solutions.append(solution)

    #     # print(unchecked_solutions)
        
    #     # print(len(feasible_solutions))
    #     if len(feasible_solutions) == 0:
    #         feasible_solutions = self.filterInfeasible(self._es.ask(), feasible_solutions)
    #     if len(feasible_solutions) < self._options['popsize']:
    #         feasible_solutions *= int(self._options['popsize'] / len(feasible_solutions)) + 1

    #     return feasible_solutions
        


    @_timeit
    def tell(self, solutions, values):
        return self._es.tell(solutions, values)

    def update(self):
        self.result = self._es.result

    # def is_optimized(self):
    #     return self._optimized

    # def _update_fbest(self):
    #     fbest = self._es.result.fbest
    #     self._optimized = self._fbest > fbest
    #     if self._optimized:
    #         self._fbest = fbest

    # def _reset_fbest(self):
    #     self._fbest = 0

    def _reset_popsize(self):
        # self._potential_popsize = self._options['popsize']
        self._options['popsize'] = self._init_popsize


    def _increase_popsize(self):
        # self._potential_popsize *= self._popsize_scale
        # self._options['popsize'] *= self._popsize_scale
        self._options['popsize'] += 100
        # print('increase popsize to:', self._potential_popsize)
        return self._options['popsize'] <= self._max_popsize
    
    # def _is_over_threshold(self):
    #     # _stop = 
    #     # # _stop = self._potential_popsize > self._max_popsize or self._es.result.iterations >= self._max_gens
    #     # print('_is_over_threshold:', _stop)
    #     return self._options['popsize'] > self._max_popsize

    # def get_sample(self):
    #     return self._es.result.xbest

    # def get_coverage(self):
    #     if self.result is None:
    #         return 0
    #     return -self.result.fbest

    # def get_generations(self):
    #     return self._es.result.iterations

    # def get_xbest(self):
    #     return self._es.result.xbest

    # def get_fbest(self):
    #     return self._fbest

    # def _reset(self):
    #     self._optimized = False
    #     # self._potential_popsize = self._options['popsize']
    #     # self._potential_popsize = self._init_popsize
    #     # maybe reset dim as well

    # def _reset

    def _current_state(self):
        # if 
        # return dict(generations = self._es.result.iterations, popsize = self._current_popsize, optimized = self.is_optimized())
        # return dict(popsize = self._current_popsize, optimized = self.is_optimized())
        # return dict(popsize = self._options['popsize'], optimized = self.is_optimized())
        return dict(seed = self._options['seed'],popsize = self._options['popsize'], generations = self.result.iterations, evaluations = self.evaluations)
        #  generations = self._generations)
        # return dict(input_size = self._input_size, popsize = self._current_popsize, optimized = self._optimized)




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
    def get_executed_paths(self, sample, current_path, total, check_interesting = False):
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

    def get_optimized_samples(self):
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
    def __init__(self, program_path, output_dir = _Program.DEFAULT_DIRS['output'], log_dir = _Program.DEFAULT_DIRS['log'], seed = 100, 
    mode = 'bytes', objective = 'branch', live_logs = True, init_popsize = 100, max_popsize = 10000, max_gens = 10000,
    timeout = 15 * 60, max_evaluations = 10**5, popsize_scale = 10, hot_restart = False):
        """Fuzzer with cmaes.

        Args:
        sampe_type = the type of a sample to optimize
        function = score function to optimize
        """
        # self._total_samples = []
        self._timeout = timeout
        # self._coverage = 0
        # self._total_coverage = 0
        # self._generations = 0
        # self._current_sample = None
        # self._total_executed_lines = set()
        # self._executed_line_sample_set = [] # {1: (block1, sample1),}
        self._interrupted = ''
        self._stop_reason = ''
        self._statuses = []
        # self._samples_map = {}
        # self._best_sample_index = (-1,0)
        # self._testsuitesize = self._calculate_testsuitesize()

        self.objective = self._select_obejctive(objective)
        self.encode = self._select_encode(mode)
        self.hot_restart = hot_restart
        # self.optimize_testsuite = self._select_optimize_teststuie(hot_restart)

        self._program = _Program(program_path, output_dir = output_dir, log_dir = log_dir, timeout=timeout, mode = mode)
        self._cmaesbuilder = CMAES_Builder(seed = seed, init_popsize= init_popsize ,input_size = self._generate_input_size(), max_popsize = max_popsize, max_gens= max_gens, popsize_scale = popsize_scale,mode = mode, max_evaluations=max_evaluations) # maybe parameter as dict
        self._samplecollector = SampleCollector()
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

    def get_current_coverage(self):
        return self._samplecollector.coverage()

    def get_total_coverage(self):
        return self._samplecollector.total_coverage()

    def _reset(self):
        self._samplecollector.reset_optimized()

    # def _reset_samples(self):
    #     # self._logger.resister(self)
    #     self._generations = 0
    #     self._samples = []
    #     # self._sample_map = {}
    #     # self._cmaesbuilder._reset()

    def _encode_real(self, sample):
        def parseToFeasible(x):
            result = int(min(max(x,0),256) * ((2**32) / (2 ** 8)))
            if result >= 2**32:
                result -=1
            return result

        out = b''
        for input_comp in sample:
            out += parseToFeasible(input_comp).to_bytes(4, 'little', signed = False)
        return out

    def _encode(self, sample: np.ndarray):
        def parseToFeasible(x):
            if x < 0:
                return 0
            if x >=256:
                return 255
            return int(x)
            
        out = bytes(np.frompyfunc(parseToFeasible,1,1)(sample).tolist())
        return out

    def _run_sample(self, sample, returncode_check = False):
        if sample is None:
            return
        returncode = self._program._run(self.encode(sample))
        if returncode_check:
            self._check_verifier_error(returncode)

    def _run_samples(self, samples, returncode_check = False):
        for sample in samples:
            self._run_sample(sample, returncode_check)

    def check_lines(self, lines, sample, cov):
        if not lines.issubset(self._total_executed_lines):
            self._total_executed_lines.update(lines)
            self._executed_line_sample_set.append((lines, sample))
    
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

    def _f_line(self, sample, interesting = False):
        self._run_sample(sample)
        lines, total = self._program.get_lines_total() # independently executed lines from sample, total number of lines
        # penalty = self.penalize(sample)
        lines = self._samplecollector.get_executed_paths(sample, lines, total, interesting)

        return -round(100 * len(lines)/total, _Program.COV_DIGITS)

    def _f_branch(self, sample, interesting = False):
        self._run_sample(sample)
        branches, total = self._program.get_branches_total() # independently executed lines from sample, total number of lines
        # penalty = 0
        branches = self._samplecollector.get_executed_paths(sample, branches, total, interesting)

        return -round(100 * len(branches)/total, _Program.COV_DIGITS)
    
    def _current_state(self):
        # if self._samplecollector:
        # return dict(testcase = len(self._samples), coverage = self._coverage, generations = self._generations)
        return dict(testcase = (len(self._samplecollector.optimized_sample_holders), self._samplecollector.get_total_size()),
         coverage = (self.get_current_coverage(), self.get_total_coverage()))
        #  generations = self._generations)
        # return dict(testcase = len(self._samples), coverage = self._coverage)


    def _stop(self):
        if self._interrupted:
            self._stop_reason = self._interrupted
        elif self.get_total_coverage() == 100 or self.get_current_coverage() == 100:
            self._stop_reason = 'coverage is 100%'
        # elif self._cmaesbuilder._options['popsize'] > self._cmaesbuilder._max_popsize:
        #     self._stop_reason = 'max popsize is reached'
        elif self._cmaesbuilder.evaluations > self._cmaesbuilder.max_evaluations:
            self._stop_reason = 'evaluations is over max_evaluations' 
        return self._stop_reason

    def time(self):
        return time.time() - _init_time

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
        return self._samplecollector.update_best(es.result.xbest, es.result.stds, self._interrupted)


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

    def generate_testsuite(self):
        self._program._compile_program()
        self.optimize_testsuite()
        self._program._timeout = None
        return self._samplecollector.get_total_samples()

    def last_report(self):
        if os.path.isfile(self._program.pname+'.gcda'):
            os.remove(self._program.pname+'.gcda')

        total_samples = self._samplecollector.get_total_samples()
        self._run_samples(_total_samples, returncode_check=True)
        line, branch = self._program.get_line_and_branch_coverages()

        self._total_coverage = branch
        self._logger.report_final()
        self._logger.report_time_log()
        self._logger.print_logs()

        print('testsuite:', _total_samples)
        print('total sample len:', len(_total_samples))
        print('line_coverage:', round(line/100, 4))
        print('branch_coverage:', round(branch/100,4))
        print('total_eval:', self._cmaesbuilder.evaluations)


def parse_argv_to_fuzzer_kwargs():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-od', '--output_dir', type = str, default =_Program.DEFAULT_DIRS['output'], 
        help = 'directory for complied and executable programs')
    arg_parser.add_argument('-ld', '--log_dir', type = str, default =_Program.DEFAULT_DIRS['log'],
        help = 'directory for logs')
    arg_parser.add_argument('-ip', '--init_popsize', type = int, default = CMAES_Builder.DEFAULTS['init_popsize'],
        help = 'initial population size for CMA-ES to start with')
    arg_parser.add_argument('-mp', '--max_popsize', type = int, default = CMAES_Builder.DEFAULTS['max_popsize'],
        help = 'maximum population size for CMA-ES')
    arg_parser.add_argument('-m', '--mode', type = str, default = CMAES_Builder.DEFAULTS['mode'],
        help = 'type of input vectors that CMA-ES-Fuzzer work with')
    arg_parser.add_argument('-me', '--max_evaluations', type = int, default = CMAES_Builder.DEFAULTS['max_evaluations'],
        help = 'maximum evaluations for CMA-ES-Fuzzer')
    arg_parser.add_argument('-s', '--seed', type = int, default = CMAES_Builder.DEFAULTS['seed'],
        help = 'seed to control the randomness')
    arg_parser.add_argument('-o', '--objective', type = str,
        help = 'type of objective function for CMA-ES-Fuzzer')
    arg_parser.add_argument('-t', '--timeout', type = int, default = 60*15 - 30,
        help = 'timeout in seconds')
    arg_parser.add_argument('-hr', '--hot_restart', action ='store_true',
        help = 'activate hot restart while optimizing input vectors')
    arg_parser.add_argument('-ll', '--live_logs', action = 'store_true',
        help = 'write logs in log files whenever it changes')
    arg_parser.add_argument('program_path', type = str,
        help = 'reletive program path to test')
    args = arg_parser.parse_known_args()
    return vars(args[0])

def main():
    kwargs = parse_argv_to_fuzzer_kwargs()
    fuzzer = Fuzzer(**kwargs)
    t = fuzzer.generate_testsuite()
    fuzzer.last_report()


if __name__ == "__main__":
    main()
