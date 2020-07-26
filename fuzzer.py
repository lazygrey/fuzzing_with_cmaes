import cma
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import subprocess
import os
import time
import functools
import sys
import random
# import threading 

# random.seed(0)

_init_time = time.time()
_time_log = {}

def _timeit(f):
        @functools.wraps(f)
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

    DEFAULT_PATHS = {'log' : 'logs/', 'output' : 'output/'}

    def __init__(self, path, verifier_path = 'verifiers/__VERIFIER.c', verifier_input_path = 'verifiers/__VERIFIER_input_size.c', output_path = DEFAULT_PATHS['output'], log_path = DEFAULT_PATHS['log'], timeout = None):
        self.path = path
        self.verifier_path = verifier_path
        self.verifier_input_path = verifier_input_path
        self.output_path = output_path
        self.log_path = log_path

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
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.isdir(self.log_path):
            os.mkdir(self.log_path)


    def timeout(self):
        if self._timeout:
            return self._timeout - time.time() + _init_time
        return None

    def _compile_program(self):
        return subprocess.run(['gcc',self.path , self.verifier_path, '-o', self.output_path + self.pname, '--coverage']).returncode

    def _compile_input_size(self):
        return subprocess.run(['gcc', self.path, self.verifier_input_path, '-o', self.output_path + self.pname + '_input_size']).returncode

    def get_input_size(self):
        # initialize inputsize.txt
        output = '0'
        with open('inputsize.txt', 'w') as f:
            f.write(output)
        returncode = subprocess.run(self.output_path + self.pname + '_input_size').returncode
        with open('inputsize.txt', 'r') as f:
            output = f.read()

        # output = process.stdout.decode()
        # print('inputsize output',output)
        # TODO: think what it means for the output to be ''
        # if output == '':
        #     output = 0

        return int(output), returncode        

    @_timeit
    def _reset(self):
        if os.path.isfile(self.pname + '.gcda'):
            os.remove(self.pname+'.gcda')
        else:
            print('WARNING: No gcda file at %s!' % self.timeout())
            # maybe program reached error?

    @_timeit
    def _run(self, sample_bytes):
        # outputs/test <- sample_bytes
        print(self.timeout())
        return subprocess.run(self.output_path + self.pname, input = sample_bytes, timeout=self.timeout()).returncode

    @_timeit
    def _gcov(self):
        # gcov test
        return subprocess.run(['gcov', self.pname + '.gcda'], capture_output = True, timeout=self.timeout()).stdout.decode()

    def _cov(self, output):
        if len(output) == 0:
            return 0.0
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


class CMAES_outputer:
    MIN_INPUT_SIZE = 2

    def __init__(self, mean = [128], sigma = 64, input_size = 1000, init_popsize = 7, max_popsize = 1000, max_gens = 1000):
        # print('inputdim =', input_size)
        self._input_size = input_size
        self._options = dict(bounds = [0, 255.99], popsize = init_popsize, verb_disp = 0, seed = 123)
        self._args = dict(x0 = input_size * mean, sigma0 = sigma, inopts = self._options)
        self._max_popsize = max_popsize
        self._max_gens = max_gens
        self._es = None #
        self._fbest = 0
        self._prev_fbest = 0
        self._optimized = False
        self._potential_popsize = init_popsize
        self._current_popsize = init_popsize
        self._popsize_scale = 2

    def cmaes(self):
        self._reset()
        self._es = cma.CMAEvolutionStrategy(**self._args)
        return self

    def init_cmaes(self):
        # self._reset()
        self._es = cma.CMAEvolutionStrategy(**self._args)
        # return self

    @_timeit
    def stop(self):
        return self._es.stop() or self._es.result.iterations >= self._max_gens or self._es.result.fbest == -100

    def reset_stop_conditions(self):
        self._es.stop().clear()

    @_timeit
    def ask(self):
        self._current_popsize = self._potential_popsize
        return self._es.ask(self._current_popsize)

    @_timeit
    def tell(self, solutions, values):
        return self._es.tell(solutions, values)

    def is_optimized(self):
        return self._optimized
    
    # def _optimize(self):
    #     es = self
    #     while not _Timer.timeit(es.stop)():
    #         solutions = _Timer.timeit(es.ask)()
    #         # print(len(solutions))
    #         values = [self._f(x) for x in solutions]
    #         # print(len(values))
    #         _Timer.timeit(es.tell)(solutions, values)
    #         print('values:',values )
    #         print('fbest:',es.get_coverage())
    #         print('iterations:', es._es.result.iterations)
    #         # print('evaluations:', es.result.evals_best)
    #         # es.tell(solutions, values)
    #         # print('\n')

    #     es.reset_stop_conditions()
    #     es._update_fbest()
    #     return self._optimized

    # def update(self):
    #     print('updating')
    #     self._fbest = self._es.result.fbest
    #     # print(self._optimized)
    #     self._optimized = self._prev_fbest > self._fbest
    #     if self._optimized:
    #         self._prev_fbest = self._fbest
    #     else:
    #         self._potential_popsize *= self._popsize_scale
    #         print('increase popsize to:', self._potential_popsize)

    def _update_fbest(self):
        fbest = self._es.result.fbest
        self._optimized = self._fbest > fbest
        if self._optimized:
            self._fbest = fbest

    def _reset_fbest(self):
        self._fbest = 0

    def _reset_popsize(self):
        self._potential_popsize = self._options['popsize']


    def _increase_popsize(self):
        self._potential_popsize *= self._popsize_scale
        # print('increase popsize to:', self._potential_popsize)
        return self._potential_popsize <= self._max_popsize
    
    def _is_over_threshold(self):
        # _stop = 
        # # _stop = self._potential_popsize > self._max_popsize or self._es.result.iterations >= self._max_gens
        # print('_is_over_threshold:', _stop)
        return self._potential_popsize > self._max_popsize

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
        self._potential_popsize = self._options['popsize']
        # maybe reset dim as well

    # def _reset

    def _current_state(self):
        # if 
        # return dict(generations = self._es.result.iterations, popsize = self._current_popsize, optimized = self.is_optimized())
        return dict(popsize = self._current_popsize, optimized = self.is_optimized())
        # return dict(input_size = self._input_size, popsize = self._current_popsize, optimized = self._optimized)

class FuzzerLogger:
    def __init__(self):
        self._fuzzer = None
        self._log = dict(testcase = 0, popsize = 0, generations = 0, coverage = 0, optimized = False, time = 0)
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
        self._log_path = fuzzer._program.log_path
        # example: logs/test.txt
        self._filename = self._log_path + fuzzer._program.pname +'.txt'
        # write initial parameters of Fuzzer.
        # program_path, max_sample_size, max_gen, max_popsize, timout, 
        initial_parameters = [fuzzer._cmaesoutputer._input_size, fuzzer._cmaesoutputer._max_popsize, fuzzer._cmaesoutputer._popsize_scale, fuzzer._cmaesoutputer._max_gens, fuzzer._timeout]
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

    def report_changes(self):
        if not self._fuzzer:
            return
        # precov = self._log['coverage']
        # if precov != self._fuzzer.get_coverage()
        self._log.update(self._fuzzer._cmaesoutputer._current_state())
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
            f.writelines("     %s     " % str(item) for item in self._log.values())
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




class Fuzzer:

    _VERIFIER_ERROS = {_Program.SAFE : 'SAFE', _Program.ERROR : 'ERROR', _Program.ASSUME : 'ASSUME_ERROR'}


    # def __init__(self, function, mean, sigma, options, program_path = 'test.c', sample_size = 1, max_sample_size = 10, resetable = True, max_popsize = 1000, input_size = 1000):
    def __init__(self, program_path, output_path = _Program.DEFAULT_PATHS['output'], log_path = _Program.DEFAULT_PATHS['log'], max_test = 10, max_popsize = 1000, max_gens = 1000 ,resetable = True, timeout = 15 * 60):
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
        self._interrupted = ''
        self._stop_reason = ''
        self._statuses = []
        self._program = _Program(program_path, output_path = output_path, log_path = log_path, timeout=timeout)
        self._cmaesoutputer = CMAES_outputer(init_popsize= 7 ,input_size = self._generate_input_size(), max_popsize = max_popsize, max_gens= max_gens) # maybe parameter as dict
        self._logger = FuzzerLogger().resister(self)
        # self._sample_map = {}

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
        input_size, returncode = self._program.get_input_size()
        self._check_runtime_error(returncode)
        #TODO: check input_szie < 2
        if input_size < CMAES_outputer.MIN_INPUT_SIZE:
            exit('ERROR: input_size: ' + str(input_size) + ', Input size must be greater than ' + str(CMAES_outputer.MIN_INPUT_SIZE) + '!')
        return input_size


    def get_coverage(self):
        # return self._cmaesoutputer.get_coverage()
        return self._coverage

    def _reset_samples(self):
        # self._logger.resister(self)
        if self._resetable:
            self._generations = 0
            self._samples = []
            self._sample_map = {}
            # self._cmaesoutputer._reset()

    def _encode(self, sample: np.ndarray) -> int:
        return bytes(sample.astype(int).tolist())

    def _run_all_samples(self):
        for sample in self._samples:
            self._program._run(self._encode(sample))

    def _run_samples(self, samples, returncode_check = False):
        for sample in samples:
            returncode = self._program._run(self._encode(sample))
            if returncode_check:
                self._check_verifier_error(returncode)
            # print("RETURNCODE!!!:",returncode)

    def _f(self, sample:np.ndarray):
        # self._run_all_samples()
        self._run_samples(self._samples + [sample])
        cov = self._program.get_coverage()
        return -cov
    
    def _current_state(self):
        return dict(testcase = len(self._samples), coverage = self._coverage, generations = self._generations)
        # return dict(testcase = len(self._samples), coverage = self._coverage)


    def _update(self):
        # pass
        # self._coverage = -self._cmaesoutputer.get_fbest()
        # self._coverage = self._cmaesoutputer.get_coverage()
        self._generations = self._cmaesoutputer.get_generations()
        # self._current_sample = self._cmaesoutputer.get_xbest()

    # def _update_coverage(self):


    def _stop(self):
        if self._interrupted:
            self._stop_reason = self._interrupted
        elif self._cmaesoutputer._fbest == -100:
            self._stop_reason = 'coverage is 100%'
        elif self._cmaesoutputer._potential_popsize > self._cmaesoutputer._max_popsize:
            self._stop_reason = 'max popsize is reached'

        return self._stop_reason
    def time(self):
        return time.time() - _init_time
        

    @_timeit
    def optimize_sample(self):
        es = self._cmaesoutputer
        try:
            while not es.stop():
                solutions = es.ask()
                # print(len(solutions))
                values = [self._f(x) for x in solutions]
                # print(len(values))
                es.tell(solutions, values)
                # self._update()
                self._generations = self._cmaesoutputer.get_generations()

                # print('values:',values )
                print('fbest:',self._cmaesoutputer._es.result.fbest)
                # print('iterations:', self._cmaesoutputer.get_generations())
                # print('evaluations:', es.result.evals_best)
                # es.tell(solutions, values)
                # print('\n')
        except (subprocess.TimeoutExpired,  KeyboardInterrupt) as e:
            self._interrupted = e.__class__.__name__

        es.reset_stop_conditions()
        es._update_fbest()
        self._coverage = -es.get_fbest()

        return es.is_optimized()

    @_timeit
    def optimize_testsuite(self):
        self._program._compile_program()
        count = 0
        while not self._stop() :
            self._cmaesoutputer.init_cmaes()
            optimized = self.optimize_sample()
            if optimized:
                self._current_sample = self._cmaesoutputer.get_xbest()
                self._samples.append(self._current_sample)
                self._total_samples.append(self._current_sample)
                self._logger.report_changes()

                # print('current_sample:', self._current_sample)
                # self._sample_map[str(sample)] = self.get_coverage()
                count = 0
                # maybe better if we increase if it was not optimized
                # self._cmaesoutputer._reset_popsize()
            else:
                self._logger.report_changes()
                # coverage is the same or lower than -fbest
                count += 1


            # # idea 2
            # if count > 0 and len(self._samples) > 0:
            #     # maybe rset fbest as wel
            #     self._cmaesoutputer._reset_fbest()
            #     del self._samples[-1]
            #     # del self._sample_map.keys()[-1]

            if count > 0:
                # self._total_samples += self._samples
                # self._coverage = 0
                self._cmaesoutputer._reset_fbest()
                if self._cmaesoutputer._increase_popsize():
                    # self._cmaesoutputer._reset_fbest()
                    self._reset_samples()
                    # pass
            # else:

                
            # if not self._cmaesoutputer._is_over_threshold():
            # else:
                # break


        return self._samples
        
    def get_testsuite(self):
        self.optimize_testsuite()
        self._program._timeout = None

        return self._samples


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
        # self._program._reset()
        # self._program.verifier_path = 'programs/__VERIFIER_with_error.c'
        # self._program._compile()
        self._run_samples(self._total_samples, returncode_check=True)
        gcov = self._program._gcov() 
        print(gcov)
        self._total_coverage = self._program._cov(gcov)
        self._logger.report_final()
        self._logger.report_time_log()


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
#         t = fuzzer.get_testsuite()
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
        t = fuzzer.get_testsuite()
        print('testsuite:', t)
        fuzzer.last_report()
        # fuzzer.gcov()
        # fuzzer.report()

def main_test():
    # subprocess.run('gcc programs/test.c programs/__VERIFIER_input_size.c -o test')

    #            1,2,3,4,5,6,7,8,9,0,11,12,13,14,15,16,17,18,19
    inp = bytes([0,0,0,0,0,0,0,0,0,0,254,254,254,0,0,0,0,0,113])
    subprocess.run('output/pals_STARTPALS_ActiveStandby.ufo.BOUNDED-10.pals', input = inp)

    # time_with_none = 0
    # time_with_19dim = 0
    # for _ in range(10):
    #     init_time = time.time()
    #     runtime += time.time() - init_time
    # print("tme")

    # init_time = time.time()

def parse_argv_to_fuzzer_kwargs():
    argvsize = len(sys.argv)
    example = 'e.g.: python3 fuzzer.py <program_path> [-op <output_path>] [-lp <log_path>] [-mp <max_popsize>] [-t timeout]'
    if argvsize == 1:
        exit('ERROR: No program_path is given!\n' + example)

    commands = {'-op' : 'output_path', '-lp' : 'log_path', '-mp' : 'max_popsize', '-t' : 'timeout'}
    current_command = None
    command_type_dict = {'output_path' : str, 'log_path' : str, 'max_popsize' : int, 'timeout' : int}
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
    # programs = [path+'test.c']


    fuzzer = Fuzzer(**parse_argv_to_fuzzer_kwargs())
    t = fuzzer.get_testsuite()
    print('testsuite:', t)
    fuzzer.last_report()
    # for program in programs:
        # fuzzer = Fuzzer(None, [0], None, None, program_path = program, max_sample_size = sample_size, input_size = input_size)

if __name__ == "__main__":
    # main_sv3()
    main()
    # main_test()
    # main_test_inputsize()
    # main_sv3()
    # main2()
    # simple_run('test2', bytes([79, 185]))
    # simple_run('test2', bytes([141,249]))
    
