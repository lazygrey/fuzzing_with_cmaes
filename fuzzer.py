import cma
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import subprocess
import os
import time
import functools

class _Timer:
    logs = {}
    def timeit(f):
        @functools.wraps(f)
        def newf(*args, **kwargs):
            startTime = time.time()
            output = f(*args, **kwargs)
            elapsedTime = time.time() - startTime

            if not f.__name__ in _Timer.logs:
                _Timer.logs[f.__name__] = 0.
            _Timer.logs[f.__name__] += elapsedTime * 1000

            return output

            # print('function [{}] finished in {} ms'.format(
            #     f.__name__, int(elapsedTime * 1000)))
        return newf

    def reset():
        _Timer.logs = {}
    


    

class _Program:
    def __init__(self, path):
        self.path = path
        self.pname = path[:-2] # *.c
        # self.program = path + program
        self._compiled = False

    def compile(self):
        if not self._compiled:
            subprocess.run(['gcc', self.path, '__VERIFIER.c', '-o', self.pname, '--coverage'])
            self._compiled = True

    @_Timer.timeit
    def _reset(self):
        if os.path.isfile(self.pname + '.gcda'):
            os.remove(self.pname+'.gcda')
        else:
            print('!No gcda file!')
            # maybe program reached error?
        

    @_Timer.timeit
    def _run(self, sample_bytes):
        subprocess.run(self.pname, input = sample_bytes)

    @_Timer.timeit
    def _gcov(self):
        return subprocess.run(['gcov', self.pname], capture_output = True).stdout.decode()

    @_Timer.timeit
    def _cov(self, output):
        start, end = 0 , 0
        for i in range(len(output)):
            if output[i] == ':':
                start = i + 1
            elif output[i] == '%':
                end = i
                break
        return float(output[start:end])


    def get_coverage(self, sample_bytes):
        # self.timer.timeit(self._run)
        # output = self.timer.timeit(self._gcov)
        # self.timer.timeit(self._reset)
        # return self.timer.timeit(self._cov)(output)
        
        self._run(sample_bytes)
        output = self._gcov()
        self._reset()
        return self._cov(output)


class Fuzzer:
    def __init__(self, function, mean, sigma, options, program_path = 'test.c', sample_size = 1, reset = False, max_popsize = 100):
        """Fuzzer with cmaes.

        Args:
        sampe_type = the type of a sample to optimize
        function = score function to optimize
        """
        # n = sample_dim[sample_type]
        # mean = n * [128]
        # sigma = 64
        # options = {'bounds' : [0, 256]}
        self._args = sample_size * mean, sigma, options
        self._mean, self._sigma, self._options = self._args
        self._scale = 1
        # self._cma = cma.CMAEvolutionStrategy(mean, sigma, options)
        self._function = function
        # self._program_path = program_path
        self._program = _Program(program_path)
        self._sample_size = sample_size
        self._fbest = 0
        self._optimized = False
        self._resetable = reset
        self._max_popsize = max_popsize
        self._samples = []

    def get_coverage(self):
        return -self._fbest

    def reset_cov(self):
        if self._resetable:
            self._samples = []

    def encode(self, sample: np.ndarray) -> int:
        # assert len(bytes) == 4, "Integer should have 4 bytes"
        # check
        # if np.any( bytes < 0) or np.any(bytes > 255):
        # check
        # print("bytes before: ", bytes)
        # bytes = np.where(bytes < 0, 0., bytes)
        # bytes = np.where(bytes > 255, 255., bytes)
        # print("bytes after: ", bytes)

        # result = int.from_bytes(sample.astype(int).tolist(), 'big', signed = True)
        # print('bytes:', bytes)
        # print('int:', result)
        # return str(result).encode()
        out = bytes(sample.astype(int).tolist())
        # print(out)
        return out


    def _f(self, sample:np.ndarray):
        for prev_sample in self._samples:
            self._program._run(self.encode(prev_sample))

        cov = self._program.get_coverage(self.encode(sample))
        return -cov


    def get_samples(self):
        ess = []
        for i in range(1, int(len(self._mean)/self._sample_size)):
            mean = self._sample_size * self._mean[-i-1:]
            ess.append(cma.CMAEvolutionStrategy(mean, self._sigma, self._options))
        samples = []
        for es in ess:
            while not es.stop():
                solutions = es.ask()
                values = []
                for x in solutions:
                    fit = self._f(x)
                    values.append(fit)
                    if fit < self._fbest:
                        self._fbest = fit
                        samples.append(x)
                es.tell(solutions, values)

        return samples

    def get_samples2(self):
        ess = []

        for i in range(1, int(len(self._mean)/self._sample_size)):
            mean = self._sample_size * self._mean[-i-1:]
            ess.append(cma.CMAEvolutionStrategy(mean, self._sigma, self._options))
        samples = []
        for es in ess:
            while not es.stop():
                solutions = es.ask()
                values = []
                for x in solutions:
                    fit = self._f(x)
                    values.append(fit)
                    if fit < self._fbest:
                        self._optimized = True
                        self._fbest = fit
                        samples.append(x)
                es.tell(solutions, values)

        if not self._optimized and self._options['popsize'] < self._max_popsize:
            self.reset_cov()
            self._options['popsize'] *= 10

            if self._options['popsize'] >= self._max_popsize:
                self._options['popsize'] = self._max_popsize
                self._options['mean'] = 2 * [128]

            print('increasing popsize to', self._options['popsize'])

            return self.get_samples2()

        print(len(samples))
        self._optimized = False
        return samples

    def get_sample2(self):
        ess = []
        for i in range(1, int(len(self._mean)/self._sample_size)):
            mean = self._sample_size * self._mean[-i-1:]
            ess.append(cma.CMAEvolutionStrategy(mean, self._sigma, self._options))

        fbest = 0
        xbest = None
        stds = None
        for es in ess:
            while not es.stop():
                solutions = es.ask()
                values = [self._f(x) for x in solutions]
                es.tell(solutions, values)
            # check if fbest is optimized
            print(solutions)
            print(values)
            if fbest <= es.result.fbest:
                return xbest, stds, fbest
            xbest, stds, fbest = es.result.xbest, es.result.stds, es.result.fbest

    def get_sample3(self):
        # with fixed var
        pass

    def get_sample(self):
        # with this interface, cmaes minimize the values
        f = self._f
        if self._function:
            f = self._function

        self._program.compile()

        es = cma.CMAEvolutionStrategy(*self._args)
        g = 0
        while not _Timer.timeit(es.stop)():
            g += 1
            print(g)
            solutions = _Timer.timeit(es.ask)()
            values = [f(x) for x in solutions]
            _Timer.timeit(es.tell)(solutions, values)
            # es.tell(solutions, values)
            print('\n')
        """
        'CMAEvolutionStrategyResult', [
            'xbest',
            'fbest',
            'evals_best',
            'evaluations',
            'iterations',
            'xfavorite',
            'stds',
            'stop',
        ]
        """
        # return sample, es.result.stds, es.result.fbest
        self._fbest = es.result.fbest
        sample = es.result.xbest
        self._samples.append(sample)
        return sample

    def get_logs(self):
        logs = _Timer.logs
        _Timer.reset()
        return logs


def bytes_to_int(bytes: np.ndarray) -> int:
    # assert len(bytes) == 4, "Integer should have 4 bytes"
    # check
    # if np.any( bytes < 0) or np.any(bytes > 255):
    # check
    # print("bytes before: ", bytes)
    # bytes = np.where(bytes < 0, 0., bytes)
    # bytes = np.where(bytes > 255, 255., bytes)
    # print("bytes after: ", bytes)

    result = int.from_bytes(bytes.astype(int).tolist(), 'little', signed = True)
    # print('bytes:', bytes)
    # print('int:', result)
    return result




def program(input : int):
    coverage = 0
    x = input

    if x > 0:
        if x > 1000:
            if x < 100000:
                return 4

            else:
                return 3

        else:
            return 2
    else:
        return 1


def function(sample: np.ndarray):
    coverage = program(bytes_to_int(sample))
    return -coverage


def function1(sample : np.ndarray):
    # any input has same coverage
    return 50


def function2(sample : np.ndarray):
    # check if x3 < 50 will be found

    x1, x2, x3 = sample
    if x1 > 128:
        if x2 < 128:
            if x3 < 50:
                return 0.
            else:
                return 1.
        else:
            return 2.
    else:
        return 3.


def function3(sample: np.ndarray):
    # check if x3 == 128 will be found

    x1, x2, x3 = sample
    if 128 <= x1 < 129 and 128 <= x2 < 129:
        return 0.
    else:
        return 3.

def function4(sample: np.ndarray):
    # check if more 0 or more 255
    x1, x2, x3 = sample
    if x1 > 128:
        if x2 > 128:
            if 0 <= x3 < 1 or 255 < x3 <= 256:
                return 0.
            else:
                return 1.
        else:
            return 2.
    else:
        return 3.

def function5(sample: np.ndarray):
    # check if more 100 or more 200
    x1, x2, x3 = sample
    if x1 > 128:
        if x2 > 128:
            if x3 < 101 or 200 < x3 <= 201:
                return 0.
            else:
                return 1.
        else:
            return 2.
    else:
        return 3.

def function6(sample: np.ndarray):
    # check if f4 with 256 (max) and f6 with 255 different

    x1, x2, x3 = sample
    if x1 > 128:
        if x2 > 128:
            if 254 <= x3 < 256 or 0 <= x3 < 1 :
                return 0.
            else:
                return 1.
        else:
            return 2.
    else:
        return 3.

def function7(sample: np.ndarray):
    # check if minimum or maximum will be found more often

    x1, x2, x3 = sample
    if 0 <= x1 <= 1 or  255 <= x1 <= 256:
       #   or \
       # 85 <= x3 < 86 or 170 <= x3 < 171:
         return 0.
    else:
        return 3.

def function8(sample: np.ndarray):
    # check if minimum or maximum will be found more often

    x1, x2, x3 = sample
    if 0 <= x1 <= 10 or  246 <= x1 <= 256:
       #   or \
       # 85 <= x3 < 86 or 170 <= x3 < 171:
         return 0.
    else:
        return 3.


# def parse_bytes_to_sample(sample_type: SampleType):
    # maybe a general method or class to parse byte to sample in a right type
    # pass



def plot_samples(xs, ys, zs):
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.scatter3D(xs, ys, zs, cmap='Greens')
    plt.show()

def dist_samples(xs, ys, zs, kwargs):
    # hist_samples = plt.figure()
    # plt.figure()
    fig, (axes1, axes2) = plt.subplots(2, 3, figsize=(10,5), dpi=100, sharex=True, sharey=False)
    data = [xs, ys, zs]
    colors = ['g', 'b', 'r']
    labels = ['x1', 'x2', 'x3']
    # kwargs = dict(alpha = 0.5, bins = 100)
    kkwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

    # # density
    for i, (ax1, ax2) in enumerate(zip(axes1.flatten(), axes2.flatten())):
        sns.distplot(data[i], color = colors[i], axlabel = labels[i], ax = ax1, **kkwargs)
        ax2.hist(data[i], **kwargs, color = colors[i])

    # hist
    # for i, ax in enumerate(axes.flatten()):

    plt.tight_layout()

def dist_samples_one(sample):
    # hist_samples = plt.figure()
    # plt.figure()
    # fig, axes = plt.subplots(1, 3, figsize=(10,2.5), dpi=100, sharex=True, sharey=False)
    kwargs = dict(alpha = 0.5, bins = 50)

    # sns.distplot(samples, color = 'g', label = 'Input')
    plt.hist(sample, **kwargs, color='g')

    # plt.tight_layout()
    plt.legend()

def dist_values(values, kwargs):
    hist_values = plt.figure()
    # kwargs = dict(alpha = 0.5, bins = 50)
    plt.hist(values, **kwargs, color = 'g')

def visualize_results(samples, values, fname, i, sample_size):

    xs, ys, zs = [],[],[]
    for sample in samples:
        x, y, z = sample
        xs.append(x)
        ys.append(y)
        zs.append(z)
    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

    kwargs = dict(alpha = 0.5, bins = sample_size)

    plt.figure(2*i - 1)
    dist_samples(xs, ys, zs, kwargs)
    plt.suptitle(fname)
    # dist_samples_one(np.array(samples))
    plt.figure(2*i)
    dist_values(np.array(values), kwargs)
    plt.suptitle(fname)

def visualize_results1(samples, coverages, pname):
    plt.plot(samples, coverages)
    plt.grid(True)
    plt.suptitle(pname)
    plt.xlabel('sample')
    plt.ylabel('coverage')
    plt.show()


def main1():
    mean = 3 * [128]
    sigma = 64
    options = dict(bounds = [0, 256], popsize = 10000)
    # functions = [function1, function2, function3, function7, function8]
    functions = [function3]
    sample_size = 3
    # functions = [function]

    for i, function in enumerate(functions, 1):
        fuzzer = Fuzzer(function, mean, sigma, options)
        samples = []
        values = []
        stds=[]
        for _ in range(sample_size):
            sample, std, value= fuzzer.get_sample()
            samples.append(sample)
            # values.append(function(sample))
            values.append(value)
            stds.append(std)
        # print(samples)
        # print(values)
        # print(stds)
        visualize_results(samples, values, function.__name__, i, sample_size)

    plt.show()


def get_coverage(program, input):
    gcc = 'gcc'
    path = '../test/'
    program = 'test.c'

    subprocess.run('./test', input = input)
    output = subprocess.run(['gcov', program], stdout = subprocess.PIPE).stdout.decode('utf-8')

    start, end = 0 , 0
    for i in range(len(output)):
        if output[i] == ':':
            start = i + 1
        elif output[i] == '%':
            end = i
            break

    return float(output[start:end])

def main2():
    # subprocess.run('gcc ./test.c -o ./test --coverage')
    #
    # input = np.zeros(4).astype(int).tolist()
    #
    # program = './test.c'
    # coverage = get_coverage(program, bytes(input))
    # print(coverage)
    # print(get_coverage(program, '1'.encode('utf-8')))

    # fuzzer = Fuzzer()
    # subprocess.run('del *.gcda', shell = True)

    # subprocess.run('gcc test1.c -o test1')
    # for i in range(256):
    #     input = bytes([i,0,0,0])
    #     output = subprocess.run('./test1', stdout=subprocess.PIPE, input = input)
    #     print('generated output:',output.stdout)
        
    # print(input)
    subprocess.run('gcc testingsize.c __VERIFIER.c -o testingsize --coverage')
    for i in range(1):
        input = b'\x1a\x00\x00\x00\x1a\x00\x00\x00'
        output = subprocess.run(['./testingsize'], capture_output = True, input = input)
        print(output.stdout.decode())
        subprocess.run('gcov testingsize')
        os.remove('testingsize.gcda')

def main3():
    mean = 4 * [128]
    sigma = 64
    options = dict(bounds = [0, 256], popsize = 10, verb_disp = 0)
    generations = 4
    programs = ['./test.c']
    max_popsize = 100

    for i, program in enumerate(programs, 1):
        samples = []
        values = []
        stds=[]
        fuzzer = Fuzzer(None, mean, sigma, options, program_path = program, sample_size = 1, max_popsize = max_popsize)
        for i in range(generations):
            # sample, std, value = fuzzer.get_sample2()
            # subprocess.run('del test.gcda', shell = True)
            # for x in np.split(sample, 1):
            #     print(x)
            #     samples.append(bytes_to_int(x))c
            # values.append(value)
            # stds.append(std)
            # for sample in fuzzer.get_samples():
            #     samples.append(bytes_to_int(sample))
            print('generation:', i)
            print('coverage:', fuzzer.get_coverage())
            for sample in fuzzer.get_samples2():
                samples.append(bytes_to_int(sample))
        # visualize_results(samples, values, function.__name__, i, sample_size)
    print('last samples:', samples)
    print('last coverage:',fuzzer.get_coverage())
    # print(values)
    # print(stds)

    # plt.show()

def test0(sample: np.ndarray):
    x = bytes_to_int(sample)
    if x > 0:
        return 87.74
        if x < 100:
            return 100.0
        
    return 67.14

def main4():
    mean = 2 * [128]
    sigma = 64
    options = dict(bounds = [0, 256], popsize = 10, verb_disp = 0)
    programs = ['./test_2max.c']
    sample_size = 8

    for program in programs:
        fuzzer = Fuzzer(None, mean, sigma, options, program_path = program)
        samples = {}
        inputs = {}
        bs = []
        for i in range(sample_size):
            sample = fuzzer.get_sample()
            samples[bytes_to_int(sample)] = fuzzer.get_coverage()
            inputs[str(sample)] = fuzzer.get_coverage()
            bs.append(sample)

    for b in bs:
        subprocess.run('test_2max', input = bytes(b.astype(int).tolist()))
    subprocess.run(['gcov', 'test_2max'])
    print(fuzzer.get_logs())
    print('samples:', samples)
    print('inputs:', inputs)

def main5():
    program = _Program('./test_2max.c')
    program.compile()
    samples = []
    inputs = []
    coverages = []
    s = 2 ** 0
    for i in range(int(256/s)):
        print(i)
        for j in range(int(256/s)):
            input = bytes([s*j,s*i])
            sample = int.from_bytes(input, 'little', signed = True)
            cov = program.get_coverage(input)
            inputs.append(input)
            samples.append(sample)
            coverages.append(cov)
    # visualize_results1(samples, coverages, 'test1.c')
    samples, coverages = zip(*sorted(zip(samples, coverages)))
    print(*zip(samples,coverages))
    for input in inputs:
        subprocess.run('./test_2max', input = input)
    subprocess.run('gcov ./test_2max')
    visualize_results1(samples, coverages, 'test1.c')

def simple_run(program, input):
    # subprocess.run(['gcc', program, '-o', program[:-2]])
    subprocess.run(program, input = input)

def test2(x):
    y = -32768
    z = 8192
    if (x < y + z):
        return 43.75
    y += z
    if (y <= x and x < y + z):
        return 58.33
    y += z
    if (y <= x and x < y + z):
        return 50.0
    y += z
    if (y <= x and x < y + z):
        return 50.0
    y += z
    if (y <= x and x < y + z):
        return 45.83
    y += z
    if (y <= x and x < y + z):
        return 54.17
    
    y += z
    if (y <= x and x < y + z):
        return 45.83
    
    y += z
    if (y <= x):
        return 43.75

    return 0

def main6():
    samples = []
    coverages = []
    s = 2 ** 0
    for i in range(int(256/s)):
        print(i)
        for j in range(int(256/s)):
            input = bytes([s*j,s*i])
            sample = int.from_bytes(input, 'little', signed = True)
            cov = test2(sample)
            samples.append(sample)
            coverages.append(cov)
    # visualize_results1(samples, coverages, 'test1.c')
    samples, coverages = zip(*sorted(zip(samples, coverages)))
    visualize_results1(samples, coverages, 'test2.c')


def main_sv():
    mean = 100000 * [128]
    sigma = 64
    options = dict(bounds = [0, 256], popsize = 10, verb_disp = 0)
    pname = 'data_structures_set_multi_proc_ground-1'
    # programs = ['./data_structures_set_multi_proc_ground-1.c']
    # programs = ['./data_structures_set_multi_proc_ground-2.c'] #true
    # programs = ['standard_init1_ground-1.c'] 
    # programs = ['standard_copy1_ground-1.c'] # True
    # programs = ['standard_copy1_ground-2.c']
    # programs = ['relax-2.c'] # True
    programs = [pname+'.c']
    
    sample_size = 1

    for program in programs:
        fuzzer = Fuzzer(None, mean, sigma, options, program_path = program)
        samples = {}
        inputs = {}
        bs = []
        for i in range(sample_size):
            sample = fuzzer.get_sample()
            samples[bytes_to_int(sample)] = fuzzer.get_coverage()
            inputs[str(sample)] = fuzzer.get_coverage()
            bs.append(sample)

    for b in bs:
        subprocess.run(pname, input = bytes(b.astype(int).tolist()))
    subprocess.run(['gcov', pname])
    print(fuzzer.get_logs())
    # print('samples:', samples)
    # print('inputs:', inputs)


if __name__ == "__main__":
    main_sv()
    # main2()
    # simple_run('test2', bytes([79, 185]))
    # simple_run('test2', bytes([141,249]))
    
