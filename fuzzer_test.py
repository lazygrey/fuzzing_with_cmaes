import numpy as np
import fuzzer


program_path = 'examples/test.c'
log_dir = 'logs/fuzzer_test/'
timeout = 60
init_popsize = 10
cmaes_fuzzer = fuzzer.Fuzzer(program_path, log_dir=log_dir, timeout = timeout, init_popsize=init_popsize)
cmaes_fuzzer._program._compile_program()

def assert_best_sample_holder_update(best_sample_holder, xbest):
    if best_sample_holder.sample is None:
        exit('Test Failed: sample is never optimized')

    if xbest is not None and not cmaes_fuzzer._interrupted and (np.any(xbest != best_sample_holder.sample)):
        print(xbest, best_sample_holder.sample)
        exit('given sample is not the same as best sample')
    
    

def assert_gcov_coverage_matches_calculated_coverage(cov, calculated_cov):
    if cov != calculated_cov:
        exit('Test Failed: gcov coverage (%f) does not match with calculated coverage (%f)' % (cov, calculated_cov))

def _f_test(sample):
    samples = cmaes_fuzzer._samplecollector.get_optimized_samples()
    cmaes_fuzzer._run_samples(samples + [sample])
    _, cov = cmaes_fuzzer._program.get_line_and_branch_coverages()
    
    calculated_cov = -cmaes_fuzzer._f_branch(sample)

    assert_best_sample_holder_update(cmaes_fuzzer._samplecollector.best_sample_holder, cmaes_fuzzer._cmaesbuilder.result.xbest)
    assert_gcov_coverage_matches_calculated_coverage(cov, calculated_cov)

    return -calculated_cov

def test():
    cmaes_fuzzer.objective = _f_test
    cmaes_fuzzer.optimize_testsuite()
    if cmaes_fuzzer._interrupted:
        print("Test interrupted with '%s'" % cmaes_fuzzer._interrupted)
    else:
        print('Test Passed')


if __name__ == "__main__":
    test()
