import numpy as np
import fuzzer

kwargs = fuzzer.parse_argv_to_fuzzer_kwargs()
kwargs['log_dir'] = 'logs/fuzzer_test/'

def assert_best_sample_holder_has_the_best_sample(sample, best_sample, interrupted):
    if not interrupted and (np.any(sample != best_sample)):
        print(sample, best_sample)
        exit('Test Failed: given sample is not the same as the best sample in sample collector')

def assert_gcov_coverage_matches_calculated_coverage(cov, calculated_cov):
    # return
    if cov != calculated_cov:
        exit('Test Failed: gcov coverage (%f) does not match with calculated coverage (%f)' % (cov, calculated_cov))

def assert_duplicates(sample_holders):
    for i, sample_holder1 in enumerate(sample_holders):
        for j, sample_holder2 in enumerate(sample_holders):
            if i == j:
                continue

            if sample_holder2.path.issubset(sample_holder1.path):
                print([sh.sample for sh in sample_holders])
                print(sample_holders[i].sample, sample_holders[j].sample)
                exit('Test Failed: duplicates for interesting sample holders with (%d, %d) indices' % (i, j))


class TestSampleCollector(fuzzer.SampleCollector):
    def __init__(self, fuzzer):
        super(TestSampleCollector, self).__init__()
        self.fuzzer = fuzzer

    def check_interesting(self, sample, current_path):
        super(TestSampleCollector, self).check_interesting(sample, current_path)
        assert_duplicates(self.total_sample_holders)

    def add_best(self, sample, stds):
        assert_best_sample_holder_has_the_best_sample(sample, self.best_sample_holder.sample, self.fuzzer._interrupted)

        return super(TestSampleCollector, self).add_best(sample, stds)

class TestFuzzer(fuzzer.Fuzzer):
    def __init__(self, **kwargs):
        super(TestFuzzer, self).__init__(**kwargs)
        self._samplecollector = TestSampleCollector(self)

    def get_gcov_coverages(self, sample):
        samples = self._samplecollector.get_optimized_samples()
        self._run_samples(samples + [sample])
        return self._program.get_line_and_branch_coverages()

    def _f_line(self, sample):
        line_cov, _ = self.get_gcov_coverages(sample)
        calculated_cov = -super(TestFuzzer, self)._f_line(sample)

        assert_gcov_coverage_matches_calculated_coverage(line_cov, calculated_cov)

        return -calculated_cov

    def _f_branch(self, sample):
        _, branch_cov = self.get_gcov_coverages(sample)
        calculated_cov = -super(TestFuzzer, self)._f_branch(sample)

        assert_gcov_coverage_matches_calculated_coverage(branch_cov, calculated_cov)

        return -calculated_cov

def test():
    test_fuzzer = TestFuzzer(**kwargs)
    t = test_fuzzer.generate_testsuite()
    # print('interesting:', [s.sample for s in test_fuzzer._samplecollector.interesting_sample_holders])
    # print('interesting:', 100 * len(test_fuzzer._samplecollector.interesting_paths) / test_fuzzer._samplecollector._total_path_length)
    test_fuzzer.last_report()

    if test_fuzzer._interrupted:
        print("Test interrupted with '%s'" % test_fuzzer._interrupted)
    else:
        print('Test Passed')


if __name__ == "__main__":
    test()
