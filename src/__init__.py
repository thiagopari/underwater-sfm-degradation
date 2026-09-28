import warnings

# numpy + Apple Accelerate raise spurious floating-point warnings from matmul
# (numpy issue #26669); the results are finite and correct.
warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)
