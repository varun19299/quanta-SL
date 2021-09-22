"""
Usage:

Just @profile any function to benchmark
"""
import atexit

import line_profiler

# Makes the line profiler decorator
# available
# See: https://lothiraldan.github.io/2018-02-18-python-line-profiler-without-magic/

profile = line_profiler.LineProfiler()

# Before exiting print stats
atexit.register(profile.print_stats)
