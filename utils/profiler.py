import line_profiler
import atexit

# Makes the line profiler decorator
# available
# See: https://lothiraldan.github.io/2018-02-18-python-line-profiler-without-magic/

profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)