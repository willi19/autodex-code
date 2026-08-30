"""Low-latency multi-object basket demo.

The package intentionally keeps policy (catalogue selection, retry and
verification) separate from the robot-specific runner so its safety decisions
can be tested without a robot, cameras or CUDA.
"""
