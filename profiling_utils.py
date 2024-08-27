# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import time
import logging
import torch
import torch.distributed
from functools import partial
from torch.profiler import tensorboard_trace_handler

WARMUP = 3

logger = logging.getLogger()

# adapted from https://github.com/pytorch/torchtitan


def trace_handler(
    prof: torch.profiler.profiler.profile,
    rank,
    export_memory_timeline,
    output_dir,
    metric="self_cuda_time_total",
    with_stack=True,
    group_by_stack=0,
    group_by_input_shape=False,
    row_limit=25,
):
    curr_trace_dir_name = "iteration_" + str(prof.step_num)
    curr_trace_dir = os.path.join(output_dir, curr_trace_dir_name)
    if not os.path.exists(curr_trace_dir):
        os.makedirs(curr_trace_dir, exist_ok=True)

    # Export chrome / tensorboard trace
    logger.info(f"Dumping traces at step {prof.step_num}")
    begin = time.monotonic()

    # Use tensorboard trace handler rather than directly exporting chrome traces since
    # tensorboard doesn't seem to be able to parse traces when with prof.export_chrome_trace
    exporter = tensorboard_trace_handler(
        curr_trace_dir, worker_name=f"rank{rank}", use_gzip=True
    )
    exporter(prof)
    # prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json")

    logger.info(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")

    # Construct the memory timeline file.
    if export_memory_timeline:
        try:
            prof.export_memory_timeline(
                f"{curr_trace_dir}/rank{rank}_memory-timeline.html"
            )
        except Exception:
            logger.info(
                "Failed to export memory timeline to html, retrying as gzipped json."
            )
            try:
                prof.export_memory_timeline(
                    f"{curr_trace_dir}/rank{rank}_memory-timeline.json.gz"
                )
            except Exception:
                logger.info(
                    "Failed to export memory timeline to gzipped json. Saving profiler timeline object instead."
                )
                from torch.profiler._memory_profiler import MemoryProfileTimeline

                memory_profile = MemoryProfileTimeline(prof._memory_profile())
                torch.save(
                    memory_profile, f"{curr_trace_dir}/rank{rank}_memory-timeline.pt"
                )

    # Dump stack traces
    if with_stack:
        prof.export_stacks(f"{curr_trace_dir}/rank{rank}_stacks.txt", metric=metric)

    # Export event averages
    key_avgs = prof.key_averages(
        group_by_input_shape=group_by_input_shape, group_by_stack_n=group_by_stack
    ).table(sort_by=metric, row_limit=row_limit)
    with open(f"{curr_trace_dir}/rank{rank}_key_averages.txt", "w") as f:
        print(key_avgs, file=f)
    if rank == 0:
        print(f"Saving profiling results to {curr_trace_dir}")

    # TODO: Is this necessary?
    torch.distributed.barrier()


@contextlib.contextmanager
def profiling_context(
    profile,
    model_name,
    train_type,
    profiling_output,
    warmup_steps,
    active_steps,
    repeat,
    profiling_frequency,
    wait_steps,
    rank,
    export_memory_timeline,
    with_stack,
    with_shapes,
):
    enable_profiling = profile

    if enable_profiling:
        model_name = model_name.split("/")[-1]
        train_type = train_type
        output_dir = (
            profiling_output if profiling_output else f"./{model_name}_{train_type}"
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Profiling enabled. Traces will be saved at {output_dir}")

        warmup = warmup_steps
        active = active_steps
        repeat = repeat

        if repeat == 0:
            steps_per_cycle = profiling_frequency
            wait = steps_per_cycle - (active + warmup)
        else:
            wait = wait_steps
            steps_per_cycle = wait + warmup + active
        assert (
            wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"
        logger.info(
            f"Profiler schedule - steps per cycle: {steps_per_cycle} wait: {wait} warmup: {warmup} active: {active} repeat: {repeat if repeat !=0 else 'inf'}"
        )

        profile_memory = export_memory_timeline
        with_stack = with_stack or export_memory_timeline
        with_shapes = with_shapes or export_memory_timeline
        callback = partial(
            trace_handler,
            rank=rank,
            export_memory_timeline=export_memory_timeline,
            output_dir=output_dir,
            with_stack=with_stack,
            group_by_input_shape=with_shapes,
            group_by_stack=5 if with_stack else 0,
        )

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=with_stack,
            profile_memory=profile_memory,
            record_shapes=with_shapes,
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=repeat
            ),
            on_trace_ready=callback,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
            if with_stack
            else None,
        ) as torch_profiler:
            yield torch_profiler
    else:

        class FakeProfiler:
            """
            Fake profiler object when profiling is not enabled.

            """

            def __enter__(self):
                return self

            def __exit__(self, *args, **kwargs):
                pass

            def step(self):
                pass

        yield FakeProfiler()
