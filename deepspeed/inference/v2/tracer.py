from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

@dataclass
class BatchTraceHolder:
    batch_id: int
    num_layers: int
    is_empty_run: bool
    seen_tokens: Any = field(default_factory=list)
    in_flight_tokens: Any = field(default_factory=list)
    traces: Any = field(default_factory=list)

@dataclass
class BatchTraceSummary:
    batch_id: int
    is_empty_run: bool
    num_layers: int
    seen_tokens: Any
    in_flight_tokens: Any
    layer_exec_times: Any
    embed: int
    unembed: int

@dataclass
class LayerExecTime:
    attn: int = 0
    moe: int = 0
    moe_a2a_1: int = 0
    moe_a2a_2: int = 0
    moe_ffn: int = 0
    moe_a2a_3: int = 0

class Tracer:

    def __init__(self):
        self._batch_counter = 0
        self._batch_traces = []
        self._cur_batch_trace = None

    def init_batch(self, is_empty_run, num_layers):
        batch_id = self._batch_counter
        self._batch_counter += 1
        self._cur_batch_trace = BatchTraceHolder(batch_id, num_layers, is_empty_run)
        self._batch_traces.append(self._cur_batch_trace)

    def add_sequence(self, seq_desc):
        self._cur_batch_trace.seen_tokens.append(seq_desc.seen_tokens)
        self._cur_batch_trace.in_flight_tokens.append(seq_desc.in_flight_tokens)

    def add_trace(self, name, start_event, end_event):
        self._cur_batch_trace.traces.append((name, start_event, end_event))

    def _summarize(self, batch_trace):

        traces = batch_trace.traces
        if batch_trace.is_empty_run:
            embed = 0
            unembed = 0
        else:
            embed = int(traces[0][1].elapsed_time(traces[0][2]) * 1000)
            unembed = int(traces[-1][1].elapsed_time(traces[-1][2]) * 1000)
            traces = traces[1:-1]
        layer_exec_times = []
        records_per_layer = len(traces) // batch_trace.num_layers
        for layer_idx in range(batch_trace.num_layers):
            exec_time = LayerExecTime()
            for r in traces[layer_idx * records_per_layer:(layer_idx + 1) * records_per_layer]:
                setattr(exec_time, r[0], int(r[1].elapsed_time(r[2]) * 1000))
            layer_exec_times.append(layer_exec_times)

        return BatchTraceSummary(
            batch_id=batch_trace.batch_id,
            is_empty_run=batch_trace.is_empty_run,
            num_layers=batch_trace.num_layers,
            seen_tokens=batch_trace.seen_tokens,
            in_flight_tokens=batch_trace.in_flight_tokens,
            layer_exec_times=layer_exec_times,
            embed=embed,
            unembed=unembed
        )

    def batch_summaries(self):
        for t in self._batch_traces:
            yield self._summarize(t)
        # return [self._summarize(t) for t in self._batch_traces]

TRACER = None
def set_tracer(tracer):
    global TRACER
    TRACER = tracer

def get_tracer():
    global TRACER
    return TRACER

@contextmanager
def record(name):
    global TRACER
    try:
        if TRACER:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        yield
    finally:
        if TRACER:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            TRACER.add_trace(name, start_event, end_event)
