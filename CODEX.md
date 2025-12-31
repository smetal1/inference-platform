# Codex: Serverless ML Inference Without Warm GPUs

Codex is a low-latency, minimal-downtime serverless framework for ML model hosting that **never keeps GPUs warm or idle**. GPUs are allocated **only when there is active work** and released within seconds of idleness.

---

## 1. High-Level Architecture Diagram (Textual)

```
                    ┌──────────────────────────────────────────────────┐
                    │                    Control Plane                 │
                    │                                                  │
                    │  ┌────────────┐  ┌───────────────┐  ┌──────────┐ │
Ingress ───────────▶│  │ API Gateway │→│  Scheduler    │→│ GPU Alloc │ │
REST/gRPC/Streaming │  └────────────┘  └───────────────┘  └──────────┘ │
                    │        │                 │              │       │
                    │        ▼                 ▼              ▼       │
                    │  ┌────────────┐  ┌───────────────┐  ┌──────────┐ │
                    │  │ Model Reg. │  │ Artifact Mgr  │  │ Autoscale│ │
                    │  └────────────┘  └───────────────┘  └──────────┘ │
                    │        │                 │              │       │
                    └────────┼─────────────────┼──────────────┼───────┘
                             │                 │              │
                             ▼                 ▼              ▼
                    ┌──────────────────────────────────────────────────┐
                    │                     Data Plane                   │
                    │ ┌─────────────┐  ┌──────────────┐  ┌───────────┐ │
                    │ │ Runtime Mgr │→│ Execution     │→│ Observ/Bill │ │
                    │ │ (LLM/Vision)│  │ Plane (K8s)   │  │ Engine     │ │
                    │ └─────────────┘  └──────────────┘  └───────────┘ │
                    │         ▲                 │                     │
                    │         │                 ▼                     │
                    │   ┌────────────┐     ┌───────────────┐          │
                    │   │ GPU Worker │◀────│ Model Runtime │          │
                    │   └────────────┘     └───────────────┘          │
                    └──────────────────────────────────────────────────┘
```

---

## 2. Component Responsibilities

### API Gateway (REST, gRPC, Streaming)
- Auth, tenant routing, rate limits, priority class (interactive/batch).
- Streams tokens and maintains request context.

### Model Registry (Versioned, Immutable)
- Stores model metadata and versioned artifacts.
- Immutable refs: `model://org/model@sha256:...`.

### Scheduler (GPU-aware, SLO-driven)
- Computes placement based on VRAM, architecture, cost, and latency SLO.
- Implements fairness and queue prioritization.
- Issues GPU claims only when queue is non-empty.

### GPU Allocator (On-demand)
- Requests GPU nodes or on-demand GPU pods via Kubernetes device plugins.
- Releases GPU resources after idle timeout.

### Runtime Manager (LLM/Vision/Multimodal)
- Manages runtime lifecycle (init/load/infer/unload).
- Supports vLLM/TGI for LLMs, TensorRT/ONNX for vision.

### Artifact Manager (Weights/Engines/Tokenizers)
- Maintains CPU/SSD caches of model artifacts.
- Handles layered container caching, mmap, and TensorRT engine reuse.

### Execution Plane (Kubernetes or equivalent)
- Runs GPU workers as ephemeral pods.
- Enforces cgroups/namespaces for isolation.

### Autoscaler (Event-driven, No Warm Pools)
- Scales up GPU workers based on queue depth, arrival rate, and SLO risk.
- Scales down aggressively after idle timeout.

### Observability & Billing Engine
- Collects latency metrics, TTFT, queue time, GPU utilization.
- Metering per second/token with GPU-aware pricing.

---

## 3. Control Plane vs Data Plane

**Control Plane**
- API Gateway, Scheduler, GPU Allocator, Model Registry, Artifact Manager, Autoscaler.
- Decides where/when to allocate GPUs and which model/runtime to run.

**Data Plane**
- Runtime Manager, Execution Plane, GPU Workers, Observability & Billing.
- Executes inference, streams responses, reports metrics.

---

## 4. Request Lifecycle (Ingress → Inference → Teardown)

1. **Ingress**: API Gateway authenticates request, assigns priority class.
2. **Queueing**: Request is enqueued with SLO deadline and latency budget.
3. **Scheduling**: Scheduler evaluates model compatibility and SLO risk.
4. **GPU Claim**: GPU Allocator provisions a GPU worker *only when queue exists*.
5. **Runtime Init**: Runtime Manager pulls container layers; Artifact Manager prefetches model artifacts to CPU/SSD.
6. **Load**: Model weights are loaded via parallel chunking and mmap; TensorRT engines reused.
7. **Inference**: Streaming responses; micro-batching in millisecond windows.
8. **Teardown**: Idle timeout triggers `unload()` and GPU release (seconds-level).

---

## 5. Failure and Retry Paths

- **GPU node failure**: Scheduler retries on another GPU node with hedged acquisition if latency risk grows.
- **OOM/Runtime crash**: Auto rollback to last healthy model version; retry on alternate GPU type.
- **Artifact mismatch**: Invalid cache → force re-fetch + checksum validation.
- **SLO violation**: Scheduler re-prioritizes, may drop batch jobs in favor of interactive.

---

## 6. GPU Scheduling & Execution Model

### Scheduling Goals
- Meet SLOs without warm GPUs.
- Avoid fragmentation using bin-packing on VRAM.
- Maintain tenant fairness with weighted fair queues.

### GPU Selection Criteria
- VRAM fit and architecture compatibility.
- Cost per token/second.
- Latency risk score (queue delay + cold start estimate).

### Hedged Acquisition
- If SLO risk exceeds threshold, acquire a second GPU in parallel and cancel the slower one.

### Pseudocode: Scheduler

```python
def schedule(request):
    model = registry.resolve(request.model_ref)
    queue = get_queue(model, request.priority)
    queue.enqueue(request)

    if queue.depth > 0:
        candidates = gpu_inventory.filter(
            vram>=model.vram_req,
            arch_in=model.supported_arches
        )
        ranked = rank_candidates(
            candidates,
            cost_weight=model.cost_weight,
            slo_weight=request.slo_deadline
        )
        gpu = ranked.first()
        claim_gpu(gpu, request.model_ref)

        if slo_risk(queue, request) > RISK_THRESHOLD:
            hedge = ranked.second()
            if hedge:
                claim_gpu(hedge, request.model_ref, hedged=True)
```

### GPU Fragmentation Avoidance
- Bin-pack by VRAM class and model size.
- Prefer consolidating same-model requests into the same worker to reuse weights.

### Preemption and Failure Handling
- Batch jobs can be preempted if interactive SLO is threatened.
- Failures trigger immediate reschedule with stateful request IDs.

### Multi-tenant Fairness
- Weighted fair queues per tenant.
- Hard caps on concurrent GPU slots.

---

## 7. Low Latency Without Warm GPUs

### Techniques
- **Layered container + artifact caching**: GPU pods pull minimal base layers; large weights cached on SSD.
- **Parallel chunked loading**: Split weights into shards; load in parallel.
- **Memory-mapped / lazy loading**: `mmap` weights to reduce copy time.
- **TensorRT / engine reuse**: Cached optimized engines reused across requests.
- **Streaming inference**: Emit tokens as soon as first batch is ready.
- **Micro-batching**: 1–5ms window to batch requests without large latency hit.
- **Priority queues**: Interactive > batch, ensures fast TTFT.
- **Predictive artifact prefetch**: CPU/SSD prefetch based on traffic signals (no GPU reservation).

### True Cold Start
- CPU-only prefetch executes immediately when new traffic is detected.
- First GPU allocation occurs only when request arrives; model materialization happens fast via cached artifacts.

### TTFT Minimization
- Stream tokens as soon as first decoder step is ready.
- Pipeline model load and runtime init in parallel.

---

## 8. Runtime Interface Contract

```python
class ModelRuntime:
    def init(self, runtime_config): ...
    def load(self, model_ref): ...
    def infer(self, request): ...
    def health(self): ...
    def unload(self): ...
```

### Supported Runtimes
- LLMs: vLLM/TGI-style.
- Vision: TensorRT, ONNX.
- Multimodal: pipeline composition.
- CPU fallback (optional).

### Precision Policies
- `fp16`, `bf16`, `int8`, `int4` supported per model spec.

### KV Cache Strategy
- Shared cache per worker, TTL-limited.
- Session affinity via request routing, no permanent GPU pinning.

---

## 9. Deployment & Zero-Downtime Upgrades

### Blue/Green & Canary
- New version deployed as parallel runtime pods.
- Traffic split via API Gateway.
- Automatic rollback on latency regression, error rate, or OOM spikes.

### Versioned Endpoints
- `/v1/models/{name}/versions/{hash}/infer`
- `/v1/models/{name}/infer` resolves to active version.

---

## 10. Autoscaling Logic (Event-Driven, No Warm Pools)

```python
def autoscale(model):
    q = queue_depth(model)
    rps = arrival_rate(model)
    slo = slo_risk(model)

    if q > 0 and (rps > MIN_RPS or slo > SLO_RISK):
        scale_up_gpu_workers(model, target=compute_target(q, rps))

    if idle_time(model) > IDLE_TIMEOUT:
        scale_down_gpu_workers(model, aggressive=True)
```

### Predictive Scaling (CPU/SSD Only)
- Uses arrival rate and model popularity to prefetch artifacts.
- Does **not** reserve GPUs; only optimizes load time.

---

## 11. Observability & SLO Metrics

- TTFT, p50/p95/p99 latency
- Queue vs compute time
- Cold start frequency
- GPU utilization, fragmentation
- Retry/failure rates
- Per-request tracing with scheduling decision introspection

---

## 12. Billing & Cost Model

- Metered per token or per second.
- GPU-type aware pricing.
- Queue time not billed (configurable).
- Retries billed only on success (optional).
- Tenant caps and throttles.

---

## 13. Developer Experience

### Declarative Model YAML Spec

```yaml
name: llama-3-8b
version: 3.1.0
runtime: vllm
precision: bf16
vram_gb: 20
artifact:
  weights_uri: s3://models/llama-3-8b/weights/
  tokenizer_uri: s3://models/llama-3-8b/tokenizer.json
  trt_engine_uri: s3://models/llama-3-8b/trt.engine
slo:
  p95_ms: 300
  ttft_ms: 150
tenant:
  max_concurrency: 4
```

### CLI Workflow
- `codex deploy model.yaml`
- `codex update model.yaml`
- `codex rollback model-name --to version-hash`

### CI/CD Hooks
- Validate model spec and artifacts.
- Canary deploy with automatic rollback thresholds.

---

## 14. Why No Warm GPUs Are Needed

Codex eliminates the need for warm GPUs by **front-loading the expensive steps on CPU/SSD**:
- Artifacts and containers are cached ahead of time without GPU reservation.
- Model weights are memory-mapped and loaded in parallel shards.
- Streaming inference masks load latency by returning tokens as soon as the first decoding step is ready.
- Micro-batching preserves throughput without significant TTFT penalty.

This keeps GPUs active only when serving live requests and releases them within seconds of idleness, achieving low latency and minimal downtime without warm pools.
