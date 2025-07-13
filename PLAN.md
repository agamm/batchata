# Batch Processing Library Rehaul

## Project Goal
Rehaul the existing batch processing library to create a clean, simple API for running large-scale AI batch jobs across multiple providers (Anthropic, OpenAI, Gemini). The library should handle cost limits, concurrency, state persistence, and graceful error handling.

## User-Facing API
The library exposes a simple builder pattern:

```python
# Simple usage
batch = Batch(state_file="./", results_dir="./output", max_concurrent=10)
    .defaults(model="claude-3-sonnet")
    .add_cost_limit(usd=15)

for file in files:
    batch.add_job(file=file, prompt="Summarize")

run = batch.run(wait=True)
print(run.status())  # Shows progress and costs

# Advanced usage
batch = Batch(state_file="./", results_dir="./output", max_concurrent=10)
    .add_cost_limit(usd=5)
    .defaults(model="claude-3-sonnet", temperature=0.7, max_tokens=500)
    .on_progress(lambda stats: print(f"Progress: {stats.completed}/{stats.total}"))
    .add_job(file="invoice.pdf", prompt="Extract data", response_model=Invoice)
    .add_job(messages=[{"role": "user", "content": "Hello"}], model="gpt-4")

run = batch.run()
while not run.is_complete:
    run.status(print=True)
    time.sleep(5)

results = run.results()  # Dict[job_id, JobResult]
```

## Architecture

### Core Classes

**Job** - Immutable configuration for a single AI task:
```python
@dataclass
class Job:
    """Configuration for a single AI job.
    
    Either provide messages OR file+prompt, not both.
    """
    id: str  # Unique identifier
    messages: Optional[List[Dict]]  # Chat messages
    file: Optional[Path]  # File input
    prompt: Optional[str]  # Prompt for file
    model: str  # Model name (e.g., "claude-3-sonnet")
    temperature: float = 0.7
    max_tokens: int = 1000
    response_model: Optional[Type[BaseModel]] = None  # For structured output
    enable_citations: bool = False
```

**JobResult** - Result from a completed job:
```python
@dataclass
class JobResult:
    """Result from a completed AI job."""
    job_id: str
    response: str  # Raw text response
    parsed_response: Optional[Union[BaseModel, Dict]]  # Structured output or error dict
    citations: Optional[List[Citation]]  # Extracted citations
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: Optional[str] = None  # Error message if failed
```

**ModelConfig** - Provider-specific model capabilities:
```python
@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str  # e.g., "claude-3-sonnet"
    max_input_tokens: int
    max_output_tokens: int
    batch_discount: float  # e.g., 0.5 for 50% off
    supports_images: bool = False
    supports_files: bool = False  # PDFs, docs, etc.
    supports_citations: bool = False
    supports_structured_output: bool = True
    file_types: List[str] = field(default_factory=list)  # [".pdf", ".docx"]
```

**Provider (ABC)** - Interface for AI providers:
```python
class Provider(ABC):
    """Abstract base class for AI providers."""
    
    models: Dict[str, ModelConfig]  # Available models
    
    @abstractmethod
    def validate_job(self, job: Job) -> None:
        """Validate job constraints and message format.
        
        Raises:
            ValueError: If job violates provider/model constraints
        """
        pass
    
    @abstractmethod
    def create_batch(self, jobs: List[Job]) -> 'BatchRequest':
        """Create and submit a batch of jobs.
        
        Returns:
            BatchRequest with provider's batch ID
        """
        pass
    
    # ... other abstract methods
```

**ProviderRegistry** - Maps models to providers:
```python
class ProviderRegistry:
    """Registry for model to provider mapping.
    
    Automatically populated when providers are imported.
    """
    _instance = None
    _providers: Dict[str, Provider] = {}
    _model_to_provider: Dict[str, str] = {}
    
    @classmethod
    def register(cls, provider: Provider):
        """Register a provider and its models."""
        provider_name = provider.__class__.__name__
        cls._providers[provider_name] = provider
        for model_name in provider.models:
            cls._model_to_provider[model_name] = provider_name
    
    @classmethod
    def get_provider(cls, model: str) -> Provider:
        """Get provider instance for a model.
        
        Raises:
            ValueError: If model is not registered
        """
        provider_name = cls._model_to_provider.get(model)
        if not provider_name:
            raise ValueError(f"No provider found for model: {model}")
        return cls._providers[provider_name]
```

**BatchRequest** - Represents a batch submitted to a provider:
```python
class BatchRequest:
    """A batch of jobs submitted to a provider."""
    id: str  # Provider's batch ID
    provider: Provider
    jobs: List[Job]
    status: str  # "pending", "running", "complete", "failed"
    submitted_at: datetime
    
    def get_status(self) -> str:
        """Update and return current status."""
        pass
    
    def get_results(self) -> List[JobResult]:
        """Retrieve results when complete."""
        pass
```

**Batch** - User-facing configuration builder:
```python
class Batch:
    """Builder for batch job configuration."""
    
    def __init__(self, state_file: str, results_dir: str, max_concurrent: int):
        """Initialize batch configuration."""
        pass
    
    def defaults(self, **kwargs) -> 'Batch':
        """Set default parameters for all jobs."""
        pass
    
    def add_job(self, **kwargs) -> 'Batch':
        """Add a job to the batch."""
        pass
    
    def run(self, wait: bool = False) -> 'BatchRun':
        """Execute the batch."""
        pass
```

**BatchRun** - Manages execution of multiple batches:
```python
class BatchRun:
    """Manages the execution of a batch job.
    
    Supports graceful shutdown via signal handlers.
    """
    
    def __init__(self, ...):
        self._shutdown_event = threading.Event()
        self._setup_signal_handlers()
    
    @property
    def is_complete(self) -> bool:
        """Whether all jobs are complete."""
        pass
    
    def status(self, print: bool = False) -> Dict:
        """Get current execution statistics."""
        pass
    
    def results(self) -> Dict[str, JobResult]:
        """Get all completed results."""
        pass
    
    def shutdown(self, wait_for_active: bool = True):
        """Gracefully shutdown the batch run.
        
        Args:
            wait_for_active: If True, wait for active batches to complete
        """
        pass
```

### Internal Classes

**ConcurrentExecutor** - Manages parallel batch submissions:
```python
class ConcurrentExecutor:
    """Manages concurrent execution with cost and rate limits.
    
    Uses ThreadPoolExecutor internally to manage parallel batch submissions
    while respecting max_concurrent limit and checking cost limits before
    each submission.
    """
    
    def __init__(self, max_concurrent: int, cost_limit_usd: Optional[float]):
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.cost_tracker = CostTracker(cost_limit_usd)
        self.active_futures: Dict[Future, BatchRequest] = {}
        self._shutdown = False
    
    def submit_if_allowed(self, provider: Provider, jobs: List[Job]) -> Optional[Future]:
        """Submit jobs if cost limit allows."""
        if self._shutdown:
            return None
            
        estimated_cost = provider.estimate_cost(jobs)
        if self.cost_tracker.can_proceed(estimated_cost):
            future = self.executor.submit(provider.create_batch, jobs)
            return future
        return None
    
    def get_completed(self) -> List[Tuple[BatchRequest, List[JobResult]]]:
        """Check for completed batches and return their results."""
        pass
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self._shutdown = True
        self.executor.shutdown(wait=wait)
```

**CostTracker** - Tracks costs and enforces limits:
```python
class CostTracker:
    """Tracks cumulative costs and enforces limits."""
    
    def __init__(self, limit_usd: Optional[float]):
        self.limit_usd = limit_usd
        self.used_usd = 0.0
        self._lock = threading.Lock()
    
    def can_proceed(self, estimated_cost_usd: float) -> bool:
        """Check if we can proceed with given cost."""
        pass
    
    def track(self, actual_cost_usd: float):
        """Track actual cost after completion."""
        pass
```

## Testing Requirements

### Mocking Strategy
Every major component must be easily mockable for unit testing:

```python
# providers/mock.py
class MockProvider(Provider):
    """Mock provider for testing.
    
    Allows configuration of responses, delays, and failures.
    """
    def __init__(self):
        super().__init__()
        self.mock_responses: Dict[str, List[JobResult]] = {}
        self.mock_delays: Dict[str, float] = {}
        self.mock_failures: Dict[str, Exception] = {}
    
    def set_mock_response(self, job_id: str, result: JobResult):
        """Configure mock response for a job."""
        pass

# Example test
def test_batch_run_with_mock():
    mock_provider = MockProvider()
    mock_provider.set_mock_response("job_1", JobResult(...))
    
    # Override registry for testing
    with mock.patch.object(ProviderRegistry, 'get_provider', return_value=mock_provider):
        batch = Batch(...)
        run = batch.run(wait=True)
        assert run.results()["job_1"].cost_usd == 0.50
```

### Mock Interfaces
- **MockProvider**: Simulates provider behavior with configurable responses
- **MockBatchRequest**: Simulates batch lifecycle with controllable status transitions
- **MockCostTracker**: Allows testing cost limit scenarios
- **MockStateManager**: Tests state persistence without file I/O

## Implementation Guidelines

### Code Style
- **Keep it simple and readable** - Avoid clever tricks or overly complex abstractions
- **Don't write unnecessary code** - Only implement what's needed for the API
- **Use descriptive names** - `cost_usd` not just `cost`, `is_complete` not `done`
- **Fail fast with clear errors** - Validate early and provide helpful error messages

### Documentation
Every class and method must have comprehensive pydoc documentation:

```python
def create_batch(self, jobs: List[Job]) -> BatchRequest:
    """Create and submit a batch of jobs to the provider.
    
    Args:
        jobs: List of jobs to include in the batch. All jobs must
            use models available from this provider.
    
    Returns:
        BatchRequest object with the provider's batch ID.
        
    Raises:
        ValueError: If any job uses an unsupported model.
        ProviderError: If batch submission fails.
        
    Example:
        >>> jobs = [Job(messages=[...], model="claude-3-sonnet")]
        >>> batch_req = provider.create_batch(jobs)
        >>> print(batch_req.id)
        'batch_abc123'
    """
```

### State Management
- Save state after each batch submission to `state_file`
- State should include: pending jobs, active batch IDs, completed results
- Must be able to resume from any point if process crashes
- Use JSON for state serialization

### Error Handling
- Individual job failures don't fail the entire batch
- Network errors trigger retries with exponential backoff
- Cost limit exceeded stops new submissions but lets active batches complete
- Provider errors are wrapped in consistent exception types

### Graceful Shutdown
- Handle SIGINT and SIGTERM signals
- Stop accepting new jobs immediately
- Optionally wait for active batches to complete
- Save current state before exiting
- Clean up resources (threads, temp files)

### Provider Implementation
Each provider (Anthropic, OpenAI, Gemini) should:
1. Validate jobs against model capabilities
2. Format messages according to provider's API
3. Handle structured output instructions
4. Parse responses safely (catch parsing errors)
5. Extract token counts and calculate costs
6. Auto-register with ProviderRegistry on import

### File Structure
```
/
├── core/
│   ├── __init__.py
│   ├── job.py          # Job, JobResult classes
│   ├── batch.py        # Batch configuration builder
│   ├── run.py          # BatchRun execution manager
│   └── executor.py     # ConcurrentExecutor
├── providers/
│   ├── __init__.py
│   ├── base.py         # Provider ABC, ModelConfig, ProviderRegistry
│   ├── anthropic.py    # AnthropicProvider
│   ├── openai.py       # OpenAIProvider
│   ├── gemini.py       # GeminiProvider
│   └── mock.py         # MockProvider for testing
├── utils/
│   ├── __init__.py
│   ├── cost.py         # CostTracker
│   └── state.py        # State persistence
├── exceptions.py       # All custom exceptions
├── types.py           # Type definitions, Citation, etc.
└── tests/
    ├── __init__.py
    ├── test_job.py
    ├── test_batch.py
    ├── test_providers.py
    └── test_integration.py
```

## Implementation Steps

### Step 1: Core Data Models (Commit 1)
**Files:** `types.py`, `exceptions.py`, `core/job.py`
- Define `Job` and `JobResult` dataclasses
- Define all custom exceptions
- Define type aliases and common types (Citation, etc.)
- Full test coverage for data models
- **Deliverable:** Can create and validate job configurations

### Step 2: Provider Framework (Commit 2)
**Files:** `providers/base.py`, `providers/mock.py`
- Implement `Provider` ABC with all abstract methods
- Implement `ModelConfig` dataclass
- Implement `ProviderRegistry` singleton
- Create `MockProvider` for testing
- Full test coverage for provider framework
- **Deliverable:** Provider interface defined and mockable

### Step 3: Batch Configuration (Commit 3)
**Files:** `core/batch.py`
- Implement `Batch` builder class
- Implement `BatchConfig` dataclass
- Handle defaults and job accumulation
- Full test coverage for configuration building
- **Deliverable:** Can build batch configurations with fluent API

### Step 4: Cost and State Management (Commit 4)
**Files:** `utils/cost.py`, `utils/state.py`
- Implement thread-safe `CostTracker`
- Implement `StateManager` with JSON serialization
- Add resume capability from saved state
- Full test coverage for both components
- **Deliverable:** Cost tracking and state persistence working

### Step 5: Concurrent Executor (Commit 5)
**Files:** `core/executor.py`
- Implement `ConcurrentExecutor` with ThreadPoolExecutor
- Handle max concurrent limits
- Integrate cost checking before submission
- Add graceful shutdown support
- Full test coverage with mock provider
- **Deliverable:** Concurrent execution framework complete

### Step 6: Batch Request and Run (Commit 6)
**Files:** `providers/base.py` (update), `core/run.py`
- Implement `BatchRequest` class
- Implement `BatchRun` with full execution logic
- Add signal handlers for graceful shutdown
- Integrate all components (executor, state, cost)
- Full test coverage for execution flow
- **Deliverable:** Complete execution pipeline working

### Step 7: First Provider - Anthropic (Commit 7)
**Files:** `providers/anthropic.py`
- Implement `AnthropicProvider`
- Handle message formatting for Claude
- Implement structured output preparation
- Parse responses and extract tokens/costs
- Full test coverage with mocked API calls
- **Deliverable:** Anthropic provider fully functional

### Step 8: Additional Providers (Commit 8)
**Files:** `providers/openai.py`, `providers/gemini.py`
- Implement `OpenAIProvider`
- Implement `GeminiProvider`
- Ensure consistent behavior across providers
- Full test coverage for each provider
- **Deliverable:** All providers implemented

### Step 9: Integration and Polish (Commit 9)
**Files:** `__init__.py` files, README.md
- Add proper exports in `__init__.py` files
- Create comprehensive integration tests
- Add end-to-end examples
- Write README with usage examples
- Performance testing with mock provider
- **Deliverable:** Production-ready library

### Step 10: Advanced Features (Commit 10)
**Files:** Various updates
- Add retry logic with exponential backoff
- Add progress callbacks with detailed stats
- Add result streaming for long-running batches
- Add batch result export formats (CSV, JSON)
- **Deliverable:** Feature-complete library

## Key Requirements
1. The user-facing API (`Batch` and `BatchRun`) must remain simple and intuitive
2. Cost tracking must be accurate and respect limits
3. Concurrent execution must respect max_concurrent limit
4. State persistence enables resuming long-running batches
5. Each provider handles its own API peculiarities internally
6. All costs must be in USD with clear `_usd` suffix
7. Token counts must be tracked for transparency
8. Every component must be easily mockable for testing
9. Provider resolution from model name must be automatic
10. Graceful shutdown must save state and clean up resources

Remember: A senior engineer should look at this code and think "this is clean, well-documented, easy to test, and easy to maintain."