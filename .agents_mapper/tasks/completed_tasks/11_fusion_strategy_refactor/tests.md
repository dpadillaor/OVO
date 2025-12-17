## TDD: Tests for the Strategy Pattern

To ensure the correct implementation of the Strategy pattern in the fusion logic, we propose a set of unit tests focused on the **infrastructure** of the pattern, not on the mathematical validity of each strategy. The goal is to guarantee that the system selects and uses the appropriate strategy in a decoupled and extensible way.

### 1. Factory Test
**Goal:** Verify that, given a configuration (string), the correct strategy class is instantiated.

- Create a test that calls a function like `create_fusion_strategy` and checks that the result is an instance of the expected class (`SemanticGeometricFusion`, `GeometricOnlyFusion`, etc.).
- This allows new strategies to be added in the future without modifying the OVO core.w

### 2. Delegation (Mocking) Test
**Goal:** Verify that the `OVO` class actually delegates the fusion decision to the selected strategy, and does not execute hardcoded logic.

- Use a **Mock** (spy double) that simulates the fusion strategy.
- Inject the Mock into `OVO` and execute the relevant method (e.g., `update_map`).
- Check that the Mock's `same_instance` method was called with the expected arguments.
- This ensures that the "wiring" of the Strategy pattern works, without depending on the internal logic of each strategy.

### 3. Interface (Contract) Test
**Goal:** Ensure that all strategies comply with the contract (they have the `same_instance` method with the correct signature).

- Check that strategy classes inherit from the abstract base class (`FusionStrategy`) or implement the required method.
- If a new strategy does not implement the method, the test should fail (ideally using `abc` to enforce this at import time).

### Notes on TDD and Mocks
- Tests should avoid instantiating the full `OVO` class if it is heavy; it is recommended to isolate the logic or mock costly dependencies.
- The use of Mocks allows checking delegation without running mathematical logic or loading heavy models.
- These tests should be written before implementing the real strategies, following the TDD cycle: **Red → Green → Refactor**.