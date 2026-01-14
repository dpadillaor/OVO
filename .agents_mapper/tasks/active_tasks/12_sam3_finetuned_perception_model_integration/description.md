# Purpose

Integrate the SAM3 finetuned Perception model into the OVO semantic mapping system. The META Perception model repository already exists in `thirdParty/` and base models are integrated into OVO. However, SAM3 uses a finetuned version of the Perception model that needs to be extracted and integrated.

This task follows a staged approach:
1. **Research Phase**: Review the SAM3 repository in `thirdParty/` to understand how the finetuned model is loaded/used and identify the extraction method.
2. **User Review**: Present findings for user validation before proceeding.
3. **Planning Phase**: Design a smooth, structured integration plan that fits OVO's existing architecture.
4. **User Approval**: Review the integration plan with the user.
5. **Implementation Phase**: Execute the integration following the approved plan.

## Task files
- `strategy.md`: Describes the architecture and the strategy to be implemented.
- `tests.md`: Summarizes the proposed TDD tests to validate the implementation.
