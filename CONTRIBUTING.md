# Contributing to Monocle
Contributions to Monocle are welcome from everyone. The following are a set of guidelines for contributing to Monocle. Following these guidelines makes contributing to this project easy and transparent. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request. Please refer to our code of conduct when making contributions. 

## How can you contribute?
### Contributing code
If you encounter a bug, you can
- (Recommended) File an issue about the bug.
- Provide clear and concrete ways/scripts to reproduce the bug.
- Provide possible solutions for the bug.
- Pull a request to fix the bug.

If you're interested in existing issues, you can
- (Recommended) Provide answers for issue labeled question.
- Provide help for issues labeled bug, improvement, and enhancement by
-- (Recommended) Ask questions, reproduce the issue, or provide solutions.
-- Pull a request to fix the issue.

If you require a new feature or major enhancement, you can
- (Recommended) File an issue about the feature/enhancement with reasons.
- Pull a request to implement the feature.

If you are a reviewer/approver of Monocle, you can
- Participate in PR review process.
- Instruct newcomers in the community to complete the PR process.

If you want to become a contributor of Monocle, submit your pull requests. All contributions to this project must be accompanied by acknowledgment of, and agreement to, the [Developer Certificate of Origin](https://github.com/apps/dco). All submissions will be reviewed as quickly as possible.

## Pull Request Process
### CI/CD Workflow Approval
For security reasons, GitHub requires manual approval for workflows on pull requests from forked repositories:
- After submitting your PR, automated tests may show "action_required" status
- A maintainer with write access will review and approve the workflow to run
- This is a standard security measure to prevent malicious code execution
- Once approved, all unit and integration tests will run automatically

### PR Status Checks
Before your PR can be merged, it must:
- Pass all unit tests in `apptrace/tests/unit/`
- Pass all integration tests in `apptrace/tests/integration/`
- Be marked as "Ready for review" (not in draft status)
- Receive approval from at least one maintainer
- Have all review comments addressed

### Testing Your Changes
To run tests locally before submitting:
```bash
cd apptrace
pip install -e '.[dev]'
export PYTHONPATH=./src:./tests
pytest tests/unit/
```

 
