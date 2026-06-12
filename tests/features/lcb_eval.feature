Feature: LiveCodeBench rescue eval
  Competition problems (LiveCodeBench hard split) where 32B-class open models
  score pass@1 = 0. Candidate programs read stdin and write stdout; a solution
  passes a problem only if it matches the expected output on every test case.
  The rescue flow is shared with the SWE suite: baseline screens first, the
  engine attempts only real failures.

  Scenario: A correct stdin program passes all cases
    Given an lcb problem that doubles each input number
    When a correct stdin solution is verified
    Then the stdin verification passes

  Scenario: A wrong stdin program fails
    Given an lcb problem that doubles each input number
    When an incorrect stdin solution is verified
    Then the stdin verification fails

  Scenario: A hanging stdin program fails by timeout
    Given an lcb problem that doubles each input number
    When a non-terminating stdin solution is verified
    Then the stdin verification fails

  Scenario: Output comparison tolerates trailing whitespace
    Given an lcb problem that doubles each input number
    When a correct solution with trailing whitespace is verified
    Then the stdin verification passes

  Scenario: The rescue flow works with stdin verification
    Given a mocked LLM where the baseline fails the doubling problem but the engine solves it
    When the lcb rescue eval runs
    Then the lcb engine attempts 1 problem
    And the lcb rescue count is 1
