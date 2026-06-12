Feature: LLM call resilience
  An engine run is hundreds of sequential LLM calls; the runtime retries
  transient failures with exponential backoff instead of letting a single
  network error destroy the whole run. Persistent failures still surface.

  Scenario: Transient LLM failures are retried instead of killing the run
    Given an LLM transport that fails 2 times before succeeding
    When an agent call runs through the runtime
    Then the call succeeds after 3 attempts

  Scenario: A persistent LLM failure still surfaces
    Given an LLM transport that always fails
    When an agent call runs through the runtime
    Then the call fails after exhausting 3 attempts

  Scenario: A rate-limited call waits out the quota window instead of burning retries
    Given an LLM transport that is rate limited once before succeeding
    When an agent call runs through the runtime
    Then the call succeeds after 2 attempts
    And the retry waited at least 45 seconds

  Scenario: Rate limits do not consume the transient-failure retry budget
    Given an LLM transport that is rate limited 5 times before succeeding
    When an agent call runs through the runtime
    Then the call succeeds after 6 attempts

  Scenario: Concurrent calls can be capped for tightly-quota'd backends
    Given the runtime concurrency cap is 1
    When 4 agent calls run concurrently through the runtime
    Then no more than 1 call was in flight at once
