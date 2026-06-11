"""HumanEval-style code problems with ground-truth unit tests.

Each problem is a Python function specification; verification executes the
candidate implementation against the asserts, so scoring needs no LLM judge.
Problems are adapted from the OpenAI HumanEval benchmark (MIT license).
"""

from pydantic import BaseModel, Field


class CodeProblem(BaseModel):
    """One code-generation problem with executable acceptance tests."""

    id: str = Field(..., description="Short slug identifying the problem.")
    prompt: str = Field(..., description="Function stub with docstring.")
    entry_point: str = Field(..., description="Name of the required function.")
    tests: str = Field(..., description="Assert-based acceptance tests.")


SWE_PROBLEMS = [
    CodeProblem(
        id="has-close-elements",
        entry_point="has_close_elements",
        prompt='''def has_close_elements(numbers: list[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each
    other than the given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
        tests="""assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) is True
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) is False
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) is True
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) is False
assert has_close_elements([1.0], 1.0) is False
assert has_close_elements([], 1.0) is False
""",
    ),
    CodeProblem(
        id="separate-paren-groups",
        entry_point="separate_paren_groups",
        prompt='''def separate_paren_groups(paren_string: str) -> list[str]:
    """Input to this function is a string containing multiple groups of nested
    parentheses. Your goal is to separate those groups into separate strings
    and return the list of those. Separate groups are balanced (each open
    brace is properly closed) and not nested within each other. Ignore any
    spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
''',
        tests="""assert separate_paren_groups('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
assert separate_paren_groups('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
assert separate_paren_groups('(()(())((())))') == ['(()(())((())))']
assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
""",
    ),
    CodeProblem(
        id="below-zero",
        entry_point="below_zero",
        prompt='''def below_zero(operations: list[int]) -> bool:
    """You're given a list of deposit and withdrawal operations on a bank
    account that starts with zero balance. Your task is to detect if at any
    point the balance of account falls below zero, and at that point function
    should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
''',
        tests="""assert below_zero([]) is False
assert below_zero([1, 2, -3, 1, 2, -3]) is False
assert below_zero([1, 2, -4, 5, 6]) is True
assert below_zero([1, -1, 2, -2, 5, -5, 4, -4]) is False
assert below_zero([1, -1, 2, -2, 5, -5, 4, -5]) is True
""",
    ),
    CodeProblem(
        id="make-palindrome",
        entry_point="make_palindrome",
        prompt='''def make_palindrome(string: str) -> str:
    """Find the shortest palindrome that begins with a supplied string.
    Algorithm idea is simple:
    - Find the longest postfix of supplied string that is a palindrome.
    - Append to the end of the string reverse of a string prefix that comes
      before the palindromic suffix.
    >>> make_palindrome('')
    ''
    >>> make_palindrome('cat')
    'catac'
    >>> make_palindrome('cata')
    'catac'
    """
''',
        tests="""assert make_palindrome('') == ''
assert make_palindrome('x') == 'x'
assert make_palindrome('xyz') == 'xyzyx'
assert make_palindrome('xyx') == 'xyx'
assert make_palindrome('jerry') == 'jerryrrej'
""",
    ),
    CodeProblem(
        id="remove-duplicates",
        entry_point="remove_duplicates",
        prompt='''def remove_duplicates(numbers: list[int]) -> list[int]:
    """From a list of integers, remove all elements that occur more than once.
    Keep order of elements left the same as in the input.
    >>> remove_duplicates([1, 2, 3, 2, 4])
    [1, 3, 4]
    """
''',
        tests="""assert remove_duplicates([]) == []
assert remove_duplicates([1, 2, 3, 4]) == [1, 2, 3, 4]
assert remove_duplicates([1, 2, 3, 2, 4, 3, 5]) == [1, 4, 5]
assert remove_duplicates([1, 1, 1]) == []
""",
    ),
    CodeProblem(
        id="is-prime",
        entry_point="is_prime",
        prompt='''def is_prime(n: int) -> bool:
    """Return true if a given number is prime, and false otherwise.
    >>> is_prime(6)
    False
    >>> is_prime(101)
    True
    >>> is_prime(11)
    True
    >>> is_prime(1)
    False
    """
''',
        tests="""assert is_prime(6) is False
assert is_prime(101) is True
assert is_prime(11) is True
assert is_prime(13441) is True
assert is_prime(61) is True
assert is_prime(4) is False
assert is_prime(1) is False
assert is_prime(0) is False
assert is_prime(-7) is False
""",
    ),
    CodeProblem(
        id="prime-fib",
        entry_point="prime_fib",
        prompt='''def prime_fib(n: int) -> int:
    """prime_fib returns the n-th number that is a Fibonacci number and it's
    also prime.
    >>> prime_fib(1)
    2
    >>> prime_fib(2)
    3
    >>> prime_fib(3)
    5
    >>> prime_fib(4)
    13
    >>> prime_fib(5)
    89
    """
''',
        tests="""assert prime_fib(1) == 2
assert prime_fib(2) == 3
assert prime_fib(3) == 5
assert prime_fib(4) == 13
assert prime_fib(5) == 89
assert prime_fib(6) == 233
assert prime_fib(7) == 1597
""",
    ),
    CodeProblem(
        id="longest",
        entry_point="longest",
        prompt='''def longest(strings: list[str]) -> str | None:
    """Out of list of strings, return the longest one. Return the first one in
    case of multiple strings of the same length. Return None in case the input
    list is empty.
    >>> longest([])

    >>> longest(['a', 'b', 'c'])
    'a'
    >>> longest(['a', 'bb', 'ccc'])
    'ccc'
    """
''',
        tests="""assert longest([]) is None
assert longest(['x', 'y', 'z']) == 'x'
assert longest(['x', 'yyy', 'zzzz', 'www', 'kkkk', 'abc']) == 'zzzz'
""",
    ),
]
