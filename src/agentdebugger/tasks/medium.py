"""Medium task — a red herring in an authentication module.

Four of ten tests fail, and every failure message names ``authenticate_user``.
``authenticate_user`` is correct. The bug is one call down the stack in
``hash_password``, which wraps its hex digest in ``str(bytes(...))`` and so
returns ``"b'5f4dcc…'"`` instead of ``"5f4dcc…"``.

The task measures whether an agent traces a symptom back to its cause or stops
at the first frame the error mentions. The grader scores the hypothesis
accordingly: blaming only ``authenticate_user`` earns nothing.
"""

from agentdebugger.tasks.harness import build_test_runner
from agentdebugger.tasks.models import GroundTruth, Task

BUGGY_CODE = '''import hashlib


def hash_password(password: str) -> str:
    """Hash a password using MD5 and return the hex digest string."""
    password_bytes = password.encode('utf-8')
    hash_obj = hashlib.md5(password_bytes)
    hex_digest = hash_obj.hexdigest()
    return str(bytes(hex_digest, 'ascii'))


def validate_password(password: str, stored_hash: str) -> bool:
    """Check if password matches the stored hash."""
    computed_hash = hash_password(password)
    return computed_hash == stored_hash


def authenticate_user(username: str, password: str, user_db: dict) -> bool:
    """Return True if the user exists, is active, and the password matches."""
    if username not in user_db:
        return False
    user = user_db[username]
    if not user.get('active', False):
        return False
    return validate_password(password, user['password_hash'])
'''

FIXED_CODE = '''import hashlib


def hash_password(password: str) -> str:
    """Hash a password using MD5 and return the hex digest string."""
    password_bytes = password.encode('utf-8')
    hash_obj = hashlib.md5(password_bytes)
    return hash_obj.hexdigest()


def validate_password(password: str, stored_hash: str) -> bool:
    """Check if password matches the stored hash."""
    computed_hash = hash_password(password)
    return computed_hash == stored_hash


def authenticate_user(username: str, password: str, user_db: dict) -> bool:
    """Return True if the user exists, is active, and the password matches."""
    if username not in user_db:
        return False
    user = user_db[username]
    if not user.get('active', False):
        return False
    return validate_password(password, user['password_hash'])
'''

TEST_SUITE = '''import hashlib


def _make_hash(password):
    """How the registration system stores passwords: hexdigest, directly."""
    return hashlib.md5(password.encode('utf-8')).hexdigest()


def _build_user_db():
    return {
        'alice': {'password_hash': _make_hash('password123'), 'active': True},
        'bob': {'password_hash': _make_hash('securepass'), 'active': True},
        'charlie': {'password_hash': _make_hash('charlie_pw'), 'active': False},
        'diana': {'password_hash': _make_hash('d1@n@_pass'), 'active': True},
    }


def test_hash_returns_string():
    assert isinstance(hash_password("test"), str)

def test_hash_deterministic():
    assert hash_password("same_input") == hash_password("same_input")

def test_hash_different_inputs():
    assert hash_password("password1") != hash_password("password2")

def test_unknown_user_rejected():
    assert authenticate_user('unknown', 'password123', _build_user_db()) is False

def test_inactive_user_rejected():
    assert authenticate_user('charlie', 'charlie_pw', _build_user_db()) is False

def test_wrong_password_rejected():
    assert authenticate_user('alice', 'wrong_password', _build_user_db()) is False

def test_alice_correct_password():
    result = authenticate_user('alice', 'password123', _build_user_db())
    assert result is True, "authenticate_user('alice', 'password123') returned %r" % (result,)

def test_bob_correct_password():
    result = authenticate_user('bob', 'securepass', _build_user_db())
    assert result is True, "authenticate_user('bob', 'securepass') returned %r" % (result,)

def test_diana_correct_password():
    result = authenticate_user('diana', 'd1@n@_pass', _build_user_db())
    assert result is True, "authenticate_user('diana', 'd1@n@_pass') returned %r" % (result,)

def test_validate_password_direct():
    stored = _make_hash('mypassword')
    result = validate_password('mypassword', stored)
    assert result is True, "validate_password with the correct password returned %r" % (result,)
'''

TEST_NAMES = (
    "test_hash_returns_string",
    "test_hash_deterministic",
    "test_hash_different_inputs",
    "test_unknown_user_rejected",
    "test_inactive_user_rejected",
    "test_wrong_password_rejected",
    "test_alice_correct_password",
    "test_bob_correct_password",
    "test_diana_correct_password",
    "test_validate_password_direct",
)

TASK = Task(
    task_id="medium",
    name="Red herring in an authentication module",
    difficulty="medium",
    description=(
        "An authentication module hashes passwords with MD5, validates them by comparing "
        "hashes, and authenticates users against a user database. Four tests fail: "
        "authenticate_user returns False for users whose password is correct. "
        "Find the root cause, state a hypothesis, and fix it."
    ),
    buggy_code=BUGGY_CODE,
    test_suite=TEST_SUITE,
    test_runner=build_test_runner(TEST_NAMES),
    tests_total=len(TEST_NAMES),
    max_attempts=7,
    max_steps=15,
    ground_truth=GroundTruth(
        bug_location="hash_password",
        bug_type="bytes_str_conversion",
        hypothesis_keywords=("hash_password", "bytes", "str(", "hexdigest", "encoding", "b'"),
        fixed_code=FIXED_CODE,
        red_herring_keyword="authenticate_user",
        reference_hypothesis=(
            "authenticate_user is correct; the error message names it only because it is "
            "where the False surfaces. The bug is in hash_password on line 8, which wraps "
            "its hex digest in str(bytes(...)) and so returns the literal text \"b'5f4dcc…'\" "
            "instead of the digest \"5f4dcc…\". Stored hashes are written by hexdigest() "
            "directly, so the comparison in validate_password can never match. Returning "
            "hash_obj.hexdigest() unwrapped fixes all four failures."
        ),
    ),
)
