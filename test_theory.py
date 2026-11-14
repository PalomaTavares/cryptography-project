from theory import rosenfeld


cases = [
    (0.10, 1, 0.20),
    (0.10, 2, 0.05600),
    (0.10, 5, 0.00178),
    (0.20, 3, 0.11584),
    (0.20, 6, 0.02331),
    (0.30, 4, 0.25207),
    (0.30, 10, 0.06511),
    (0.40, 5, 0.53314),
    (0.40, 10, 0.37218),
    (0.50, 3, 1.0),
]

for q, n, expected in cases:
    got = rosenfeld(q, n)
    assert abs(got - expected) < 1e-5, f"q={q}, n={n}: got {got}, expected {expected}"

print("All Rosenfeld tests passed")
