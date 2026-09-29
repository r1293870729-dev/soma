"""Fail-closed checks; metadata does not substitute for actual access controls."""
def assert_disjoint_roles(roles: dict[str, list[str]]) -> None:
    seen = {}
    for role, ids in roles.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample ID inside role {role}")
        for sample in ids:
            if sample in seen:
                raise ValueError(f"Role overlap: {sample}: {seen[sample]} / {role}")
            seen[sample] = role

def assert_arrived(sample_arrival_steps: list[int], current_step: int) -> None:
    if current_step < 0 or any(t < 0 or t > current_step for t in sample_arrival_steps):
        raise ValueError("Future/unarrived data cannot enter a candidate branch")
