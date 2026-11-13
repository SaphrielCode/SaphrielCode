"""Strict sampled resource policy for the controlled two-pass experiment."""
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Policy:
    available_reserve_bytes: int = 2 * 1024**3
    immediate_available_bytes: int = 1024**3
    process_resident_limit_bytes: int = 4 * 1024**3
    consecutive_samples: int = 3
    sampling_interval_seconds: float = 1.0


@dataclass
class Guard:
    policy: Policy = field(default_factory=Policy)
    low_available_count: int = 0
    high_process_counts: dict = field(default_factory=dict)

    def observe(self, available_bytes, processes):
        self.low_available_count = self.low_available_count + 1 if available_bytes < self.policy.available_reserve_bytes else 0
        self.high_process_counts = {
            p["pid"]: self.high_process_counts.get(p["pid"], 0) + 1
            for p in processes if p["resident_bytes"] > self.policy.process_resident_limit_bytes}
        if available_bytes < self.policy.immediate_available_bytes:
            return "available_system_ram_below_1_GiB_first_sample"
        if self.low_available_count >= self.policy.consecutive_samples:
            return "available_system_ram_below_2_GiB_three_consecutive_samples"
        if any(count >= self.policy.consecutive_samples for count in self.high_process_counts.values()):
            return "process_resident_ram_above_4_GiB_three_consecutive_samples"
        return None

    def snapshot(self):
        return {"policy": asdict(self.policy), "low_available_count": self.low_available_count,
                "high_process_counts": self.high_process_counts}
