import json
import random
from pathlib import Path

OUTPUT_PATH = Path("data/train.jsonl")
NUM_SAMPLES = 2000
MAX_FLOWS = 5
MU_MIN = 0.05


def normalize_mu(values):
    total = sum(values)
    values = [v / total for v in values]

    # make sure no value is below MU_MIN
    values = [max(v, MU_MIN) for v in values]
    total = sum(values)
    values = [round(v / total, 4) for v in values]

    return values


def generate_mu(avg_congestion, min_lifetime, privacy_fraction):
    """
    mu = [M1, M2, M3, M4, M5]
    M1 = connection
    M2 = lifetime/latency quality
    M3 = QoS
    M4 = load balance
    M5 = privacy
    """

    m1 = 0.20
    m2 = 0.20
    m3 = 0.20
    m4 = 0.20
    m5 = 0.20

    if min_lifetime < 0.4:
        m1 += 0.20

    if avg_congestion > 0.6:
        m4 += 0.20
        m3 += 0.10

    if privacy_fraction > 0.4:
        m5 += 0.25

    return normalize_mu([m1, m2, m3, m4, m5])


def generate_privacy(qos, congestion, rwnd):
    privacy = 0.2

    if qos >= 2:
        privacy += 0.2

    if congestion > 0.6:
        privacy += 0.1

    if rwnd < 0.4:
        privacy += 0.1

    privacy += random.uniform(-0.1, 0.1)

    return round(min(max(privacy, 0.0), 1.0), 3)


def generate_sample(sample_id):
    num_flows = random.randint(1, MAX_FLOWS)

    flows = []
    privacy_values = {}

    lifetimes = []
    congestions = []
    rwnds = []
    privacy_count = 0

    for flow_id in range(1, num_flows + 1):
        qos = random.randint(0, 3)
        lifetime_ratio = round(random.uniform(0.2, 1.5), 3)
        congestion = round(random.uniform(0.0, 1.0), 3)
        rwnd = round(random.uniform(0.1, 1.0), 3)

        privacy = generate_privacy(qos, congestion, rwnd)

        if privacy > 0.5:
            privacy_count += 1

        flows.append({
            "flow_id": flow_id,
            "qos": qos,
            "lifetime_ratio": lifetime_ratio,
            "congestion": congestion,
            "rwnd": rwnd
        })

        privacy_values[str(flow_id)] = privacy
        lifetimes.append(lifetime_ratio)
        congestions.append(congestion)
        rwnds.append(rwnd)

    avg_congestion = round(sum(congestions) / len(congestions), 3)
    min_lifetime = round(min(lifetimes), 3)
    avg_rwnd = round(sum(rwnds) / len(rwnds), 3)
    privacy_fraction = round(privacy_count / num_flows, 3)

    mu = generate_mu(avg_congestion, min_lifetime, privacy_fraction)

    return {
        "input": {
            "sample_id": sample_id,
            "num_flows": num_flows,
            "avg_congestion": avg_congestion,
            "min_lifetime": min_lifetime,
            "avg_rwnd": avg_rwnd,
            "privacy_fraction": privacy_fraction,
            "flows": flows
        },
        "output": {
            "mu": mu,
            "privacy": privacy_values
        }
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i in range(NUM_SAMPLES):
            sample = generate_sample(i + 1)
            f.write(json.dumps(sample) + "\n")

    print(f"Dataset generated: {OUTPUT_PATH}")
    print(f"Samples: {NUM_SAMPLES}")


if __name__ == "__main__":
    main()