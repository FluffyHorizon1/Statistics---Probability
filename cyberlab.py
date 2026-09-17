"""cyberlab.py -- one synthetic security dataset for the whole of CITY2161.

Every workshop, every staff-guide program and every number quoted in the
lecture notes is computed from this file, so a learner can reproduce any
figure in the notes by editing a program they already have open.

Why synthetic?  A public benchmark cannot give two things this module needs:
it is not reproducible to the digit across library versions, and nobody knows
its ground truth.  Here we do.  `make_flows(return_truth=True)` also returns
what really happened in each host-hour, which is how Topic 10 measures label
noise: the labelling detector misses about 15% of attack hours, exactly as a
real alert-derived label would.

Views
-----
make_flows()            raw flows + alerts          (Topic 2 SQL work)
host_hour_matrix()      one row per host-hour       (Topics 1, 3-10)

Columns in the host-hour matrix
-------------------------------
FEATURES  n_flows, n_distinct_ports, n_distinct_dsts, total_bytes,
          mean_duration, failed_logins           (numeric)
          top_service                            (categorical)
LABEL     label          1 if a TRUE alert fired in the host-hour
METADATA  n_alerts       alert count -- derived from the label's source, so
                         it LEAKS; Topics 2 and 9 use it to show that
          attack_family  portscan / exfil / bruteforce / benign, read from the
                         alert signature -- metadata for held-out-family tests,
                         never a feature
          true_kind      (only when truth is passed) what the host really did,
                         alerted or not -- for measuring label noise
"""
import numpy as np
import pandas as pd

NUM = ["n_flows", "n_distinct_ports", "n_distinct_dsts", "total_bytes",
       "mean_duration", "failed_logins"]
CAT = ["top_service"]
FAMILY = {1: "portscan", 2: "exfil", 3: "bruteforce"}


def make_flows(seed=42, n_hosts=60, n_hours=120, attack_frac=0.08, return_truth=False):
    """Return (flows, alerts) DataFrames, or (flows, alerts, truth) when
    return_truth=True.  One latent behaviour per host-hour drives the traffic;
    attacks are slightly more frequent later (a trend Topic 9 exploits for the
    temporal split and drift work)."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2024-03-01", periods=n_hours, freq="h")
    flow_rows, alert_rows, truth_rows, fid = [], [], [], 0
    for host in range(n_hosts):
        host_id = f"h{host:03d}"
        for hi, ts in enumerate(hours):
            p = attack_frac * (0.6 + 0.8 * hi / n_hours)   # rising trend
            kind = "benign"
            if rng.random() < p:
                kind = rng.choice(["portscan", "exfil", "bruteforce"],
                                  p=[0.5, 0.25, 0.25])
            if kind == "benign":
                quirk = rng.random()
                if quirk < 0.04:            # nightly backup: looks like exfil
                    n = 1 + rng.poisson(4)
                    ports = rng.choice([443, 22], size=n)
                    dsts = rng.integers(0, 2, size=n)
                    bytes_ = rng.lognormal(12.5, 0.9, size=n).astype(int)
                    dur = rng.lognormal(2.5, 0.8, size=n)
                    fail = np.zeros(n, int)
                elif quirk < 0.08:          # busy admin: looks like a scan
                    n = 20 + rng.poisson(30)
                    ports = rng.integers(1, 400, size=n)
                    dsts = rng.integers(0, 40, size=n)
                    bytes_ = rng.lognormal(7.5, 1.0, size=n).astype(int)
                    dur = rng.lognormal(-1.0, 1.0, size=n)
                    fail = np.zeros(n, int)
                else:                       # ordinary traffic
                    n = 1 + rng.poisson(rng.gamma(2.0, 6.0))
                    ports = rng.choice([80, 443, 22, 53, 25], size=n,
                                       p=[0.4, 0.3, 0.08, 0.15, 0.07])
                    dsts = rng.integers(0, 8, size=n)
                    bytes_ = rng.lognormal(7.5, 1.3, size=n).astype(int)
                    dur = rng.lognormal(-1.0, 1.0, size=n)
                    fail = (rng.random(n) < 0.01).astype(int)
            elif kind == "portscan":
                inten = rng.uniform(0.3, 1.0)               # some scans stealthy
                n = 10 + rng.poisson(int(150 * inten))
                ports = rng.integers(1, int(64 + 960 * inten), size=n)
                dsts = rng.integers(0, int(10 + 190 * inten), size=n)
                bytes_ = rng.lognormal(4.0, 0.6, size=n).astype(int)
                dur = rng.lognormal(-3.0, 0.5, size=n)
                fail = np.zeros(n, int)
            elif kind == "exfil":
                inten = rng.uniform(0.4, 1.0)               # low-and-slow vs bulk
                n = 1 + rng.poisson(3)
                ports = rng.choice([443, 22], size=n)
                dsts = np.zeros(n, int)
                bytes_ = rng.lognormal(11.0 + 3.0 * inten, 0.8, size=n).astype(int)
                dur = rng.lognormal(3.0, 0.8, size=n)
                fail = np.zeros(n, int)
            else:  # bruteforce
                inten = rng.uniform(0.3, 1.0)
                n = 8 + rng.poisson(int(70 * inten))
                ports = np.full(n, 22)
                dsts = rng.integers(0, 3, size=n)
                bytes_ = rng.lognormal(5.0, 0.5, size=n).astype(int)
                dur = rng.lognormal(-1.5, 0.5, size=n)
                fail = (rng.random(n) < 0.6 + 0.35 * inten).astype(int)
            svc = {80: "http", 443: "http", 53: "dns", 22: "ssh", 25: "smtp"}
            for j in range(n):
                port = int(ports[j])
                proto = "udp" if port == 53 else "tcp"
                flow_rows.append((fid, host_id, ts,
                    f"10.0.{dsts[j] // 256}.{dsts[j] % 256}", port, proto,
                    svc.get(port, "other"), int(bytes_[j]), float(dur[j]),
                    int(fail[j])))
                fid += 1
            fired = kind != "benign" and rng.random() < 0.85
            if kind == "benign" and rng.random() < 0.02:   # false-positive alert
                fired = True
            if fired:
                sig = {"portscan": 1, "exfil": 2, "bruteforce": 3, "benign": 4}[kind]
                alert_rows.append((len(alert_rows), host_id, ts, sig,
                    "high" if kind != "benign" else "low",
                    1 if kind != "benign" else 0))
            truth_rows.append((host_id, ts, str(kind)))
    flows = pd.DataFrame(flow_rows, columns=["flow_id", "host_id", "ts",
        "dst_ip", "dst_port", "protocol", "service", "bytes", "duration",
        "is_failed_login"])
    alerts = pd.DataFrame(alert_rows, columns=["alert_id", "host_id", "ts",
        "signature_id", "severity", "is_true_attack"])
    if return_truth:
        truth = pd.DataFrame(truth_rows, columns=["host_id", "ts", "true_kind"])
        return flows, alerts, truth
    return flows, alerts


def host_hour_matrix(flows, alerts, truth=None):
    """Aggregate flows to one row per (host_id, hour) with a binary label."""
    g = flows.groupby(["host_id", "ts"])
    X = g.agg(n_flows=("flow_id", "size"),
              n_distinct_ports=("dst_port", "nunique"),
              n_distinct_dsts=("dst_ip", "nunique"),
              total_bytes=("bytes", "sum"),
              mean_duration=("duration", "mean"),
              failed_logins=("is_failed_login", "sum")).reset_index()
    top_svc = g["service"].agg(lambda s: s.value_counts().index[0]).rename("top_service")
    X = X.merge(top_svc.reset_index(), on=["host_id", "ts"])
    # n_alerts is derived from the label's source -> it LEAKS. Kept out of the
    # feature list on purpose; Topics 2 and 9 use it to demonstrate leakage.
    ac = alerts.groupby(["host_id", "ts"]).size().rename("n_alerts").reset_index()
    X = X.merge(ac, on=["host_id", "ts"], how="left")
    X["n_alerts"] = X["n_alerts"].fillna(0).astype(int)
    true_alerts = alerts[alerts.is_true_attack == 1]
    fam = (true_alerts.groupby(["host_id", "ts"])["signature_id"].first()
           .map(FAMILY).rename("attack_family").reset_index())
    X = X.merge(fam, on=["host_id", "ts"], how="left")
    X["attack_family"] = X["attack_family"].fillna("benign")
    X["label"] = (X["attack_family"] != "benign").astype(int)
    if truth is not None:
        X = X.merge(truth, on=["host_id", "ts"], how="left")
    return X.sort_values(["ts", "host_id"], kind="stable").reset_index(drop=True)


def load(return_truth=False):
    """The host-hour matrix in one call: X = cyberlab.load()."""
    if return_truth:
        flows, alerts, truth = make_flows(return_truth=True)
        return host_hour_matrix(flows, alerts, truth)
    return host_hour_matrix(*make_flows())