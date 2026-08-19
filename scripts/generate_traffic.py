"""
Traffic generator - FinSentinel M3 monitoring validation.

Sends real prediction requests to the running API in two phases:
a mixed baseline, then a deliberately skewed burst. Used to verify
end-to-end that predictions feed the metrics and that the PSI drift
detector reacts to a distribution shift - the monitoring stack is
only validated if it has been exercised with real traffic.
"""

import sys
import time
from collections import Counter

import requests

API = "http://localhost:8000"

BASELINE = [
    "Le titre progresse apres la publication des resultats",
    "Chiffre d affaires en hausse de douze pour cent",
    "L action recule legerement en cloture",
    "Perspectives revues a la baisse pour le trimestre",
    "La societe maintient ses objectifs annuels",
    "Marge operationnelle stable sur un an",
    "Le conseil approuve le versement du dividende",
    "Endettement en augmentation apres l acquisition",
    "Les analystes relevent leur objectif de cours",
    "Trafic commercial conforme aux attentes",
    "Le groupe annonce une restructuration",
    "Croissance organique soutenue en Europe",
]

DRIFT = [
    "Effondrement brutal du titre apres l annonce",
    "Pertes massives et faillite imminente",
    "Licenciements en serie dans toutes les filiales",
    "Le groupe suspend son dividende en urgence",
    "Chute historique de la valorisation boursiere",
    "Defaut de paiement sur la dette obligataire",
    "Fermeture definitive de plusieurs usines",
    "Fraude comptable revelee par les auditeurs",
    "Alerte sur resultats pour la troisieme fois",
    "Retrait immediat de la cotation",
]


def predict(text: str) -> str | None:
    try:
        r = requests.post(f"{API}/predict", json={"text": text}, timeout=30)
        r.raise_for_status()
        return r.json()["label"]
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return None


def drift_state() -> tuple[float, float]:
    body = requests.get(f"{API}/metrics", timeout=10).text
    score = alert = 0.0
    for line in body.splitlines():
        if line.startswith("finsentinel_data_drift_score "):
            score = float(line.split()[1])
        elif line.startswith("finsentinel_data_drift_alert "):
            alert = float(line.split()[1])
    return score, alert


def run_phase(name: str, texts: list[str]) -> None:
    print(f"\n{name} ({len(texts)} requests)")
    labels = Counter()
    for text in texts:
        label = predict(text)
        if label:
            labels[label] += 1
        time.sleep(0.2)
    print(f"  labels: {dict(labels)}")
    score, alert = drift_state()
    print(f"  PSI: {score}  |  drift alert: {int(alert)}")


def main() -> None:
    try:
        requests.get(f"{API}/health", timeout=10).raise_for_status()
    except Exception:
        sys.exit(f"API not reachable at {API} - is the stack running?")

    run_phase("PHASE 1 - mixed baseline", BASELINE)
    run_phase("PHASE 2 - deliberate distribution shift", DRIFT)


if __name__ == "__main__":
    main()