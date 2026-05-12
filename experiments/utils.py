# -*- coding: utf-8 -*-
"""Shared utilities for Aletheia experiments."""
import os, csv, sys

CSV_PATH = r"C:\tmp\experiment_control.csv"

def phase_complete(phase_num):
    """Handle phase completion: beep or hibernate based on CSV.
    When running via runner (ALETHEIA_RUNNER=1), skip beeps.
    Runner handles final beep itself.
    """
    hibernate = False
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r') as f:
            for row in csv.DictReader(f):
                if int(row['phase']) == phase_num:
                    hibernate = int(row.get('hibernate', 0)) == 1
                    break
    if hibernate:
        print(f"[Phase {phase_num}] Complete. Hibernate requested.")
        # Hibernation handled by runner scripts (not tracked in git)
        sys.exit()

    # Only beep if running standalone (not via runner)
    if os.environ.get('ALETHEIA_RUNNER') != '1':
        try:
            import winsound, time
            for _ in range(5):
                winsound.Beep(1000, 500)
                time.sleep(0.3)
        except Exception:
            pass
        print(f"[Phase {phase_num}] Complete. Beep!")
    else:
        print(f"[Phase {phase_num}] Complete.")

