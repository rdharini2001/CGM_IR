from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_documented_training_and_forward_commands():
    r=(ROOT/'README.md').read_text().lower()
    assert 'bash run_all.sh --train' in r
    assert 'bash run_all.sh --verify' in r
    assert 'python predict.py' in r
    assert 'published' in r and 'constraints' in r
    assert 'not estimates of accuracy on new patients' in r
    assert (ROOT/'docs/TRAINING.md').exists()
    assert (ROOT/'docs/METHODS.md').exists()

def test_bibliography():
    s=(ROOT/'REFERENCES.bib').read_text()
    assert '10.1111/1467-9868.00294' in s
    assert '10.1109/ICNN.1995.488968' in s
