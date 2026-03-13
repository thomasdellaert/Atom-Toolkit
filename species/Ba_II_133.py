from atomtoolkit import Q_, IO, transition_strengths
from atomtoolkit import a0, e, MHz
import numpy as np

NAME = "133Ba II"
I = 1/2
NUM_LEVELS = 300
ALLOWED_TRANSITIONS = (True, False, False)

_energy_level_df = IO.load_NIST_data('Ba II')

Ba133 = IO.generate_full_from_dataframe(_energy_level_df,
                                        name=NAME,
                                        I=I,
                                        num_levels=NUM_LEVELS,
                                        allowed=ALLOWED_TRANSITIONS)

# Define Hyperfine Coefficients # TODO: Move this to a CSV
Ba133.levels["6s 2S1/2"].hfA = Q_(-9925.45355459, "MHz")
Ba133.levels["5d 2D5/2"].hfA = Q_(29.7565, "MHz")

# List of dipole matrix elements out of the ground state #TODO: Move this to a CSV
mels = {
    "6p 2P*1/2": 3.3251,
    "6p 2P*3/2": 4.7017,
    "7p 2P*1/2": 0.0605607,
    "7p 2P*3/2": 0.0869283,
    "8p 2P*1/2": 0.0868183,
    "8p 2P*3/2": 0.0330815,
    "9p 2P*1/2": 0.0721177,
    "9p 2P*3/2": 0.0437074,
    "10p 2P*1/2": 0.0586589,
    "10p 2P*3/2": 0.0407040,
    "11p 2P*1/2": 0.0484327,
    "11p 2P*3/2": 0.0358371,
    "12p 2P*1/2": 0.0407219,
    "12p 2P*3/2": 0,
}

# List of dipole matrix elements out of the D5/2 state #TODO: Move this to a CSV
melsd = {
    "6p 2P*3/2": 4.1028,
    "7p 2P*3/2": 0.4512685,
    "8p 2P*3/2": 0.2231382,
    "9p 2P*3/2": 0.1437971,
    "10p 2P*3/2": 0.1039872,
    "11p 2P*3/2": 0.0802920,
    "12p 2P*3/2": 0.0646935,
    "13p 2P*3/2": 0.0537146,
    "14p 2P*3/2": 0.0456094,
    "4f 2F*5/2": 0.9983269,
    "4f 2F*7/2": 4.4743822,
    "5f 2F*5/2": 0.3046073,
    "5f 2F*7/2": 1.4289390,
    "6f 2F*5/2": 0.0431679,
    "6f 2F*7/2": 0.0748619,
    "7f 2F*5/2": 0.2048909,
    "7f 2F*7/2": 0.7601925,
    "8f 2F*5/2": 0.2140143,
    "8f 2F*7/2": 0.8650676,
    "9f 2F*5/2": 0.1535997,
    "9f 2F*7/2": 0.6753982,
    "10f 2F*5/2": 0.0921940,
    "10f 2F*7/2": 0.4486550,
    "11f 2F*5/2": 0.0501564,
    "11f 2F*7/2": 0.2755798,
    "12f 2F*5/2": 0.0246295,
    "12f 2F*7/2": 0.1591081,
}

for s, t in Ba133.linked_levels('6s 2S1/2').items():
    mel = mels[s.name] * a0 * e
    A = transition_strengths.dipole_mel_to_A(mel, t.E_lower.term.J, t.E_upper.term.J, t.omega)
    t.A = A

for s, t in Ba133.linked_levels('5d 2D5/2').items():
    try:
        mel = melsd[s.name] * a0 * e
        A = transition_strengths.dipole_mel_to_A(mel, t.E_lower.term.J, t.E_upper.term.J, t.omega)
        t.A = A
    except KeyError:
        t.A = Q_(0, "Hz")

if __name__ == '__main__':
    Ba133.save_json('Ba_II_133.json')