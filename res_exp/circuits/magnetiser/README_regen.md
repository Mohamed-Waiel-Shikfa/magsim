# Regenerative thyristor model (task 6)

`magnetiser.lib` models the CLA100E1200HB as a gate-controlled conductance: it conducts as soon as the gate
drive rises and needs that drive to keep conducting. With the board's anode-fired gate (RG = 2.2 kΩ from the
anode, through the VO4258H, to the gate) conduction collapses V_AK, the gate current stops at 30.5 mA and both
thyristors sit half-on (V_AK ≈ 55–70 V): I_pk = 1381 A and the bank keeps 172.7 V (10 µH / 60 mΩ test coil).
A real thyristor fires after the gate-controlled delay and then latches regeneratively while I_A > I_H.

| file | content |
|---|---|
| `magnetiser_regen.lib` | same subcircuit names and pins; VO4258H and DCROWBAR unchanged; CLA100E1200HB rewritten as a latch |
| `magnetiser_regen_optodelay.lib` | as `magnetiser_regen.lib` plus a charge-control VO4258H turn-on delay |

Parameters (Littelfuse/IXYS CLA100E1200HB rev. 20210601d; page/table cited in the `.lib` comments):
I_GT = 40 mA (p.2, 25 °C), I_L = 150 mA, I_H = 100 mA (p.2), gate-controlled delay shaped on p.5 Fig. 5
(t_gd ≈ 1.0 µs + 3.3 µs·(45 mA/(I_G − I_GT))^1.55; 2.1 + 1.0 µs at I_G = 100 mA), on-state refitted to the p.2
table at 25 °C (1.37 V at 100 A, 1.78 V at 200 A; 3.83 mΩ slope, extrapolated to kA: an assumption), off-state
1e-8 S (p.2 leakage). No forward current flows until the latch state crosses its threshold, so V_AK stays high
while the gate charges.

Results (board + fixed 10 µH / 60 mΩ coil, 450 V; `tests/test_switching.py`):

| model | RG | I_pk | bank after | note |
|---|---|---|---|---|
| original | 2.2 kΩ | 1381 A | 172.7 V | half-on, I_G max 30.5 mA |
| regenerative | 2.2 kΩ | 1959 A | 0.1 V | fires ~3 µs after GP16, V_AK ≈ 1.3 V at 15 µs |
| regenerative | 470 Ω | 1959 A | 0.1 V | I_G max 461 mA |
| regenerative + opto delay | 2.2 kΩ | 1959 A | 0.1 V | fires ~43 µs later (I_F ≈ 3.9 mA, VO4258 p.5 Fig. 8) |

Minimum firing voltage with RG = 2.2 kΩ (anode-fired): I_G = (V_AK − V_GK − V_triac)/RG − V_GK/RGK must exceed
I_GT, i.e. V_AK ≳ 40 mA·2.2 kΩ + 1.1 V + 0.3 V + 2.4 V ≈ 92 V per thyristor (simulated: I_G = 40.3 mA at
V_AK = 92.5 V). Before firing the two thyristors of a diagonal share the bus through the snubbers, so the bus
needs ≳ 185 V; to fire inside the 200 µs gate pulse I_G must exceed ~44 mA (t_gd < 200 µs): 180–190 V does not
fire, 200 V fires late (~170 µs), 250 V after ~8 µs.

The user decides which model represents the board; reference scenarios default to `magnetiser_regen.lib`,
`scenarios/switch_r1_n52_foil_origlib.yaml` runs R1 with the original model.
