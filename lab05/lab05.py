import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temp = ctrl.Antecedent(np.arange(15, 36, 1), 'temperatura')
wilgotnosc = ctrl.Antecedent(np.arange(0, 101, 1), 'wilgotnosc')
woda = ctrl.Consequent(np.arange(0, 26, 1), 'ilosc_wody')

temp['chlodno'] = fuzz.trimf(temp.universe, [15, 15, 25])
temp['cieplo'] = fuzz.trimf(temp.universe, [20, 25, 30])
temp['goraco'] = fuzz.trimf(temp.universe, [25, 35, 35])

wilgotnosc['sucho'] = fuzz.trimf(wilgotnosc.universe, [0, 0, 50])
wilgotnosc['przecietnie'] = fuzz.trimf(wilgotnosc.universe, [25, 50, 75])
wilgotnosc['mokro'] = fuzz.trimf(wilgotnosc.universe, [50, 100, 100])

woda['malo'] = fuzz.trimf(woda.universe, [0, 0, 10])
woda['srednio'] = fuzz.trimf(woda.universe, [5, 12.5, 20])
woda['duzo'] = fuzz.trimf(woda.universe, [15, 25, 25])

rule1 = ctrl.Rule(temp['chlodno'] & wilgotnosc['sucho'], woda['srednio'])
rule2 = ctrl.Rule(temp['chlodno'] & wilgotnosc['przecietnie'], woda['srednio'])
rule3 = ctrl.Rule(temp['chlodno'] & wilgotnosc['mokro'], woda['malo'])
rule4 = ctrl.Rule(temp['cieplo'] & wilgotnosc['sucho'], woda['duzo'])
rule5 = ctrl.Rule(temp['cieplo'] & wilgotnosc['przecietnie'], woda['srednio'])
rule6 = ctrl.Rule(temp['cieplo'] & wilgotnosc['mokro'], woda['malo'])
rule7 = ctrl.Rule(temp['goraco'] & wilgotnosc['sucho'], woda['duzo'])
rule8 = ctrl.Rule(temp['goraco'] & wilgotnosc['przecietnie'], woda['duzo'])
rule9 = ctrl.Rule(temp['goraco'] & wilgotnosc['mokro'], woda['srednio'])

watering_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
watering = ctrl.ControlSystemSimulation(watering_ctrl)

wilgotnosc_range = np.arange(0, 101, 5)
temp_range = np.arange(15, 36, 1)
wyniki = np.zeros((len(temp_range), len(wilgotnosc_range)))

for i, t in enumerate(temp_range):
    for j, w in enumerate(wilgotnosc_range):
        watering.input['temperatura'] = t
        watering.input['wilgotnosc'] = w
        watering.compute()
        wyniki[i, j] = watering.output['ilosc_wody']

plt.figure(figsize=(10, 8))
plt.imshow(wyniki, cmap='viridis', extent=[0, 100, 15, 35], origin='lower', aspect='auto')
plt.colorbar(label='Ilość wody (l/dzień)')
plt.xlabel('Wilgotność (%)')
plt.ylabel('Temperatura (°C)')
plt.title('Heatmapa ilości wody do podlewania')
plt.show()

temp.view()
wilgotnosc.view()
woda.view()