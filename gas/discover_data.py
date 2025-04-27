import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/gsalc.csv', header=None)

df.loc[:, 'experience'] = [i % 5 + 1 for i in range(len(df))]

df_gas_response = df.iloc[:, 2:-1]

sensor_names = ['TGS2603', 'TGS2630', 'TGS813', 'TGS822', 'MQ-135', 'MQ-137', 'MQ-138', '2M012', 'VOCS-P', '2SH12']

# ----------------------------------------------------------------------------------------------------------------------
# REORDER DATA SET BY SENSOR TYPE (10), BY GAS (6), BY CONCENTRATION (3), BY RUN (5) = 900 TIME SERIES AT 1 HZ
# ----------------------------------------------------------------------------------------------------------------------

# sensor_name, gas type, concentration
df_ordered = pd.DataFrame(
    columns=[(name, row[1][0], row[1][1], row[1]['experience']) for name in sensor_names for row in
             list(df.iterrows())])

for row in list(df.iterrows()):

    gas_type = row[1][0]
    concentration = row[1][1]
    experience = row[1]['experience']

    for idx, name in enumerate(sensor_names):
        start = idx * 900 + 2
        end = start + 900
        df_ordered[(name, gas_type, concentration, experience)] = row[1].iloc[start:end].to_numpy()

# ----------------------------------------------------------------------------------------------------------------------
# PLOT A GAS TYPE AND SPECIFIC CONCENTRATION FOR ALL EXPERIENCE AND ALL SENSORS.
# ----------------------------------------------------------------------------------------------------------------------

list_gases = ['ethanol', 'acetone', 'toluene', 'ethyl acetate', 'isopropanol', 'hexane']
concentrations = ['50ppb', '100ppb', '200ppb']

for gas_type, concentration in [(t, c) for t in list_gases for c in concentrations]:

    fig, ax = plt.subplots(10, 1, figsize=(15, 20))

    for idx, sensor in enumerate(sensor_names):
        for experience in [1, 2, 3, 4, 5]:
            ax[idx].plot(df_ordered[(sensor, gas_type, concentration, experience)].to_numpy(),label=f'{experience}')
        ax[idx].set_title(sensor)
        ax[idx].legend()

    fig.suptitle(f'{gas_type}::{concentration}')

    plt.tight_layout()
    plt.show()
    plt.close()
