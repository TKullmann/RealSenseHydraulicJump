import pandas as pd

def extract_xlsx():
    file_path = '../Wechselsprung_Auswertung.xlsx'
    data = pd.read_excel(file_path, sheet_name='Auswertung_WS6', skiprows=8)
    extracted_data = data.iloc[:, [1, 4]]
    extracted_data.columns = ['Strecke', 'Wassertiefe']
    return extracted_data

