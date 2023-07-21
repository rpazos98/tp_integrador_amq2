import pandas as pd

def transform(data: pd.DataFrame):
    data['Outlet_Establishment_Year'] = 2020 - data['Outlet_Establishment_Year']

    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
    
    productos = list(data[data['Item_Weight'].isnull()]['Item_Identifier'].unique())
    for producto in productos:
        moda = (data[data['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0,0]
        data.loc[data['Item_Identifier'] == producto, 'Item_Weight'] = moda  

    outlets = list(data[data['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
    
    for outlet in outlets:
        data.loc[data['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'
    
    data.loc[data['Item_Type'] == 'Household', 'Item_Fat_Content'] = 'NA'
    data.loc[data['Item_Type'] == 'Health and Hygiene', 'Item_Fat_Content'] = 'NA'
    data.loc[data['Item_Type'] == 'Hard Drinks', 'Item_Fat_Content'] = 'NA'
    data.loc[data['Item_Type'] == 'Soft Drinks', 'Item_Fat_Content'] = 'NA'
    data.loc[data['Item_Type'] == 'Fruits and Vegetables', 'Item_Fat_Content'] = 'NA'

    data['Item_Type'] = data['Item_Type'].replace({'Others': 'Non perishable', 'Health and Hygiene': 'Non perishable', 'Household': 'Non perishable',
    'Seafood': 'Meats', 'Meat': 'Meats',
    'Baking Goods': 'Processed Foods', 'Frozen Foods': 'Processed Foods', 'Canned': 'Processed Foods', 'Snack Foods': 'Processed Foods',
    'Breads': 'Starchy Foods', 'Breakfast': 'Starchy Foods',
    'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})

    # FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
    data.loc[data['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'

    data['Item_MRP'] = pd.qcut(data['Item_MRP'], 4, labels = [1, 2, 3, 4])

    dataframe = data.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()

    dataframe['Outlet_Size'] = dataframe['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
    dataframe['Outlet_Location_Type'] = dataframe['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}) # Estas categorias se ordenaron asumiendo la categoria 2 como más lejos

    dataframe = pd.get_dummies(dataframe, columns=['Outlet_Type'])

    # Eliminación de variables que no contribuyen a la predicción por ser muy específicas
    dataset = dataframe.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

    return dataset