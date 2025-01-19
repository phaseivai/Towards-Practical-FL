import os
import pandas as pd
import numpy as np
import pickle
import warnings
import argparse

'''This file preprocesses all the raw data from different studies'''

def find_max_lesion(df):

    '''
    This function finds the most significant lesion per person, and the corresponding values 
    for the Italy.xlsx dataset
    '''

    # Fill all lesions of a single person
    df['Block'] = (df['age'].notna()).cumsum()
    df['age'] = df.groupby('Block')['age'].ffill()
    
    result = []
    
    # Group by the block identifier and process each group
    for key, group in df.groupby('Block'):
        
        # Fill NaN values in the PIRADS column with the maximum value of the group
        max_b = group['PIRADS'].max()
        group['PIRADS'] = group['PIRADS'].astype(float).fillna(max_b)
        
        # Find the maximum value of GLEASON where the PIRADS equals its maximum
        max_c = group[group['PIRADS'] == max_b]['sig_cancer'].max()
        result.append([max_c, group['age'].iloc[0], group['PSA'].iloc[0], group['PV'].iloc[0], max_b])
    
    # Convert the result list to a DataFrame
    result_df = pd.DataFrame(result, columns=['sig_cancer', 'age', 'PSA', 'PV', 'PIRADS'])

    return result_df


def is_anomaly(val):
    
    """This function (for Turkey datasets) returns True if the ISUP grade starts with 3 and is 
    followed by zeros (which is not correct) or is equal to 0 for the CHina1.csv dataset"""
    
    return str(val).startswith('3') and str(val)[1:] == '0' * (len(str(val)) - 1) or val == 0


warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

parser = argparse.ArgumentParser(description="A script with argparse defaults.")
parser.add_argument("--folder", type=str, default="data", help="Folder to store the processed datasets")
# Parse arguments
args = parser.parse_args()


# Toggle split_countries_to_silos flag to not split by hospital, but genew_cutoffraphically 
split_countries_to_silos = False
# Toggle to get a 3+3 cutoff for non-significant cancer based on GGG/
# Theres is a tendency these days to consider 3+4 as not significant cancer, hence the variable name.
new_cutoff = False

# create data directory
directory = args.folder

# the preprocessed data is stored in this dictionary
silos = {}
silos_test = {}

for root, dirs, files in os.walk(directory):
    for file_name in files:
        file_path = os.path.join(root, file_name)

        # both Koreas were recorded in the same hospital by the same people
        if 'Korea1' in  file_name:
            if not new_cutoff:
                df = pd.read_csv(file_path)
                df = df[['sig_cancer', 'Age', 'PSA', 'PV', 'PIRADS']]
                df.columns = ['sig_cancer', 'age', 'PSA', 'PV', 'PIRADS']
                df['5ARI'] = 0
                df = df.astype({'sig_cancer': float, 'age': float, 'PSA': float, 'PV': float, 'PIRADS': float, '5ARI': float})
                df = df.dropna(subset = ['sig_cancer'])
                df.loc[ (df['age'] <= 20) , 'age'] = np.nan
                df = df.dropna(subset = ['age'])
                silos['Korea1'] = df
            else:
                # No raw GGG
                pass
        
        # both Koreas were recorded in the same hospital by the same people
        if 'Korea2' in file_name:
            if not new_cutoff:
                df = pd.read_excel(file_path)
                df = df[['sig_cancer', 'age', 'PSA', 'PV', 'PIRADS']]
                df['5ARI'] = 0
                df = df.astype({'sig_cancer': float, 'age': float, 'PSA': float, 'PV': float, 'PIRADS': float, '5ARI': float})
                # for now filling nans for prostate volume with median
                df['PV'] = df['PV'].fillna(df['PV'].median())
                df.loc[ (df['age'] <= 20) , 'age'] = np.nan
                df = df.dropna(subset = ['age'])
                silos['Korea2'] = df
            else:
                # No raw GGG
                pass

        # Italy has multiple lessions for a single person, so we take the "worst" per patient
        if 'Italy' in file_name:
            df = pd.read_excel(file_path)
            columns = df.iloc[0]
            # gleason_column = columns[26]
            gleason_column = columns.iloc[26]
            # age_column = columns[10]
            age_column = columns.iloc[10]
            # PSA_column = columns[5]
            PSA_column = columns.iloc[5]
            # PV1_column = columns[11]
            PV1_column = columns.iloc[11]
            # PV2_column = columns[12]
            PV2_column = columns.iloc[12]
            # PIRADS_column = columns[16]
            PIRADS_column = columns.iloc[16]
            df = df.iloc[1:456]
            df.columns = columns
            
            if not new_cutoff:
                # Convert gleason to significant cancer or not. Threshhold is 3+4
                df.loc[ (df[gleason_column] == '0') | (df[gleason_column] == '6 (3+3)'), gleason_column] = 0
                df.loc[ (df[gleason_column] == '8 (4+4)') | (df[gleason_column] == '8 (5+3)') | (df[gleason_column] == '9 (5+4)') | 
                (df[gleason_column] == '7 (4+3)') | (df[gleason_column] == '7 (3+4)'), gleason_column] = 1
            else:
                # Convert gleason to significant cancer or not. Threshhold is 4+3
                df.loc[ (df[gleason_column] == '0') | (df[gleason_column] == '6 (3+3)') |
                (df[gleason_column] == '7 (3+4)'), gleason_column] = 0
                df.loc[ (df[gleason_column] == '8 (4+4)') | (df[gleason_column] == '8 (5+3)') | (df[gleason_column] == '9 (5+4)') |
                (df[gleason_column] == '7 (4+3)') , gleason_column] = 1

            # For those patients missing mri volume, replace it with the calculated one.
            df.loc[ (df[PV2_column] == '?'), PV2_column] =  df.loc[ (df[PV2_column] == '?'), PV1_column]
            
            df = df[[gleason_column, age_column, PSA_column, PV2_column, PIRADS_column]]
            df.columns = ['sig_cancer', 'age', 'PSA', 'PV', 'PIRADS']
            df = df.reset_index(drop=True)
            
            df = find_max_lesion(df)
            df['5ARI'] = 0
            df = df.astype({'sig_cancer': float, 'age': float, 'PSA': float, 'PV': float, 'PIRADS': float, '5ARI': float})
            silos['Italy'] = df

        # Spain has 10 relatively small silos collected in the Barcelona area.
        if 'Spain' in file_name:
            if not new_cutoff:
                # people with isup grade >= 2
                df = pd.read_csv(file_path, delimiter='\t')
                df = df[['C','sPC', 'ED', 'PSA', 'VP', 'PIR']]
                df.columns = ['C', 'sig_cancer', 'age', 'PSA', 'PV', 'PIRADS']
                df['5ARI'] = 0
                df = df.astype({'sig_cancer': float, 'age': float, 'PSA': float, 'PV': float, 'PIRADS': float, '5ARI': float})
                if split_countries_to_silos:
                    for key, group in df.groupby('C'):
                        group = group.drop(columns=['C'])
                        silos['Spain'+ str(key)] = group
                        # print(len(group))
                else:
                    df = df.drop(columns = ['C'])
                    silos['Spain'] = df
            else:
                # No raw GGG
                pass

        # For this silo we convert prostate weight to prostate volume, by assuming prostate density roughly been 1 gm/cm^3
        if 'Turkey' in file_name:
            df = pd.read_csv(file_path)
            # sig_cancer_column = df.columns[0]
            age_column = df.columns[1]
            PV_column = df.columns[10]
            sig_cancer_column = df.columns[11]
            df = df.drop_duplicates(subset=[age_column, PV_column, 'psa'])
            df = df[[sig_cancer_column, age_column, 'psa', PV_column]]
            df[sig_cancer_column] = pd.to_numeric(df[sig_cancer_column], errors='coerce')
            df = df.dropna(subset = [sig_cancer_column])
            # df[sig_cancer_column] = df[sig_cancer_column].fillna(0)
            if not new_cutoff:
                # people with isup grade >= 2
                df[sig_cancer_column] = np.where(df[sig_cancer_column] >= 2, 1, 0)
            else:
                # people with isup grade >= 3
                df[sig_cancer_column] = np.where(df[sig_cancer_column] >= 3, 1, 0)
            df.columns = ['sig_cancer', 'age', 'PSA', 'PV']
            df['PIRADS'] = 3
            df['5ARI'] = 0
            df = df.astype({'sig_cancer': float, 'age': float, 'PSA': float, 'PV': float, 'PIRADS': float, '5ARI': float})
            silos['Turkey'] = df

        # Since silos have been collected in different countries, we do not merge it even when split_countries_to_silos = False
        if 'China1' in file_name:
            df = pd.read_csv(file_path)
            df = df[['Chinese1Dutch2', 'HPCa', 'age', 'PSA', 'PV_MRI', 'PV_TRUS', 'Overall_PIRADS', 
                     'HighestGleasonSBx', 'HighestGleasonTBx']]
            df = df.iloc[:-5]
            # Specify the columns to check and correct for empty values that are not nans
            columns_to_check = ['age', 'PSA', 'PV_MRI', 'PV_TRUS', 'Overall_PIRADS', ]
            df[columns_to_check] = df[columns_to_check].apply(lambda col: pd.to_numeric(col, errors='coerce'))
            # For now drop nans in PSA
            df = df.dropna(subset = ['PSA'])
            # For those patients missing mri volume, replace it with the ultrasound volume.
            df['PV_MRI'] = df['PV_MRI'].fillna(df['PV_TRUS'])
            df = df.drop(columns=['PV_TRUS'])
            
            if new_cutoff:
                # Find people with 3+4 who were assigned csPC and shange to zero
                df = df.astype({'HPCa': float, 'HighestGleasonSBx': float, 'HighestGleasonTBx': float})
                below_43 = [0, 33, 34]
                df.loc[(df['HPCa'] == 1) & \
                (df['HighestGleasonTBx'].isin(below_43)) & (df['HighestGleasonSBx'].isin(below_43)),'HPCa'] = 0

                df = df[~((df['HPCa'] == 1) & 
                          (df['HighestGleasonTBx'].apply(is_anomaly)) & (df['HighestGleasonSBx'].apply(is_anomaly)))]

            df = df[['Chinese1Dutch2', 'HPCa', 'age', 'PSA', 'PV_MRI', 'Overall_PIRADS']]
            df.columns = ['Chinese1Dutch2', 'sig_cancer', 'age', 'PSA', 'PV', 'PIRADS']
            df['5ARI'] = 0
            df = df.astype({'sig_cancer': float, 'age': float, 'PSA': float, 'PV': float, 'PIRADS': float, '5ARI': float})
                
            
            for key, group in df.groupby('Chinese1Dutch2'):
                group = group.drop(columns=['Chinese1Dutch2'])
                if key == '1':
                    silos['China1'] = group
                elif key == '2':
                    silos['Netherlands'] = group
                else:
                    raise ValueError("No Silo with this code")

        if 'USA1' in file_name:
            df = pd.read_excel(file_path)
            df = df[['cs_Overall', 'Overall_GGG', 'Age_at_Biopsy ', 'PSA', 'MRI_Prostate_Volume', 'Highest_PIRADS_OVERALL']]
            df = df.astype({'Overall_GGG': float})
            if new_cutoff:
                # only patients with Isup grade 3 or higher
                df['cs_Overall'] = np.where(df['Overall_GGG'] >= 3, 1, 0)
            df =  df[['cs_Overall', 'Age_at_Biopsy ', 'PSA', 'MRI_Prostate_Volume', 'Highest_PIRADS_OVERALL']]
            df.columns = ['sig_cancer', 'age', 'PSA', 'PV', 'PIRADS']
            df['5ARI'] = 0
            df = df.astype({'sig_cancer': float, 'age': float, 'PSA': float, 'PV': float, 'PIRADS': float, '5ARI': float})
            silos['USA1'] = df

        if 'Finland' in file_name:
            df = pd.read_csv(file_path, delimiter='\t')
            df = df[['AgeAtStartOfAs', 'PSA at cancer diagnosis', 'Prostate size grams at diagnosis', 'MRI1highestPIRADSscore']]
            df.columns = ['age', 'PSA', 'PV', 'PIRADS']
            df = df.replace(',', '.', regex=True)
            df['sig_cancer'] = 0
            df['5ARI'] = 0
            
            df['PSA'] = pd.to_numeric(df['PSA'], errors='coerce')
            df = df.dropna(subset=['PSA'])
            
            df = df.astype({'sig_cancer': float, 'age': float, 'PSA': float, 'PV': float, 'PIRADS': float, '5ARI': float})
            df = df[['sig_cancer', 'age', 'PSA', 'PV', 'PIRADS', '5ARI']]
            # silos_test['Finland1'] = df
            silos['Finland'] = df


        if 'Germany2' in file_name:
            df = pd.read_excel(file_path)
            if not new_cutoff:
                df = df.drop(df.columns[0], axis=1)
                df = df.replace(',', '.', regex=True)
                df = df[['center','outcome', 'Age', 'PSA', 'Volume', 'PI.RADSv2']]
                df.columns = ['center','sig_cancer', 'age', 'PSA', 'PV', 'PIRADS']
                df['5ARI'] = 0
                df = df.astype({'sig_cancer': float, 'age': float, 'PSA': float, 'PV': float, 'PIRADS': float, '5ARI': float})
    
                for key, group in df.groupby('center'):
                        group = group.drop(columns=['center'])
                        if key == 'London':
                            silos['UK'] = group
                        elif key == 'Heidelberg':
                            silos['Germany2'] = group
                        else:
                            raise ValueError("No Silo with this code")
            else:
                # No raw GGG
                pass

        if 'USA2' in file_name:
            df = pd.read_excel(file_path)
            df['MRProstateVol'] = df['MRProstateVol'].fillna(df['ProstateVolumeUS'])
            df = df[['GleasonSum', 'PrimaryGleason', 'SecondaryGleason', 'Age', 'PSA', 'MRProstateVol', 'OVERALL SUSPICION PIRADS V2']]
            
            if not new_cutoff:
                # ISUP grade 3+4
                df['GleasonSum'] = np.where(df['GleasonSum'] >= 7, 1, 0)
            else:
                # ISUP grade 4+3
                df['GleasonSum'] = np.where( (df['GleasonSum'] >= 7) &
                                            ((df['PrimaryGleason'] != 3) & (df['SecondaryGleason'] != 4)), 1, 0)

            df = df[['GleasonSum', 'Age', 'PSA', 'MRProstateVol', 'OVERALL SUSPICION PIRADS V2']]
            df.columns = ['sig_cancer', 'age', 'PSA', 'PV', 'PIRADS']
            df['5ARI'] = 0
            df = df.astype({'sig_cancer': float, 'age': float, 'PSA': float, 'PV': float, 'PIRADS': float, '5ARI': float})
            df.loc[ (df['age'] <= 20) , 'age'] = np.nan
            df = df.dropna(subset = ['age'])
            silos['USA2'] = df

            
        if 'Germany1' in file_name:
            df = pd.read_excel(file_path, sheet_name = 'Baseline')
            # df2 = pd.read_excel(file_path, sheet_name = 'Biopsy')
            # both sheets have PIRADS, taking the biopsy one
            # df1 = df1.drop(columns=['PIRADS'])
            # df2 = df2.iloc[:-2]
            # df = pd.concat([df1,df2], axis = 1)
            df['sig_cancer'] = 0 
            if not new_cutoff:
                # ISUP grade 3+4
                df.loc[ (df['ISUP'] >= 2) , 'sig_cancer'] = 1
            else:
                # ISUP grade 4+3
                df.loc[ (df['ISUP'] >= 3) , 'sig_cancer'] = 1
            df = df[['sig_cancer', 'Age', 'PSA', 'Volume', 'PIRADS']]
            df.columns = ['sig_cancer', 'age', 'PSA', 'PV', 'PIRADS']
            df['5ARI'] = 0
            # For now filling nans in volume with the mean
            df['PV'] = df['PV'].fillna(df['PV'].median())
            # For now drop nans in PSA
            df = df.dropna(subset = ['PSA'])
            
            df = df.astype({'sig_cancer': float, 'age': float, 'PSA': float, 'PV': float, 'PIRADS': float, '5ARI': float})
            silos['Germany1'] = df


        if 'China2' in file_name:
            df = pd.read_csv(file_path)
            df =  df[['GS', 'age', 't-PSA', 'PV', 'PI-RADS score']]
            df.columns = ['sig_cancer', 'age', 'PSA', 'PV', 'PIRADS']
            df['5ARI'] = 0
            df['sig_cancer'] = df['sig_cancer'].fillna(0)
            if not new_cutoff:
                # ISUP grade 3+4
                ggg0 = [0, '3+3=6', '3+3+6' , ' ']
            else:
                # ISUP grade 4+3
                ggg0 = [0, '3+4=7', '3+3=6', '3+3+6' , ' ']
            df.loc[df['sig_cancer'].isin(ggg0),'sig_cancer'] = 0
            df.loc[~df['sig_cancer'].isin(ggg0),'sig_cancer'] = 1
            
            df = df.astype({'sig_cancer': float, 'age': float, 'PSA': float, 'PV': float, 'PIRADS': float, '5ARI': float})
            silos['China2'] = df

# Although both Korea1 and Korea2 can be considered a different silo, since both were gathered in a same hospital, we treat it as one.
if not new_cutoff:
    # No raw GGG
    silos['Korea'] = pd.concat([silos['Korea1'], silos['Korea2']], axis = 0, ignore_index=True)
    for a in ['Korea1', 'Korea2']:
        silos.pop(a)
        
with open(args.folder+'/data.pkl', 'wb') as file:
    pickle.dump(silos, file)