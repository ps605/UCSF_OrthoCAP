

#!/usr/bin/env python3
"""
Created on Thu Sep 14 09:06:58 2023

@author: erinarchibeck
"""
#K-Score Calculation Code
#Last Updated Date: 1/16/23
#this version applies PCA first and then performs GPA on the transformed PCA data (leaving the weights alone)
#KDI is then calculated by Weighted Sum of PC Values:  summing the (transformed aligned data * weights) for every time point 

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from sklearn.preprocessing import StandardScaler
import glob
#import random
import re

###############################################################################
#                               Reading Files                                 #
###############################################################################
#Feel free to change this to read your files as needed 

def read_csv_files(file_list):
    data = []
    fixed_length = 100  # Desired fixed length to normalize data

    for file in file_list:
        match = re.search(r'(Control_\d{3}|\d{3}-\d{5})', file)
        patient_id = match.group(1)


        try:
            df = pd.read_csv(file, skiprows=1)
            df.dropna(inplace=True)
            num_rows_per_activity = len(df)
            indices = np.linspace(0, num_rows_per_activity - 1, fixed_length, dtype=int)
            df_interpolated = df.iloc[indices]

        except Exception as e:
            print(f"Error processing file '{file}': {str(e)}")


        data.append((patient_id, df_interpolated, num_rows_per_activity))
        #data_activity.append((patient_id,df_interpolated, num_rows_per_activity))# Add the ID and dataframe to 'data'

    return data


###############################################################################
#               Principal Component Analysis (for every patient)              #
###############################################################################

#PCA is performed indivudally on every patient to include temporal and positional data 
#In this specific one, the first 15 columns are the torso vposition alues and the next 18 are the leg positions 

def perform_pca(data):
    pca_data = []  # Initialize list to store PCA weights and components
    pca_contribution=[]
    df_standardd=[]
    
    for patient_id, df, num_rows in data:  # df=Data Frame  
        scaler = StandardScaler()
        df_standard = scaler.fit_transform(df)
        df_standardd.append((patient_id, df_standard))
        
        pca = PCA()  # Create PCA object
        pca.fit(df_standard)
        
        pca_weights = pca.explained_variance_
        pca_components = pca.components_
        pc_values_average = pca.transform(df_standard)
        
        pc_values_torso=np.dot(df_standard[:, :15], pca_components[:,:15].T) 

        pc_values_legs=np.dot(df_standard[:, -18:], pca_components[:,-18:].T)
        pca_contribution.append((patient_id,pc_values_torso,pc_values_legs))
        #if patient_id.startswith("Controlx002"):
            #TEST2=np.dot(df_standard, pca_components.T) #correct, same as PC_VALUES_AVERAGE
            #print(TEST2)
        
        pca_data.append((patient_id, pca_weights, pca_components, pc_values_average))
       
    return pca_data, pca_contribution, df_standardd

#EXPLANATION OF THESE VALUES (33 TOTAL POSITIONS FOR KINECT DATA, 11 3D JOINTS)    
#PCA_data (pca_weights): index 1 is 33, indicating the weight  for all 33 PCs,how much each data point aligns with each principal component
#PCA_data (pca_components): index 2 is 33x33, with each row corresponding to a PC, AND each column corresponds to original joint data in order listed 
        # The values in pca_components indicate the weights of each joint to the corresponding principal component.
#PCA_data (pc_values): index 3 is 200x33, which is the transformed data (33 joints) at each time point (200 time points)



def organize_pc_values(pca_data_average): #organize so X,Y,Z (3D) for each joint 
#The data needed to be organzied like this for the Procrustes step (GPA)
    organized_pc_values=[] # Initialize 

    for patient_id, _, _, pc_values in pca_data_average:
        num_time_points = pc_values.shape[0]
        pc_values_reshaped = pc_values.reshape(num_time_points, 30, 3)
        organized_pc_values.append((patient_id, pc_values_reshaped))

    return organized_pc_values


###############################################################################
#         Generalized Procrystes Analysis (on the patient's PC weights)       #
###############################################################################

#Perform GPA on the transformed data (pc_values) in 3D space 
#Here, I am performing GPA by using the Control Data as the refernce frame
#This can be also be changed to use all data as reference frame 

#PROCRUSTES ANALYSIS 
def perform_gpa(organized_pc_values):
    # Take the data for each patient at t=0
    first_slices = []
    first_slices_c = []

    for patient_id, pc_values_reshaped in organized_pc_values:
        first_slice = pc_values_reshaped[0]  # Extract the first slice (t=0)
        first_slices.append(first_slice)
        if patient_id.startswith("C"):  # Check if patient ID starts with "C"
            first_slices_c.append(first_slice)

    # Stack the first slices into a single array
    first_slices_stacked_c = np.stack(first_slices_c, axis=0)


    # Calculate the mean data for the first slices
    mean_data = np.mean(first_slices_stacked_c, axis=0)  # 11x3 (11 joints in 3D space)

    # Initialize lists to store the results
    aligned_pca_data_t0 = []
    aligned_pca_data = []
    transformation_matrices=[]
    total_aligned_pca_data=[]

    # Perform Procrustes analysis on each patient's data
    for patient_index in range(len(organized_pc_values)):
        transformation, aligned_pca_data, _= procrustes(mean_data, first_slices[patient_index])
        transformation_matrix=transformation
        transformation_matrix = np.array(transformation_matrix).reshape(30, 3)
        aligned_pca_data_t0.append(aligned_pca_data)
        transformation_matrices.append(transformation_matrix)
        
        # Extract the patient's data at all time points
        patient_pca_data = organized_pc_values[patient_index][1]

        # Initialize a list to store the transformed patient's data at all time points
        transformed_patient_pca_data_all_time = []

        # Apply the transformation matrix to each time point
        for time_point_pca_data in patient_pca_data:
            transformed_time_point_pca_data = time_point_pca_data * transformation_matrix #11x3 matrix 
            transformed_patient_pca_data_all_time.append(transformed_time_point_pca_data)

        # Append the transformed data for this patient to the list
        total_aligned_pca_data.append(transformed_patient_pca_data_all_time)

    return total_aligned_pca_data
        

def organize_aligned_pca_values(total_aligned_pca_data, pca_data):
   #Now that the data has gone through GPA, I re-organized it to the original state 
    organized_aligned_pca_data = []  # Initialize list to store organized data

    for patient_index, (patient_id, pca_weights, pca_components, pc_values) in enumerate(pca_data):
        aligned_pca_data = total_aligned_pca_data[patient_index]
        aligned_pca_data = np.array(aligned_pca_data)  # Convert the list to a NumPy array

        # Reshape the aligned PCA weights to (200, 33) matrix as required
        num_time_points = aligned_pca_data.shape[0]
        aligned_pca_data_reshaped = np.empty((num_time_points, 90))

        # Iterate over the 11 joints and fill the aligned_pca_weights_reshaped matrix
        for joint_index in range(30):
            start_col = joint_index * 3
            end_col = start_col + 3
            aligned_pca_data_reshaped[:, start_col:end_col] = aligned_pca_data[:, joint_index, :]

        organized_aligned_pca_data.append((patient_id, pca_weights, pca_components, aligned_pca_data_reshaped))

    return organized_aligned_pca_data


#Now, the data has been transformed (PCA) and aligned (GPA)
#ORGANIZED_ALIGNED_PCA_DATA HAS:
#PCA_data (pca_weights): index 1 is 33, indicating the weight  for all 33 PCs,how much each data point aligns with each principal component
#PCA_data (pca_components): index 2 is 33x33, with each row corresponding to a PC, AND each column corresponds to original joint data in order listed 
        # The values in pca_components indicate the weights of each joint to the corresponding principal component.
#PCA_data (pc_values): index 3 is 100x33, which is the transformed data (33 joints) at each time point (100 time points)


###############################################################################
#               Contrution (LEGS & TORSO)                                     #
###############################################################################
#This portion is specifically if you want to calculate K-Scores seperately for the legs and torso

def calculate_contribution(pca_contribution, pca_data):
    ka_contribution = []
    for patient_id, pca_weights, pca_components, pc_values in pca_data:
        for patient_id1, pc_values_torso, pc_values_legs in pca_contribution:
            if patient_id == patient_id1:
                ka_value_torso = np.dot(pca_weights, pc_values_torso.T)
                ka_value_legs = np.dot(pca_weights, pc_values_legs.T)
                ka_value_fullbody = np.dot(pca_weights, pc_values.T)
                ka_contribution.append((patient_id, ka_value_torso, ka_value_legs, ka_value_fullbody))
    return ka_contribution
###############################################################################
#                              Kinematic Profile                              #
###############################################################################

#Calculates K-sCORE score at every time point using: sum_n(PC_n(t)*lambda_n(t)): 

#This calculates the k-profile (yields a data point at every time point)
def calculate_kprofile(organized_aligned_pca_data, df_standard):
    k_profile = []
    pc_values_test=[] 
    pc_values_AVERAGE=[]
    num_patients = 0
    accumulated_pca_components = np.zeros_like(organized_aligned_pca_data[0][2])
    num_patients += 1

    for patient_id, pca_weights, pca_components, pc_values in organized_aligned_pca_data:
        accumulated_pca_components += pca_components
        num_patients += 1
    average_pca_components = accumulated_pca_components / num_patients

    for patient_id1, data in df_standard:
        scaler = StandardScaler()
        df_standardD = scaler.fit_transform(data)
        pc_values_AVG=(patient_id1, np.dot(df_standardD, average_pca_components.T)) #THIS GIVES SAME RESULTS AS .TRANSFORM BEFORE GPA 
        pc_values_AVERAGE.append(pc_values_AVG)

    for patient_id, pca_weights, pca_components, pc_values in organized_aligned_pca_data:
        for patient_id1, pcAVG in pc_values_AVERAGE:
            if patient_id==patient_id1:
                ka_value=(patient_id, np.dot(pca_weights, pcAVG.T))
        # Check if the first entry in ka_value is not negative
        # Sometimes, it is negative just due to nature of PCA (signs have no significance)
            if ka_value[1][0] <= 0:
                ka_value = (patient_id, -ka_value[1])  # Multiply the entire matrix by -1
        
        k_profile.append(ka_value)   
        
    return k_profile, pc_values_test  #THIS YIELDS A 200, 1 (200 time points for each patient)


def calculate_mean(k_profile):
    # Separate Data 
    control_data =  [(patient_id, score) for patient_id, score in k_profile if patient_id.startswith("C")]
    # Extract the values from each group's data
    control_values = [data[1] for data in control_data]
    
    control_mean = np.mean(control_values, axis=0)
    control_std = np.std(control_values, axis=0)
    
    
    return (control_mean, control_std)

###############################################################################
#                              Kinematic Score (K-Score)                      #
###############################################################################
#Calculates total K-Score score for every activity by area under curve:
def calculate_k_scores(k_profile, control_mean, data):
    total_k_scores = []
    control_length = [tup[2] for tup in data if tup[0].startswith("C")]
    mean_control_length=np.mean(control_length)

#this is to incorporate the time component into the score (how fast is a participant performing activity)
    for patient_id, k_score in k_profile:
        for patient_id1, df, num_rows_per_activity in data:
            if patient_id1 == patient_id:
                patient_num_rows = num_rows_per_activity
                time_component=patient_num_rows/mean_control_length 
                break
                        
        else:
            patient_num_rows=len(k_profile) #367
            
            fixed_length=100
            indices = np.linspace(0, patient_num_rows- 1, fixed_length, dtype=int)
            k_score = k_profile[indices]
            time_component=(patient_num_rows/mean_control_length)  #this is not changing
        
#This is the main component for calculating:6.5 and /1000 are arbitrary depending on how you would like to display the data (just transforming), we can change as neccesarry 
        total_k_score = (6.5* np.sum(np.abs(k_score - control_mean)) * time_component)/1000  
        total_k_score=100-total_k_score #change so 100 is ideal, smaller the worst
        total_k_scores.append((patient_id, total_k_score))
    return (total_k_scores, k_score)


def kdi_by_patient(total_kdi):
    kdi_by_patient = {}  # Dictionary to store KDI scores by patient_ID_activity#

    for patient_id, kdi_score in total_kdi:
        # Extract the patient ID before '_activity'
        patient_id_activity = patient_id.split('_activity')[0]

        if patient_id_activity not in kdi_by_patient:
            # Create a new entry for this patient ID
            kdi_by_patient[patient_id_activity] = []

        # Append the KDI score for this patient ID
        kdi_by_patient[patient_id_activity].append(kdi_score)

    return kdi_by_patient


def standard_deviation_patient(kdi_by_patient):
    std_deviation_by_patient = {}  # Dictionary to store standard deviations by patient_ID_activity#
    std_deviation_c = []  
    std_deviation_LBP = [] 
    kscore_std_avg=[]

    for patient_id_activity, kdi_scores in kdi_by_patient.items():
        # Calculate the standard deviation for the KDI scores
        std_dev = np.std(kdi_scores, ddof=1)  # Use ddof=1 for sample standard deviation

        # Store the standard deviation in the dictionary
        std_deviation_by_patient[patient_id_activity] = std_dev
        if patient_id_activity.startswith("C"):
            std_deviation_c.append(std_dev)
        else:
            std_deviation_LBP.append(std_dev)
            
    kscore_control_avgstd=np.mean(std_deviation_c)
    kscore_control_stdstd=np.std(std_deviation_c)
    kscore_LBP_avgstd=np.mean(std_deviation_LBP)
    kscore_LBP_stdstd=np.std(std_deviation_LBP)
    kscore_std_avg.append(kscore_control_avgstd)
    kscore_std_avg.append(kscore_control_stdstd)
    kscore_std_avg.append(kscore_LBP_avgstd)
    kscore_std_avg.append(kscore_LBP_stdstd)
   
    
    return std_deviation_by_patient, kscore_std_avg



###############################################################################
#                     Enter File & Directory Information HERE                 #
###############################################################################
#EDIT THIS TO YOUR FILES 
csv_files = []

non_random_csv_files = glob.glob('./*.csv') + \
                     glob.glob('./*.csv')

# Filter files that start with '10'
controls = [file for file in non_random_csv_files if file.startswith('./CSV_T2/')]
group_10_files = [file for file in non_random_csv_files if file.startswith('./')]

# Combine all filenames
csv_files = controls + group_10_files

#csv_files = glob.glob(f"/Users/erinarchibeck/Desktop/KDI/Kinect/CSV_files/[ch]*_l_*000.csv")
data = read_csv_files(csv_files)

#Sequence 
pca_data, pca_contribution, datastandard = perform_pca(data)
organized_pc_values =organize_pc_values(pca_data)
total_aligned_pca_data= perform_gpa(organized_pc_values)
aligned_pca_data=organize_aligned_pca_values(total_aligned_pca_data, pca_data)
ka_contribution=calculate_contribution(pca_contribution, aligned_pca_data)
control_Contribution = [(patient_id, ka_value_torso,ka_value_legs) for patient_id, ka_value_torso,ka_value_legs,_ in ka_contribution if patient_id.startswith("C")]
LBP_Contribution = [(patient_id, ka_value_torso,ka_value_legs) for patient_id, ka_value_torso,ka_value_legs,_ in ka_contribution if patient_id.startswith("10")]
k_profile, pc_test_score = calculate_kprofile(aligned_pca_data, datastandard)
control_mean, control_std= calculate_mean(k_profile)
total_kscore, k_score =calculate_k_scores(k_profile, control_mean, data)

plt.plot(np.arange(1,101,1), k_score)
plt.xlabel('Normalised time (%)')
plt.ylabel('K-Score')
plt.savefig('./Out/Figures/' + csv_files[0][2:13] + '_k_score.png')