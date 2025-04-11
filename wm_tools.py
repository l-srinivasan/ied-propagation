#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri Sep 20 15:23:28 2024
@author: Leela Srinivasan

Functions to aid WM analysis
"""

import os
import subprocess
import pickle
import pandas as pd
import numpy as np
import mne
mne.set_log_level(False)
import mne.channels

     
def map_parcels_to_networks(filename):
    """
    

    Parameters
    ----------
    filename : pickle file
        file containing Schaefer parcel names.

    Returns
    -------
    None.

    """    
    # Load parcel labels and map them to Yeo networks
    USER_DIR='path_to_labels'
    filename = USER_DIR+'schaefer_400_parcel_labels.pickle'
    with open(filename, 'rb') as f: 
        parcel_names = pickle.load(f)
    network_map = {'FPN': parcel_names[0:22] + parcel_names[200:230], # Frontoparietal
                   'DMN': parcel_names[22:74] + parcel_names[230:269], # Default Mode Network
                   'DAN': parcel_names[74:97] + parcel_names[269:292], # Dorsal Attention Network
                   'LIN': parcel_names[97:110] + parcel_names[292:305], # Limbic
                   'SAN': parcel_names[110:132] + parcel_names[305:330], # Salience
                   'SMN': parcel_names[132:169] + parcel_names[330:370], # Somatomotor
                   'VIN': parcel_names[169:200] + parcel_names[370:400], # Visual
                   'Medial Wall': parcel_names[400:402]}
    return network_map, parcel_names


def mark_source_and_sink_activity(df, source_cluster, sink_cluster, suma2mne, hemi):
    """
    

    Parameters
    ----------
    df : df
        sliced df containing sensors that could be involved in white matter connections (thresholded by timing).
    source_cluster : int
        source cluster number.
    sink_cluster : int
        sink cluster number.
    suma2mne : df
        df with relationship betwen SUMA XYZ vertices and MNE virtual sensors.
    hemi : string
        hemisphere (lh/rh).

    Returns
    -------
    None.

    """    
    # Convert merged clusters file to 1D file
    cmd="ConvertDset -o_1D -input {} -prefix {}"
    cmd=cmd.format('colored_clusters.gii','to_modify')
    subprocess.run(cmd,shell=True)


    # Subset desired pairing
    f1='to_modify.1D.dset'
    to_modify=pd.DataFrame(np.genfromtxt(f1).astype(int))
    subset_df = df[df['Cluster Pair'] == (1,2)]
    
    
    # Modify source nodes to 2 to demonstrate strength/involvement in white matter connections
    for sensor in subset_df['Source Sensor'].values:
        corresp_nodes = suma2mne[suma2mne.virtual_sensor_node == float(sensor)].SUMA_vertex.values
        to_modify.loc[corresp_nodes, 0] = 2
        
        
    # Modify sink nodes to 1 demonstrate strength/involvement in white matter connections
    for sensor in subset_df['Destination Sensor'].values:
        corresp_nodes = suma2mne[suma2mne.virtual_sensor_node == float(sensor)].SUMA_vertex.values
        to_modify.loc[corresp_nodes, 0] = 1
    
    
    #Convert back to gifti for SUMA viewing
    f2='modified_merge.1D.dset'
    to_modify[0].to_csv(f2, index=False, header=False)
    cmd="ConvertDset -o_gii -input {} -prefix {}"
    cmd=cmd.format(f2,'mod_{}_merged_clusters'.format(hemi))
    subprocess.run(cmd,shell=True)
    
    
    #Clear temporary files
    os.remove(f1)
    os.remove(f2)
    

def generate_parcel_vertex_mapping(freesurfer_dir, wdir, stc, subject, session):
    """
    

    Parameters
    ----------
    freesurfer_dir : string
        path to freesurfer directory for all subjects.
    wdir : string
        path to working directory/surface_clusters directory.
    stc : mne object
        source reconstructed time series data.
    subject : string
        p***.
    session : string
        clinical/altclinical.

    Returns
    -------
    None.

    """
    SENSORS_PER_HEMI=2562
    
    
    #Alter naming convention to BIDS for MNE
    os.rename(os.path.join(freesurfer_dir,'sub-'+subject+'_'+session),os.path.join(freesurfer_dir,subject))
    for hemi in ['lh', 'rh']:

        
        #Read cortical parcellation labels from a FreeSurfer annotation file
        parcellation_type = 'Schaefer2018_400Parcels_7Networks_order'
        read_labels = mne.read_labels_from_annot('fsaverage', parc = parcellation_type, subjects_dir=freesurfer_dir)
        subject_labels = mne.morph_labels(read_labels, subject_to = subject,subjects_dir = freesurfer_dir)


        # Iterate over each parcel in subject_labels, finding vertices corresponding to the label vertices
        parcel_vertex_mapping = {}
        for parcel_num, label in enumerate(subject_labels):
            idx = np.nonzero(np.isin(stc.vertices[0 if hemi == 'lh' else 1], label.vertices))[0] 
            parcel_vertex_mapping[parcel_num] = [sensor + SENSORS_PER_HEMI if hemi == 'rh' else sensor for sensor in idx]
            
            
        # Save dictionary mapping to dSPM folder
        np.save(os.path.join(wdir, "{}_parcel_vertex_mapping.npy".format(hemi)), parcel_vertex_mapping)


    # Restore Freesurfer folder naming convention
    os.rename(os.path.join(freesurfer_dir,subject),os.path.join(freesurfer_dir,'sub-'+subject+'_'+session))
