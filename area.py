#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:08:43 2025
@author: Leela Srinivasan

Functions to read and manipulate AFNI SurfClust outputs
"""

import os
import sys
import re
import pandas as pd


def get_fs_path(subj, deriv_dir):
    """

    Parameters
    ----------
    subj : str
        subject with FreeSurfer recon-all run

    Returns
    -------
    subj_fs_dir : str
        path to subject specific FreeSurfer folder.
    session : str
        clinical/altclinical.

    """
    for file in os.listdir(deriv_dir):
        if subj+'_' in file:
            subj_fs_dir=os.path.join(deriv_dir,file)
            if 'alt' in file:
                session='ses-altclinical'
            else:
                session='ses-clinical'
    return subj_fs_dir, session

                
def list_to_textfile(path, input_list):
    """

    Parameters
    ----------
    path : str
        desired path.
    input_list : list
        desired list.

    Returns
    -------
    None.

    """
    with open(path, 'w') as f:
        for line in input_list:
            f.write(f"{line}\n")

              
def get_subject_info(f):
    """
    

    Parameters
    ----------
    f : str
        path to file.

    Returns
    -------
    subject_info : list
        list of subject info.

    """
    col=pd.read_csv(f, header=None)[0]
    subject_info=[eval(x) for x in col]
    return subject_info
    

def split_row(input_string):
    """

    Parameters
    ----------
    input_string : str
        AFNI Column header string.

    Returns
    -------
    list
        AFNI Column names, split from original header.

    """
    return [x for x in re.split('  ', input_string) if x not in ['', ' ']]


def verify_clusters(f):
    """

    Parameters
    ----------
    f : str
        Path to txt file output.

    Returns
    -------
    bool
        True if clusters exist, False if not.

    """
    
    with open(f, "r") as text_file:
        contents=text_file.readlines()
        if contents[0]=='Empty cluster list.\n':
            return False
    return True
        
        
def afnisummary_to_df(f):
    """

    Parameters
    ----------
    f : str
        Path to txt file output.

    Returns
    -------
    df : df
        df containing table information from AFNI text report.
    total_voxels : str
        String integer value containing total PVS voxels in the nifti volume.

    """
    
    #Read table, skip opening lines and remove unwanted hashed lines, preserving summary footer
    table=pd.read_csv(f, delimiter='\t', skiprows=25)
    
    #Append split strings to new df with corresponding columns
    df=pd.DataFrame(columns=split_row(table.columns[0]))
    for i in range(0, len(table)):
        df.loc[i]=split_row(table.loc[i, :].values[0])
        
    return df


def df_to_csv(df, outname):
    """

    Parameters
    ----------
    df : df
        df containing table information from AFNI text report.
    outname : str
        path to output file..

    Returns
    -------
    None.

    """
    df.to_csv(outname)


