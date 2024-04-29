import numpy as np
import requests
import tarfile
import gzip
import tempfile
import json
import time
import pickle
import os
import sys
from tqdm import trange
from Bio.PDB import PDBParser, DSSP

def get_structure_count(tar_path):
    count = 0
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".pdb.gz"):
                count += 1
    return count

def stream_structures(tar_path=None):
    if tar_path is None:
        tar_path = input("Enter the path to the tar archive of Alphafold structures: ")

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".pdb.gz"):
                fileobj = tar.extractfile(member)
                if fileobj:
                    with gzip.open(fileobj, "rt", encoding="utf-8") as gz_text:
                        string_data = gz_text.read()

                    # Create a temporary file and write the pdb data to it
                    temp_pdb_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
                    with open(temp_pdb_file.name, "w") as temp_file:
                        temp_file.write(string_data)

                parser = PDBParser()
                base_name = os.path.basename(member.name).split(".pdb.gz")[0]
                structure = parser.get_structure(base_name, temp_pdb_file.name)
                model = structure[0]

                yield (temp_pdb_file.name, model, base_name)

def get_pae(base_name, attempts = 3, verbose = False):
    # Retrieves Predicted Alignment Error (in angstroms) from Alphafold DB, which is not available on FTP (why, Google?)
    v_idx = base_name.index("_v")
    pred_name = base_name[0:v_idx]
    pred_name = pred_name.split("-model")[0]
    suffix = base_name[v_idx:].split(".pdb")[0]
    pae_url = f"https://alphafold.ebi.ac.uk/files/{pred_name}-predicted_aligned_error{suffix}.json"
    print(f"\tRequesting {pae_url}") if verbose else None

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    pae = None
    status_code = None
    for i in np.arange(attempts):
        response = requests.get(pae_url, headers=headers)
        status_code = response.status_code
        if status_code == 200:
            pae_data = json.loads(response.text)
            pae = np.array(pae_data[0]["predicted_aligned_error"])
            break
        elif status_code == 404:
            break
        else:
            print(f"\tHTTP Error: {response.status_code} (Attempt #{i+1})")

    if status_code not in [200, 404]:
        raise Exception(f"\tUnexpected HTTP status code from final attempt: {status_code}")

    return pae, status_code

def run_dssp(tar_path, dssp_executable="/usr/bin/dssp", forbidden_codes = ("H","B","E","G","I","T"),
             retrieve_pae = True, pae_thres = 5, verbose = False):

    structure_count = get_structure_count(tar_path)
    dssp_results = {}
    excluded_results = {}

    with trange(structure_count, desc = f"Running DSSP for structures in tar archivee...") as pbar:
        for i, (pdb_file, model, base_name) in enumerate(stream_structures(tar_path)):
            dssp = DSSP(model, pdb_file, dssp=dssp_executable)
            dssp_vals = list(dssp.property_dict.values())
            dssp_codes = [val[2] for val in dssp_vals]
            forbidden_dssp_mask = np.isin(dssp_codes, forbidden_codes)

            high_confidence_forbidden = forbidden_dssp_mask
            if retrieve_pae:
                pae, status_code = get_pae(base_name, verbose=verbose)
                if pae is not None:
                    pae_mask = np.less_equal(pae, pae_thres)
                    high_confidence_forbidden = np.logical_and(pae_mask, forbidden_dssp_mask)

            excluded_results[base_name] = high_confidence_forbidden

            dssp_results[base_name] = (dssp, pae)
            os.unlink(pdb_file)  # Remove the temporary file after use

            pbar.update()

    return dssp_results, excluded_results

if __name__ == "__main__":
    tar_path = input("Enter the path to the tar archive of Alphafold structures:  ")
    dssp_results, excluded_results = run_dssp(tar_path)
    with open("alphadssp_excluded_results.pkl", "wb") as file:
        pickle.dump((dssp_results, excluded_results), file)