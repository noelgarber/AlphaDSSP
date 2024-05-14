import numpy as np
import tarfile
import gzip
import tempfile
import json
import pickle
import os
from multiprocessing import Pool
from functools import partial
from tqdm import trange
from Bio.PDB import MMCIFParser, DSSP

def get_structure_count(tar_paths, extension = ".cif.gz"):
    count = 0
    if isinstance(tar_paths, str):
        tar_paths = [tar_paths]

    for path in tar_paths:
        with tarfile.open(path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(extension):
                    count += 1

    return count

def stream_structures(tar_paths):
    # Prompt user for paths if none are passed
    if tar_paths is None:
        tar_paths = []
        while True:
            if len(tar_paths) == 0:
                prompt = "Enter path to alphafold tar file or shard:  "
            else:
                prompt = "Enter another path, or hit enter when done:  "
            path = input(prompt)
            if path != "":
                tar_paths.append(path)
            else:
                break

    # Stream the structures as pairs of mmCIF and JSON files
    for tar_path in tar_paths:
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".cif.gz"):
                    # Extract the mmCIF file
                    fileobj = tar.extractfile(member)
                    if fileobj:
                        with gzip.open(fileobj, "rt", encoding="utf-8") as gz_text:
                            string_data = gz_text.read()

                        # Create a temporary file and write the pdb data to it
                        temp_cif_file = tempfile.NamedTemporaryFile(delete=False, suffix=".cif")
                        with open(temp_cif_file.name, "w") as temp_file:
                            temp_file.write(string_data)

                    # Parse the file into a structure
                    parser = MMCIFParser()
                    base_name = os.path.basename(member.name).split(".cif.gz")[0]
                    structure = parser.get_structure(base_name, temp_cif_file.name)
                    model = structure[0]

                    # Extract the accompanying JSON file
                    confidence_json_name = base_name + ".json.gz"
                    confidence_json_name = confidence_json_name.replace("model", "confidence")
                    confidence_json_file = tar.extractfile(confidence_json_name)
                    with gzip.open(confidence_json_file, "rt") as gzip_file:
                        confidence_dict = json.load(gzip_file)
                        confidence_vals = confidence_dict["confidenceScore"]
                        confidence_vals = np.array(confidence_vals)

                    yield (temp_cif_file.name, model, base_name, confidence_vals)

def run_dssp(entry_tuple, dssp_executable="/usr/bin/dssp", forbidden_codes = ("H","B","E","G","I","T"), plddt_thres=70):
    # Function for single entry
    cif_file, model, base_name, plddt_vals = entry_tuple
    dssp = DSSP(model, cif_file, dssp=dssp_executable)
    dssp_codes = [val[2] for val in list(dssp.property_dict.values())]
    forbidden_dssp_mask = np.isin(dssp_codes, forbidden_codes)

    # Only consider forbidden secondary structures if model confidence passes a given threshold
    plddt_mask = np.greater_equal(plddt_vals, plddt_thres)
    high_confidence_forbidden = np.logical_and(plddt_mask, forbidden_dssp_mask)
    results = (high_confidence_forbidden, "".join(dssp_codes))
    os.unlink(cif_file)  # Remove the temporary file after use

    return (base_name, results)

def run_dssp_parallel(tar_paths, dssp_executable="/usr/bin/dssp", forbidden_codes = ("H","B","E","G","I","T"),
                      plddt_thres=70):

    print(f"Getting structure count...")
    structure_count = get_structure_count(tar_paths)
    excluded_results = {}

    processes = os.cpu_count() - 1
    pool = Pool(processes=processes)
    func = partial(run_dssp, dssp_executable = dssp_executable, forbidden_codes = forbidden_codes,
                   plddt_thres = plddt_thres)

    with trange(structure_count, desc = f"Running DSSP for structures in tar archive...") as pbar:
        for i, output in enumerate(pool.imap_unordered(func, stream_structures(tar_paths))):
            base_name, results = output
            excluded_results[base_name] = results
            pbar.update()

    return excluded_results

current_file_path = __file__
directory_path = os.path.dirname(os.path.abspath(current_file_path))
alphadssp_path = os.path.join(directory_path, "alphadssp_excluded_results.pkl")

def generate_dssp(tar_dir = None, dssp_executable="/usr/bin/dssp", forbidden_codes = ("H","B","E","G","I","T"), plddt_thres=70):
    if os.path.exists(alphadssp_path):
        with open(alphadssp_path, "rb") as file:
            excluded_results = pickle.load(file)
    else:
        if tar_dir is None:
            tar_dir = input("Enter the path to the folder containing tar shards of Alphafold structures:  ")
        tar_paths = [os.path.join(tar_dir, filename) for filename in os.listdir(tar_dir)]
        excluded_results = run_dssp_parallel(tar_paths, dssp_executable, forbidden_codes, plddt_thres)
        with open("../alphadssp_excluded_results.pkl", "wb") as file:
            pickle.dump(excluded_results, file)

    return excluded_results

if __name__ == "__main__":
    generate_dssp()